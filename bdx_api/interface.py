import time
import numpy as np
import onnxruntime as ort
from abc import ABC, abstractmethod
from pathlib import Path
from pynput import keyboard

from bdx_api.config import load_policy_config, RobotPolicyConfig


# ==========================================
# Utils
# ==========================================

def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q. q is [x, y, z, w]."""
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


# ==========================================
# Keyboard Controller
# ==========================================

class KeyboardController:
    """Non-blocking keyboard controller using pynput.

    Controls:
        ↑ / ↓      - Increase / Decrease forward speed
        ← / →      - Increase / Decrease lateral speed
        Z / X       - Turn Left / Right
        SPACE       - Stop all movement
        ESC         - Quit
    """

    def __init__(self, lin_vel_step=0.1, ang_vel_step=0.1,
                 lin_vel_x_range=(-1.0, 1.0),
                 lin_vel_y_range=(-0.4, 0.4),
                 ang_vel_z_range=(-1.0, 1.0)):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0

        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range

        self._quit = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.up:
            self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_step, self.lin_vel_x_range[1])
        elif key == keyboard.Key.down:
            self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_step, self.lin_vel_x_range[0])
        elif key == keyboard.Key.left:
            self.lin_vel_y = min(self.lin_vel_y + self.lin_vel_step, self.lin_vel_y_range[1])
        elif key == keyboard.Key.right:
            self.lin_vel_y = max(self.lin_vel_y - self.lin_vel_step, self.lin_vel_y_range[0])
        elif key == keyboard.Key.space:
            self.lin_vel_x = 0.0
            self.lin_vel_y = 0.0
            self.ang_vel_yaw = 0.0
        elif key == keyboard.Key.esc:
            self._quit = True
        else:
            try:
                if key.char == 'z':
                    self.ang_vel_yaw = min(self.ang_vel_yaw + self.ang_vel_step, self.ang_vel_z_range[1])
                elif key.char == 'x':
                    self.ang_vel_yaw = max(self.ang_vel_yaw - self.ang_vel_step, self.ang_vel_z_range[0])
            except AttributeError:
                pass

    @property
    def should_quit(self):
        return self._quit

    def get_command(self):
        return np.array([
            round(self.lin_vel_x, 2),
            round(self.lin_vel_y, 2),
            round(self.ang_vel_yaw, 2),
        ], dtype=np.float32)


# ==========================================
# Abstract Robot Backend
# ==========================================

class RobotBackend(ABC):
    """Abstract interface that both sim and real hardware implement."""

    @abstractmethod
    def get_imu_angular_velocity(self) -> np.ndarray:
        """Returns 3D angular velocity from IMU [rad/s]."""
        ...

    @abstractmethod
    def get_projected_gravity(self) -> np.ndarray:
        """Returns 3D projected gravity vector (unit length, pointing down)."""
        ...

    @abstractmethod
    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        """Returns joint positions [rad] for given joint indices."""
        ...

    @abstractmethod
    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        """Returns joint velocities [rad/s] for given joint indices."""
        ...

    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None:
        """Send position targets to all actuators."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance one timestep (sim) or sleep to maintain loop rate (real)."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Return False to exit the control loop."""
        ...

    @abstractmethod
    def get_joint_map(self, joint_names: list[str]) -> np.ndarray:
        """Map policy joint names → backend actuator indices."""
        ...

    @abstractmethod
    def get_default_pos_array(self, num_actuators: int) -> np.ndarray:
        """Return a zero array sized to the backend's actuator count."""
        ...


# ==========================================
# Policy Runner (Backend-Agnostic)
# ==========================================

class PolicyRunner:
    """Runs the ONNX policy on any RobotBackend."""

    def __init__(self, backend: RobotBackend, model_path: Path, decimation: int = 4):
        self.backend = backend

        # Load config from ONNX metadata
        self.cfg: RobotPolicyConfig = load_policy_config(model_path)

        # Load ONNX runtime
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

        self.decimation = decimation

        # Joint mapping
        self.joint_map = backend.get_joint_map(self.cfg.joint_names)
        self.num_actuators = len(self.joint_map)

        # Default positions (full actuator array)
        self.default_pos = backend.get_default_pos_array(self.num_actuators)
        for i, name in enumerate(self.cfg.joint_names):
            mj_id = self.joint_map[i]
            self.default_pos[mj_id] = self.cfg.default_joint_pos[i]

        self.action_scale = np.array(self.cfg.action_scale, dtype=np.float32)
        self.actions = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
        self.dof_targets = self.default_pos.copy()

        # Keyboard
        self.kb = KeyboardController(
            lin_vel_step=0.1,
            ang_vel_step=0.1,
            lin_vel_x_range=(-1.0, 1.0),
            lin_vel_y_range=(-0.4, 0.4),
            ang_vel_z_range=(-1.0, 1.0),
        )

    def _build_observation(self, cmd: np.ndarray) -> np.ndarray:
        """Build the observation vector from sensor data + command."""
        ang_vel = self.backend.get_imu_angular_velocity()
        projected_gravity = self.backend.get_projected_gravity()

        dof_pos = self.backend.get_joint_positions(self.joint_map)
        dof_vel = self.backend.get_joint_velocities(self.joint_map)

        obs_parts = []
        for term in self.cfg.observation_names:
            term = term.strip()
            if term == "projected_gravity":
                obs_parts.append(projected_gravity)
            elif term == "base_ang_vel":
                obs_parts.append(ang_vel)
            elif term in ("joint_pos", "dof_pos"):
                obs_parts.append(dof_pos - self.default_pos[self.joint_map])
            elif term in ("joint_vel", "dof_vel"):
                obs_parts.append(dof_vel)
            elif term == "actions":
                obs_parts.append(self.actions)
            elif term in ("command", "commands"):
                obs_parts.append(cmd)

        return np.concatenate(obs_parts).astype(np.float32)

    def run(self):
        """Main control loop."""
        it = 0
        print_every = int(1.0 / (0.005 * self.decimation))  # ~once per second

        print("=" * 50)
        print("  KEYBOARD CONTROLS")
        print("-" * 50)
        print("  ↑ / ↓     Forward / Backward")
        print("  ← / →     Strafe Left / Right")
        print("  Z / X     Turn Left / Right")
        print("  SPACE     Stop all movement")
        print("  ESC       Quit")
        print("=" * 50)

        while self.backend.is_running() and not self.kb.should_quit:
            if it % self.decimation == 0:
                cmd = self.kb.get_command()

                if it % (self.decimation * print_every) == 0:
                    print(f"\r  Fwd: {cmd[0]:+.2f}  Lat: {cmd[1]:+.2f}  Yaw: {cmd[2]:+.2f}", end="", flush=True)

                obs = self._build_observation(cmd)
                self.actions = self.session.run(None, {self.input_name: obs.reshape(1, -1)})[0][0]
                self.dof_targets[self.joint_map] = (
                    self.default_pos[self.joint_map] + self.action_scale * self.actions
                )

            self.backend.set_joint_targets(self.dof_targets)
            self.backend.step()
            it += 1

        print("\nShutting down.")