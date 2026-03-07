import os
import sys
import time
import signal
import atexit
import threading
import numpy as np
import onnxruntime as ort
from abc import ABC, abstractmethod
from pathlib import Path

from bdx_api.config import load_policy_config, RobotPolicyConfig
from bdx_api.logger import RuntimeLogger


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
# Keyboard Controller (works over SSH)
# ==========================================

class KeyboardController:
    """Non-blocking keyboard controller using raw terminal input.

    Works over SSH without X11. Falls back to pynput if available
    and a display is present.

    Controls:
        W / S       - Increase / Decrease forward speed
        A / D       - Increase / Decrease lateral speed (strafe)
        Q / E       - Turn Left / Right
        SPACE       - Stop all movement
        ESC / Ctrl+C - Quit
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
        self._lock = threading.Lock()
        self._old_term_settings = None

        # Start background key reader
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """Background thread: reads keypresses from stdin."""
        import tty
        import termios
        import select

        fd = sys.stdin.fileno()
        self._old_term_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._quit:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        # Could be ESC or arrow key sequence
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                # Arrow keys: A=up B=down C=right D=left
                                if ch3 == 'A':
                                    ch = 'w'
                                elif ch3 == 'B':
                                    ch = 's'
                                elif ch3 == 'D':
                                    ch = 'a'
                                elif ch3 == 'C':
                                    ch = 'd'
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # Plain ESC
                            self._quit = True
                            break
                    elif ch == '\x03':  # Ctrl+C
                        self._quit = True
                        break

                    self._handle_key(ch.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)

    def _handle_key(self, ch):
        with self._lock:
            if ch == 'w':
                self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_step, self.lin_vel_x_range[1])
            elif ch == 's':
                self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_step, self.lin_vel_x_range[0])
            elif ch == 'a':
                self.lin_vel_y = min(self.lin_vel_y + self.lin_vel_step, self.lin_vel_y_range[1])
            elif ch == 'd':
                self.lin_vel_y = max(self.lin_vel_y - self.lin_vel_step, self.lin_vel_y_range[0])
            elif ch == 'q':
                self.ang_vel_yaw = min(self.ang_vel_yaw + self.ang_vel_step, self.ang_vel_z_range[1])
            elif ch == 'e':
                self.ang_vel_yaw = max(self.ang_vel_yaw - self.ang_vel_step, self.ang_vel_z_range[0])
            elif ch == ' ':
                self.lin_vel_x = 0.0
                self.lin_vel_y = 0.0
                self.ang_vel_yaw = 0.0

    def restore_terminal(self):
        """Restore terminal settings. Safe to call multiple times."""
        if self._old_term_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term_settings)
            except Exception:
                pass

    @property
    def should_quit(self):
        return self._quit

    def get_command(self):
        with self._lock:
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
        ...

    @abstractmethod
    def get_projected_gravity(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def get_joint_map(self, joint_names: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def get_default_pos_array(self, num_actuators: int) -> np.ndarray:
        ...

    @abstractmethod
    def standup(self, duration: float = 2.0) -> None:
        """Smoothly move the robot to its default standing pose with safe gains."""
        ...

    @abstractmethod
    def hold_standing_tick(self) -> None:
        """One control tick holding the standing pose with safe gains."""
        ...

    @abstractmethod
    def activate_policy_gains(self) -> None:
        """Switch from safe standing gains to trained policy Kp/Kd."""
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

        # Logger
        self.logger = RuntimeLogger(joint_names=self.cfg.joint_names)
        self._interrupted = False
        self._sigint_count = 0

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

    def _handle_sigint(self, signum, frame):
        """Handle Ctrl+C gracefully. Double Ctrl+C force-quits."""
        self._sigint_count += 1
        if self._sigint_count >= 2:
            print("\n\n  Double Ctrl+C — force quitting!")
            self._cleanup()
            os._exit(1)
        print("\n\n  Ctrl+C received — stopping and plotting...")
        print("  (Press Ctrl+C again to force-quit)")
        self._interrupted = True

    def _cleanup(self):
        """Stop the keyboard listener and restore terminal."""
        try:
            self.kb.restore_terminal()
        except Exception:
            pass

    def run(self):
        """Main control loop."""
        # Register Ctrl+C handler
        old_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        atexit.register(self._cleanup)

        # === Stage 1: Stand up ===
        print("\n" + "=" * 50)
        print("  READY TO STAND")
        print("=" * 50)
        input(">>> Press ENTER to stand up (safe standing gains)...")
        self.backend.standup(duration=2.0)

        # === Stage 2: Hold standing pose ===
        print("\nRobot is standing with safe gains. Place it on the floor.")
        print(">>> Press ENTER to deploy the policy (switches to trained Kp/Kd)...")

        enter_pressed = threading.Event()
        def _wait_for_enter():
            input()
            enter_pressed.set()
        waiter = threading.Thread(target=_wait_for_enter, daemon=True)
        waiter.start()

        while not enter_pressed.is_set() and self.backend.is_running():
            self.backend.hold_standing_tick()

        # === Stage 3: Deploy policy ===
        self.backend.activate_policy_gains()
        print("\n" + "=" * 50)
        print("  DEPLOYING POLICY WITH TRAINED GAINS")
        print("-" * 50)
        for i, name in enumerate(self.cfg.joint_names):
            print(f"  {name:20s}  Kp={self.cfg.joint_stiffness[i]:8.3f}  Kd={self.cfg.joint_damping[i]:8.3f}")
        print("=" * 50)

        it = 0
        
        # [DEBUG] Policy Frequency Tracking
        policy_counter = 0
        policy_start_time = time.time()
        actual_policy_freq = 0.0

        print("=" * 50)
        print("  KEYBOARD CONTROLS")
        print("-" * 50)
        print("  W / S       Forward / Backward")
        print("  A / D       Strafe Left / Right")
        print("  Q / E       Turn Left / Right")
        print("  SPACE       Stop all movement")
        print("  ESC         Quit (no plot)")
        print("  Ctrl+C      Quit + Plot graphs")
        print("=" * 50)
        print("\n") # Spacing for the updating line

        while self.backend.is_running() and not self.kb.should_quit and not self._interrupted:
            if it % self.decimation == 0:
                cmd = self.kb.get_command()

                # Read sensors
                ang_vel = self.backend.get_imu_angular_velocity()
                projected_gravity = self.backend.get_projected_gravity()
                dof_pos = self.backend.get_joint_positions(self.joint_map)
                dof_vel = self.backend.get_joint_velocities(self.joint_map)

                # Build observation
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

                obs = np.concatenate(obs_parts).astype(np.float32)

                # Run policy
                self.actions = self.session.run(None, {self.input_name: obs.reshape(1, -1)})[0][0]
                self.dof_targets[self.joint_map] = (
                    self.default_pos[self.joint_map] + self.action_scale * self.actions
                )

                # Log data
                self.logger.log(
                    cmd=cmd,
                    target_pos=self.dof_targets[self.joint_map],
                    actual_pos=dof_pos,
                    actual_vel=dof_vel,
                    ang_vel=ang_vel,
                    projected_gravity=projected_gravity,
                    actions=self.actions,
                )

                # ========================================================
                # [DEBUG] Calc Policy Freq & Update Print
                # ========================================================
                policy_counter += 1
                if policy_counter >= 10:  # Calc every 10 inferences (approx 0.2s)
                    now = time.time()
                    actual_policy_freq = policy_counter / (now - policy_start_time)
                    policy_counter = 0
                    policy_start_time = now
                
                # Get Backend Freq (Joint Cmds)
                backend_freq = 0.0
                if hasattr(self.backend, 'get_actual_frequency'):
                    backend_freq = self.backend.get_actual_frequency()

                # Print Status (updates in place)
                # We print every decimation step so it looks smooth
                status_str = (
                    f"Cmds(Backend): {backend_freq:5.1f} Hz | "
                    f"Policy: {actual_policy_freq:5.1f} Hz | "
                    f"CMD [X:{cmd[0]:+0.1f} Y:{cmd[1]:+0.1f} W:{cmd[2]:+0.1f}]"
                )
                
                sys.stdout.write(f"\r{status_str}   ") # Spaces to clear end of line
                sys.stdout.flush()
                # ========================================================

            self.backend.set_joint_targets(self.dof_targets)
            self.backend.step()
            it += 1

        # Cleanup
        self._cleanup()

        # Restore old signal handler
        signal.signal(signal.SIGINT, old_handler)

        print(f"\n\nRun complete. {len(self.logger.entries)} samples logged.")

        if self._interrupted and len(self.logger.entries) > 1:
            print("\nGenerating plots...")
            self.logger.plot(save_dir="logs")
        elif len(self.logger.entries) > 1:
            print("(Press Ctrl+C next time to auto-plot, or run logger.plot() manually)")

        print("Shutting down.")