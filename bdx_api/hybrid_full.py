"""
Full hybrid backend: MuJoCo rendering + ALL real robot data.

Overrides joint positions, joint velocities, gyro, and projected gravity
with live data received over UDP from robot_sender.py on the Jetson.
MuJoCo is used only for visualization (setting qpos to match the real robot).
"""

import atexit
import numpy as np
import mujoco
from pathlib import Path

from bdx_api.sim import MujocoBackend
from bdx_api.robot_receiver import RobotStateReceiver


class HybridFullBackend(MujocoBackend):
    """MuJoCo sim driven entirely by real robot state over UDP."""

    def __init__(
        self,
        xml_path: str,
        model_path: Path,
        sim_dt: float = 0.005,
        fixed: bool = False,
        robot_port: int = 5006,
    ):
        super().__init__(
            xml_path=xml_path,
            model_path=model_path,
            sim_dt=sim_dt,
            fixed=fixed,
        )

        # Disable gravity — we're only visualizing, not simulating physics
        self.model.opt.gravity[:] = [0, 0, 0]

        # Pre-compute MuJoCo actuator IDs for each policy joint (avoid re-lookup every step)
        self._mj_joint_ids = []
        for name in self.cfg.joint_names:
            mj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.strip())
            self._mj_joint_ids.append(mj_id)

        # Start the UDP robot state receiver
        self.robot_receiver = RobotStateReceiver(port=robot_port)
        self.robot_receiver.start()

        # Build mapping from policy joint order → sender packet order
        self._sender_mapping = self.robot_receiver.build_joint_mapping(self.cfg.joint_names)

        self._connected_warned = False
        atexit.register(self.stop_receiver)

    # --- Override ALL sensor methods to use real data ---

    def get_imu_angular_velocity(self) -> np.ndarray:
        if not self.robot_receiver.is_connected():
            if not self._connected_warned:
                print("[HybridFull] WARNING: No robot data yet — using defaults")
                self._connected_warned = True
            return np.zeros(3, dtype=np.float32)
        self._connected_warned = False
        data = self.robot_receiver.get_data()
        return data["gyro"]

    def get_projected_gravity(self) -> np.ndarray:
        if not self.robot_receiver.is_connected():
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        data = self.robot_receiver.get_data()
        return data["projected_gravity"]

    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        if not self.robot_receiver.is_connected():
            return super().get_joint_positions(joint_ids)
        data = self.robot_receiver.get_data()
        all_pos = data["joint_pos"]
        # Map from policy joint order (via joint_ids) to sender packet order
        result = np.zeros(len(joint_ids), dtype=np.float32)
        for i, jid in enumerate(joint_ids):
            sender_idx = self._sender_mapping[jid]
            if sender_idx >= 0:
                result[i] = all_pos[sender_idx]
        return result

    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        if not self.robot_receiver.is_connected():
            return super().get_joint_velocities(joint_ids)
        data = self.robot_receiver.get_data()
        all_vel = data["joint_vel"]
        result = np.zeros(len(joint_ids), dtype=np.float32)
        for i, jid in enumerate(joint_ids):
            sender_idx = self._sender_mapping[jid]
            if sender_idx >= 0:
                result[i] = all_vel[sender_idx]
        return result

    def set_joint_targets(self, targets: np.ndarray) -> None:
        # No-op: we're mirroring real robot state, not actuating in sim
        pass

    def step(self) -> None:
        """Mirror real robot pose into MuJoCo for visualization (no physics)."""
        if self.robot_receiver.is_connected():
            data = self.robot_receiver.get_data()
            all_pos = data["joint_pos"]
            # Set each joint's qpos from real data
            for i, mj_id in enumerate(self._mj_joint_ids):
                sender_idx = self._sender_mapping[i]
                if sender_idx >= 0 and mj_id >= 0:
                    self.data.qpos[7 + mj_id] = all_pos[sender_idx]

        # Lock base in place
        self.data.qpos[:3] = [0.0, 0.0, 0.33]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        # Forward kinematics only — no physics simulation
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def stop_receiver(self):
        """Cleanly shut down the robot state receiver."""
        try:
            self.robot_receiver.stop()
        except Exception:
            pass
