import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from bdx_api.config import load_policy_config, STANDUP_GAINS
from bdx_api.interface import RobotBackend, quat_rotate_inverse


class MujocoBackend(RobotBackend):
    """MuJoCo simulation backend."""

    def __init__(self, xml_path: str, model_path: Path, sim_dt: float = 0.005,
                 fixed: bool = False):
        self.cfg = load_policy_config(model_path)
        self.fixed = fixed

        # --- Build model via MjSpec (rewrite actuators to PD control) ---
        spec = mujoco.MjSpec.from_file(xml_path)

        for i, name in enumerate(self.cfg.joint_names):
            name = name.strip()
            kp = self.cfg.joint_stiffness[i]
            kd = self.cfg.joint_damping[i]

            act = None
            for a in spec.actuators:
                if a.name == name:
                    act = a
                    break

            if act is None:
                raise ValueError(f"Actuator '{name}' not found in {xml_path}")

            try:
                if act.ctrllimited:
                    act.forcerange = act.ctrlrange
                    act.forcelimited = True
                    act.ctrllimited = False
            except Exception:
                pass

            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = kp
            act.biasprm[0] = 0.0
            act.biasprm[1] = -kp
            act.biasprm[2] = -kd

        self.model = spec.compile()
        self.model.opt.timestep = sim_dt

        # If fixed, disable gravity so robot floats in place
        if fixed:
            self.model.opt.gravity[:] = [0, 0, 0]

        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        # Set initial floating pose (freejoint still exists)
        self.data.qpos[:3] = [0.0, 0.0, 0.33]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Apply default joint positions
        joint_map = self.get_joint_map(self.cfg.joint_names)
        for i, mj_id in enumerate(joint_map):
            self.data.qpos[7 + mj_id] = self.cfg.default_joint_pos[i]

        mujoco.mj_forward(self.model, self.data)

        # If fixed, lock the base in place by zeroing its velocities each step
        self._lock_base = fixed

        # Launch viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.cam.elevation = -20

        # --- Store actuator index map for runtime gain swapping ---
        self._act_ids = {}  # joint_name -> mujoco actuator index
        for name in self.cfg.joint_names:
            name = name.strip()
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._act_ids[name] = aid

        # Start with standup gains so the robot is stiff from the start
        self._apply_standup_gains()

    # --- Gain swapping helpers ---

    def _apply_standup_gains(self):
        """Set MuJoCo actuator gains to the stiff STANDUP_GAINS."""
        for i, name in enumerate(self.cfg.joint_names):
            name = name.strip()
            aid = self._act_ids[name]
            if name in STANDUP_GAINS:
                kp, kd = STANDUP_GAINS[name]
            else:
                kp = self.cfg.joint_stiffness[i]
                kd = self.cfg.joint_damping[i]
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kd

    def _apply_policy_gains(self):
        """Set MuJoCo actuator gains to the trained ONNX Kp/Kd."""
        for i, name in enumerate(self.cfg.joint_names):
            name = name.strip()
            aid = self._act_ids[name]
            kp = self.cfg.joint_stiffness[i]
            kd = self.cfg.joint_damping[i]
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kd

    # --- RobotBackend implementation ---

    def get_imu_angular_velocity(self) -> np.ndarray:
        if self.fixed:
            return np.zeros(3, dtype=np.float32)
        return self.data.sensor("imu_ang_vel").data.astype(np.float32)

    def get_projected_gravity(self) -> np.ndarray:
        if self.fixed:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        wxyz = self.data.qpos[3:7].astype(np.float32)
        quat_xyzw = wxyz[[1, 2, 3, 0]]
        return quat_rotate_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0])).astype(np.float32)

    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        return self.data.qpos[7:].astype(np.float32)[joint_ids]

    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        return self.data.qvel[6:].astype(np.float32)[joint_ids]

    def set_joint_targets(self, targets: np.ndarray) -> None:
        self.data.ctrl[:] = targets

    def step(self) -> None:
        if self._lock_base:
            # Keep base position and orientation fixed
            self.data.qpos[:3] = [0.0, 0.0, 0.33]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[:6] = 0.0
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def get_joint_map(self, joint_names: list[str]) -> np.ndarray:
        ids = []
        for name in joint_names:
            name = name.strip()
            mj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mj_id == -1:
                raise ValueError(f"Actuator '{name}' not found in MuJoCo model")
            ids.append(mj_id)
        return np.array(ids, dtype=int)

    def get_default_pos_array(self, num_actuators: int) -> np.ndarray:
        return np.zeros(self.model.nu, dtype=np.float32)

    # ==========================================
    # Standup / Hold / Deploy (called by PolicyRunner)
    # ==========================================

    def standup(self, duration: float = 2.0) -> None:
        """Hold default pose for `duration` seconds to let the sim settle."""
        import time as _time
        print(f"Standing up (settling sim for {duration}s)...")
        default_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        joint_map = self.get_joint_map(self.cfg.joint_names)
        for i, mj_id in enumerate(joint_map):
            default_ctrl[mj_id] = self.cfg.default_joint_pos[i]
        self.data.ctrl[:] = default_ctrl

        num_steps = int(duration / self.model.opt.timestep)
        for _ in range(num_steps):
            self.step()
        print("Standup complete.")

    def hold_standing_tick(self) -> None:
        """One sim step holding the default pose."""
        self.step()

    def activate_policy_gains(self) -> None:
        """Switch MuJoCo actuator gains from standup to trained policy values."""
        self._apply_policy_gains()