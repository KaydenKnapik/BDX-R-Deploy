import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from bdx_api.config import load_policy_config, STANDUP_GAINS
from bdx_api.interface import RobotBackend, quat_rotate_inverse

class MujocoBackend(RobotBackend):
    def __init__(self, xml_path: str, model_path: Path, sim_dt: float = 0.005,
                 fixed: bool = False, legs_only: bool = False):
        self.cfg = load_policy_config(model_path)
        self.fixed = fixed
        self.legs_only = legs_only

        self.policy_joints = self.cfg.joint_names
        self.num_policy_joints = len(self.policy_joints)
        self.all_joint_names = list(self.policy_joints)

        if self.legs_only:
            head_joints =["Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll"]
            for hj in head_joints:
                if hj not in self.all_joint_names:
                    self.all_joint_names.append(hj)

        spec = mujoco.MjSpec.from_file(xml_path)

        for i, name in enumerate(self.all_joint_names):
            name = name.strip()
            is_head = self.legs_only and i >= self.num_policy_joints

            if is_head:
                kp, kd = STANDUP_GAINS.get(name, (20.0, 1.0))
            else:
                kp = self.cfg.joint_stiffness[i]
                kd = self.cfg.joint_damping[i]

            act = None
            for a in spec.actuators:
                if a.name == name:
                    act = a
                    break

            if act is None:
                if is_head: continue 
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

        if fixed:
            self.model.opt.gravity[:] = [0, 0, 0]

        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:3] =[0.0, 0.0, 0.33]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # --- NEW: Bulletproof QPOS/QVEL mapping mapping ---
        self._act2qpos = {}
        self._act2qvel = {}
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name:
                jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, act_name)
                if jnt_id != -1:
                    self._act2qpos[i] = self.model.jnt_qposadr[jnt_id]
                    self._act2qvel[i] = self.model.jnt_dofadr[jnt_id]

        joint_map = self.get_joint_map(self.policy_joints)
        for i, mj_id in enumerate(joint_map):
            qpos_idx = self._act2qpos.get(mj_id, -1)
            if qpos_idx != -1:
                self.data.qpos[qpos_idx] = self.cfg.default_joint_pos[i]

        mujoco.mj_forward(self.model, self.data)
        self._lock_base = fixed

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=False, show_right_ui=False,
        )
        self.viewer.cam.elevation = -20

        self._act_ids = {}
        for name in self.all_joint_names:
            name = name.strip()
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                self._act_ids[name] = aid

        self._apply_standup_gains()

    def _apply_standup_gains(self):
        for i, name in enumerate(self.all_joint_names):
            name = name.strip()
            if name not in self._act_ids: continue
            aid = self._act_ids[name]

            if name in STANDUP_GAINS:
                kp, kd = STANDUP_GAINS[name]
            else:
                if i < self.num_policy_joints:
                    kp = self.cfg.joint_stiffness[i]
                    kd = self.cfg.joint_damping[i]
                else:
                    kp, kd = 20.0, 1.0

            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kd

    def _apply_policy_gains(self):
        for i, name in enumerate(self.all_joint_names):
            name = name.strip()
            if name not in self._act_ids: continue
            aid = self._act_ids[name]

            is_head = self.legs_only and i >= self.num_policy_joints
            if is_head:
                if name in STANDUP_GAINS:
                    kp, kd = STANDUP_GAINS[name]
                else:
                    kp, kd = 20.0, 1.0
            else:
                kp = self.cfg.joint_stiffness[i]
                kd = self.cfg.joint_damping[i]

            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kd

    def get_imu_angular_velocity(self) -> np.ndarray:
        if self.fixed: return np.zeros(3, dtype=np.float32)
        return self.data.sensor("imu_ang_vel").data.astype(np.float32)

    def get_projected_gravity(self) -> np.ndarray:
        if self.fixed: return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        wxyz = self.data.qpos[3:7].astype(np.float32)
        quat_xyzw = wxyz[[1, 2, 3, 0]]
        return quat_rotate_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0])).astype(np.float32)

    # --- NEW: Safe fetching of joint positions using exact mapped indices ---
    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        res = np.zeros(len(joint_ids), dtype=np.float32)
        for i, act_id in enumerate(joint_ids):
            qpos_idx = self._act2qpos.get(act_id, 0)
            res[i] = self.data.qpos[qpos_idx]
        return res

    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        res = np.zeros(len(joint_ids), dtype=np.float32)
        for i, act_id in enumerate(joint_ids):
            qvel_idx = self._act2qvel.get(act_id, 0)
            res[i] = self.data.qvel[qvel_idx]
        return res

    def set_joint_targets(self, targets: np.ndarray) -> None:
        self.data.ctrl[:] = targets

    def step(self) -> None:
        if self._lock_base:
            self.data.qpos[:3] =[0.0, 0.0, 0.33]
            self.data.qpos[3:7] =[1.0, 0.0, 0.0, 0.0]
            self.data.qvel[:6] = 0.0
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def is_running(self) -> bool: return self.viewer.is_running()

    def get_joint_map(self, joint_names: list[str]) -> np.ndarray:
        ids =[]
        for name in joint_names:
            name = name.strip()
            mj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mj_id == -1:
                raise ValueError(f"Actuator '{name}' not found in MuJoCo model")
            ids.append(mj_id)
        return np.array(ids, dtype=int)

    def get_default_pos_array(self, num_actuators: int) -> np.ndarray:
        return np.zeros(self.model.nu, dtype=np.float32)

    def standup(self, duration: float = 2.0) -> None:
        import time as _time
        print(f"Standing up (settling sim for {duration}s)...")
        default_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        joint_map = self.get_joint_map(self.policy_joints)
        for i, mj_id in enumerate(joint_map):
            default_ctrl[mj_id] = self.cfg.default_joint_pos[i]
        self.data.ctrl[:] = default_ctrl

        num_steps = int(duration / self.model.opt.timestep)
        for _ in range(num_steps):
            self.step()
        print("Standup complete.")

    def hold_standing_tick(self) -> None:
        self.step()

    def activate_policy_gains(self) -> None:
        self._apply_policy_gains()