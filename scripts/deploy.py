import os
import sys
import json
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from pynput import keyboard

# --- Utils ---
def quat_rotate_inverse(q, v):
    """Rotates a vector v by the inverse of quaternion q. Assumes q is [x, y, z, w]"""
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def parse_metadata_value(val):
    """Converts ONNX metadata strings into Python lists or floats."""
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        pass
    cleaned = val.replace('[', '').replace(']', '').strip()
    if ',' in cleaned:
        parts = cleaned.split(',')
        try:
            return [float(p.strip()) for p in parts]
        except ValueError:
            return [p.strip() for p in parts]
    elif ' ' in cleaned:
        parts = cleaned.split()
        try:
            return [float(p.strip()) for p in parts]
        except ValueError:
            pass
    try:
        return float(val)
    except ValueError:
        return val

def get_item(src, idx):
    if isinstance(src, list): return float(src[idx])
    return float(src)


# --- Keyboard Controller ---
class KeyboardController:
    """Non-blocking keyboard controller using pynput.
    
    Controls:
        ↑ / ↓      - Increase / Decrease forward speed
        ← / →      - Increase / Decrease lateral speed
        Z / X       - Turn Left / Right
        SPACE       - Stop all movement
        ESC         - Quit
    
    Each tap increments/decrements by one step. Starts at 0,0,0.
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

        self._listener = keyboard.Listener(
            on_press=self._on_press,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        # Arrow keys
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
            # Character keys
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
        # Round to avoid float drift
        return np.array([
            round(self.lin_vel_x, 2),
            round(self.lin_vel_y, 2),
            round(self.ang_vel_yaw, 2),
        ], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml", help="Path to the Mujoco XML file")
    parser.add_argument("--checkpoint", type=str, default="models/best.onnx", help="ONNX model path")
    parser.add_argument("--sim_dt", type=float, default=0.005, help="Simulation timestep")
    parser.add_argument("--decimation", type=int, default=4, help="Control decimation")
    parser.add_argument("--max_lin_vel", type=float, default=1.0, help="Max linear velocity")
    parser.add_argument("--max_ang_vel", type=float, default=1.5, help="Max angular velocity")
    args = parser.parse_args()

    # --- 1. Load ONNX Model & Metadata ---
    model_path = args.checkpoint
    print(f"Loading ONNX model from {model_path}")

    try:
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name
        meta = ort_session.get_modelmeta().custom_metadata_map
        model_cfg = {k: parse_metadata_value(v) for k, v in meta.items()}
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        sys.exit(1)

    # --- 2. Parameters ---
    sim_dt = args.sim_dt
    decimation = args.decimation

    joint_names = model_cfg.get("joint_names")
    obs_names = model_cfg.get("observation_names")
    if not joint_names or not obs_names:
        print("CRITICAL: Metadata missing joint_names or observation_names.")
        sys.exit(1)

    # --- 3. MjSpec Init (Rewrite Actuators to Position Control) ---
    xml_path = args.xml
    spec = mujoco.MjSpec.from_file(xml_path)

    for i, name in enumerate(joint_names):
        name = name.strip()
        kp = get_item(model_cfg.get("joint_stiffness"), i)
        kd = get_item(model_cfg.get("joint_damping"), i)

        act = None
        for a in spec.actuators:
            if a.name == name:
                act = a
                break
        
        if act is None:
            print(f"Error: Actuator for joint {name} not found in Mujoco XML.")
            sys.exit(1)

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

    mj_model = spec.compile()
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    # --- 4. Map Model Joints to Mujoco ID & Defaults ---
    mj_default_pos = np.zeros(mj_model.nu, dtype=np.float32)
    model_action_scale = np.zeros(len(joint_names), dtype=np.float32)
    model_to_mj_map = []

    print("\n" + "="*60)
    print("DEBUG REPORT (Configuration)")
    print("-" * 60)
    for i, name in enumerate(joint_names):
        name = name.strip()
        mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if mj_id == -1:
            for j in range(mj_model.nu):
                if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, j):
                    mj_id = j
                    break
        
        if mj_id == -1:
            print(f"Error: Joint {name} not found in Mujoco XML.")
            sys.exit(1)
        
        model_to_mj_map.append(mj_id)

        kp = get_item(model_cfg.get("joint_stiffness"), i)
        kd = get_item(model_cfg.get("joint_damping"), i)
        
        mj_default_pos[mj_id] = get_item(model_cfg.get("default_joint_pos"), i)
        model_action_scale[i] = get_item(model_cfg.get("action_scale"), i)
        
        print(f"{name:<18} | Kp: {kp:<5.1f} | Kd: {kd:<4.1f} | Default: {mj_default_pos[mj_id]:.3f}")

    print("="*60 + "\n")
    model_to_mj_map = np.array(model_to_mj_map, dtype=int)

    # --- 5. Set Initial Simulation State ---
    mj_data.qpos[:3] = [0.0, 0.0, 0.33]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj_data.qpos[7:] = mj_default_pos 
    mujoco.mj_forward(mj_model, mj_data)

    # --- 6. Keyboard Controller ---
    kb = KeyboardController(
        lin_vel_step=0.1,
        ang_vel_step=0.1,
        lin_vel_x_range=(-1.0, 1.0),
        lin_vel_y_range=(-0.4, 0.4),
        ang_vel_z_range=(-1.0, 1.0),
    )

    # --- Runtime Variables ---
    actions_model = np.zeros(len(joint_names), dtype=np.float32)
    dof_targets = mj_default_pos.copy()
    it = 0
    print_interval = int(1.0 / (sim_dt * decimation))  # ~once per second

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print("=" * 50)
        print("  KEYBOARD CONTROLS")
        print("-" * 50)
        print("  ↑ / ↓     Increase / Decrease forward speed")
        print("  ← / →     Increase / Decrease lateral speed")
        print("  Z / X     Turn Left / Right")
        print("  SPACE     Stop all movement")
        print("  ESC       Quit")
        print("=" * 50)
        
        while viewer.is_running() and not kb.should_quit:
            # --- Control Decision (Decimated) ---
            if it % decimation == 0:
                # Update keyboard velocities
                cmd = kb.get_command()

                # Print command periodically
                if it % (decimation * print_interval) == 0:
                    print(f"\rCmd: fwd={cmd[0]:+.2f}  lat={cmd[1]:+.2f}  yaw={cmd[2]:+.2f}", end="", flush=True)

                # Build observation
                quat_wxyz = mj_data.qpos[3:7].astype(np.float32)
                quat = quat_wxyz[[1, 2, 3, 0]]

                base_ang_vel = mj_data.sensor("imu_ang_vel").data.astype(np.float32)
                projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                
                mj_dof_pos = mj_data.qpos[7:].astype(np.float32)
                mj_dof_vel = mj_data.qvel[6:].astype(np.float32)
                
                model_dof_pos = mj_dof_pos[model_to_mj_map]
                model_dof_vel = mj_dof_vel[model_to_mj_map]
                
                obs_parts = []
                for term in obs_names:
                    term = term.strip()
                    if term == "projected_gravity":
                        obs_parts.append(projected_gravity)
                    elif term == "base_ang_vel":
                        obs_parts.append(base_ang_vel)
                    elif term in ["joint_pos", "dof_pos"]:
                        obs_parts.append(model_dof_pos - mj_default_pos[model_to_mj_map])
                    elif term in ["joint_vel", "dof_vel"]:
                        obs_parts.append(model_dof_vel)
                    elif term == "actions":
                        obs_parts.append(actions_model)
                    elif term in ["command", "commands"]:
                        obs_parts.append(cmd)
                
                obs = np.concatenate(obs_parts).astype(np.float32)
                
                actions_model = ort_session.run(None, {input_name: obs.reshape(1, -1)})[0][0]
                dof_targets[model_to_mj_map] = mj_default_pos[model_to_mj_map] + model_action_scale * actions_model

            # --- Step Sim ---
            mj_data.ctrl[:] = dof_targets
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            it += 1

        print("\nShutting down.")