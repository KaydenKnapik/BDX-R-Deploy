#!/usr/bin/env python3
"""
CSV Replay — Watch your physical robot's run inside MuJoCo.

Usage:
    python scripts/replay_csv.py
    python scripts/replay_csv.py --csv logs/latest_run.csv --fps 50
"""

import time
import csv
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from scipy.spatial.transform import Rotation
from bdx_api.config import load_policy_config

def gravity_to_quat(projected_gravity, yaw_rad=0.0):
    """Convert projected gravity + yaw back into a MuJoCo quaternion."""
    g = np.array(projected_gravity)
    norm = np.linalg.norm(g)
    if norm > 1e-6:
        g = g / norm
    else:
        g = np.array([0.0, 0.0, -1.0])

    target = np.array([0.0, 0.0, -1.0])
    rot_tilt, _ = Rotation.align_vectors([target], [g])
    rot_yaw = Rotation.from_euler("z", yaw_rad)
    rot_combined = rot_yaw * rot_tilt
    q_xyzw = rot_combined.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def main():
    parser = argparse.ArgumentParser(description="Replay CSV log in MuJoCo")
    parser.add_argument("--csv", type=str, default="logs/latest_run.csv")
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml")
    parser.add_argument("--model", type=str, default="models/new.onnx")
    parser.add_argument("--fps", type=float, default=50.0) # matches policy rate
    args = parser.parse_args()

    # Load Config and MuJoCo Model
    cfg = load_policy_config(Path(args.model))
    spec = mujoco.MjSpec.from_file(args.xml)
    mj_model = spec.compile()
    
    # Disable gravity so the robot floats
    mj_model.opt.gravity[:] = [0, 0, 0]
    mj_data = mujoco.MjData(mj_model)

    # Get MuJoCo IDs for the joints
    joint_ids =[]
    for name in cfg.joint_names:
        mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.strip())
        joint_ids.append(mj_id)

    # Read CSV
    frames =[]
    try:
        with open(args.csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(row)
    except FileNotFoundError:
        print(f"File not found: {args.csv}")
        return

    print(f"Loaded {len(frames)} frames. Launching Viewer...")
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    yaw = 0.0
    dt = 1.0 / args.fps

    # Replay Loop
    for row in frames:
        if not viewer.is_running():
            break

        # Extract data from CSV
        grav = [float(row[f"proj_grav_{i}"]) for i in range(3)]
        gyro = [float(row[f"ang_vel_{i}"]) for i in range(3)]
        pos = [float(row[f"pos_{i}"]) for i in range(len(joint_ids))]

        # Integrate yaw from gyro exactly how the policy experiences it
        yaw += gyro[2] * dt

        # Set robot base position in air and apply orientation
        mj_data.qpos[:3] = [0, 0, 0.4]
        mj_data.qpos[3:7] = gravity_to_quat(grav, yaw)

        # Set real-world recorded joint positions
        for i, mj_id in enumerate(joint_ids):
            if mj_id >= 0:
                mj_data.qpos[7 + mj_id] = pos[i]

        # Forward kinematics (no physics simulation, just visuals)
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()
        
        # maintain playback speed
        time.sleep(dt)

    print("Playback complete. Close viewer to exit.")
    while viewer.is_running():
        time.sleep(0.1)

if __name__ == "__main__":
    main()