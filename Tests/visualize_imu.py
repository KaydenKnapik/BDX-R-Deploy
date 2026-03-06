#!/usr/bin/env python3
"""
IMU Visualizer — see your real IMU data mirrored on the MuJoCo robot.

No policy runs. The robot floats in place and its base orientation is set
directly from the projected gravity vector streamed over UDP. Tilt the real
robot and the sim robot mirrors it in real-time.

This lets you verify that all three axes (pitch, roll, yaw) are mapped
correctly without the walking policy fighting against you.

Usage:
    python scripts/visualize_imu.py
    python scripts/visualize_imu.py --imu_port 5005
"""

import os
import sys
import time
import signal
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from scipy.spatial.transform import Rotation

from bdx_api.robot_receiver import RobotStateReceiver
from bdx_api.config import load_policy_config


def gravity_to_quat(projected_gravity: np.ndarray, yaw_rad: float = 0.0) -> np.ndarray:
    """
    Convert a projected gravity vector + yaw angle into a quaternion (MuJoCo wxyz).

    Projected gravity gives us pitch and roll (tilt relative to world gravity).
    Yaw must come from gyro integration since gravity has no yaw info.
    """
    # Normalize
    g = projected_gravity / (np.linalg.norm(projected_gravity) + 1e-8)

    # Get pitch/roll from gravity: find rotation that maps g -> [0,0,-1]
    target = np.array([0.0, 0.0, -1.0])
    rot_tilt, _ = Rotation.align_vectors([target], [g])

    # Apply yaw rotation on top
    rot_yaw = Rotation.from_euler("z", yaw_rad)
    rot_combined = rot_yaw * rot_tilt

    # Convert to MuJoCo wxyz
    q_xyzw = rot_combined.as_quat()  # [x, y, z, w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return q_wxyz


def main():
    parser = argparse.ArgumentParser(description="BDX-R IMU Visualizer")
    parser.add_argument("--model", type=str, default="models/walk.onnx",
                        help="Path to ONNX model (for joint config)")
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml",
                        help="Path to MuJoCo XML")
    parser.add_argument("--robot_port", type=int, default=5006,
                        help="UDP port for robot state (robot_sender.py)")
    args = parser.parse_args()

    model_path = Path(args.model)
    cfg = load_policy_config(model_path)

    # --- Load MuJoCo model ---
    spec = mujoco.MjSpec.from_file(args.xml)

    # Set up PD actuators (same as sim backend) so default pose holds
    for i, name in enumerate(cfg.joint_names):
        name = name.strip()
        kp = cfg.joint_stiffness[i]
        kd = cfg.joint_damping[i]
        act = None
        for a in spec.actuators:
            if a.name == name:
                act = a
                break
        if act is None:
            continue
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
    mj_model.opt.timestep = 0.005

    # Disable gravity so the robot floats
    mj_model.opt.gravity[:] = [0, 0, 0]

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    # Initial pose
    mj_data.qpos[:3] = [0.0, 0.0, 0.33]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Set default joint positions and lock them via ctrl
    for i, name in enumerate(cfg.joint_names):
        name = name.strip()
        mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if mj_id >= 0:
            mj_data.qpos[7 + mj_id] = cfg.default_joint_pos[i]
            mj_data.ctrl[mj_id] = cfg.default_joint_pos[i]

    mujoco.mj_forward(mj_model, mj_data)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data,
        show_left_ui=False,
        show_right_ui=False,
    )
    viewer.cam.elevation = -20

    # --- Start robot state receiver ---
    receiver = RobotStateReceiver(port=args.robot_port)
    receiver.start()

    print("=" * 55)
    print("  IMU VISUALIZER — No policy, pure orientation mirror")
    print("-" * 55)
    print(f"  Listening for robot state on UDP port {args.robot_port}")
    print("  Tilt the real robot and watch the sim follow.")
    print("  Press Ctrl+C or close the viewer to exit.")
    print("=" * 55)

    should_quit = False

    def _sigint(signum, frame):
        nonlocal should_quit
        should_quit = True

    signal.signal(signal.SIGINT, _sigint)

    last_print = 0.0
    last_time = time.time()
    integrated_yaw = 0.0  # radians, accumulated from gyro Z

    try:
        while viewer.is_running() and not should_quit:
            now = time.time()
            dt = min(now - last_time, 0.1)  # cap dt to avoid jumps
            last_time = now

            data = receiver.get_data()
            gyro = data["gyro"]
            grav = data["projected_gravity"]
            connected = receiver.is_connected()

            if connected:
                # Integrate gyro Z for yaw
                integrated_yaw += float(gyro[2]) * dt

                # Set base orientation from gravity (pitch/roll) + integrated yaw
                q_wxyz = gravity_to_quat(grav, integrated_yaw)
                mj_data.qpos[3:7] = q_wxyz

            # Lock position and zero velocities
            mj_data.qpos[:3] = [0.0, 0.0, 0.33]
            mj_data.qvel[:6] = 0.0

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            # Print IMU values at ~5 Hz
            now = time.time()
            if now - last_print > 0.2:
                status = "\033[92mLIVE\033[0m" if connected else "\033[91mWAITING\033[0m"
                euler = Rotation.from_quat(
                    [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
                ).as_euler("xyz", degrees=True) if connected else np.zeros(3)
                yaw_deg = np.degrees(integrated_yaw)
                sys.stdout.write(
                    f"\r  [{status}]  "
                    f"Gyro: [{gyro[0]:+7.3f} {gyro[1]:+7.3f} {gyro[2]:+7.3f}]  "
                    f"Grav: [{grav[0]:+6.3f} {grav[1]:+6.3f} {grav[2]:+6.3f}]  "
                    f"RPY: [{euler[0]:+6.1f}\u00b0 {euler[1]:+6.1f}\u00b0 {yaw_deg:+6.1f}\u00b0]"
                    "    "
                )
                sys.stdout.flush()
                last_print = now

    finally:
        receiver.stop()
        print("\n\nIMU visualizer stopped.")


if __name__ == "__main__":
    main()
