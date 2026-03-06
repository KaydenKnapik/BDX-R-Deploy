#!/usr/bin/env python3
"""
Test Robot Deploy — MuJoCo sim driven by ALL real robot data.

The ONNX policy runs in MuJoCo, but joint positions, joint velocities,
gyro, and projected gravity all come from the real robot via UDP
(from robot_sender.py on the Jetson).

The MuJoCo viewer mirrors the real robot's pose AND shows what the
policy would command as joint targets.

Usage:
    python scripts/test_robot_deploy.py
    python scripts/test_robot_deploy.py --model models/walk.onnx
    python scripts/test_robot_deploy.py --robot_port 5006 --fixed
"""

import os
import sys
import signal
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="BDX-R Test Deploy — MuJoCo + Real Robot State"
    )
    parser.add_argument(
        "--model", type=str, default="models/walk.onnx",
        help="Path to ONNX policy model",
    )
    parser.add_argument(
        "--xml", type=str, default="xml/bdxr.xml",
        help="Path to MuJoCo XML",
    )
    parser.add_argument(
        "--sim_dt", type=float, default=0.005,
        help="Simulation timestep",
    )
    parser.add_argument(
        "--decimation", type=int, default=4,
        help="Control decimation (policy runs every N sim steps)",
    )
    parser.add_argument(
        "--robot_port", type=int, default=5006,
        help="UDP port for robot state from the Jetson (robot_sender.py)",
    )
    parser.add_argument(
        "--fixed", action="store_true",
        help="Lock the robot base in place (no falling)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)

    from bdx_api.hybrid_full import HybridFullBackend

    print("=" * 55)
    print("  FULL HYBRID MODE: MuJoCo + Real Joints + Real IMU")
    print("-" * 55)
    print(f"  Listening for robot state on UDP port {args.robot_port}")
    print(f"  Model: {args.model}")
    print(f"  Fixed base: {args.fixed}")
    print("=" * 55)

    backend = HybridFullBackend(
        xml_path=args.xml,
        model_path=model_path,
        sim_dt=args.sim_dt,
        fixed=args.fixed,
        robot_port=args.robot_port,
    )

    from bdx_api.interface import PolicyRunner

    runner = PolicyRunner(
        backend=backend,
        model_path=model_path,
        decimation=args.decimation,
    )

    def _force_exit(signum, frame):
        print("\n  Force exit.")
        backend.stop_receiver()
        os._exit(1)

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n  KeyboardInterrupt caught at top level.")
    finally:
        signal.signal(signal.SIGINT, _force_exit)
        backend.stop_receiver()
        print("Robot state receiver stopped. Goodbye.")


if __name__ == "__main__":
    main()
