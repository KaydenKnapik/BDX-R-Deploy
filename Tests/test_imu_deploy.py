#!/usr/bin/env python3
"""
Test IMU Deploy — MuJoCo simulation driven by real IMU data from the robot.

Runs the ONNX policy in MuJoCo, but angular velocity and projected gravity
come from the robot's real IMU (streamed over UDP from the Jetson).

Everything else (joint physics, rendering) is pure simulation.

Usage:
    python scripts/test_imu_deploy.py
    python scripts/test_imu_deploy.py --model models/walk.onnx
    python scripts/test_imu_deploy.py --imu_port 5005 --fixed
"""

import os
import sys
import signal
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="BDX-R Test Deploy — MuJoCo + Real IMU"
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
        "--imu_port", type=int, default=5005,
        help="UDP port for IMU data from the Jetson",
    )
    parser.add_argument(
        "--fixed", action="store_true",
        help="Lock the robot base in place (no falling)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)

    # Use the hybrid backend: MuJoCo physics + real IMU
    from bdx_api.hybrid import HybridIMUBackend

    print("=" * 50)
    print("  HYBRID MODE: MuJoCo + Real IMU")
    print(f"  Listening for IMU on UDP port {args.imu_port}")
    print(f"  Model: {args.model}")
    print(f"  Fixed base: {args.fixed}")
    print("=" * 50)

    backend = HybridIMUBackend(
        xml_path=args.xml,
        model_path=model_path,
        sim_dt=args.sim_dt,
        fixed=args.fixed,
        imu_port=args.imu_port,
    )

    from bdx_api.interface import PolicyRunner

    runner = PolicyRunner(
        backend=backend,
        model_path=model_path,
        decimation=args.decimation,
    )

    def _force_exit(signum, frame):
        """Last-resort handler: if we're stuck during cleanup, just die."""
        print("\n  Force exit.")
        backend.stop_imu()
        os._exit(1)

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n  KeyboardInterrupt caught at top level.")
    finally:
        # Install a hard-kill handler so a Ctrl+C during cleanup always works
        signal.signal(signal.SIGINT, _force_exit)
        backend.stop_imu()
        print("IMU receiver stopped. Goodbye.")


if __name__ == "__main__":
    main()
