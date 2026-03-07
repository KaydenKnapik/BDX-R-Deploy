import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="BDX-R Deploy — sim or real")
    parser.add_argument("--backend", type=str, default="sim", choices=["sim", "real"],
                        help="Backend: 'sim' for MuJoCo, 'real' for hardware")
    parser.add_argument("--model", type=str, default="models/new.onnx",
                        help="Path to ONNX policy model")
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml",
                        help="Path to MuJoCo XML (sim only)")
    parser.add_argument("--sim_dt", type=float, default=0.005,
                        help="Simulation timestep (sim only)")
    parser.add_argument("--decimation", type=int, default=4,
                        help="Control decimation (policy runs every N steps)")
    parser.add_argument("--i2c_bus", type=int, default=7,
                        help="I2C bus number for BNO055 IMU (real only)")
    args = parser.parse_args()

    model_path = Path(args.model)

    if args.backend == "sim":
        from bdx_api.sim import MujocoBackend
        backend = MujocoBackend(
            xml_path=args.xml,
            model_path=model_path,
            sim_dt=args.sim_dt,
        )
    elif args.backend == "real":
        from bdx_api.hardware import HardwareBackend
        backend = HardwareBackend(
            model_path=model_path,
            loop_dt=args.sim_dt * args.decimation,  # real loop runs at policy rate
            i2c_bus=args.i2c_bus,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    from bdx_api.interface import PolicyRunner
    runner = PolicyRunner(
        backend=backend,
        model_path=model_path,
        decimation=args.decimation,
    )
    runner.run()


if __name__ == "__main__":
    main()