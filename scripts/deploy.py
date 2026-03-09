import os
import csv
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
            loop_dt=args.sim_dt,  # Fixed decimation bug!
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
    
    # Run the control loop
    runner.run()

    # ==========================================
    # CSV EXPORT (Flight Recorder)
    # ==========================================
    entries = runner.logger.entries
    if len(entries) > 1:
        os.makedirs("logs", exist_ok=True)
        csv_path = "logs/latest_run.csv"
        print(f"\nSaving {len(entries)} frames to {csv_path}...")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Build Header
            num_joints = len(entries[0]["actual_pos"])
            header =["tick"]
            header +=[f"cmd_{i}" for i in range(3)]
            header +=[f"ang_vel_{i}" for i in range(3)]
            header +=[f"proj_grav_{i}" for i in range(3)]
            header +=[f"pos_{i}" for i in range(num_joints)]
            header +=[f"vel_{i}" for i in range(num_joints)]
            header +=[f"action_{i}" for i in range(num_joints)]
            writer.writerow(header)

            # Write Data
            for i, entry in enumerate(entries):
                row = [i]
                row.extend(entry["cmd"])
                row.extend(entry["ang_vel"])
                row.extend(entry["projected_gravity"])
                row.extend(entry["actual_pos"])
                row.extend(entry["actual_vel"])
                row.extend(entry["actions"])
                writer.writerow(row)
        print("CSV saved successfully.")

if __name__ == "__main__":
    main()