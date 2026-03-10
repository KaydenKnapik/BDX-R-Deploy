import os
import csv
import argparse
import numpy as np
from pathlib import Path

# --- NEW: Import the config loader to print metadata ---
from bdx_api.config import load_policy_config

def main():
    parser = argparse.ArgumentParser(description="BDX-R Deploy — sim or real")
    parser.add_argument("--backend", type=str, default="sim", choices=["sim", "real"],
                        help="Backend: 'sim' for MuJoCo, 'real' for hardware")
    parser.add_argument("--model", type=str, default="models/newlegs.onnx",
                        help="Path to ONNX policy model")
    
    # --- XML PATHS ---
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml",
                        help="Path to MuJoCo XML (sim only, full robot)")
    parser.add_argument("--xml_headless", type=str, default="xml/bdxr_legs.xml",
                        help="Path to headless MuJoCo XML (used if --legs is set)") 
    
    parser.add_argument("--sim_dt", type=float, default=0.005,
                        help="Simulation timestep (sim only)")
    parser.add_argument("--decimation", type=int, default=4,
                        help="Control decimation (policy runs every N steps)")
    parser.add_argument("--i2c_bus", type=int, default=7,
                        help="I2C bus number for BNO055 IMU (real only)")
    
    # --- LEGS ONLY FLAG ---
    parser.add_argument("--legs", action="store_true", 
                        help="Hold the head joints fixed and only deploy the legs policy")
    
    args = parser.parse_args()
    model_path = Path(args.model)

    # ==========================================
    # PRINT ONNX METADATA
    # ==========================================
    try:
        cfg = load_policy_config(model_path)
        print("\n" + "=" * 60)
        print("  ONNX POLICY METADATA LOADED")
        print("=" * 60)
        print(f"  Model Path : {model_path}")
        print(f"  Obs dim    : {cfg.obs_dim}")
        print(f"  Action dim : {cfg.action_dim}")
        print(f"  Joints ({len(cfg.joint_names)}): {cfg.joint_names}")
        print(f"  Stiffness  : {cfg.joint_stiffness}")
        print(f"  Damping    : {cfg.joint_damping}")
        print(f"  Defaults   : {[round(p, 3) for p in cfg.default_joint_pos]}")
        print(f"  Action Scl : {cfg.action_scale}")
        print(f"  Obs names  : {cfg.observation_names}")
        print(f"  Cmd names  : {cfg.command_names}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[WARNING] Could not print ONNX metadata: {e}\n")

    # 1. Decide which XML to use based on the flag
    if args.legs:
        active_xml = args.xml_headless
        print(f"Deploying in LEGS ONLY mode. Holding head stiff.")
        print(f"Using XML: {active_xml}")
    else:
        active_xml = args.xml
        print(f"Deploying FULL 14-joint model.")
        print(f"Using XML: {active_xml}")

    # 2. Initialize the correct backend
    if args.backend == "sim":
        from bdx_api.sim import MujocoBackend
        backend = MujocoBackend(
            xml_path=active_xml,
            model_path=model_path,
            sim_dt=args.sim_dt,
            legs_only=args.legs  # <-- Pass the flag to the backend
        )
    elif args.backend == "real":
        from bdx_api.hardware import HardwareBackend
        backend = HardwareBackend(
            model_path=model_path,
            loop_dt=args.sim_dt,
            i2c_bus=args.i2c_bus,
            legs_only=args.legs  # <-- Pass the flag to the backend
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # 3. Initialize the Runner
    from bdx_api.interface import PolicyRunner
    runner = PolicyRunner(
        backend=backend,
        model_path=model_path,
        decimation=args.decimation,
    )
    
    # 4. Run the control loop
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
            header = ["tick"]
            header +=[f"cmd_{i}" for i in range(3)]
            header +=[f"ang_vel_{i}" for i in range(3)]
            header += [f"proj_grav_{i}" for i in range(3)]
            header += [f"pos_{i}" for i in range(num_joints)]
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