#!/usr/bin/env python3
"""
Deploy Pace Chirp Sequence via Hardware Abstraction Layer (HAL).
Supports both MuJoCo (for safe testing) and the Real Robot.
KP/KD gains are HARDCODED to match Isaac Lab config.
"""
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# --- TRAJECTORY SHAPING (Must match Isaac Lab exactly!) ---
# 10 Leg Joints: L_Yaw, L_Roll, L_Pitch, L_Knee, L_Ankle, R_Yaw, R_Roll, R_Pitch, R_Knee, R_Ankle
BIAS = np.array([0.0, 0.0, 0.3, -0.6, 0.3,   0.0, 0.0, 0.3, -0.6, 0.3])
SCALE = np.array([0.2, 0.2, 0.5, 0.5, 0.5,   0.2, 0.2, 0.5, 0.5, 0.5])
DIR = np.array([1.0, 1.0, 1.0, -1.0, 1.0,   -1.0, -1.0, 1.0, -1.0, 1.0])

# --- HARDCODED GAINS (Must match Isaac Lab actuator configs) ---
# Order:[Yaw, Roll, Pitch, Knee, Ankle] for Left Leg, then Right Leg
HARDCODED_KP = np.array([
    78.957, 78.957, 78.957, 78.957, 16.581,  # Left Leg Gains
    78.957, 78.957, 78.957, 78.957, 16.581   # Right Leg Gains
])
HARDCODED_KD = np.array([
    5.027, 5.027, 5.027, 5.027, 1.056,       # Left Leg Gains
    5.027, 5.027, 5.027, 5.027, 1.056        # Right Leg Gains
])

def main():
    parser = argparse.ArgumentParser(description="Pace Data Collection (MuJoCo or Real)")
    parser.add_argument("--sim", action="store_true", help="Run in MuJoCo instead of real hardware")
    parser.add_argument("--xml", type=str, default="xml/bdxr_legs.xml", help="Path to MuJoCo XML")
    
    parser.add_argument("--model", type=str, default="models/newlegs.onnx", help="Path to ONNX policy model")
    parser.add_argument("--legs", action="store_true", help="Only use leg joints")
    
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--min_frequency", type=float, default=0.1)
    parser.add_argument("--max_frequency", type=float, default=10.0)
    parser.add_argument("--sim_dt", type=float, default=0.0025, help="Simulation timestep (400Hz)")
    args = parser.parse_args()

    if args.sim:
        print("[INFO] Starting MuJoCo Backend (Fixed Base)")
        from bdx_api.sim import MujocoBackend
        backend = MujocoBackend(xml_path=args.xml, model_path=Path(args.model), sim_dt=args.sim_dt, fixed=False, legs_only=args.legs)
        
        # Patch step to fix base in air (with gravity) so the robot hangs
        orig_step = backend.step
        def fixed_base_step():
            backend.data.qpos[:3] =[0.0, 0.0, 0.4]
            backend.data.qpos[3:7] =[1.0, 0.0, 0.0, 0.0]
            orig_step()
        backend.step = fixed_base_step
    else:
        print("[INFO] Starting Real Hardware Backend")
        from bdx_api.hardware import HardwareBackend
        backend = HardwareBackend(model_path=Path(args.model), loop_dt=args.sim_dt, legs_only=args.legs)

    # --- OVERWRITE GAINS (Bypassing Frozen Dataclass) ---
    print("[INFO] Overwriting backend config with hardcoded Pace KP/KD values.")
    object.__setattr__(backend.cfg, "joint_stiffness", HARDCODED_KP)
    object.__setattr__(backend.cfg, "joint_damping", HARDCODED_KD)
    
    # FIX: Make the standup position perfectly match the very first frame of the chirp sequence
    STARTUP_POS = BIAS * DIR * SCALE
    object.__setattr__(backend.cfg, "default_joint_pos", STARTUP_POS)
    
    # In MuJoCo, we also have to force it to re-apply these newly overwritten gains to the compiled model
    if args.sim:
        backend._apply_policy_gains()

    # 1. Standup to the STARTUP_POS (safely energize motors)
    backend.standup(duration=2.0)
    time.sleep(1.0) # Settle

    # 3. CHIRP SEQUENCE
    print("\n🚀 STARTING CHIRP SEQUENCE! Keep hands clear!")
    num_steps = int(args.duration / args.sim_dt)
    
    time_data, des_pos_data, act_pos_data = [], [],[]
    
    try:
        for step in range(num_steps):
            t = step * args.sim_dt
            
            # Linear chirp math (Exact match to Isaac Lab equations)
            phase = 2 * np.pi * (args.min_frequency * t + ((args.max_frequency - args.min_frequency) / (2 * args.duration)) * t ** 2)
            chirp_signal = np.sin(phase)
            
            # Apply the same shaping math as Isaac Lab
            leg_targets = (chirp_signal + BIAS) * DIR * SCALE
            
            # Send to backend (Dynamically sizes to 10 for MuJoCo headless, or 14 for Real Robot!)
            full_targets = backend.get_default_pos_array(10)
            full_targets[:10] = leg_targets
            backend.set_joint_targets(full_targets)
            backend.step()
            
            # Record Data (Safely grab only the first 10 policy leg joints)
            act_pos = backend.get_joint_positions(np.arange(10))
            
            time_data.append(t)
            des_pos_data.append(leg_targets)
            act_pos_data.append(act_pos)
            
            if step % int(1.0 / args.sim_dt) == 0:
                freq = args.min_frequency + (args.max_frequency - args.min_frequency) * (t / args.duration)
                print(f"  [INFO] Progress: {t:.1f}s / {args.duration}s (Freq: {freq:.2f} Hz)")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted early!")

    # 4. Save Data
    print("\nSaving data to tensor file...")
    time_tensor = torch.tensor(time_data, dtype=torch.float32)
    des_pos_tensor = torch.tensor(np.array(des_pos_data), dtype=torch.float32)
    act_pos_tensor = torch.tensor(np.array(act_pos_data), dtype=torch.float32)

    save_dir = Path("data/bdxr_real")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "chirp_data.pt"
    
    torch.save({
        "time": time_tensor,
        "dof_pos": act_pos_tensor,
        "des_dof_pos": des_pos_tensor,
    }, save_path)
    
    print(f"🎉 Success! Data saved to: {save_path.absolute()}")

if __name__ == "__main__":
    main()