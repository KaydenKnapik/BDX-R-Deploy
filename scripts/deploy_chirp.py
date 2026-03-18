#!/usr/bin/env python3
"""
Deploy Pace Chirp Sequence via Hardware Abstraction Layer (HAL).
"""
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Set Matplotlib backend to 'Agg' for headless environments (like Jetson Orin Nano)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- THE SEQUENTIAL ORDER YOU REQUESTED ---
SEQUENTIAL_NAMES =[
    "Left_Hip_Yaw", "Left_Hip_Roll", "Left_Hip_Pitch", "Left_Knee", "Left_Ankle",
    "Right_Hip_Yaw", "Right_Hip_Roll", "Right_Hip_Pitch", "Right_Knee", "Right_Ankle"
]

# --- TRAJECTORY SHAPING (Sequential: Left Leg, then Right Leg) ---
BIAS = np.zeros(10)
SCALE = np.array([0.15, 0.15, 0.25, 0.3, 0.4,   0.15, 0.15, 0.25, 0.3, 0.4])
#SCALE = np.array([0.1, 0.1, 0.1, 0.1, 0.15,   0.1, 0.1, 0.1, 0.1, 0.15,])
#DIR = np.array([1.0, 1.0, 1.0, -1.0, 1.0,   -1.0, -1.0, 1.0, -1.0, 1.0])
DIR = np.array([1.0, 1.0,  1.0, -1.0,  1.0,       -1.0, -1.0, -1.0,  1.0, -1.0])

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
    
    parser.add_argument("--model", type=str, default="models/policy.pt", help="Path to policy model")
    parser.add_argument("--legs", action="store_true", help="Only use leg joints")
    parser.add_argument("--no_imu", action="store_true", help="Disable IMU initialization")
    
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--min_frequency", type=float, default=0.1)
    parser.add_argument("--max_frequency", type=float, default=10.0)
    parser.add_argument("--sim_dt", type=float, default=0.0025, help="Simulation timestep (400Hz)")
    args = parser.parse_args()

    if args.sim:
        print("[INFO] Starting MuJoCo Backend (Fixed Base)")
        from bdx_api.sim import MujocoBackend
        backend = MujocoBackend(
            xml_path=args.xml,          
            model_path=Path(args.model),
            sim_dt=args.sim_dt,         
            legs_only=args.legs
        )
        orig_step = backend.step
        def fixed_base_step():
            backend.data.qpos[:3] =[0.0, 0.0, 0.4]
            backend.data.qpos[3:7] =[1.0, 0.0, 0.0, 0.0]
            orig_step()
        backend.step = fixed_base_step
    else:
        print("[INFO] Starting Real Hardware Backend")
        from bdx_api.hardware import HardwareBackend
        backend = HardwareBackend(
            model_path=Path(args.model),
            loop_dt=args.sim_dt,
            legs_only=args.legs,
            use_imu=not args.no_imu 
        )

    # =================================================================
    # SAFELY MAP SEQUENTIAL DATA TO THE BACKEND'S EXPECTED ORDER
    # =================================================================
    policy_order = backend.cfg.joint_names  # The order defined in policy.yaml
    mapped_kp = np.zeros(10)
    mapped_kd = np.zeros(10)
    mapped_startup = np.zeros(10)
    
    seq_startup = BIAS * DIR * SCALE
    
    # Map the arrays!
    for backend_idx, joint_name in enumerate(policy_order):
        seq_idx = SEQUENTIAL_NAMES.index(joint_name)
        mapped_kp[backend_idx] = HARDCODED_KP[seq_idx]
        mapped_kd[backend_idx] = HARDCODED_KD[seq_idx]
        mapped_startup[backend_idx] = seq_startup[seq_idx]

    print(f"[INFO] Backend loaded order: {policy_order}")
    print("[INFO] Translating sequential script arrays to backend format...")

    # Overwrite backend config with mapped values
    object.__setattr__(backend.cfg, "joint_stiffness", mapped_kp.tolist())
    object.__setattr__(backend.cfg, "joint_damping", mapped_kd.tolist())
    object.__setattr__(backend.cfg, "default_joint_pos", mapped_startup.tolist())
    
    if args.sim and hasattr(backend, '_apply_policy_gains'):
        backend._apply_policy_gains()

# 1. Standup to the STARTUP_POS safely
    backend.standup(duration=2.0)
    
    # --- NEW: FLUSH THE BACKLOG ---
    # Standup sent thousands of messages without reading, filling the buffer with old data.
    # We MUST flush the bus so we only read fresh data going forward.
    if hasattr(backend, 'activate_policy_gains'):
        backend.activate_policy_gains()
    
    # 2. ACTIVELY HOLD the standup position for 2.0s
    print("[INFO] Actively holding start position and warming up sensor filters...")
    hold_steps = int(2.0 / args.sim_dt)
    
    start_targets = backend.get_default_pos_array(10)
    for backend_idx, joint_name in enumerate(policy_order):
        seq_idx = SEQUENTIAL_NAMES.index(joint_name)
        start_targets[backend_idx] = seq_startup[seq_idx]
        
    for step in range(hold_steps):
        backend.set_joint_targets(start_targets)
        
        # SLEEP FIRST: Gives the motors 2.5ms to receive the command and send a reply
        backend.step() 
        
        # READ SECOND: Now we read the fresh reply that just arrived
        backend.get_joint_positions(np.arange(10)) 
        
        if step % int(1.0 / args.sim_dt) == 0:
            print(f"  [INFO] Holding... {step * args.sim_dt:.1f}s")

    # 3. CHIRP SEQUENCE
    print("\n🚀 STARTING CHIRP SEQUENCE! Keep hands clear!")
    num_steps = int(args.duration / args.sim_dt)
    
    # ==========================================
    # PRE-COMPUTE TRAJECTORY (Just like ETH script)
    # ==========================================
    # Create time array for the whole duration
    t_array = np.linspace(0, args.duration, num_steps, endpoint=False)
    
    # Compute phase and sine wave for the whole trajectory at once
    phase_array = 2 * np.pi * (args.min_frequency * t_array + ((args.max_frequency - args.min_frequency) / (2 * args.duration)) * t_array ** 2)
    chirp_array = np.sin(phase_array)
    
    # Broadcast arrays to shape (num_steps, 10 joints)
    # This evaluates to exactly: (chirp + BIAS) * DIR * SCALE
    all_seq_targets = (chirp_array[:, None] + BIAS[None, :]) * DIR[None, :] * SCALE[None, :]
    
    time_data, des_pos_data, act_pos_data = [], [],[]
    
    try:
        for step in range(num_steps):
            t = t_array[step]
            
            # 1. Grab pre-computed target (Extremely fast, zero math in loop)
            seq_targets = all_seq_targets[step]
            
            # 2. Translate sequential targets into backend format
            full_targets = backend.get_default_pos_array(10)
            for backend_idx, joint_name in enumerate(policy_order):
                seq_idx = SEQUENTIAL_NAMES.index(joint_name)
                full_targets[backend_idx] = seq_targets[seq_idx]
                
            backend.set_joint_targets(full_targets)
            backend.step()
            
            # 3. Read actual positions and translate back to sequential order for saving
            backend_act_pos = backend.get_joint_positions(np.arange(10))
            seq_act_pos = np.zeros(10)
            for backend_idx, joint_name in enumerate(policy_order):
                seq_idx = SEQUENTIAL_NAMES.index(joint_name)
                seq_act_pos[seq_idx] = backend_act_pos[backend_idx]
            
            time_data.append(t)
            des_pos_data.append(seq_targets) # Save as sequential
            act_pos_data.append(seq_act_pos) # Save as sequential
            
            if step % int(1.0 / args.sim_dt) == 0:
                freq = args.min_frequency + (args.max_frequency - args.min_frequency) * (t / args.duration)
                print(f"  [INFO] Progress: {t:.1f}s / {args.duration}s (Freq: {freq:.2f} Hz)")

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted early!")

    # 4. Save Data
    print("\nSaving data to tensor file (Sequential Order)...")
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
        "joint_names": SEQUENTIAL_NAMES # Save the list to confirm what order was used!
    }, save_path)
    
    print(f"🎉 Success! Data saved to: {save_path.absolute()}")

    # 5. Save Plots (Headless compatible)
    print("\nGenerating and saving plots...")
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to numpy arrays once for plotting
    t_plot = time_tensor.numpy()
    act_plot = act_pos_tensor.numpy()
    des_plot = des_pos_tensor.numpy()

    for i, joint_name in enumerate(SEQUENTIAL_NAMES):
        plt.figure()
        plt.plot(t_plot, act_plot[:, i], label=f"{joint_name} actual")
        plt.plot(t_plot, des_plot[:, i], label=f"{joint_name} target", linestyle='dashed')
        
        plt.title(f"Joint {joint_name} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        # Save the plot instead of showing it
        plot_file_path = plots_dir / f"{i:02d}_{joint_name}_trajectory.png"
        plt.savefig(plot_file_path)
        plt.close()  # Free memory after saving each plot

    print(f"🖼️ Plots successfully saved to: {plots_dir.absolute()}")


if __name__ == "__main__":
    main()