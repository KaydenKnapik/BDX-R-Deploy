#!/usr/bin/env python3
"""
System ID Deploy — Test Sim vs Real gap.
Includes safe interpolation, custom sine amplitudes, and CSV overlay plotting.
"""
import os
import csv
import argparse
import numpy as np
from pathlib import Path

# --- Headless Plotting for SSH/Jetson ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd

from bdx_api.config import load_policy_config

def get_joint_amplitudes(joint_names, global_scale=1.0):
    amps = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        n = name.lower()
        if "yaw" in n or "roll" in n:
            amps[i] = 0.05
        elif "pitch" in n:
            amps[i] = 0.20
        elif "knee" in n:
            amps[i] = 0.40
        elif "ankle" in n:
            amps[i] = 0.30
        else:
            amps[i] = 0.10
    return amps * global_scale

def plot_results(csv_path, joint_names, compare_csv=None):
    """Generates a tall PNG graphing all joints, optionally overlaying a second CSV."""
    print(f"\nGenerating graph headlessly...")
    df_main = pd.read_csv(csv_path)
    
    df_compare = None
    if compare_csv and os.path.exists(compare_csv):
        print(f"Overlaying comparison data from: {compare_csv}")
        df_compare = pd.read_csv(compare_csv)
    elif compare_csv:
        print(f"[WARNING] Compare CSV not found at {compare_csv}")

    num_to_plot = len(joint_names)
    # Make the figure tall enough to fit all joints comfortably
    fig, axs = plt.subplots(num_to_plot, 1, figsize=(12, 2.5 * num_to_plot), sharex=True)
    if num_to_plot == 1: axs = [axs]

    for i in range(num_to_plot):
        ax = axs[i]
        
        # Plot Main Command and Main Actual
        ax.plot(df_main['time'], df_main[f'cmd_{i}'], label='Command', linestyle='--', color='black', linewidth=2)
        ax.plot(df_main['time'], df_main[f'pos_{i}'], label='Sim Actual' if compare_csv else 'Actual', color='blue', alpha=0.7, linewidth=2)
        
        # Overlay Compare Actual if provided
        if df_compare is not None:
            # We assume df_compare has the same time steps
            ax.plot(df_compare['time'], df_compare[f'pos_{i}'], label='Real Actual', color='red', alpha=0.7, linewidth=2)

        ax.set_title(joint_names[i], fontsize=12, fontweight='bold')
        ax.set_ylabel("Pos (rad)")
        ax.legend(loc="upper right")
        ax.grid(True)

    axs[-1].set_xlabel("Time (s)")
    title = "System ID: Sim vs Real Comparison" if compare_csv else "System ID: Commanded vs Actual"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    plot_file = csv_path.replace('.csv', '.png')
    plt.savefig(plot_file, dpi=150)
    print(f"[SUCCESS] Graph saved to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="BDX-R System ID")
    parser.add_argument("--backend", type=str, default="sim", choices=["sim", "real"])
    parser.add_argument("--model", type=str, default="models/newlegs.onnx")
    parser.add_argument("--xml", type=str, default="xml/bdxr_legs.xml")
    
    parser.add_argument("--mode", type=str, default="sine", choices=["constant", "sine"])
    parser.add_argument("--amp_scale", type=float, default=1.0, help="Multiplier for sine joint amplitudes")
    parser.add_argument("--freq", type=float, default=1.0, help="Frequency of sine wave (Hz)")
    parser.add_argument("--constant_pos", type=float, default=0.0, help="Offset (rad) to command to all joints in 'constant' mode")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of test (seconds)")
    
    # New Argument for Overlaying Data
    parser.add_argument("--compare_csv", type=str, default=None, help="Path to a Real CSV to overlay on the Sim graph")
    
    parser.add_argument("--height", type=float, default=0.4, help="Base height in air (sim only)")
    parser.add_argument("--sim_dt", type=float, default=0.005, help="Sim/Hardware dt (200Hz)")
    parser.add_argument("--decimation", type=int, default=4, help="Policy update rate (4 = 50Hz)")
    args = parser.parse_args()

    model_path = Path(args.model)
    cfg = load_policy_config(model_path)

    if args.backend == "sim":
        from bdx_api.sim import MujocoBackend
        backend = MujocoBackend(xml_path=args.xml, model_path=model_path, sim_dt=args.sim_dt, fixed=False, legs_only=True)
        orig_step = backend.step
        def fixed_base_step():
            backend.data.qpos[:3] =[0.0, 0.0, args.height]
            backend.data.qpos[3:7] =[1.0, 0.0, 0.0, 0.0]
            backend.data.qvel[:6] = 0.0
            orig_step()
        backend.step = fixed_base_step
    else:
        from bdx_api.hardware import HardwareBackend
        backend = HardwareBackend(model_path=model_path, loop_dt=args.sim_dt, legs_only=True)

    joint_map = backend.get_joint_map(cfg.joint_names)
    num_policy_joints = len(cfg.joint_names)
    target_pos_array = backend.get_default_pos_array(num_policy_joints)
    default_pos = np.array(cfg.default_joint_pos, dtype=np.float32)

    print("\n[PHASE 1] Safely interpolating from limp to standing position (3s)...")
    start_pos = backend.get_joint_positions(joint_map)
    standup_steps = int(3.0 / args.sim_dt)
    for step in range(standup_steps):
        alpha = step / float(standup_steps)
        current_target = (1.0 - alpha) * start_pos + (alpha * default_pos)
        for i, mapped_id in enumerate(joint_map): target_pos_array[mapped_id] = current_target[i]
        backend.set_joint_targets(target_pos_array)
        backend.step()

    print("\n[PHASE 2] Switching to compliant Policy Gains & Settling (1.5s)...")
    backend.activate_policy_gains()
    settle_steps = int(1.5 / args.sim_dt)
    for _ in range(settle_steps):
        backend.set_joint_targets(target_pos_array)
        backend.step()

    print(f"\n[PHASE 3] Starting {args.mode.upper()} test for {args.duration}s...")
    log_data =[]
    t_sim = 0.0
    total_steps = int(args.duration / args.sim_dt)
    active_cmd = np.array(cfg.default_joint_pos, dtype=np.float32)
    joint_amps = get_joint_amplitudes(cfg.joint_names, args.amp_scale)

    try:
        for step_idx in range(total_steps):
            if step_idx % args.decimation == 0:
                if args.mode == "constant":
                    # Use the command line argument to shift all joints!
                    offset = np.ones(num_policy_joints) * args.constant_pos
                elif args.mode == "sine":
                    offset = joint_amps * np.sin(2 * np.pi * args.freq * t_sim)
                
                for i, mapped_id in enumerate(joint_map):
                    cmd_val = cfg.default_joint_pos[i] + offset[i]
                    target_pos_array[mapped_id] = cmd_val
                    active_cmd[i] = cmd_val
                
            backend.set_joint_targets(target_pos_array)
            actual_pos = backend.get_joint_positions(joint_map)
            actual_vel = backend.get_joint_velocities(joint_map)

            log_data.append({"time": t_sim, "cmd": active_cmd.copy(), "pos": actual_pos.copy(), "vel": actual_vel.copy()})
            backend.step()
            t_sim += args.sim_dt

    except KeyboardInterrupt:
        print("\nTest interrupted early by user.")

    if hasattr(backend, "viewer") and backend.viewer is not None: backend.viewer.close()

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/sys_id_{args.backend}_{args.mode}.csv"
    
    print(f"Saving data to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header =["time"] +[f"cmd_{i}" for i in range(num_policy_joints)] +[f"pos_{i}" for i in range(num_policy_joints)] +[f"vel_{i}" for i in range(num_policy_joints)]
        writer.writerow(header)
        for entry in log_data:
            row = [entry["time"]] + list(entry["cmd"]) + list(entry["pos"]) + list(entry["vel"])
            writer.writerow(row)

    # Plot results! Pass in the comparison CSV if the user provided it.
    plot_results(csv_path, cfg.joint_names, args.compare_csv)

if __name__ == "__main__":
    main()