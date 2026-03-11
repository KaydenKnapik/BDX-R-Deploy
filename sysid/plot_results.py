#!/usr/bin/env python3
import argparse
import os

# --- CRUCIAL FOR SSH / JETSON ---
# Force matplotlib to use a non-interactive backend so it doesn't try to open a window
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Plot System ID CSVs Headless")
    parser.add_argument("csv_files", nargs='+', help="Path to one or more CSV files")
    args = parser.parse_args()

    # Load all provided CSVs
    dfs = {}
    for f in args.csv_files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            return
        dfs[os.path.basename(f).replace('.csv', '')] = pd.read_csv(f)

    # Use the first file as the reference to find how many joints we have
    df_ref = list(dfs.values())[0]
    cmd_cols = [c for c in df_ref.columns if c.startswith('cmd_')]
    
    # Plot up to 5 joints (e.g., the left leg) to keep the image readable
    num_to_plot = min(5, len(cmd_cols))
    
    fig, axs = plt.subplots(num_to_plot, 1, figsize=(12, 3 * num_to_plot), sharex=True)
    if num_to_plot == 1:
        axs = [axs]

    # Colors for different files (e.g., sim=blue, real=red)
    colors =['blue', 'red', 'green', 'orange']

    for i in range(num_to_plot):
        ax = axs[i]
        
        # Plot the Command target (from the first file)
        ax.plot(df_ref['time'], df_ref[f'cmd_{i}'], label='Command (Target)', 
                linestyle='--', color='black', linewidth=2)

        # Plot the Actual Position for each CSV provided
        for j, (name, df) in enumerate(dfs.items()):
            c = colors[j % len(colors)]
            ax.plot(df['time'], df[f'pos_{i}'], label=f'{name} Actual', 
                    color=c, alpha=0.7, linewidth=2)

        ax.set_ylabel(f'Joint {i} Position\n(rad)')
        ax.legend(loc="upper right")
        ax.grid(True)

    axs[-1].set_xlabel("Time (seconds)")
    
    if len(dfs) > 1:
        title = "Sim vs Real: System ID Comparison"
        out_path = "logs/sys_id_comparison.png"
    else:
        title = f"System ID: {list(dfs.keys())[0]}"
        out_path = args.csv_files[0].replace('.csv', '.png')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save the image
    plt.savefig(out_path, dpi=150)
    print(f"\n[SUCCESS] Graph saved to: {out_path}")

if __name__ == "__main__":
    main()