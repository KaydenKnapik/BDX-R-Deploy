"""
Runtime data logger with plotting on exit.

Captures joint target vs actual positions, IMU data, and commands.
Plots everything when Ctrl+C is pressed or the run ends.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class LogEntry:
    timestamp: float = 0.0
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    actual_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    actual_vel: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projected_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0))


class RuntimeLogger:
    """Accumulates data each control step. Plots on request."""

    def __init__(self, joint_names: List[str]):
        self.joint_names = [n.strip() for n in joint_names]
        self.entries: List[LogEntry] = []
        self._start_time = time.time()

    def log(self, cmd: np.ndarray, target_pos: np.ndarray, actual_pos: np.ndarray,
            actual_vel: np.ndarray, ang_vel: np.ndarray, projected_gravity: np.ndarray,
            actions: np.ndarray):
        entry = LogEntry(
            timestamp=time.time() - self._start_time,
            cmd=cmd.copy(),
            target_pos=target_pos.copy(),
            actual_pos=actual_pos.copy(),
            actual_vel=actual_vel.copy(),
            ang_vel=ang_vel.copy(),
            projected_gravity=projected_gravity.copy(),
            actions=actions.copy(),
        )
        self.entries.append(entry)

    def plot(self, save_dir: str = "logs"):
        if len(self.entries) < 2:
            print("Not enough data to plot.")
            return

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        t = np.array([e.timestamp for e in self.entries])
        n_joints = len(self.joint_names)

        targets = np.array([e.target_pos for e in self.entries])
        actuals = np.array([e.actual_pos for e in self.entries])
        velocities = np.array([e.actual_vel for e in self.entries])
        ang_vels = np.array([e.ang_vel for e in self.entries])
        proj_gravs = np.array([e.projected_gravity for e in self.entries])
        cmds = np.array([e.cmd for e in self.entries])
        actions = np.array([e.actions for e in self.entries])

        # =============================================
        # Figure 1: Target vs Actual Joint Positions
        # =============================================
        cols = 2
        rows = (n_joints + cols - 1) // cols
        fig1, axes1 = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True)
        fig1.suptitle("Joint Positions: Target vs Actual", fontsize=14, fontweight='bold')
        axes1 = axes1.flatten()

        for i in range(n_joints):
            ax = axes1[i]
            ax.plot(t, targets[:, i], 'r-', linewidth=1.0, label='Target', alpha=0.8)
            ax.plot(t, actuals[:, i], 'b-', linewidth=1.0, label='Actual', alpha=0.8)
            ax.set_title(self.joint_names[i], fontsize=10)
            ax.set_ylabel("rad")
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_joints, len(axes1)):
            axes1[i].set_visible(False)

        axes1[-2].set_xlabel("Time (s)")
        if n_joints < len(axes1):
            axes1[n_joints - 1].set_xlabel("Time (s)")
        fig1.tight_layout()
        fig1.savefig(save_path / "joint_positions.png", dpi=150)
        print(f"  Saved: {save_path / 'joint_positions.png'}")

        # =============================================
        # Figure 2: Joint Position Tracking Error
        # =============================================
        fig2, axes2 = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True)
        fig2.suptitle("Joint Tracking Error (Target - Actual)", fontsize=14, fontweight='bold')
        axes2 = axes2.flatten()

        for i in range(n_joints):
            ax = axes2[i]
            error = targets[:, i] - actuals[:, i]
            ax.plot(t, error, 'g-', linewidth=1.0, alpha=0.8)
            ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
            ax.set_title(f"{self.joint_names[i]}  (RMSE={np.sqrt(np.mean(error**2)):.4f})", fontsize=10)
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)

        for i in range(n_joints, len(axes2)):
            axes2[i].set_visible(False)

        fig2.tight_layout()
        fig2.savefig(save_path / "tracking_error.png", dpi=150)
        print(f"  Saved: {save_path / 'tracking_error.png'}")

        # =============================================
        # Figure 3: Joint Velocities
        # =============================================
        fig3, axes3 = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True)
        fig3.suptitle("Joint Velocities", fontsize=14, fontweight='bold')
        axes3 = axes3.flatten()

        for i in range(n_joints):
            ax = axes3[i]
            ax.plot(t, velocities[:, i], 'm-', linewidth=1.0, alpha=0.8)
            ax.set_title(self.joint_names[i], fontsize=10)
            ax.set_ylabel("rad/s")
            ax.grid(True, alpha=0.3)

        for i in range(n_joints, len(axes3)):
            axes3[i].set_visible(False)

        fig3.tight_layout()
        fig3.savefig(save_path / "joint_velocities.png", dpi=150)
        print(f"  Saved: {save_path / 'joint_velocities.png'}")

        # =============================================
        # Figure 4: IMU Data
        # =============================================
        fig4, (ax_gyro, ax_grav) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig4.suptitle("IMU Data", fontsize=14, fontweight='bold')

        labels_xyz = ['X', 'Y', 'Z']
        colors_xyz = ['r', 'g', 'b']

        for i in range(3):
            ax_gyro.plot(t, ang_vels[:, i], color=colors_xyz[i], linewidth=1.0,
                         label=f"ω_{labels_xyz[i]}", alpha=0.8)
        ax_gyro.set_title("Angular Velocity (Gyro)")
        ax_gyro.set_ylabel("rad/s")
        ax_gyro.legend(fontsize=9)
        ax_gyro.grid(True, alpha=0.3)

        for i in range(3):
            ax_grav.plot(t, proj_gravs[:, i], color=colors_xyz[i], linewidth=1.0,
                         label=f"g_{labels_xyz[i]}", alpha=0.8)
        ax_grav.set_title("Projected Gravity")
        ax_grav.set_ylabel("(unit)")
        ax_grav.set_xlabel("Time (s)")
        ax_grav.legend(fontsize=9)
        ax_grav.grid(True, alpha=0.3)

        fig4.tight_layout()
        fig4.savefig(save_path / "imu_data.png", dpi=150)
        print(f"  Saved: {save_path / 'imu_data.png'}")

        # =============================================
        # Figure 5: Commands & Actions
        # =============================================
        fig5, (ax_cmd, ax_act) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig5.suptitle("Commands & Policy Actions", fontsize=14, fontweight='bold')

        cmd_labels = ['Fwd (x)', 'Lat (y)', 'Yaw (z)']
        for i in range(3):
            ax_cmd.plot(t, cmds[:, i], color=colors_xyz[i], linewidth=1.5,
                        label=cmd_labels[i], alpha=0.8)
        ax_cmd.set_title("Velocity Commands")
        ax_cmd.set_ylabel("m/s or rad/s")
        ax_cmd.legend(fontsize=9)
        ax_cmd.grid(True, alpha=0.3)

        for i in range(min(n_joints, actions.shape[1])):
            ax_act.plot(t, actions[:, i], linewidth=0.8, label=self.joint_names[i], alpha=0.7)
        ax_act.set_title("Raw Policy Actions")
        ax_act.set_ylabel("action")
        ax_act.set_xlabel("Time (s)")
        ax_act.legend(fontsize=7, ncol=3, loc='upper right')
        ax_act.grid(True, alpha=0.3)

        fig5.tight_layout()
        fig5.savefig(save_path / "commands_actions.png", dpi=150)
        print(f"  Saved: {save_path / 'commands_actions.png'}")

        plt.close('all')
        print(f"  All plots saved to {save_path}/")