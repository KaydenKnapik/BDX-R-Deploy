"""
Joint Direction Test Tool

Test that joint angles match between sim and real hardware.
The robot is fixed in the air (no gravity fall in sim, robot on stand for real).

Controls:
    ← / →       Select joint
    ↑ / ↓       Increase / Decrease joint angle
    0            Zero current joint
    SPACE        Zero ALL joints
    R            Go to default pose (from ONNX metadata)
    +/-          Change step size
    ESC          Quit

Each tap moves the selected joint by ±0.05 rad.
"""

import argparse
import numpy as np
from pathlib import Path
from pynput import keyboard


class JointTestController:
    """Keyboard controller for joint-by-joint testing."""

    def __init__(self, num_joints: int, joint_names: list,
                 default_pos: np.ndarray, step: float = 0.05):
        self.num_joints = num_joints
        self.joint_names = joint_names
        self.default_pos = default_pos.copy()
        self.step = step

        self.selected = 0
        self.positions = np.zeros(num_joints, dtype=np.float32)
        self._quit = False
        self._dirty = True

        self._listener = keyboard.Listener(on_press=self._on_press, suppress=False)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            self._quit = True
            return

        if key == keyboard.Key.left:
            self.selected = (self.selected - 1) % self.num_joints
            self._dirty = True
        elif key == keyboard.Key.right:
            self.selected = (self.selected + 1) % self.num_joints
            self._dirty = True
        elif key == keyboard.Key.up:
            self.positions[self.selected] += self.step
            self._dirty = True
        elif key == keyboard.Key.down:
            self.positions[self.selected] -= self.step
            self._dirty = True
        elif key == keyboard.Key.space:
            self.positions[:] = 0.0
            self._dirty = True
        else:
            try:
                if key.char == '0':
                    self.positions[self.selected] = 0.0
                    self._dirty = True
                elif key.char == 'r':
                    self.positions[:] = self.default_pos
                    self._dirty = True
                elif key.char == '+' or key.char == '=':
                    self.step = min(self.step * 2, 0.5)
                    self._dirty = True
                elif key.char == '-':
                    self.step = max(self.step / 2, 0.01)
                    self._dirty = True
            except AttributeError:
                pass

    @property
    def should_quit(self):
        return self._quit

    def print_state(self):
        if not self._dirty:
            return
        self._dirty = False

        print("\033[2J\033[H", end="")  # clear screen
        print("=" * 60)
        print("  JOINT DIRECTION TEST")
        print("=" * 60)
        print(f"  Step size: {self.step:.3f} rad  (+/- to change)")
        print("-" * 60)

        for i in range(self.num_joints):
            marker = " >> " if i == self.selected else "    "
            name = self.joint_names[i]
            pos = self.positions[i]
            default = self.default_pos[i]

            # Visual bar
            bar_width = 20
            bar_center = bar_width // 2
            bar_pos = int(np.clip(pos / 1.5 * bar_center, -bar_center, bar_center))
            bar = [" "] * bar_width
            bar[bar_center] = "|"
            idx = bar_center + bar_pos
            idx = max(0, min(bar_width - 1, idx))
            bar[idx] = "█"
            bar_str = "".join(bar)

            print(f"{marker}{name:<22} [{bar_str}]  {pos:+.3f} rad  (default: {default:+.3f})")

        print("-" * 60)
        print("  ← →  Select joint    ↑ ↓  Adjust angle")
        print("  0     Zero joint      SPACE Zero all")
        print("  R     Default pose    +/-   Step size")
        print("  ESC   Quit")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BDX-R Joint Direction Test")
    parser.add_argument("--backend", type=str, default="sim", choices=["sim", "real"],
                        help="Backend: 'sim' for MuJoCo, 'real' for hardware")
    parser.add_argument("--model", type=str, default="models/walk.onnx",
                        help="Path to ONNX policy model")
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml",
                        help="Path to MuJoCo XML (sim only)")
    parser.add_argument("--sim_dt", type=float, default=0.005,
                        help="Simulation timestep (sim only)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Angle step per keypress (rad)")
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
            fixed=True,
        )
    elif args.backend == "real":
        from bdx_api.hardware import HardwareBackend
        backend = HardwareBackend(
            model_path=model_path,
            loop_dt=0.02,
            standup_duration=2.0,
            i2c_bus=args.i2c_bus,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    from bdx_api.config import load_policy_config
    cfg = load_policy_config(model_path)

    joint_map = backend.get_joint_map(cfg.joint_names)
    default_pos = np.array(cfg.default_joint_pos, dtype=np.float32)

    ctrl = JointTestController(
        num_joints=len(cfg.joint_names),
        joint_names=[n.strip() for n in cfg.joint_names],
        default_pos=default_pos,
        step=args.step,
    )

    # Build the full target array
    full_targets = backend.get_default_pos_array(len(joint_map))

    print("Starting joint test... (press ESC to quit)")

    while backend.is_running() and not ctrl.should_quit:
        ctrl.print_state()

        # Map controller positions → backend targets
        for i in range(len(cfg.joint_names)):
            full_targets[joint_map[i]] = ctrl.positions[i]

        backend.set_joint_targets(full_targets)
        backend.step()

    print("\nTest complete.")


if __name__ == "__main__":
    main()