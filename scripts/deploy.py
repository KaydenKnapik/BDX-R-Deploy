import os
import sys
import json
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
from pynput import keyboard
from pathlib import Path

# --- Utils ---
def quat_rotate_inverse(q, v):
    """Rotates a vector v by the inverse of quaternion q. Assumes q is [x, y, z, w]"""
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def parse_metadata_value(val):
    """Converts ONNX metadata strings into Python lists or floats."""
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        pass
    cleaned = val.replace('[', '').replace(']', '').strip()
    if ',' in cleaned:
        parts = cleaned.split(',')
        try:
            return [float(p.strip()) for p in parts]
        except ValueError:
            return [p.strip() for p in parts]
    elif ' ' in cleaned:
        parts = cleaned.split()
        try:
            return [float(p.strip()) for p in parts]
        except ValueError:
            pass
    try:
        return float(val)
    except ValueError:
        return val

def get_item(src, idx):
    if isinstance(src, list): return float(src[idx])
    return float(src)


# --- Keyboard Controller ---
class KeyboardController:
    """Non-blocking keyboard controller using pynput.
    
    Controls:
        ↑ / ↓      - Increase / Decrease forward speed
        ← / →      - Increase / Decrease lateral speed
        Z / X       - Turn Left / Right
        SPACE       - Stop all movement
        ESC         - Quit
    
    Each tap increments/decrements by one step. Starts at 0,0,0.
    """

    def __init__(self, lin_vel_step=0.1, ang_vel_step=0.1,
                 lin_vel_x_range=(-1.0, 1.0),
                 lin_vel_y_range=(-0.4, 0.4),
                 ang_vel_z_range=(-1.0, 1.0)):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0

        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range

        self._quit = False

        self._listener = keyboard.Listener(
            on_press=self._on_press,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        # Arrow keys
        if key == keyboard.Key.up:
            self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_step, self.lin_vel_x_range[1])
        elif key == keyboard.Key.down:
            self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_step, self.lin_vel_x_range[0])
        elif key == keyboard.Key.left:
            self.lin_vel_y = min(self.lin_vel_y + self.lin_vel_step, self.lin_vel_y_range[1])
        elif key == keyboard.Key.right:
            self.lin_vel_y = max(self.lin_vel_y - self.lin_vel_step, self.lin_vel_y_range[0])
        elif key == keyboard.Key.space:
            self.lin_vel_x = 0.0
            self.lin_vel_y = 0.0
            self.ang_vel_yaw = 0.0
        elif key == keyboard.Key.esc:
            self._quit = True
        else:
            # Character keys
            try:
                if key.char == 'z':
                    self.ang_vel_yaw = min(self.ang_vel_yaw + self.ang_vel_step, self.ang_vel_z_range[1])
                elif key.char == 'x':
                    self.ang_vel_yaw = max(self.ang_vel_yaw - self.ang_vel_step, self.ang_vel_z_range[0])
            except AttributeError:
                pass

    @property
    def should_quit(self):
        return self._quit

    def get_command(self):
        # Round to avoid float drift
        return np.array([
            round(self.lin_vel_x, 2),
            round(self.lin_vel_y, 2),
            round(self.ang_vel_yaw, 2),
        ], dtype=np.float32)


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