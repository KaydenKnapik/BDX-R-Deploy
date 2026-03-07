import sys
import time
import enum
import math
import struct
import threading
import atexit
import numpy as np

import can

from bdx_api.config import load_policy_config, STANDUP_GAINS
from bdx_api.interface import RobotBackend
from pathlib import Path


# ==========================================
# Motor Protocol (MIT Mode)
# ==========================================

MUX_CONTROL = 0x01


class MotorMsg(enum.Enum):
    Feedback = 2


# Motor type parameters for scaling
MOTOR_TYPE_PARAMS = {
    "O2": {
        "P_MIN": -12.57, "P_MAX": 12.57,
        "V_MIN": -44.0,  "V_MAX": 44.0,
        "T_MIN": -17.0,  "T_MAX": 17.0,
        "KP_MIN": 0.0,   "KP_MAX": 500.0,
        "KD_MIN": 0.0,   "KD_MAX": 5.0,
    },
    "O3": {
        "P_MIN": -12.57, "P_MAX": 12.57,
        "V_MIN": -20.0,  "V_MAX": 20.0,
        "T_MIN": -60.0,  "T_MAX": 60.0,
        "KP_MIN": 0.0,   "KP_MAX": 5000.0,
        "KD_MIN": 0.0,   "KD_MAX": 100.0,
    },
    "O5": {
        "P_MIN": -12.57, "P_MAX": 12.57,
        "V_MIN": -50.0,  "V_MAX": 50.0,
        "T_MIN": -5.5,   "T_MAX": 5.5,
        "KP_MIN": 0.0,   "KP_MAX": 500.0,
        "KD_MIN": 0.0,   "KD_MAX": 5.0,
    },
}

MUX_ENABLE = 0x03
MUX_DISABLE = 0x04
HOST_ID = 0xFD


def _scale_to_u16(value, v_min, v_max):
    return int(65535.0 * (np.clip(value, v_min, v_max) - v_min) / (v_max - v_min))


def _send_mit_command(bus, motor_id, pos_rad, vel_rad_s, kp, kd, torque_nm, params):
    """Send a single MIT mode position command to a motor."""
    angle_u16 = _scale_to_u16(pos_rad, params["P_MIN"], params["P_MAX"])
    vel_u16 = _scale_to_u16(vel_rad_s, params["V_MIN"], params["V_MAX"])
    kp_u16 = _scale_to_u16(kp, params["KP_MIN"], params["KP_MAX"])
    kd_u16 = _scale_to_u16(kd, params["KD_MIN"], params["KD_MAX"])
    torque_u16 = _scale_to_u16(torque_nm, params["T_MIN"], params["T_MAX"])

    arb_id = ((MUX_CONTROL & 0xFF) << 24) | ((torque_u16 & 0xFFFF) << 8) | (motor_id & 0xFF)
    data = struct.pack(">HHHH", angle_u16, vel_u16, kp_u16, kd_u16)

    try:
        bus.send(can.Message(arbitration_id=arb_id, data=data, is_extended_id=True, dlc=8))
    except can.CanError as e:
        print(f"[CAN ERROR] Motor {motor_id}: {e}", file=sys.stderr)


def _flush_bus(bus):
    while True:
        msg = bus.recv(timeout=0.01)
        if msg is None:
            break


# ==========================================
# BNO055 IMU
# ==========================================

class BNO055_IMU:
    """BNO055 IMU reader running in a background thread."""

    def __init__(self, i2c_bus_num=7, calibration_samples=100, frequency=200.0):
        self.i2c_bus_num = i2c_bus_num
        self.calibration_samples = calibration_samples
        self.frequency = frequency
        self.sensor = None
        self.offsets = {}
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_data = {
            "gyro": np.zeros(3, dtype=np.float32),
            "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }

    def _initialize_sensor(self):
        from adafruit_extended_bus import ExtendedI2C as I2C
        import adafruit_bno055

        try:
            self.sensor = adafruit_bno055.BNO055_I2C(I2C(self.i2c_bus_num))
            time.sleep(2)
            return True
        except Exception as e:
            print(f"FATAL: Failed to initialize BNO055 IMU: {e}", file=sys.stderr)
            return False

    def calibrate(self):
        if not self._initialize_sensor():
            return False

        from scipy.spatial.transform import Rotation

        print("--- Starting IMU Calibration ---")
        time.sleep(2)

        gyro_data = []
        gravity_data = []
        for i in range(self.calibration_samples):
            gyro = self.sensor.gyro
            gravity = self.sensor.gravity
            if gyro is not None and gravity is not None:
                gyro_data.append(gyro)
                gravity_data.append(gravity)
            sys.stdout.write(f"\rCollecting samples... {i + 1}/{self.calibration_samples}")
            sys.stdout.flush()
            time.sleep(0.05)

        print("\n--- Calibration Calculations ---")

        if not gyro_data or not gravity_data:
            print("FATAL: No valid IMU data during calibration.", file=sys.stderr)
            return False

        try:
            self.offsets["gyro"] = tuple(np.mean(gyro_data, axis=0))

            avg_gravity = np.mean(np.array(gravity_data), axis=0)
            at_rest = avg_gravity / np.linalg.norm(avg_gravity)
            at_rest[2] = -at_rest[2]

            ideal = np.array([0.0, 0.0, -1.0])
            correction, _ = Rotation.align_vectors([ideal], [at_rest])
            self.offsets["gravity_correction_rotation"] = correction

            print(f"Gyro offset: {self.offsets['gyro']}")
            print("--- Calibration Complete ---")
            return True
        except Exception as e:
            print(f"FATAL: IMU calibration error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False

    def _read_loop(self):
        while self.running:
            loop_start = time.time()
            if self.sensor and "gyro" in self.offsets and "gravity_correction_rotation" in self.offsets:
                try:
                    raw_gyro = self.sensor.gyro
                    gravity_vector = self.sensor.gravity
                    if raw_gyro is None or gravity_vector is None:
                        continue

                    # 1. Calibrate raw gyro
                    calibrated_gyro = np.array(raw_gyro) - np.array(self.offsets["gyro"])
                    
                    # 2. MATCH ROBOT_SENDER.PY AXES FLIP!
                    calibrated_gyro[0] *= -1
                    calibrated_gyro[1] *= -1

                    # 3. Process Gravity (this part was already correct)
                    proj_grav = np.array(gravity_vector)
                    magnitude = np.linalg.norm(proj_grav)
                    if magnitude > 1e-6:
                        proj_grav /= magnitude
                    else:
                        proj_grav = np.array([0.0, 0.0, 1.0])

                    proj_grav[2] = -proj_grav[2]
                    aligned = self.offsets["gravity_correction_rotation"].apply(proj_grav)
                    aligned[0] *= -1
                    aligned[1] *= -1

                    with self.lock:
                        # Save the fixed gyro
                        self.latest_data["gyro"] = calibrated_gyro.astype(np.float32)
                        self.latest_data["projected_gravity"] = aligned.astype(np.float32)
                except Exception as e:
                    print(f"IMU read error: {e}", file=sys.stderr)

            elapsed = time.time() - loop_start
            sleep_time = (1.0 / self.frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        if not self.calibrate():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("IMU reader thread started.")
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("IMU reader thread stopped.")

    def get_latest_data(self):
        with self.lock:
            return {
                "gyro": self.latest_data["gyro"].copy(),
                "projected_gravity": self.latest_data["projected_gravity"].copy(),
            }


# ==========================================
# Joint Definition
# ==========================================

# Each entry: (policy_joint_name, can_bus_index, motor_id, motor_type)
# The ORDER here must match the ONNX joint_names from the policy metadata.
# We build this mapping dynamically in __init__ based on the ONNX joint names.

# Physical wiring table — maps joint name → (bus_idx, motor_id, motor_type)
# bus_idx: 0=can2, 1=can1, 2=can0 (matches CAN_CHANNELS order)
JOINT_WIRING = {
    # Left leg (can2 = bus 0)
    "Left_Hip_Yaw":    (0, 1,  "O3"),
    "Left_Hip_Roll":   (0, 2,  "O3"),
    "Left_Hip_Pitch":  (0, 3,  "O3"),
    "Left_Knee":       (0, 4,  "O3"),
    "Left_Ankle":      (0, 5,  "O2"),
    # Right leg (can1 = bus 1)
    "Right_Hip_Yaw":   (1, 6,  "O3"),
    "Right_Hip_Roll":  (1, 7,  "O3"),
    "Right_Hip_Pitch": (1, 8,  "O3"),
    "Right_Knee":      (1, 9,  "O3"),
    "Right_Ankle":     (1, 10, "O2"),
    # Head (can0 = bus 2)
    "Neck_Pitch":      (2, 11, "O2"),
    "Head_Pitch":      (2, 12, "O5"),
    "Head_Yaw":        (2, 13, "O5"),
    "Head_Roll":       (2, 14, "O5"),
}

# STANDUP_GAINS imported from bdx_api.config

# Joint safety limits (rad)
JOINT_LIMITS = {
    "Left_Hip_Yaw":    (-0.4, 0.4),
    "Left_Hip_Roll":   (-0.34, 0.34),
    "Left_Hip_Pitch":  (-0.75, 0.7),
    "Left_Knee":       (-0.94, 1.3),
    "Left_Ankle":      (-0.84, 1.2),
    "Right_Hip_Yaw":   (-0.4, 0.4),
    "Right_Hip_Roll":  (-0.34, 0.34),
    "Right_Hip_Pitch": (-0.75, 0.7),
    "Right_Knee":      (-1.3, 0.94),
    "Right_Ankle":     (-1.2, 0.84),
    "Neck_Pitch":      (-0.35, 0.87),
    "Head_Pitch":      (-1.05, 0.87),
    "Head_Yaw":        (-0.87, 0.87),
    "Head_Roll":       (-0.52, 0.52),
}

# CAN bus channels (order matches bus_idx in JOINT_WIRING)
CAN_CHANNELS = ["can2", "can1", "can0"]

# Low-pass filter alpha for joint readings
LPF_ALPHA = 1.0  # 1.0 = no filtering, raw values pass through


# ==========================================
# Hardware Backend
# ==========================================

class HardwareBackend(RobotBackend):
    """Real robot backend using Robstride motors over CAN + BNO055 IMU."""

    def __init__(self, model_path: Path, loop_dt: float = 0.005,
                 standup_duration: float = 2.0, i2c_bus: int = 7):

        self.cfg = load_policy_config(model_path)
        self.loop_dt = loop_dt
        self._running = True
        self._last_time = time.time()

        # --- Validate all policy joints are wired ---
        self.num_joints = len(self.cfg.joint_names)
        self._joint_wiring = []
        for name in self.cfg.joint_names:
            name = name.strip()
            if name not in JOINT_WIRING:
                raise ValueError(f"Policy joint '{name}' has no entry in JOINT_WIRING.")
            self._joint_wiring.append(JOINT_WIRING[name])

        # --- Build motor_id → motor_type lookup for feedback decoding ---
        self._motor_type_by_id = {}
        for bus_idx, motor_id, motor_type in self._joint_wiring:
            self._motor_type_by_id[motor_id] = motor_type

        # --- Initialize CAN buses ---
        print("Initializing CAN buses...")
        self.buses = []
        for ch in CAN_CHANNELS:
            bus = can.interface.Bus(interface="socketcan", channel=ch)
            self.buses.append(bus)
        atexit.register(self._shutdown_buses)

        # --- Motor state tracking ---
        self._motor_ids = set()
        for bus_idx, motor_id, _ in self._joint_wiring:
            self._motor_ids.add(motor_id)

        self._motor_states = {
            motor_id: {"pos": 0.0, "vel": 0.0, "updated_at": 0.0}
            for motor_id in self._motor_ids
        }

        # --- Filtered joint state ---
        self._filtered_positions = np.zeros(self.num_joints, dtype=np.float32)
        self._filtered_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self._initialized_filter = False

        # --- Initialize IMU ---
        print("Initializing IMU...")
        self.imu = BNO055_IMU(i2c_bus_num=i2c_bus, frequency=200.0)
        if not self.imu.start():
            raise RuntimeError("IMU initialization failed")
        atexit.register(self.imu.stop)

        # --- Enable motors ---
        print("Enabling all motors...")
        time.sleep(0.5)
        enabled_on_bus = {}  # bus_idx → set of motor_ids already enabled
        for bus_idx, motor_id, _ in self._joint_wiring:
            if bus_idx not in enabled_on_bus:
                enabled_on_bus[bus_idx] = set()
            if motor_id not in enabled_on_bus[bus_idx]:
                enable_id = (MUX_ENABLE << 24) | (HOST_ID << 8) | motor_id
                self.buses[bus_idx].send(
                    can.Message(arbitration_id=enable_id, is_extended_id=True, dlc=8)
                )
                enabled_on_bus[bus_idx].add(motor_id)
        time.sleep(0.5)

        # --- Read initial positions ---
        self._update_motor_states()
        for i, (bus_idx, motor_id, _) in enumerate(self._joint_wiring):
            self._filtered_positions[i] = self._motor_states[motor_id]["pos"]
        self._initialized_filter = True

        self._standup_duration = standup_duration
        print("\nHardware backend ready.")

    def _shutdown_buses(self):
        """Disable all motors and shutdown CAN."""
        print("Disabling all motors...")
        disabled = set()
        for bus_idx, motor_id, _ in self._joint_wiring:
            if motor_id not in disabled:
                disable_id = (MUX_DISABLE << 24) | (HOST_ID << 8) | motor_id
                try:
                    self.buses[bus_idx].send(
                        can.Message(arbitration_id=disable_id, is_extended_id=True, dlc=8)
                    )
                except Exception as e:
                    print(f"  Failed to disable motor {motor_id}: {e}")
                disabled.add(motor_id)

        for bus in self.buses:
            bus.shutdown()
        print("CAN buses shut down.")

    def _update_motor_states(self):
        """Read all pending CAN feedback messages and decode using correct motor type."""
        for bus in self.buses:
            while True:
                msg = bus.recv(timeout=0)
                if msg is None:
                    break
                if msg.is_error_frame:
                    continue
                try:
                    motor_id = (msg.arbitration_id & 0xFF00) >> 8
                    msg_type = (msg.arbitration_id & 0x1F000000) >> 24
                except Exception:
                    continue
                if msg_type != MotorMsg.Feedback.value:
                    continue
                if motor_id not in self._motor_states:
                    continue

                # Look up motor type for correct velocity scaling
                motor_type = self._motor_type_by_id.get(motor_id, "O3")
                params = MOTOR_TYPE_PARAMS[motor_type]

                try:
                    # Position decoding (same for all types: ±12.57 rad)
                    angle_raw = struct.unpack(">H", msg.data[0:2])[0]
                    p_range = params["P_MAX"] - params["P_MIN"]
                    pos_rad = (float(angle_raw) / 65535.0) * p_range + params["P_MIN"]

                    # Velocity decoding (varies by motor type)
                    vel_raw = struct.unpack(">H", msg.data[2:4])[0]
                    v_range = params["V_MAX"] - params["V_MIN"]
                    vel_rps = (float(vel_raw) / 65535.0) * v_range + params["V_MIN"]

                    self._motor_states[motor_id]["pos"] = pos_rad
                    self._motor_states[motor_id]["vel"] = vel_rps
                    self._motor_states[motor_id]["updated_at"] = time.time()
                except (struct.error, IndexError):
                    pass

    def _read_and_filter(self):
        """Read motor states and apply LPF."""
        self._update_motor_states()

        for i, (bus_idx, motor_id, _) in enumerate(self._joint_wiring):
            raw_pos = self._motor_states[motor_id]["pos"]
            raw_vel = self._motor_states[motor_id]["vel"]

            if self._initialized_filter:
                self._filtered_positions[i] = (
                    LPF_ALPHA * raw_pos + (1.0 - LPF_ALPHA) * self._filtered_positions[i]
                )
                self._filtered_velocities[i] = (
                    LPF_ALPHA * raw_vel + (1.0 - LPF_ALPHA) * self._filtered_velocities[i]
                )
            else:
                self._filtered_positions[i] = raw_pos
                self._filtered_velocities[i] = raw_vel

    def _send_targets(self, targets: np.ndarray, use_standup_gains: bool = False):
        """Send position targets with PD gains to all joints.

        Args:
            targets: Position targets (rad), one per policy joint.
            use_standup_gains: If True, use the safe STANDUP_GAINS instead of
                               the policy-trained gains from ONNX metadata.
        """
        for i, (bus_idx, motor_id, motor_type) in enumerate(self._joint_wiring):
            name = self.cfg.joint_names[i].strip()
            params = MOTOR_TYPE_PARAMS[motor_type]

            if use_standup_gains and name in STANDUP_GAINS:
                kp, kd = STANDUP_GAINS[name]
            else:
                kp = self.cfg.joint_stiffness[i]
                kd = self.cfg.joint_damping[i]

            # Apply safety limits
            if name in JOINT_LIMITS:
                lo, hi = JOINT_LIMITS[name]
                pos = float(np.clip(targets[i], lo, hi))
            else:
                pos = float(targets[i])

            _send_mit_command(
                self.buses[bus_idx], motor_id,
                pos_rad=pos, vel_rad_s=0.0,
                kp=kp, kd=kd, torque_nm=0.0,
                params=params,
            )

    # ==========================================
    # Standup / Hold / Deploy (called by PolicyRunner)
    # ==========================================

    def standup(self, duration: float = 2.0) -> None:
        dur = duration if duration else self._standup_duration
        print(f"Standing up over {dur}s...")

        self._read_and_filter()
        start_pos = self._filtered_positions.copy()
        target_pos = np.array(self.cfg.default_joint_pos, dtype=np.float32)

        num_steps = int(dur / 0.02)
        for step in range(num_steps + 1):
            alpha = step / float(num_steps)
            interp = (1.0 - alpha) * start_pos + alpha * target_pos
            self._send_targets(interp, use_standup_gains=True)
            time.sleep(0.02)

        print("Standup complete.")

    def hold_standing_tick(self) -> None:
        target_pos = np.array(self.cfg.default_joint_pos, dtype=np.float32)
        self._read_and_filter()
        self._send_targets(target_pos, use_standup_gains=True)
        time.sleep(0.02)

    def activate_policy_gains(self) -> None:
        # No-op — _send_targets already uses policy gains when
        # use_standup_gains=False (the default called by set_joint_targets).
        pass

    # ==========================================
    # RobotBackend implementation
    # ==========================================

    def get_imu_angular_velocity(self) -> np.ndarray:
        return self.imu.get_latest_data()["gyro"]

    def get_projected_gravity(self) -> np.ndarray:
        return self.imu.get_latest_data()["projected_gravity"]

    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        self._read_and_filter()
        return self._filtered_positions[joint_ids].copy()

    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        return self._filtered_velocities[joint_ids].copy()

    def set_joint_targets(self, targets: np.ndarray) -> None:
        # targets is the full default_pos-sized array from PolicyRunner
        # We need to extract just our joints (ordered by policy index)
        # PolicyRunner sets targets[joint_map[i]], and our joint_map is identity (0..N-1)
        self._send_targets(targets)

    def step(self) -> None:
        # Maintain fixed loop rate
        elapsed = time.time() - self._last_time
        sleep_time = self.loop_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.time()

    def is_running(self) -> bool:
        return self._running

    def get_joint_map(self, joint_names: list[str]) -> np.ndarray:
        # On hardware, policy joint index i maps directly to index i
        # (the _joint_wiring list is already ordered by policy joint order)
        return np.arange(len(joint_names), dtype=int)

    def get_default_pos_array(self, num_actuators: int) -> np.ndarray:
        return np.zeros(num_actuators, dtype=np.float32)