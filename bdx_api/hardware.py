import sys
import time
import enum
import math
import struct
import threading
import atexit
import numpy as np
import can
import serial

from bdx_api.config import load_policy_config, STANDUP_GAINS
from bdx_api.interface import RobotBackend
from pathlib import Path
from scipy.spatial.transform import Rotation


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
# Xsens MTi-3 IMU
# ==========================================

class MTi_Serial_IMU:
    def __init__(self, port="/dev/ttyUSB0", baudrate=2000000):
        self.port = port
        self.baudrate = baudrate
        self.latest_data = {
            "gyro": np.zeros(3),
            "projected_gravity": np.array([0.0, 0.0, -1.0]),
        }
        self.lock = threading.Lock()
        self.running = False
        self.serial = None
        self.thread = None

    def _compute_checksum(self, data: bytes) -> int:
        return (-sum(data)) & 0xFF

    def _send_message(self, mid: int, payload: bytes = b''):
        msg = bytes([0xFF, mid, len(payload)]) + payload
        cs = self._compute_checksum(msg)
        self.serial.write(bytes([0xFA]) + msg + bytes([cs]))

    def _configure_outputs(self):
        self._send_message(0x30)  # GoToConfig
        time.sleep(0.1)
        self.serial.reset_input_buffer()
        payload = bytes([
            0x20, 0x10, 0x00, 0x32,  # Quaternion (float32), 50 Hz
            0x80, 0x20, 0x00, 0x32,  # Rate of Turn (float32), 50 Hz
        ])
        self._send_message(0xC0, payload)
        time.sleep(0.2)
        self._send_message(0x10)  # GoToMeasurement
        time.sleep(0.1)
        self.serial.reset_input_buffer()
        print("[MTi-3] Output configuration set: Quaternion + RateOfTurn @ 50 Hz")

    def start(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.serial.reset_input_buffer()
        except Exception as e:
            print(f"[FATAL] Failed to open MTi-3 on {self.port}: {e}", file=sys.stderr)
            return False
        self._configure_outputs()
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"[MTi-3] Connected at {self.baudrate} baud on {self.port}")
        return True

    def _read_loop(self):
        while self.running:
            try:
                if self.serial.read(1) == b'\xFA' and self.serial.read(1) == b'\xFF' and self.serial.read(1) == b'\x36':
                    length_bytes = self.serial.read(1)
                    if not length_bytes: continue
                    length = length_bytes[0]
                    data = self.serial.read(length)
                    checksum_bytes = self.serial.read(1)
                    if len(data) < length or not checksum_bytes: continue
                    if (0xFF + 0x36 + length + sum(data) + checksum_bytes[0]) & 0xFF != 0: continue

                    idx = 0
                    gyro, quat = None, None
                    while idx < length:
                        group = data[idx]; type_id = data[idx+1]; size = data[idx+2]
                        payload = data[idx+3 : idx+3+size]
                        if group == 0x80 and (type_id == 0x20 or type_id == 0x23) and size in (12, 24):
                            gyro = np.array(struct.unpack(">3f" if size == 12 else ">3d", payload))
                        elif group == 0x20 and type_id == 0x10 and size == 16:
                            q = struct.unpack(">4f", payload)
                            quat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().apply([0.0, 0.0, -1.0])
                        idx += 3 + size

                    with self.lock:
                        if gyro is not None: self.latest_data["gyro"] = gyro
                        if quat is not None: self.latest_data["projected_gravity"] = quat
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.serial: self.serial.close()

    def get_latest_data(self):
        with self.lock:
            return {
                "gyro": self.latest_data["gyro"].copy(),
                "projected_gravity": self.latest_data["projected_gravity"].copy(),
            }


# ==========================================
# Joint Definition
# ==========================================

JOINT_WIRING = {
    "Left_Hip_Yaw":    (0, 1,  "O3"),
    "Left_Hip_Roll":   (0, 2,  "O3"),
    "Left_Hip_Pitch":  (0, 3,  "O3"),
    "Left_Knee":       (0, 4,  "O3"),
    "Left_Ankle":      (0, 5,  "O2"),
    "Right_Hip_Yaw":   (1, 6,  "O3"),
    "Right_Hip_Roll":  (1, 7,  "O3"),
    "Right_Hip_Pitch": (1, 8,  "O3"),
    "Right_Knee":      (1, 9,  "O3"),
    "Right_Ankle":     (1, 10, "O2"),
    "Neck_Pitch":      (2, 11, "O2"),
    "Head_Pitch":      (2, 12, "O5"),
    "Head_Yaw":        (2, 13, "O5"),
    "Head_Roll":       (2, 14, "O5"),
}

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

CAN_CHANNELS =["can2", "can0", "can1"]
LPF_ALPHA_POS = 1.0  # Pass raw positions (No delay)
LPF_ALPHA_VEL = 0.1  # Filter velocities (Smooths out the noise)


# ==========================================
# Hardware Backend
# ==========================================
class HardwareBackend(RobotBackend):
    def __init__(self, model_path: Path, loop_dt: float = 0.005,
                 standup_duration: float = 2.0, imu_port: str = "/dev/ttyUSB0",
                 legs_only: bool = False, use_imu: bool = True): # Add use_imu parameter
        
        self.use_imu = use_imu # Store it

        self.cfg = load_policy_config(model_path)
        self.loop_dt = loop_dt
        self._running = True
        self._last_time = time.time()
        self.legs_only = legs_only

        # Debug Freq Counters
        self._debug_tick_count = 0
        self._debug_start_time = time.time()
        self._actual_cmd_freq = 0.0

        # Asimov Latency Tracking
        self._cmd_sent_time = {}
        self._latencies = {}

        # --- NEW: Identify policy joints vs ALL physical joints ---
        self.policy_joints = self.cfg.joint_names
        self.num_policy_joints = len(self.policy_joints)
        self.all_joint_names = list(self.policy_joints)

        # If --legs is flagged, add the head joints so the backend powers them
        if self.legs_only:
            head_joints =["Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll"]
            for hj in head_joints:
                if hj not in self.all_joint_names:
                    self.all_joint_names.append(hj)

        self.num_joints = len(self.all_joint_names)

        self._joint_wiring =[]
        for name in self.all_joint_names:
            name = name.strip()
            if name not in JOINT_WIRING:
                raise ValueError(f"Policy joint '{name}' has no entry in JOINT_WIRING.")
            self._joint_wiring.append(JOINT_WIRING[name])

        self._motor_type_by_id = {}
        for bus_idx, motor_id, motor_type in self._joint_wiring:
            self._motor_type_by_id[motor_id] = motor_type
            self._latencies[motor_id] =[]

        print("Initializing CAN buses...")
        self.buses =[]
        for ch in CAN_CHANNELS:
            bus = can.interface.Bus(interface="socketcan", channel=ch)
            self.buses.append(bus)
        atexit.register(self._shutdown_buses)

        self._motor_ids = set()
        for bus_idx, motor_id, _ in self._joint_wiring:
            self._motor_ids.add(motor_id)

        self._motor_states = {
            motor_id: {"pos": 0.0, "vel": 0.0, "updated_at": 0.0}
            for motor_id in self._motor_ids
        }

        self._filtered_positions = np.zeros(self.num_joints, dtype=np.float32)
        self._filtered_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self._initialized_filter = False

        
        print("Initializing IMU...")
        if self.use_imu:
            self.imu = MTi_Serial_IMU(port=imu_port)
            if not self.imu.start():
                raise RuntimeError("IMU initialization failed")
            atexit.register(self.imu.stop)
        else:
            print("[WARN] IMU disabled. Using dummy values.")
            self.imu = None # Or a dummy class if your code calls get_latest_data()

        print("Enabling all motors...")
        time.sleep(0.5)
        enabled_on_bus = {}
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

        self._update_motor_states()
        for i, (bus_idx, motor_id, _) in enumerate(self._joint_wiring):
            self._filtered_positions[i] = self._motor_states[motor_id]["pos"]
        self._initialized_filter = True

        self._standup_duration = standup_duration
        print("\nHardware backend ready.")

    def _shutdown_buses(self):
        # ... (Identical to original) ...
        print("Disabling all motors...")
        disabled = set()
        for bus_idx, motor_id, _ in self._joint_wiring:
            if motor_id not in disabled:
                disable_id = (MUX_DISABLE << 24) | (HOST_ID << 8) | motor_id
                try:
                    self.buses[bus_idx].send(
                        can.Message(arbitration_id=disable_id, is_extended_id=True, dlc=8)
                    )
                except Exception:
                    pass
                disabled.add(motor_id)

        for bus in self.buses:
            bus.shutdown()
        print("CAN buses shut down.")

    def _update_motor_states(self):
        # ... (Identical to original) ...
        for bus in self.buses:
            while True:
                msg = bus.recv(timeout=0)
                if msg is None: break
                if msg.is_error_frame: continue
                try:
                    motor_id = (msg.arbitration_id & 0xFF00) >> 8
                    msg_type = (msg.arbitration_id & 0x1F000000) >> 24
                except Exception: continue
                if msg_type != MotorMsg.Feedback.value: continue
                if motor_id not in self._motor_states: continue

                now = time.perf_counter()
                if motor_id in self._cmd_sent_time:
                    lat = now - self._cmd_sent_time[motor_id]
                    self._latencies[motor_id].append(lat)
                    if len(self._latencies[motor_id]) > 100:
                        self._latencies[motor_id].pop(0)

                motor_type = self._motor_type_by_id.get(motor_id, "O3")
                params = MOTOR_TYPE_PARAMS[motor_type]

                try:
                    angle_raw = struct.unpack(">H", msg.data[0:2])[0]
                    p_range = params["P_MAX"] - params["P_MIN"]
                    pos_rad = (float(angle_raw) / 65535.0) * p_range + params["P_MIN"]

                    vel_raw = struct.unpack(">H", msg.data[2:4])[0]
                    v_range = params["V_MAX"] - params["V_MIN"]
                    vel_rps = (float(vel_raw) / 65535.0) * v_range + params["V_MIN"]

                    self._motor_states[motor_id]["pos"] = pos_rad
                    self._motor_states[motor_id]["vel"] = vel_rps
                    self._motor_states[motor_id]["updated_at"] = time.time()
                except (struct.error, IndexError):
                    pass

    def _read_and_filter(self):
        self._update_motor_states()

        for i, (bus_idx, motor_id, _) in enumerate(self._joint_wiring):
            raw_pos = self._motor_states[motor_id]["pos"]
            raw_vel = self._motor_states[motor_id]["vel"]

            if self._initialized_filter:
                # Use POS alpha for positions (1.0 = raw)
                self._filtered_positions[i] = (LPF_ALPHA_POS * raw_pos + (1.0 - LPF_ALPHA_POS) * self._filtered_positions[i])
                # Use VEL alpha for velocities (0.1 = smoothed)
                self._filtered_velocities[i] = (LPF_ALPHA_VEL * raw_vel + (1.0 - LPF_ALPHA_VEL) * self._filtered_velocities[i])
            else:
                self._filtered_positions[i] = raw_pos
                self._filtered_velocities[i] = raw_vel

    def _send_targets(self, targets: np.ndarray, use_standup_gains: bool = False):
        for i, (bus_idx, motor_id, motor_type) in enumerate(self._joint_wiring):
            name = self.all_joint_names[i].strip()
            params = MOTOR_TYPE_PARAMS[motor_type]

            # Determine if this is a "headless" joint that the policy knows nothing about
            is_fixed_head_joint = self.legs_only and i >= self.num_policy_joints

            if use_standup_gains or is_fixed_head_joint:
                if name in STANDUP_GAINS:
                    kp, kd = STANDUP_GAINS[name]
                else:
                    kp, kd = 20.0, 1.0  # Safe fallback
            else:
                kp = self.cfg.joint_stiffness[i]
                kd = self.cfg.joint_damping[i]

            pos = float(targets[i])
            if name in JOINT_LIMITS:
                lo, hi = JOINT_LIMITS[name]
                pos = float(np.clip(pos, lo, hi))

            self._cmd_sent_time[motor_id] = time.perf_counter()

            _send_mit_command(
                self.buses[bus_idx], motor_id,
                pos_rad=pos, vel_rad_s=0.0,
                kp=kp, kd=kd, torque_nm=0.0,
                params=params,
            )

    def standup(self, duration: float = 2.0) -> None:
        dur = duration if duration else self._standup_duration
        print(f"Standing up over {dur}s...")

        self._read_and_filter()
        start_pos = self._filtered_positions.copy()
        
        # Initialize full 14 joints to 0.0, then overwrite the policy joints with their default
        target_pos = np.zeros(self.num_joints, dtype=np.float32)
        target_pos[:self.num_policy_joints] = self.cfg.default_joint_pos

        num_steps = int(dur / 0.02)
        for step in range(num_steps + 1):
            alpha = step / float(num_steps)
            interp = (1.0 - alpha) * start_pos + alpha * target_pos
            self._send_targets(interp, use_standup_gains=True)
            time.sleep(0.02)

        print("Standup complete.")

    def hold_standing_tick(self) -> None:
        target_pos = np.zeros(self.num_joints, dtype=np.float32)
        target_pos[:self.num_policy_joints] = self.cfg.default_joint_pos
        self._read_and_filter()
        self._send_targets(target_pos, use_standup_gains=True)
        time.sleep(0.02)

    def activate_policy_gains(self) -> None:
        print("\n[DEBUG] Activating Policy: Flushing CAN Bus to prevent cold-start jerk...")
        for bus in self.buses:
            _flush_bus(bus)
        self._read_and_filter()
        print("[DEBUG] Hardware buffers flushed.")

    def get_imu_angular_velocity(self) -> np.ndarray: 
        if self.use_imu:
            return self.imu.get_latest_data()["gyro"]
        return np.zeros(3)

    def get_projected_gravity(self) -> np.ndarray: 
        if self.use_imu:
            return self.imu.get_latest_data()["projected_gravity"]
        return np.array([0.0, 0.0, -1.0])
    
    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray:
        self._read_and_filter()
        return self._filtered_positions[joint_ids].copy()

    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray:
        return self._filtered_velocities[joint_ids].copy()

    def set_joint_targets(self, targets: np.ndarray) -> None:
        self._send_targets(targets)

    def step(self) -> None:
        elapsed = time.time() - self._last_time
        sleep_time = self.loop_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_time = time.time()

        self._debug_tick_count += 1
        if self._debug_tick_count >= 50:
            now = time.time()
            duration = now - self._debug_start_time
            if duration > 0:
                self._actual_cmd_freq = self._debug_tick_count / duration
            self._debug_tick_count = 0
            self._debug_start_time = now

    def is_running(self) -> bool: return self._running

    def get_joint_map(self, joint_names: list[str]) -> np.ndarray:
        # Maps the policy's requested joints into our backend's internal list
        return np.array([self.all_joint_names.index(n) for n in joint_names], dtype=int)

    def get_default_pos_array(self, num_actuators: int) -> np.ndarray:
        # Even if the policy only asks for 10 joints, return the full 14 array size so the runner pads it for us
        return np.zeros(self.num_joints, dtype=np.float32)

    def get_actual_frequency(self) -> float: return self._actual_cmd_freq
    def get_latency_stats(self):
        stats = {}
        for mid, lats in self._latencies.items():
            if len(lats) > 0:
                stats[mid] = {"avg": np.mean(lats)*1000.0, "max": np.max(lats)*1000.0, "std": np.std(lats)*1000.0}
        return stats