#!/usr/bin/env python3
"""
Robot State UDP Sender — runs on the Jetson.
Reads all 14 joint motor positions/velocities + BNO055 IMU data
and broadcasts over UDP for use in MuJoCo sim visualization.

Usage:
    python robot_sender.py
"""

import time
import sys
import socket
import struct
import numpy as np
import threading
import can
from scipy.spatial.transform import Rotation

# =========================================================================
# CONFIGURATION
# =========================================================================
I2C_BUS = 7
CALIBRATION_SAMPLES = 100
READ_FREQUENCY = 100.0    # Hz — IMU read rate
SEND_FREQUENCY = 100.0    # Hz — UDP send rate
UDP_PORT = 5006           # Different port from imu-only sender
BROADCAST_ADDR = "255.255.255.255"

# --- CAN BUS ---
CAN_CONFIGS = [
    ('can2', [1, 2, 3, 4, 5]),
    ('can1', [6, 7, 8, 9, 10]),
    ('can0', [11, 12, 13, 14]),
]

# --- MOTOR TYPES ---
MOTOR_TYPE_PARAMS = {
    'O2': {'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -44.0, 'V_MAX': 44.0, 'T_MIN': -17.0, 'T_MAX': 17.0, 'KP_MIN': 0.0, 'KP_MAX': 500.0, 'KD_MIN': 0.0, 'KD_MAX': 5.0},
    'O3': {'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -20.0, 'V_MAX': 20.0, 'T_MIN': -60.0, 'T_MAX': 60.0, 'KP_MIN': 0.0, 'KP_MAX': 5000.0, 'KD_MIN': 0.0, 'KD_MAX': 100.0},
    'O5': {'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -50.0, 'V_MAX': 50.0, 'T_MIN': -5.5,  'T_MAX': 5.5,  'KP_MIN': 0.0, 'KP_MAX': 500.0, 'KD_MIN': 0.0, 'KD_MAX': 5.0},
}

MOTOR_ID_TO_TYPE = {
    1: 'O3', 2: 'O3', 3: 'O3', 4: 'O3', 5: 'O2',
    6: 'O3', 7: 'O3', 8: 'O3', 9: 'O3', 10: 'O2',
    11: 'O2',
    12: 'O5', 13: 'O5', 14: 'O5',
}

MOTOR_NAMES = {
    1: 'Left_Hip_Yaw',   2: 'Left_Hip_Roll',   3: 'Left_Hip_Pitch',  4: 'Left_Knee',   5: 'Left_Ankle',
    6: 'Right_Hip_Yaw',  7: 'Right_Hip_Roll',  8: 'Right_Hip_Pitch', 9: 'Right_Knee',  10: 'Right_Ankle',
    11: 'Neck_Pitch',    12: 'Head_Pitch',     13: 'Head_Yaw',       14: 'Head_Roll',
}

# Ordered list of all motor IDs (this is the order data is packed in the UDP packet)
ALL_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
NUM_JOINTS = len(ALL_MOTOR_IDS)

# =========================================================================
# MOTOR STATE PARSING (matches findangles.py approach)
# =========================================================================
motor_states = {mid: {'pos': 0.0, 'vel': 0.0} for mid in ALL_MOTOR_IDS}

def scale_value_to_u16(value, v_min, v_max):
    return int(65535.0 * (max(v_min, min(v_max, value)) - v_min) / (v_max - v_min))

def unscale_u16_to_float(val_u16, v_min, v_max):
    return (float(val_u16) / 65535.0) * (v_max - v_min) + v_min

def poll_motor_state(bus, motor_id):
    """Send a zero-torque MIT command to trigger a feedback response."""
    mtype = MOTOR_ID_TO_TYPE.get(motor_id, 'O3')
    params = MOTOR_TYPE_PARAMS[mtype]
    angle_u16 = scale_value_to_u16(0.0, params['P_MIN'], params['P_MAX'])
    vel_u16 = scale_value_to_u16(0.0, params['V_MIN'], params['V_MAX'])
    kp_u16 = scale_value_to_u16(0.0, params['KP_MIN'], params['KP_MAX'])
    kd_u16 = scale_value_to_u16(0.0, params['KD_MIN'], params['KD_MAX'])
    torque_u16 = scale_value_to_u16(0.0, params['T_MIN'], params['T_MAX'])
    arbitration_id = (0x01 << 24) | (torque_u16 << 8) | motor_id
    data = struct.pack('>HHHH', angle_u16, vel_u16, kp_u16, kd_u16)
    try:
        bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True, dlc=8))
    except can.CanError:
        pass

def update_all_motor_states_from_buffer(buses):
    for bus in buses:
        while True:
            msg = bus.recv(timeout=0)
            if msg is None:
                break
            if msg.is_error_frame:
                continue
            try:
                msg_type_val = (msg.arbitration_id & 0x1F000000) >> 24
                motor_id = (msg.arbitration_id & 0xFF00) >> 8
                if motor_id not in motor_states:
                    motor_id = msg.arbitration_id & 0xFF
            except Exception:
                continue
            if msg_type_val == 2 and motor_id in motor_states:
                try:
                    mtype = MOTOR_ID_TO_TYPE.get(motor_id, 'O3')
                    params = MOTOR_TYPE_PARAMS[mtype]
                    p_raw = struct.unpack('>H', msg.data[0:2])[0]
                    v_raw = struct.unpack('>H', msg.data[2:4])[0]
                    motor_states[motor_id]['pos'] = unscale_u16_to_float(p_raw, params['P_MIN'], params['P_MAX'])
                    motor_states[motor_id]['vel'] = unscale_u16_to_float(v_raw, params['V_MIN'], params['V_MAX'])
                except Exception:
                    pass

def flush_bus(bus):
    while True:
        msg = bus.recv(timeout=0.01)
        if msg is None:
            break

# =========================================================================
# IMU
# =========================================================================
class BNO055_IMU:
    def __init__(self, i2c_bus_num=I2C_BUS, calibration_samples=CALIBRATION_SAMPLES, frequency=READ_FREQUENCY):
        self.i2c_bus_num = i2c_bus_num
        self.calibration_samples = calibration_samples
        self.frequency = frequency
        self.sensor = None
        self.gyro_offset = np.zeros(3)
        self.gravity_correction = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_data = {
            "gyro": np.zeros(3),
            "projected_gravity": np.array([0.0, 0.0, -1.0]),
        }

    def _init_sensor(self):
        from adafruit_extended_bus import ExtendedI2C as I2C
        import adafruit_bno055
        try:
            self.sensor = adafruit_bno055.BNO055_I2C(I2C(self.i2c_bus_num))
            time.sleep(1)
            print(f"BNO055 detected on I2C bus {self.i2c_bus_num}")
            return True
        except Exception as e:
            print(f"FATAL: Could not initialize BNO055: {e}", file=sys.stderr)
            return False

    def calibrate(self):
        if not self._init_sensor():
            return False
        print(f"Calibrating IMU ({self.calibration_samples} samples) — keep still...")
        time.sleep(1)
        gyro_samples, gravity_samples = [], []
        for i in range(self.calibration_samples):
            gyro = self.sensor.gyro
            gravity = self.sensor.gravity
            if gyro is not None and gravity is not None:
                if None not in gyro and None not in gravity:
                    gyro_samples.append(gyro)
                    gravity_samples.append(gravity)
            sys.stdout.write(f"\r  Sample {i + 1}/{self.calibration_samples}")
            sys.stdout.flush()
            time.sleep(0.05)
        print()
        if len(gyro_samples) < 10 or len(gravity_samples) < 10:
            print("FATAL: Not enough valid IMU samples.", file=sys.stderr)
            return False
        self.gyro_offset = np.mean(gyro_samples, axis=0)
        avg_gravity = np.mean(gravity_samples, axis=0)
        avg_gravity_norm = avg_gravity / np.linalg.norm(avg_gravity)
        avg_gravity_norm[2] = -avg_gravity_norm[2]
        ideal = np.array([0.0, 0.0, -1.0])
        self.gravity_correction, _ = Rotation.align_vectors([ideal], [avg_gravity_norm])
        print(f"  Gyro offset : {np.array2string(self.gyro_offset, precision=4)}")
        print("IMU calibration complete.\n")
        return True

    def _read_loop(self):
        while self.running:
            t0 = time.time()
            try:
                raw_gyro = self.sensor.gyro
                raw_gravity = self.sensor.gravity
                if raw_gyro is None or raw_gravity is None:
                    continue
                if None in raw_gyro or None in raw_gravity:
                    continue
                gyro = np.array(raw_gyro) - self.gyro_offset
                gyro[0] *= -1
                gyro[1] *= -1
                grav = np.array(raw_gravity)
                mag = np.linalg.norm(grav)
                if mag > 1e-6:
                    grav /= mag
                else:
                    grav = np.array([0.0, 0.0, 1.0])
                grav[2] = -grav[2]
                grav = self.gravity_correction.apply(grav)
                grav[0] *= -1
                grav[1] *= -1
                with self.lock:
                    self.latest_data["gyro"] = gyro
                    self.latest_data["projected_gravity"] = grav
            except Exception as e:
                print(f"\nIMU read error: {e}", file=sys.stderr)
            elapsed = time.time() - t0
            sleep = (1.0 / self.frequency) - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def start(self):
        if not self.calibrate():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_data(self):
        with self.lock:
            return self.latest_data.copy()


# =========================================================================
# MAIN
# =========================================================================
def main():
    # --- Init CAN ---
    print("Initializing CAN buses...")
    buses = [can.interface.Bus(interface='socketcan', channel=port) for port, _ in CAN_CONFIGS]
    motor_ids_list = [ids for _, ids in CAN_CONFIGS]

    # --- Enable motors in backdrivable mode (same as findangles.py) ---
    print("Enabling motors in backdrivable mode...")
    time.sleep(0.3)
    for bus, motor_ids in zip(buses, motor_ids_list):
        for motor_id in motor_ids:
            flush_bus(bus)
            bus.send(can.Message(
                arbitration_id=(0x03 << 24) | (0xFD << 8) | motor_id,
                is_extended_id=True, dlc=8
            ))
            poll_motor_state(bus, motor_id)
    time.sleep(0.3)
    print(f"  {NUM_JOINTS} motors enabled.\n")

    # --- Init IMU ---
    print("Initializing IMU...")
    imu = BNO055_IMU()
    if not imu.start():
        sys.exit(1)

    # --- Init UDP ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    print(f"Broadcasting robot state on UDP port {UDP_PORT} at {SEND_FREQUENCY} Hz")
    print(f"Packet: 14 joint pos + 14 joint vel + 3 gyro + 3 gravity = 34 floats (136 bytes)")
    print("Press Ctrl+C to stop\n")

    interval = 1.0 / SEND_FREQUENCY
    packets_sent = 0

    try:
        while True:
            t0 = time.time()

            # Poll and read motors
            for bus, motor_ids in zip(buses, motor_ids_list):
                for motor_id in motor_ids:
                    poll_motor_state(bus, motor_id)
            update_all_motor_states_from_buffer(buses)

            joint_pos = np.array([motor_states[mid]['pos'] for mid in ALL_MOTOR_IDS], dtype=np.float32)
            joint_vel = np.array([motor_states[mid]['vel'] for mid in ALL_MOTOR_IDS], dtype=np.float32)

            # Read IMU
            imu_data = imu.get_data()
            gyro = imu_data["gyro"].astype(np.float32)
            grav = imu_data["projected_gravity"].astype(np.float32)

            # Pack: 14 pos + 14 vel + 3 gyro + 3 grav = 34 floats
            packet = struct.pack(f"{NUM_JOINTS * 2 + 6}f", *joint_pos, *joint_vel, *gyro, *grav)
            sock.sendto(packet, (BROADCAST_ADDR, UDP_PORT))

            packets_sent += 1
            if packets_sent % int(SEND_FREQUENCY) == 0:
                pos_str = " ".join(f"{p:+5.2f}" for p in joint_pos[:5])
                sys.stdout.write(
                    f"\rLegs: [{pos_str} ...]  "
                    f"Gyro: [{gyro[0]:+6.2f} {gyro[1]:+6.2f} {gyro[2]:+6.2f}]  "
                    f"Pkts: {packets_sent}"
                )
                sys.stdout.flush()

            elapsed = time.time() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total packets sent: {packets_sent}")
    finally:
        print("Disabling motors...")
        for bus, motor_ids in zip(buses, motor_ids_list):
            for motor_id in motor_ids:
                try:
                    flush_bus(bus)
                    bus.send(can.Message(
                        arbitration_id=(0x04 << 24) | (0xFD << 8) | motor_id,
                        is_extended_id=True, dlc=8
                    ))
                except Exception:
                    pass
                time.sleep(0.05)
        imu.stop()
        for bus in buses:
            bus.shutdown()
        sock.close()
        print("Done.")


if __name__ == "__main__":
    main()
