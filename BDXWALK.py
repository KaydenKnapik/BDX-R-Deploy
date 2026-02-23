import robstride.client
import can
import time
import numpy as np
import atexit
import sys
import torch
import os
import enum
import struct
import math
import threading
from scipy.spatial.transform import Rotation
import pygame

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
#POLICY_FILE_PATH = "60000.pt"
POLICY_FILE_PATH = "plsdontbreak.pt"
ACTION_SCALE = 0.5
GRAPH_SAVE_DIR = os.path.join(os.path.dirname(__file__), "deployment_graphs_walk")

# --- JOYSTICK CONFIGURATION (NEW) ---
LEFT_STICK_Y_AXIS = 1
JOYSTICK_DEADZONE = 0.25 # Increased slightly for better stability

# --- HIGH-PERFORMANCE CONTROL LOOP PARAMETERS ---
CONTROL_FREQUENCY = 200.0  # Hz - Main loop for sending motor commands
POLICY_FREQUENCY = 50.0   # Hz - How often we run the expensive AI policy
POLICY_UPDATE_INTERVAL = int(CONTROL_FREQUENCY / POLICY_FREQUENCY)

# --- LOW-PASS FILTER (LPF) SETUP ---
LPF_ALPHA = 0.1
ACTION_FILTER_ALPHA = 1

# --- MOTOR TYPE PARAMETERS (Unchanged) ---
MOTOR_TYPE_PARAMS = {
    'O2': { 'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -44.0,  'V_MAX': 44.0, 'T_MIN': -17.0,  'T_MAX': 17.0,  'KP_MIN': 0.0,   'KP_MAX': 500.0, 'KD_MIN': 0.0,   'KD_MAX': 5.0 },
    'O3': { 'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -20.0,  'V_MAX': 20.0, 'T_MIN': -60.0,  'T_MAX': 60.0,  'KP_MIN': 0.0,   'KP_MAX': 5000.0, 'KD_MIN': 0.0,   'KD_MAX': 100.0 }
}

# --- CAN BUS AND MOTOR ID CONFIGURATION (Unchanged) ---
CAN_CONFIGS = [
    ('can0', [19, 18, 16, 17, 20]),
    ('can1', [5, 1, 2, 3, 4]),
]

# --- CORRECTED JOINT MAP (Unchanged) ---
JOINT_MAP = [
    ('L_HipYaw',   1, 5, 'O3'), ('R_HipYaw',   0, 19, 'O3'), ('L_HipRoll',  1, 1, 'O3'), ('R_HipRoll',  0, 18, 'O3'),
    ('L_HipPitch', 1, 2, 'O3'), ('R_HipPitch', 0, 16, 'O3'), ('L_Knee',     1, 3, 'O3'), ('R_Knee',     0, 17, 'O3'),
    ('L_Ankle',    1, 4, 'O2'), ('R_Ankle',    0, 20, 'O2'),
]

# --- PD GAINS (Unchanged) ---
JOINT_GAINS = {
    'L_HipYaw':   {'kp': 78.0, 'kd': 5},   'R_HipYaw':   {'kp': 78.0, 'kd': 5}, 'L_HipRoll':  {'kp': 78.0, 'kd': 5}, 'R_HipRoll':  {'kp': 78.0, 'kd': 5},
    'L_HipPitch': {'kp': 78.0, 'kd': 5},  'R_HipPitch': {'kp': 78.0, 'kd': 5}, 'L_Knee':     {'kp': 78.0, 'kd': 5},  'R_Knee':     {'kp': 78.0, 'kd': 5},
    'L_Ankle':    {'kp': 17.0,  'kd': 1},  'R_Ankle':    {'kp': 17.0,  'kd': 1},
}
JOINT_GAINS_STANDING = {
    'L_HipYaw':   {'kp': 78.0, 'kd': 5},   'R_HipYaw':   {'kp': 78.0, 'kd': 5}, 'L_HipRoll':  {'kp': 78.0, 'kd': 5}, 'R_HipRoll':  {'kp': 78.0, 'kd': 5},
    'L_HipPitch': {'kp': 78.0, 'kd': 5},  'R_HipPitch': {'kp': 78.0, 'kd': 5}, 'L_Knee':     {'kp': 78.0, 'kd': 5},  'R_Knee':     {'kp': 78.0, 'kd': 5},
    'L_Ankle':    {'kp': 17.0,  'kd': 1},  'R_Ankle':    {'kp': 17.0,  'kd': 1},
}
JOINT_GAINS_ALMOST = JOINT_GAINS_STANDING # simplified

# --- Joint Safety Limits (Unchanged) ---
JOINT_LIMITS = {
    'L_HipYaw':   {'min': -0.4, 'max': 0.4},   'R_HipYaw':   {'min': -0.4, 'max': 0.4}, 'L_HipRoll':  {'min': -0.34, 'max': 0.34}, 'R_HipRoll':  {'min': -0.34, 'max': 0.34},
    'L_HipPitch': {'min': -0.75, 'max': 0.7},  'R_HipPitch': {'min': -0.75, 'max': 0.7}, 'L_Knee':     {'min': -0.94, 'max': 1.3},   'R_Knee':     {'min': -1.3, 'max': 0.94},
    'L_Ankle':    {'min': -0.84, 'max': 1.2},   'R_Ankle':    {'min': -1.2, 'max': 0.84},
}

TARGET_VELOCITY_RAD_S = 0.0
TARGET_TORQUE_NM = 0.0

print("--- High-Performance MIT Control Script ---")
print(f"Control Loop Freq: {CONTROL_FREQUENCY} Hz")
print(f"Policy Update Freq: {POLICY_FREQUENCY} Hz")
print("-" * 33)

NUM_JOINTS = len(JOINT_MAP)
JOINT_NAMES = [name for name, _, _, _ in JOINT_MAP]

# =========================================================================
# IMU CLASS (Unchanged)
# =========================================================================
class BNO055_IMU:
    def __init__(self, i2c_bus_num=7, calibration_samples=100, frequency=100.0):
        self.i2c_bus_num = i2c_bus_num; self.calibration_samples = calibration_samples; self.frequency = frequency; self.sensor = None; self.offsets = {}; self.running = False; self.thread = None; self.lock = threading.Lock(); self.latest_data = {"gyro": np.zeros(3), "projected_gravity": np.array([0.0, 0.0, -1.0])}
    def _initialize_sensor(self):
        from adafruit_extended_bus import ExtendedI2C as I2C; import adafruit_bno055
        try: self.sensor = adafruit_bno055.BNO055_I2C(I2C(self.i2c_bus_num)); time.sleep(2); return True
        except Exception as e: print(f"FATAL: Failed to initialize BNO055 IMU. Error: {e}", file=sys.stderr); return False
    def calibrate(self):
        if not self._initialize_sensor(): return False
        print("--- Starting IMU Calibration ---"); time.sleep(2); gyro_data = []; gravity_data = []
        for i in range(self.calibration_samples):
            gyro = self.sensor.gyro; gravity = self.sensor.gravity
            if gyro is not None and gravity is not None: gyro_data.append(gyro); gravity_data.append(gravity)
            sys.stdout.write(f"\rCollecting samples... {i+1}/{self.calibration_samples}"); sys.stdout.flush(); time.sleep(0.05)
        print("\n--- Calibration Calculations ---")
        if not gyro_data or not gravity_data: print("FATAL: Failed to collect any valid IMU data during calibration. Exiting."); return False
        try:
            self.offsets['gyro'] = tuple(np.mean(gyro_data, axis=0)); avg_gravity = np.mean(np.array(gravity_data), axis=0); at_rest_gravity_vector = avg_gravity / np.linalg.norm(avg_gravity); at_rest_gravity_vector[2] = -at_rest_gravity_vector[2]
            ideal_gravity_vector = np.array([0.0, 0.0, -1.0]); gravity_correction_rotation, _ = Rotation.align_vectors([ideal_gravity_vector], [at_rest_gravity_vector]); self.offsets['gravity_correction_rotation'] = gravity_correction_rotation
            print(f"DEBUG: Calculated gyro offset: {self.offsets['gyro']}"); print("--- Calibration Complete ---"); return True
        except Exception as e: print(f"FATAL: Error during IMU offset calculation: {e}", file=sys.stderr); import traceback; traceback.print_exc(); return False
    def _read_loop(self):
        while self.running:
            loop_start = time.time()
            if self.sensor:
                try:
                    if 'gyro' not in self.offsets or 'gravity_correction_rotation' not in self.offsets: time.sleep(0.01); continue
                    raw_gyro = self.sensor.gyro; gravity_vector = self.sensor.gravity
                    if raw_gyro is None or gravity_vector is None: continue
                    calibrated_gyro = np.array(raw_gyro) - np.array(self.offsets['gyro']); proj_grav_unaligned = np.array(gravity_vector); magnitude = np.linalg.norm(proj_grav_unaligned)
                    if magnitude > 1e-6: proj_grav_unaligned /= magnitude
                    else: proj_grav_unaligned = np.array([0.0, 0.0, 1.0])
                    proj_grav_unaligned[2] = -proj_grav_unaligned[2]; aligned_proj_gravity = self.offsets['gravity_correction_rotation'].apply(proj_grav_unaligned); aligned_proj_gravity[1] *= -1; aligned_proj_gravity[0] *= -1
                    with self.lock: self.latest_data["gyro"] = calibrated_gyro; self.latest_data["projected_gravity"] = aligned_proj_gravity
                except Exception as e: print(f"IMU read error: {e}", file=sys.stderr)
            elapsed = time.time() - loop_start; sleep_time = (1.0 / self.frequency) - elapsed
            if sleep_time > 0: time.sleep(sleep_time)
    def start(self):
        if not self.calibrate(): return False
        self.running = True; self.thread = threading.Thread(target=self._read_loop, daemon=True); self.thread.start(); print("IMU reader thread started."); return True
    def stop(self): self.running = False; self.thread.join(); print("IMU reader thread stopped.")
    def get_latest_data(self):
        with self.lock: return self.latest_data.copy()

# =========================================================================
# MOTOR/HELPER FUNCTIONS (Unchanged)
# =========================================================================
MOTOR_STATES = {motor_id: {'pos': 0.0, 'vel': 0.0, 'updated_at': 0.0} for _, _, motor_id, _ in JOINT_MAP}
class MotorMsg(enum.Enum): Feedback = 2
def parse_and_update_motor_state(msg: can.Message):
    try: msg_type_val = (msg.arbitration_id & 0x1F000000) >> 24; motor_id = (msg.arbitration_id & 0xFF00) >> 8
    except Exception: return
    if msg_type_val != MotorMsg.Feedback.value or motor_id not in MOTOR_STATES: return
    try:
        angle_raw = struct.unpack('>H', msg.data[0:2])[0]; pos_rad = (float(angle_raw) / 65535.0 * 8.0 * math.pi) - (4.0 * math.pi)
        velocity_raw = struct.unpack('>H', msg.data[2:4])[0]; vel_rps = (float(velocity_raw) / 65535.0 * 88.0) - 44.0
        state = MOTOR_STATES[motor_id]; state['pos'] = pos_rad; state['vel'] = vel_rps; state['updated_at'] = time.time()
    except (struct.error, IndexError): pass
def update_all_motor_states_from_buffer(buses):
    for bus in buses:
        while True:
            msg = bus.recv(timeout=0)
            if msg is None: break
            if not msg.is_error_frame: parse_and_update_motor_state(msg)
MUX_CONTROL = 0x01
def scale_value_to_u16(value, v_min, v_max): return int(65535.0 * (np.clip(value, v_min, v_max) - v_min) / (v_max - v_min))
def send_mit_control_command(bus, motor_id, pos_rad, vel_rad_s, kp, kd, torque_nm, params):
    angle_u16 = scale_value_to_u16(pos_rad, params['P_MIN'], params['P_MAX']); vel_u16 = scale_value_to_u16(vel_rad_s, params['V_MIN'], params['V_MAX']); kp_u16 = scale_value_to_u16(kp, params['KP_MIN'], params['KP_MAX']); kd_u16 = scale_value_to_u16(kd, params['KD_MIN'], params['KD_MAX']); torque_u16 = scale_value_to_u16(torque_nm, params['T_MIN'], params['T_MAX'])
    arbitration_id = ((MUX_CONTROL & 0xFF) << 24) | ((torque_u16 & 0xFFFF) << 8) | (motor_id & 0xFF)
    data = struct.pack('>HHHH', angle_u16, vel_u16, kp_u16, kd_u16)
    try: bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True, dlc=8))
    except can.CanError as e: print(f"Error sending MIT command to motor {motor_id}: {e}", file=sys.stderr)
def load_policy(filepath):
    print(f"--> Loading TorchScript policy from: {filepath}")
    try: model = torch.jit.load(filepath, map_location='cpu'); model.eval(); return model
    except Exception as e: print(f"FATAL: Could not load TorchScript file '{filepath}'. Exiting.\nError: {e}"); sys.exit(1)
def format_np(arr): return np.array2string(arr, precision=3, floatmode='fixed', suppress_small=True)
def flush_bus(bus):
    while True:
        msg = bus.recv(timeout=0.01)
        if msg is None: break
def safe_disable(client, bus, motor_id, retries=3):
    for attempt in range(retries):
        try: flush_bus(bus); client.disable(motor_id); return
        except Exception as e: print(f"[ERROR] Disable attempt {attempt+1} failed for motor {motor_id}: {e}")
    print(f"[FAILED] Could not disable motor {motor_id} after {retries} attempts.")
def plot_and_save_data(data_history, save_dir):
    if not data_history["timesteps"]: print("No data to plot."); return
    print(f"--> Generating plots in: {save_dir}"); os.makedirs(save_dir, exist_ok=True)
    timesteps = data_history['timesteps']; live_pos = np.array(data_history['live_joint_pos_rad']); target_pos = np.array(data_history['target_joint_pos']); imu_ang_vel_data = np.array(data_history['imu_ang_vel']); projected_gravity_data = np.array(data_history['projected_gravity'])
    print("    - Plotting Joint Positions...")
    for i in range(NUM_JOINTS):
        fig, ax = plt.subplots(figsize=(12, 6)); joint_name = JOINT_NAMES[i]
        ax.plot(timesteps, live_pos[:, i], label=f'Live Position', color='b'); ax.plot(timesteps, target_pos[:, i], label=f'Target Position', color='r', linestyle='--'); ax.set_xlabel('Timestep'); ax.set_ylabel('Position (rad)'); ax.set_title(f'Joint Performance: {joint_name}'); ax.legend(); ax.grid(True)
        try: plt.savefig(os.path.join(save_dir, f"joint_{joint_name}_position_plot.png"))
        except Exception as e: print(f"      - FAILED to save plot for {joint_name}: {e}")
        plt.close(fig)
    print("    - Plotting IMU Data...")
    fig, ax = plt.subplots(figsize=(12, 6)); ax.plot(timesteps, imu_ang_vel_data[:, 0], label='Gyro X', color='r'); ax.plot(timesteps, imu_ang_vel_data[:, 1], label='Gyro Y', color='g'); ax.plot(timesteps, imu_ang_vel_data[:, 2], label='Gyro Z', color='b'); ax.set_title('IMU: Angular Velocity'); ax.legend(); ax.grid(True); plt.savefig(os.path.join(save_dir, "imu_angular_velocity_plot.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6)); ax.plot(timesteps, projected_gravity_data[:, 0], label='Gravity X', color='r'); ax.plot(timesteps, projected_gravity_data[:, 1], label='Gravity Y', color='g'); ax.plot(timesteps, projected_gravity_data[:, 2], label='Gravity Z', color='b'); ax.set_title('IMU: Projected Gravity'); ax.legend(); ax.grid(True); plt.savefig(os.path.join(save_dir, "imu_projected_gravity_plot.png")); plt.close(fig)
    print("--> Plotting complete.")

# =========================================================================
# --- MAIN SCRIPT ---
# =========================================================================
print("\nInitializing CAN buses...")
buses = [can.interface.Bus(interface='socketcan', channel=port) for port, _ in CAN_CONFIGS]
clients = [robstride.client.Client(bus) for bus in buses]
atexit.register(lambda: [bus.shutdown() for bus in buses])

print("\nInitializing IMU...")
imu = BNO055_IMU(frequency=CONTROL_FREQUENCY)
if not imu.start(): sys.exit(1)
atexit.register(imu.stop)

# --- NEW: PYGAME AND JOYSTICK INITIALIZATION ---
print("\nInitializing Joystick...")
pygame.init()
pygame.joystick.init()
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"--> Joystick found: {joystick.get_name()}")
    # Register pygame.quit() to be called at exit
    atexit.register(pygame.quit)
else:
    print("--> WARNING: No joystick found. Velocity will remain 0.")
# --- END NEW SECTION ---

policy_model = load_policy(POLICY_FILE_PATH)

data_history = {"timesteps": [], "live_joint_pos_rad": [], "target_joint_pos": [], "joint_pos_obs": [], "joint_vel_obs": [], "last_actions": [], "velocity_command": [], "imu_ang_vel": [], "projected_gravity": []}
filtered_joint_velocities = np.zeros(NUM_JOINTS, dtype=np.float32)
filtered_joint_positions = np.zeros(NUM_JOINTS, dtype=np.float32)

try:
    print("Enabling all motors..."); time.sleep(0.5)
    for client, bus, (_, motor_ids) in zip(clients, buses, CAN_CONFIGS):
        for motor_id in motor_ids: flush_bus(bus); client.enable(motor_id)
    time.sleep(0.5)

    print("\nMoving to initial standing position...")
    time.sleep(0.2); update_all_motor_states_from_buffer(buses)
    start_joint_pos = np.array([MOTOR_STATES[motor_id]['pos'] for _, _, motor_id, _ in JOINT_MAP], dtype=np.float32)
    target_standing_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
    num_steps = 100
    for i in range(num_steps + 1):
        interpolation_alpha = i / float(num_steps)
        interpolated_pos = (1.0 - interpolation_alpha) * start_joint_pos + interpolation_alpha * target_standing_pos
        for joint_idx, (name, bus_idx, motor_id, motor_type) in enumerate(JOINT_MAP):
            gains = JOINT_GAINS[name]; params = MOTOR_TYPE_PARAMS[motor_type]
            send_mit_control_command(buses[bus_idx], motor_id, interpolated_pos[joint_idx], TARGET_VELOCITY_RAD_S, gains['kp'], gains['kd'], TARGET_TORQUE_NM, params)
        time.sleep(0.02)

    print("\n" + "="*50); print("MODE: RUNNING WALKING POLICY")
    print("Switching to WALKING gains now.")
    input("--> Robot is ready. Press ENTER to start policy.")
    time.sleep(3)

    velocity_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    new_policy_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    previous_policy_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    default_joint_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
    filtered_policy_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    loop_counter = 0; output_lines_count = 0

    while True:
        loop_start_time = time.time()

        # --- MODIFIED: JOYSTICK CONTROL FOR VELOCITY COMMAND ---
        # This replaces the old hardcoded `if loop_counter == 2000:` block
        pygame.event.pump() # Must be called to process internal pygame events
        
        commanded_y_velocity = 0.0
        if joystick:
            # Get the current value of the left stick's Y-axis
            y_axis_value = joystick.get_axis(LEFT_STICK_Y_AXIS)
            
            # A negative value means the stick is pushed forward
            if y_axis_value < -JOYSTICK_DEADZONE:
                commanded_y_velocity = -0.4
        
        # Create the final 3-element velocity command array
        velocity_command = np.array([0.0, commanded_y_velocity, 0.0], dtype=np.float32)
        # --- END MODIFICATION ---

        update_all_motor_states_from_buffer(buses)
        imu_data = imu.get_latest_data()

        raw_joint_positions = np.array([MOTOR_STATES[motor_id]['pos'] for _, _, motor_id, _ in JOINT_MAP], dtype=np.float32)
        raw_joint_velocities = np.array([MOTOR_STATES[motor_id]['vel'] for _, _, motor_id, _ in JOINT_MAP], dtype=np.float32)

        filtered_joint_velocities = LPF_ALPHA * raw_joint_velocities + (1.0 - LPF_ALPHA) * filtered_joint_velocities
        filtered_joint_positions = LPF_ALPHA * raw_joint_positions + (1.0 - LPF_ALPHA) * filtered_joint_positions

        if loop_counter % POLICY_UPDATE_INTERVAL == 0:
            previous_policy_action = np.copy(new_policy_action)
            obs_joint_pos = filtered_joint_positions - default_joint_pos
            obs_joint_vel = filtered_joint_velocities
            obs_ang_vel = imu_data["gyro"]
            obs_proj_gravity = imu_data["projected_gravity"]
            observation = np.concatenate([obs_ang_vel, obs_proj_gravity, velocity_command, obs_joint_pos, obs_joint_vel, previous_policy_action]).astype(np.float32)
            obs_tensor = torch.from_numpy(observation).unsqueeze(0)
            with torch.no_grad(): action_tensor = policy_model(obs_tensor)
            noisy_action = action_tensor.squeeze().numpy()
            filtered_policy_action = ACTION_FILTER_ALPHA * noisy_action + (1.0 - ACTION_FILTER_ALPHA) * filtered_policy_action
            new_policy_action = filtered_policy_action

        interpolation_alpha = (loop_counter % POLICY_UPDATE_INTERVAL) / float(POLICY_UPDATE_INTERVAL - 1)
        interpolated_action = (1.0 - interpolation_alpha) * previous_policy_action + interpolation_alpha * new_policy_action
        action_delta = interpolated_action * ACTION_SCALE
        target_joint_pos = default_joint_pos + action_delta
        last_action_for_log = interpolated_action

        for joint_idx, (name, _, _, _) in enumerate(JOINT_MAP):
            limits = JOINT_LIMITS[name]
            target_joint_pos[joint_idx] = np.clip(target_joint_pos[joint_idx], limits['min'], limits['max'])

        for joint_idx, (name, bus_idx, motor_id, motor_type) in enumerate(JOINT_MAP):
            gains = JOINT_GAINS[name]; params = MOTOR_TYPE_PARAMS[motor_type]
            send_mit_control_command(buses[bus_idx], motor_id, target_joint_pos[joint_idx], TARGET_VELOCITY_RAD_S, gains['kp'], gains['kd'], TARGET_TORQUE_NM, params)

        obs_joint_pos = raw_joint_positions - default_joint_pos
        obs_joint_vel = raw_joint_velocities
        obs_ang_vel = imu_data["gyro"]
        obs_proj_gravity = imu_data["projected_gravity"]
        output_string = (f"--- Real-time Policy Observations (Timestep: {loop_counter}) ---\n" f"Angular Velocity  : {format_np(obs_ang_vel)}\n" f"Projected Gravity : {format_np(obs_proj_gravity)}\n" f"Velocity Command  : {format_np(velocity_command)}\n" f"Joint Positions   : {format_np(obs_joint_pos)}\n" f"Joint Velocities  : {format_np(obs_joint_vel)}\n" f"Last Action       : {format_np(last_action_for_log)}")
        if output_lines_count > 0: sys.stdout.write(f'\033[{output_lines_count}A')
        sys.stdout.write(output_string); sys.stdout.write('\033[J'); sys.stdout.flush()
        if output_lines_count == 0: output_lines_count = output_string.count('\n') + 1

        data_history["timesteps"].append(loop_counter); data_history["live_joint_pos_rad"].append(np.copy(obs_joint_pos)); data_history["target_joint_pos"].append(np.copy(target_joint_pos)); data_history["joint_pos_obs"].append(np.copy(obs_joint_pos)); data_history["joint_vel_obs"].append(np.copy(obs_joint_vel)); data_history["last_actions"].append(np.copy(last_action_for_log)); data_history["velocity_command"].append(np.copy(velocity_command)); data_history["imu_ang_vel"].append(np.copy(obs_ang_vel)); data_history["projected_gravity"].append(np.copy(obs_proj_gravity))

        loop_counter += 1
        loop_duration = time.time() - loop_start_time
        sleep_time = (1.0 / CONTROL_FREQUENCY) - loop_duration
        if sleep_time > 0: time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print("\n--- A FATAL ERROR OCCURRED ---"); import traceback; traceback.print_exc(); print("-" * 20 + "\n")
finally:
    print("Disabling all motors...")
    for client, bus, (_, motor_ids) in zip(clients, buses, CAN_CONFIGS):
        for motor_id in motor_ids: safe_disable(client, bus, motor_id); time.sleep(0.05)
    if 'imu' in locals() and imu.running: imu.stop()
    plot_and_save_data(data_history, GRAPH_SAVE_DIR)