import robstride.client
import can
import time
import atexit
import struct

FULL_ROTATION = 2 * 3.14159

CAN_CONFIGS = [
    ('can2', [1,2,3,4,5]),
    ('can1', [6,7,8,9,10]),
    ('can0', [11,12,13,14]),
]

MOTOR_TYPE_PARAMS = {
    'O2': {'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -44.0, 'V_MAX': 44.0, 'T_MIN': -17.0, 'T_MAX': 17.0, 'KP_MIN': 0.0, 'KP_MAX': 500.0, 'KD_MIN': 0.0, 'KD_MAX': 5.0},
    'O3': {'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -20.0, 'V_MAX': 20.0, 'T_MIN': -60.0, 'T_MAX': 60.0, 'KP_MIN': 0.0, 'KP_MAX': 5000.0, 'KD_MIN': 0.0, 'KD_MAX': 100.0},
    'O5': {'P_MIN': -12.57, 'P_MAX': 12.57, 'V_MIN': -50.0, 'V_MAX': 50.0, 'T_MIN': -5.5,  'T_MAX': 5.5,  'KP_MIN': 0.0, 'KP_MAX': 500.0, 'KD_MIN': 0.0, 'KD_MAX': 5.0}
}

MOTOR_ID_TO_TYPE = {
    1: 'O3', 2: 'O3', 3: 'O3', 4: 'O3', 5: 'O2',     
    6: 'O3', 7: 'O3', 8: 'O3', 9: 'O3', 10: 'O2',    
    11: 'O2',                                        
    12: 'O5', 13: 'O5', 14: 'O5'                     
}

MOTOR_NAMES = {
    1: 'Left_Hip_Yaw',
    2: 'Left_Hip_Roll',
    3: 'Left_Hip_Pitch',
    4: 'Left_Knee',
    5: 'Left_Ankle',
    6: 'Right_Hip_Yaw',
    7: 'Right_Hip_Roll',
    8: 'Right_Hip_Pitch',
    9: 'Right_Knee',
    10: 'Right_Ankle',
    11: 'Neck_Pitch',
    12: 'Head_Pitch',
    13: 'Head_Yaw',
    14: 'Head_Roll'
}

#Left_Hip_Yaw = 1
#Left_Hip_Roll = 2
#Left_Hip_Pitch = 3
#Left_Knee = 4
#Left_Ankle = 5
#Right_Hip_Yaw = 6
#Right_Hip_Roll = 7
#Right_Hip_Pitch = 8
#Right_Knee = 9
#Right_Ankle = 10
#Neck_Pitch = 11
#Head_Pitch = 12
#Head_Yaw = 13
#Head_Roll = 14
#
#Left_Hip_Yaw = O3
#Left_Hip_Roll = O3
#Left_Hip_Pitch = O3
#Left_Knee = O3
#Left_Ankle = O2
#Right_Hip_Yaw = O3
#Right_Hip_Roll = O3
#Right_Hip_Pitch = O3
#Right_Knee = O3
#Right_Ankle = O2
#Neck_Pitch = O2
#Head_Pitch = O5
#Head_Yaw = O5
#Head_Roll = O5


#joint_names=['Left_Hip_Yaw', 'Left_Hip_Roll', 'Left_Hip_Pitch', 'Left_Knee', 'Left_Ankle', 
#'Right_Hip_Yaw', 'Right_Hip_Roll', 'Right_Hip_Pitch', 'Right_Knee', 'Right_Ankle', 'Neck_Pitch',
# 'Head_Pitch', 'Head_Yaw', 'Head_Roll'], joint_stiffness=[78.957, 78.957, 78.957, 78.957, 
#16.581, 78.957, 78.957, 78.957, 78.957, 16.581, 16.581, 2.763, 2.763, 2.763], 
#joint_damping=[5.027, 5.027, 5.027, 5.027, 1.056, 5.027, 5.027, 5.027, 5.027, 1.056, 
#1.056, 0.176, 0.176, 0.176],

def degrees(rad):
    return (rad * 360.0) / FULL_ROTATION

# --- Helper functions ---
def flush_bus(bus):
    while True:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break

def safe_write_param(client, bus, motor_id, param_name, value, retries=3):
    for attempt in range(retries):
        try:
            flush_bus(bus)
            client.write_param(motor_id, param_name, value)
            return
        except Exception as e:
            print(f"[ERROR] Write attempt {attempt+1} failed for motor {motor_id}, param '{param_name}': {e}")
    print(f"[FAILED] Could not write '{param_name}' to motor {motor_id} after {retries} attempts.")

def safe_read_param(client, bus, motor_id, param_name, retries=3):
    for attempt in range(retries):
        try:
            flush_bus(bus)
            return client.read_param(motor_id, param_name)
        except Exception as e:
            print(f"[ERROR] Read attempt {attempt+1} failed for motor {motor_id}, param '{param_name}': {e}")
    print(f"[FAILED] Could not read '{param_name}' from motor {motor_id} after {retries} attempts.")
    return None

def safe_disable(client, bus, motor_id, retries=3):
    for attempt in range(retries):
        try:
            flush_bus(bus)
            client.disable(motor_id)
            return
        except Exception as e:
            print(f"[ERROR] Disable attempt {attempt+1} failed for motor {motor_id}: {e}")
    print(f"[FAILED] Could not disable motor {motor_id} after {retries} attempts.")

motor_states = {}

def scale_value_to_u16(value, v_min, v_max):
    return int(65535.0 * (max(v_min, min(v_max, value)) - v_min) / (v_max - v_min))

def unscale_u16_to_float(val_u16, v_min, v_max):
    return (float(val_u16) / 65535.0) * (v_max - v_min) + v_min

def poll_motor_state(bus, motor_id):
    mtype = MOTOR_ID_TO_TYPE.get(motor_id, 'O3')
    params = MOTOR_TYPE_PARAMS[mtype]
    
    # 0 torque MIT Request triggers an immediate Type 2 feedback ping
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
            if msg is None: break
            if msg.is_error_frame: continue
            
            try:
                msg_type_val = (msg.arbitration_id & 0x1F000000) >> 24
                # Standard ID location
                motor_id = (msg.arbitration_id & 0xFF00) >> 8
                
                # Check for biped specific location
                if motor_id not in motor_states:
                    motor_id = msg.arbitration_id & 0xFF
            except Exception: continue
            
            if msg_type_val == 2 and motor_id in motor_states:
                try:
                    mtype = MOTOR_ID_TO_TYPE.get(motor_id, 'O3')
                    params = MOTOR_TYPE_PARAMS[mtype]
                    
                    p_raw = struct.unpack('>H', msg.data[0:2])[0]
                    v_raw = struct.unpack('>H', msg.data[2:4])[0]
                    
                    pos_rad = unscale_u16_to_float(p_raw, params['P_MIN'], params['P_MAX'])
                    vel_rad = unscale_u16_to_float(v_raw, params['V_MIN'], params['V_MAX'])
                    
                    motor_states[motor_id]['pos'] = pos_rad
                    motor_states[motor_id]['vel'] = vel_rad
                    motor_states[motor_id]['has_new_data'] = True
                except Exception: pass

# --- Setup buses and clients ---
buses = [can.interface.Bus(interface='socketcan', channel=port) for port, _ in CAN_CONFIGS]
clients = [robstride.client.Client(bus) for bus in buses]
motor_ids_list = [ids for _, ids in CAN_CONFIGS]

# Ensure proper shutdown
for bus in buses:
    atexit.register(bus.shutdown)

# Init min/max tracking
min_angles = {motor_id: float('inf') for ids in motor_ids_list for motor_id in ids}
max_angles = {motor_id: float('-inf') for ids in motor_ids_list for motor_id in ids}

# Initialize fast motor states tracking
for ids in motor_ids_list:
    for motor_id in ids:
        motor_states[motor_id] = {'pos': 0.0, 'vel': 0.0, 'has_new_data': False}

try:
    print("Enabling motors in HIGH-SPEED BACKDRIVABLE mode context...")
    for client, bus, motor_ids in zip(clients, buses, motor_ids_list):
        for motor_id in motor_ids:
            # We DONT change run modes here. Just pure enable, letting MIT mode 
            # allow us to backdrive with 0 position/velocity gains (KP/KD) and 0 torque.
            
            flush_bus(bus)
            # Send MIT mode enable command directly to stay out of the client's SDO waiting parsing logic
            bus.send(can.Message(arbitration_id=(0x03 << 24) | (0xFD << 8) | motor_id, is_extended_id=True, dlc=8))
            
            # Initial poll to make sure states populate before calculating offsets
            poll_motor_state(bus, motor_id)

    print("\nRotate the motors by hand. Press Ctrl+C to stop.")
    print("Recording min/max angles from the LOW-LEVEL OPERATIONAL encoder...")

    target_dt = 1.0 / 400.0  # 400 Hz loop target

    last_print_time = time.time()
    loops_completed = 0

    while True:
        loop_start = time.time()
        output_lines = []
        
        # 1. Quickly trigger feedback updates from all motors concurrently
        for bus, motor_ids in zip(buses, motor_ids_list):
            for motor_id in motor_ids:
                poll_motor_state(bus, motor_id)
                
        # 2. Extract feedback instantly from CAN bus without blocking SDO timeouts
        update_all_motor_states_from_buffer(buses)

        for client, bus, motor_ids in zip(clients, buses, motor_ids_list):
            for motor_id in motor_ids:
                # If we didn't receive a new packet, skip this update
                if not motor_states[motor_id]['has_new_data']:
                    continue

                rad = motor_states[motor_id]['pos']
                velocity_rps = motor_states[motor_id]['vel']
                motor_states[motor_id]['has_new_data'] = False # Reset

                min_angles[motor_id] = min(min_angles[motor_id], rad)
                max_angles[motor_id] = max(max_angles[motor_id], rad)

                motor_name = MOTOR_NAMES.get(motor_id, 'Unknown')
                output_lines.append(
                    f"Motor {motor_id:2d} {motor_name:<15} | Pos: {rad:7.3f} rad | Vel: {velocity_rps:7.3f} rad/s | "
                    f"Min: {min_angles[motor_id]:7.3f} rad | Max: {max_angles[motor_id]:7.3f} rad"
                )

        loops_completed += 1
        current_time = time.time()
        
        # Calculate actual Hz & print to terminal once per second
        if current_time - last_print_time >= 1.0:
            actual_hz = loops_completed / (current_time - last_print_time)
            header = f"--- CAN Bus Performance: {actual_hz:.2f} Hz | Target: 400.00 Hz ---"
            
            if output_lines:
                print(header)
                print("\n".join(output_lines))
                # Clear the screen lines so it prints in place (+1 for the header line)
                print("\033[F" * (len(output_lines) + 1), end='', flush=True)
                
            loops_completed = 0
            last_print_time = current_time

        elapsed = time.time() - loop_start
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\nFinal angle ranges (from operational mode encoder):")
    for motor_id in sorted(min_angles.keys()):
        if min_angles[motor_id] != float('inf'):
            motor_name = MOTOR_NAMES.get(motor_id, 'Unknown')
            range_rad = max_angles[motor_id] - min_angles[motor_id]
            print(f"Motor {motor_id:2d} {motor_name:<15}: Min = {min_angles[motor_id]:.3f} rad, "
                  f"Max = {max_angles[motor_id]:.3f} rad, "
                  f"Range = {range_rad:.3f} rad")

finally:
    print("\nDisabling all motors...")
    for client, bus, motor_ids in zip(clients, buses, motor_ids_list):
        for motor_id in motor_ids:
            try:
                flush_bus(bus)
                bus.send(can.Message(arbitration_id=(0x04 << 24) | (0xFD << 8) | motor_id, is_extended_id=True, dlc=8))
            except Exception as e:
                print(f"Disable error for {motor_id}: {e}")