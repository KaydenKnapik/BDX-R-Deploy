import can
import time
import struct
import math
import numpy as np
import sys
import traceback

# --- CONFIGURATION ---
HOST_ID = 0xFD

CAN_CONFIGS = [
    ('can1', [1,2,3,4,5]),
    ('can2', [6,7,8,9,10]),
    ('can0', [11,12,13,14]),
]

# Per-motor gains (stiffness=KP, damping=KD) from policy config
MOTOR_GAINS = {
    1:  {'kp': 150.957, 'kd': 5.027},  # Left_Hip_Yaw
    2:  {'kp': 150.957, 'kd': 5.027},  # Left_Hip_Roll
    3:  {'kp': 150.957, 'kd': 5.027},  # Left_Hip_Pitch
    4:  {'kp': 150.957, 'kd': 5.027},  # Left_Knee
    5:  {'kp': 50.581, 'kd': 1.056},  # Left_Ankle
    6:  {'kp': 150.957, 'kd': 5.027},  # Right_Hip_Yaw
    7:  {'kp': 150.957, 'kd': 5.027},  # Right_Hip_Roll
    8:  {'kp': 150.957, 'kd': 5.027},  # Right_Hip_Pitch
    9:  {'kp': 150.957, 'kd': 5.027},  # Right_Knee
    10: {'kp': 50.581, 'kd': 1.056},  # Right_Ankle
    11: {'kp': 16.581, 'kd': 1.056},  # Neck_Pitch
    12: {'kp':  2.763, 'kd': 0.176},  # Head_Pitch
    13: {'kp':  2.763, 'kd': 0.176},  # Head_Yaw
    14: {'kp':  2.763, 'kd': 0.176},  # Head_Roll
}

# --- PROTOCOL CONSTANTS ---
MUX_ENABLE = 0x03
MUX_CONTROL = 0x01
MUX_DISABLE = 0x04

# --- MOTOR PARAMETERS ---
MOTOR_TYPE_PARAMS = {
    'O2': { 
        'name': 'Robstride O2',
        'P_MIN': -12.57, 'P_MAX': 12.57,
        'V_MIN': -44.0,  'V_MAX': 44.0,
        'T_MIN': -17.0,  'T_MAX': 17.0,
        'KP_MIN': 0.0,   'KP_MAX': 500.0,
        'KD_MIN': 0.0,   'KD_MAX': 5.0,
    },
    'O3': { 
        'name': 'Robstride O3',
        'P_MIN': -12.57, 'P_MAX': 12.57,
        'V_MIN': -20.0,  'V_MAX': 20.0,
        'T_MIN': -60.0,  'T_MAX': 60.0,
        'KP_MIN': 0.0,   'KP_MAX': 5000.0,
        'KD_MIN': 0.0,   'KD_MAX': 100.0,
    },
    'O5': { 
        'name': 'Robstride O5',
        'P_MIN': -12.57, 'P_MAX': 12.57,
        'V_MIN': -50.0,  'V_MAX': 50.0,
        'T_MIN': -5.5,   'T_MAX': 5.5,
        'KP_MIN': 0.0,   'KP_MAX': 500.0,
        'KD_MIN': 0.0,   'KD_MAX': 5.0,
    }
}

# Map IDs to their Types
MOTOR_ID_TO_TYPE_MAP = {
    1: 'O3',
    2: 'O3',
    3: 'O3',
    4: 'O3',
    5: 'O2',
    6: 'O3',
    7: 'O3',
    8: 'O3',
    9: 'O3',
    10: 'O2',
    11: 'O2',
    12: 'O5',
    13: 'O5',
    14: 'O5'
}

def scale_value_to_u16(value, v_min, v_max):
    """Clips and scales a float value to a 16-bit unsigned integer."""
    scaled = 65535.0 * (np.clip(value, v_min, v_max) - v_min) / (v_max - v_min)
    return int(scaled)

def send_control_command(bus, motor_id, pos, vel, kp, kd, torque, params):
    """
    Builds and sends the MIT control command using the correct scaling
    parameters for the specified motor type.
    """
    # 1. Scale all values using the provided params dictionary
    angle_u16 = scale_value_to_u16(pos, params['P_MIN'], params['P_MAX'])
    vel_u16 = scale_value_to_u16(vel, params['V_MIN'], params['V_MAX'])
    kp_u16 = scale_value_to_u16(kp, params['KP_MIN'], params['KP_MAX'])
    kd_u16 = scale_value_to_u16(kd, params['KD_MIN'], params['KD_MAX'])
    torque_u16 = scale_value_to_u16(torque, params['T_MIN'], params['T_MAX'])

    # 2. Build the CAN Arbitration ID
    mux_part = (MUX_CONTROL & 0xFF) << 24
    torque_part = (torque_u16 & 0xFFFF) << 8
    id_part = motor_id & 0xFF
    arbitration_id = mux_part | torque_part | id_part

    # 3. Build the 8-byte Data Payload (Big-Endian)
    data = struct.pack('>HHHH', angle_u16, vel_u16, kp_u16, kd_u16)

    msg = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True, dlc=8)
    try:
        bus.send(msg)
    except can.CanOperationError:
        pass  # Skip if buffer full, next cycle will retry

def get_motor_params(motor_id):
    m_type = MOTOR_ID_TO_TYPE_MAP.get(motor_id)
    if not m_type:
        raise ValueError(f"Motor ID {motor_id} not found in TYPE MAP.")
    return MOTOR_TYPE_PARAMS[m_type]

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    buses = {}

    print("="*50)
    print("Holding all motors at 0.0 until Ctrl+C is pressed.")
    print("="*50)

    input("Ensure motors are powered. Press Enter to START...")

    try:
        # Connect to all CAN buses
        for interface, motor_ids in CAN_CONFIGS:
            try:
                buses[interface] = can.interface.Bus(channel=interface, bustype='socketcan')
                print(f"Connected to {interface}.")
            except Exception as e:
                print(f"Failed to connect to {interface}: {e}")

        # --- STEP 1: ENABLE MOTORS ---
        print("\n[1] Enabling Motors...")
        for interface, motor_ids in CAN_CONFIGS:
            if interface not in buses: continue
            bus = buses[interface]
            for mid in motor_ids:
                enable_id = (MUX_ENABLE << 24) | (HOST_ID << 8) | mid
                bus.send(can.Message(arbitration_id=enable_id, is_extended_id=True, dlc=8))
        
        time.sleep(1) # Wait for them to wake up

        # --- STEP 2: HOLD AT ZERO ---
        print(f"\n[2] Moving all motors to 0.0 rad and holding...")
        print("Press Ctrl+C to stop.")
        
        while True:
            for interface, motor_ids in CAN_CONFIGS:
                if interface not in buses: continue
                bus = buses[interface]
                # Drain any incoming responses to free buffer space
                while bus.recv(timeout=0) is not None:
                    pass
                for mid in motor_ids:
                    params = get_motor_params(mid)
                    gains = MOTOR_GAINS[mid]
                    send_control_command(
                        bus, mid, 0.0, 0.0, gains['kp'], gains['kd'], 0.0, params
                    )
            time.sleep(0.0025) # 400Hz loop rate

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Exiting...")
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        traceback.print_exc()

    finally:
        print(f"\n[Final] Disabling all motors...")
        for interface, motor_ids in CAN_CONFIGS:
            if interface in buses:
                bus = buses[interface]
                for mid in motor_ids:
                    disable_id = (MUX_DISABLE << 24) | (HOST_ID << 8) | mid
                    try:
                        bus.send(can.Message(arbitration_id=disable_id, is_extended_id=True, dlc=8))
                    except:
                        pass
                bus.shutdown()
        print("Sequence Complete.")