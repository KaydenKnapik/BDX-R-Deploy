#!/usr/bin/env python3
"""
Robot State UDP Receiver — runs on the laptop.
Receives 14 joint positions + 14 joint velocities + 3 gyro + 3 gravity
from robot_sender.py over UDP.

Usage:
    from bdx_api.robot_receiver import RobotStateReceiver
    rx = RobotStateReceiver()
    rx.start()
    state = rx.get_data()
"""

import socket
import struct
import time
import sys
import threading
import numpy as np

UDP_PORT = 5006
NUM_JOINTS = 14
PACKET_FLOATS = NUM_JOINTS * 2 + 6  # 14 pos + 14 vel + 3 gyro + 3 grav = 34
PACKET_BYTES = PACKET_FLOATS * 4     # 136 bytes

# Motor names in the order they are packed by robot_sender.py (motor IDs 1..14)
SENDER_MOTOR_NAMES = [
    "Left_Hip_Yaw",    # motor 1
    "Left_Hip_Roll",   # motor 2
    "Left_Hip_Pitch",  # motor 3
    "Left_Knee",       # motor 4
    "Left_Ankle",      # motor 5
    "Right_Hip_Yaw",   # motor 6
    "Right_Hip_Roll",  # motor 7
    "Right_Hip_Pitch", # motor 8
    "Right_Knee",      # motor 9
    "Right_Ankle",     # motor 10
    "Neck_Pitch",      # motor 11
    "Head_Pitch",      # motor 12
    "Head_Yaw",        # motor 13
    "Head_Roll",       # motor 14
]


class RobotStateReceiver:
    """Thread-safe UDP receiver for full robot state from the Jetson."""

    def __init__(self, port=UDP_PORT):
        self.port = port
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_data = {
            "joint_pos": np.zeros(NUM_JOINTS, dtype=np.float32),
            "joint_vel": np.zeros(NUM_JOINTS, dtype=np.float32),
            "gyro": np.zeros(3, dtype=np.float32),
            "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        self.packets_received = 0
        self.last_packet_time = 0.0
        self.sock = None

    def _receive_loop(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        except Exception:
            pass
        self.sock.bind(("", self.port))
        self.sock.settimeout(1.0)

        while self.running:
            try:
                data, _ = self.sock.recvfrom(256)
                if len(data) == PACKET_BYTES:
                    values = struct.unpack(f"{PACKET_FLOATS}f", data)
                    joint_pos = np.array(values[0:NUM_JOINTS], dtype=np.float32)
                    joint_vel = np.array(values[NUM_JOINTS:NUM_JOINTS*2], dtype=np.float32)
                    gyro = np.array(values[NUM_JOINTS*2:NUM_JOINTS*2+3], dtype=np.float32)
                    grav = np.array(values[NUM_JOINTS*2+3:NUM_JOINTS*2+6], dtype=np.float32)
                    with self.lock:
                        self.latest_data["joint_pos"] = joint_pos
                        self.latest_data["joint_vel"] = joint_vel
                        self.latest_data["gyro"] = gyro
                        self.latest_data["projected_gravity"] = grav
                        self.packets_received += 1
                        self.last_packet_time = time.time()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"RobotStateReceiver error: {e}", file=sys.stderr)

        self.sock.close()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"Robot state receiver listening on UDP port {self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_data(self):
        with self.lock:
            return {k: v.copy() for k, v in self.latest_data.items()}

    def is_connected(self):
        with self.lock:
            return (time.time() - self.last_packet_time) < 2.0

    @staticmethod
    def build_joint_mapping(policy_joint_names: list[str]) -> np.ndarray:
        """
        Build an index array that maps from policy joint order → sender packet order.

        Returns an array where mapping[i] is the index into the sender's 14-joint
        arrays for policy joint i. Returns -1 for joints not found in the sender.
        """
        sender_lookup = {name: idx for idx, name in enumerate(SENDER_MOTOR_NAMES)}
        mapping = np.full(len(policy_joint_names), -1, dtype=int)
        for i, name in enumerate(policy_joint_names):
            name = name.strip()
            if name in sender_lookup:
                mapping[i] = sender_lookup[name]
            else:
                print(f"[RobotStateReceiver] WARNING: policy joint '{name}' not in sender data")
        return mapping
