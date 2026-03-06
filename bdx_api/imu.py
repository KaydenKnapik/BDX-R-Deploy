#!/usr/bin/env python3
"""
IMU UDP Receiver — runs on the laptop.
Receives gyro + projected gravity from the Jetson over UDP.

Two ways to use:

1) Standalone test (just prints values):
       python imu_receiver.py

2) Import into your MuJoCo script:
       from imu_receiver import IMUReceiver
       imu = IMUReceiver()
       imu.start()
       ...
       data = imu.get_data()  # {"gyro": np.array, "projected_gravity": np.array}
       ...
       imu.stop()
"""

import socket
import struct
import time
import sys
import threading
import numpy as np

# --- CONFIGURATION ---
UDP_PORT = 5005  # Must match imu_sender.py


class IMUReceiver:
    """Thread-safe UDP receiver for real-time IMU data from the Jetson."""

    def __init__(self, port=UDP_PORT):
        self.port = port
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_data = {
            "gyro": np.zeros(3),
            "projected_gravity": np.array([0.0, 0.0, -1.0]),
        }
        self.packets_received = 0
        self.last_packet_time = 0.0
        self.sock = None

    def _receive_loop(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.port))
        self.sock.settimeout(1.0)  # So we can check self.running periodically

        while self.running:
            try:
                data, _ = self.sock.recvfrom(64)
                if len(data) == 24:  # 6 floats * 4 bytes
                    values = struct.unpack("6f", data)
                    gyro = np.array(values[0:3])
                    grav = np.array(values[3:6])
                    with self.lock:
                        self.latest_data["gyro"] = gyro
                        self.latest_data["projected_gravity"] = grav
                        self.packets_received += 1
                        self.last_packet_time = time.time()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receiver error: {e}", file=sys.stderr)

        self.sock.close()

    def start(self):
        """Start the receiver thread."""
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"IMU receiver listening on UDP port {self.port}")

    def stop(self):
        """Stop the receiver thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_data(self):
        """Get the latest IMU data. Thread-safe."""
        with self.lock:
            return self.latest_data.copy()

    def is_connected(self):
        """Check if we've received data recently (within 2s)."""
        with self.lock:
            return (time.time() - self.last_packet_time) < 2.0


def fmt(arr):
    return np.array2string(arr, precision=4, floatmode="fixed", suppress_small=True, sign=" ")


def main():
    """Standalone mode: print incoming IMU data to the terminal."""
    receiver = IMUReceiver()
    receiver.start()

    print("Waiting for IMU data from Jetson...\n")
    lines_printed = 0

    try:
        while True:
            data = receiver.get_data()
            gyro = data["gyro"]
            grav = data["projected_gravity"]
            connected = receiver.is_connected()

            status = "LIVE" if connected else "WAITING..."
            text = (
                f"Status            : {status}\n"
                f"Gyroscope (rad/s) : {fmt(gyro)}\n"
                f"Projected Gravity : {fmt(grav)}\n"
                f"Packets received  : {receiver.packets_received}"
            )

            if lines_printed > 0:
                sys.stdout.write(f"\033[{lines_printed}A")
            sys.stdout.write(text)
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            if lines_printed == 0:
                lines_printed = text.count("\n") + 1

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        receiver.stop()
        print("Done.")


if __name__ == "__main__":
    main()
