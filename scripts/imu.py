import serial
import struct
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class MTi_Serial_IMU:
    def __init__(self, port="/dev/ttyUSB0", baudrate=2000000, debug=False):
        self.port = port
        self.baudrate = baudrate
        self.latest_data = {
            "gyro": np.zeros(3),
            "projected_gravity": np.array([0.0, 0.0, -1.0])
        }
        self.lock = threading.Lock()
        self.running = False
        self.serial = None
        self.debug = debug
        self._debug_printed = False

    def _compute_checksum(self, data: bytes) -> int:
        return (-sum(data)) & 0xFF

    def _send_message(self, mid: int, payload: bytes = b''):
        msg = bytes([0xFF, mid, len(payload)]) + payload
        cs = self._compute_checksum(msg)
        self.serial.write(bytes([0xFA]) + msg + bytes([cs]))

    def _configure_outputs(self):
        """Put device in config mode, set outputs, go back to measurement."""
        # GoToConfig
        self._send_message(0x30)
        time.sleep(0.1)
        self.serial.reset_input_buffer()

        # SetOutputConfiguration: pairs of [XDI_hi, XDI_lo, freq_hi, freq_lo]
        # 0x2010 = Quaternion (float32)  @ 100 Hz
        # 0x8020 = RateOfTurn (float32)  @ 100 Hz
        payload = bytes([
            0x20, 0x10, 0x00, 0x32,  # Quaternion, 50 Hz
            0x80, 0x20, 0x00, 0x32,  # Rate of Turn, 50 Hz
        ])
        self._send_message(0xC0, payload)
        time.sleep(0.2)

        # GoToMeasurement
        self._send_message(0x10)
        time.sleep(0.1)
        self.serial.reset_input_buffer()
        print("[MTi-3] Output configuration set: Quaternion + RateOfTurn @ 50 Hz")

    def start(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.serial.reset_input_buffer()
        except Exception as e:
            print(f"[FATAL] Failed to open MTi-3 on {self.port}: {e}")
            return False

        self._configure_outputs()

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"MTi-3 Serial IMU connected at {self.baudrate} baud!")
        return True

    def _read_loop(self):
        while self.running:
            try:
                # 1. Look for the Xsens MTData2 header
                if self.serial.read(1) == b'\xFA' and self.serial.read(1) == b'\xFF' and self.serial.read(1) == b'\x36':
                    
                    # 2. Get length
                    length_bytes = self.serial.read(1)
                    if not length_bytes: continue
                    length = length_bytes[0]
                    
                    # 3. Read payload and checksum
                    data = self.serial.read(length)
                    checksum_bytes = self.serial.read(1)
                    if len(data) < length or not checksum_bytes: continue
                    
                    # Validate Checksum
                    packet_sum = (0xFF + 0x36 + length + sum(data) + checksum_bytes[0]) & 0xFF
                    if packet_sum != 0: continue # Bad packet
                        
                    # 4. Parse the data chunks
                    idx = 0
                    gyro, quat = None, None
                    debug_xdi = []

                    while idx < length:
                        group = data[idx]
                        type_id = data[idx+1]
                        size = data[idx+2]
                        payload = data[idx+3 : idx+3+size]

                        if self.debug and not self._debug_printed:
                            debug_xdi.append(f"0x{group:02X}{type_id:02X} size={size}")

                        # Rate of Turn (0x8020) -> 12 bytes (float32) or 24 bytes (double)
                        if group == 0x80 and (type_id == 0x20 or type_id == 0x23) and size in (12, 24):
                            fmt = ">3f" if size == 12 else ">3d"
                            gyro = np.array(struct.unpack(fmt, payload))
                            
                        # Quaternion (0x2010) -> 16 bytes
                        elif group == 0x20 and type_id == 0x10 and size == 16:
                            q = struct.unpack(">4f", payload)
                            r = Rotation.from_quat([q[1], q[2], q[3], q[0]]) # w,x,y,z -> x,y,z,w
                            quat = r.inv().apply([0.0, 0.0, -1.0])
                            
                        idx += 3 + size
                        
                    if self.debug and not self._debug_printed and debug_xdi:
                        print(f"\n[DEBUG] XDI blocks in packet: {', '.join(debug_xdi)}")
                        self._debug_printed = True

                    # 5. Update thread-safe dictionary
                    with self.lock:
                        if gyro is not None: self.latest_data["gyro"] = gyro
                        if quat is not None: self.latest_data["projected_gravity"] = quat
                            
            except Exception:
                pass # Ignore serial blips

    def get_latest_data(self):
        with self.lock:
            return {
                "gyro": self.latest_data["gyro"].copy(),
                "projected_gravity": self.latest_data["projected_gravity"].copy(),
            }

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.serial: self.serial.close()

# --- TEST IT ---
if __name__ == "__main__":
    WINDOW = 250  # samples to show (~5s at 50Hz)

    imu = MTi_Serial_IMU(debug=True)
    if not imu.start():
        exit(1)

    t_buf   = deque(maxlen=WINDOW)
    gx_buf  = deque(maxlen=WINDOW)
    gy_buf  = deque(maxlen=WINDOW)
    gz_buf  = deque(maxlen=WINDOW)
    t_start = time.time()

    px_buf = deque(maxlen=WINDOW)
    py_buf = deque(maxlen=WINDOW)
    pz_buf = deque(maxlen=WINDOW)

    fig, (ax_g, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_g.set_title("Angular Velocity (rad/s)")
    ax_g.set_ylabel("rad/s")
    ax_g.set_ylim(-10, 10)
    line_gx, = ax_g.plot([], [], label="X", color="tab:red")
    line_gy, = ax_g.plot([], [], label="Y", color="tab:green")
    line_gz, = ax_g.plot([], [], label="Z", color="tab:blue")
    ax_g.legend(loc="upper right")
    ax_g.grid(True, alpha=0.3)

    ax_p.set_title("Projected Gravity (unit vector)")
    ax_p.set_xlabel("Time (s)")
    ax_p.set_ylabel("component")
    ax_p.set_ylim(-1.1, 1.1)
    line_px, = ax_p.plot([], [], label="X", color="tab:red")
    line_py, = ax_p.plot([], [], label="Y", color="tab:green")
    line_pz, = ax_p.plot([], [], label="Z", color="tab:blue")
    ax_p.legend(loc="upper right")
    ax_p.grid(True, alpha=0.3)

    def update(_):
        data = imu.get_latest_data()
        g = data["gyro"]
        p = data["projected_gravity"]
        t = time.time() - t_start
        t_buf.append(t)
        gx_buf.append(g[0]); gy_buf.append(g[1]); gz_buf.append(g[2])
        px_buf.append(p[0]); py_buf.append(p[1]); pz_buf.append(p[2])
        t_arr = list(t_buf)
        line_gx.set_data(t_arr, list(gx_buf))
        line_gy.set_data(t_arr, list(gy_buf))
        line_gz.set_data(t_arr, list(gz_buf))
        line_px.set_data(t_arr, list(px_buf))
        line_py.set_data(t_arr, list(py_buf))
        line_pz.set_data(t_arr, list(pz_buf))
        if len(t_arr) > 1:
            ax_p.set_xlim(t_arr[0], max(t_arr[-1], t_arr[0] + 1))
        return line_gx, line_gy, line_gz, line_px, line_py, line_pz

    ani = animation.FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
    fig.suptitle("MTi-3 Live IMU Data")
    try:
        plt.tight_layout()
        plt.show()
    finally:
        imu.stop()