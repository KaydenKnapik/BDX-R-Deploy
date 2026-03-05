"""
Hybrid backend: MuJoCo simulation + real IMU data from the robot.

Uses MuJoCo for joint physics (positions, velocities, stepping, rendering)
but overrides IMU readings (angular velocity, projected gravity) with live
data received over UDP from the Jetson via IMUReceiver.
"""

import atexit
import numpy as np
from pathlib import Path

from bdx_api.sim import MujocoBackend
from bdx_api.imu import IMUReceiver


class HybridIMUBackend(MujocoBackend):
    """MuJoCo sim with real IMU data injected from the robot."""

    def __init__(
        self,
        xml_path: str,
        model_path: Path,
        sim_dt: float = 0.005,
        fixed: bool = False,
        imu_port: int = 5005,
    ):
        # Initialize MuJoCo simulation as normal
        super().__init__(
            xml_path=xml_path,
            model_path=model_path,
            sim_dt=sim_dt,
            fixed=fixed,
        )

        # Start the UDP IMU receiver
        self.imu_receiver = IMUReceiver(port=imu_port)
        self.imu_receiver.start()

        self._imu_connected_warned = False

        # Ensure IMU receiver is stopped even on abnormal exit
        atexit.register(self.stop_imu)

    # --- Override IMU methods to use real data ---

    def get_imu_angular_velocity(self) -> np.ndarray:
        if not self.imu_receiver.is_connected():
            if not self._imu_connected_warned:
                print("[HybridIMU] WARNING: No IMU data yet — using zeros")
                self._imu_connected_warned = True
            return np.zeros(3, dtype=np.float32)
        self._imu_connected_warned = False
        data = self.imu_receiver.get_data()
        return data["gyro"].astype(np.float32)

    def get_projected_gravity(self) -> np.ndarray:
        if not self.imu_receiver.is_connected():
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        data = self.imu_receiver.get_data()
        return data["projected_gravity"].astype(np.float32)

    def stop_imu(self):
        """Cleanly shut down the IMU receiver."""
        try:
            self.imu_receiver.stop()
        except Exception:
            pass
