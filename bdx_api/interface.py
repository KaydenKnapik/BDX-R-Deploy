import os
import sys
import time
import signal
import atexit
import threading
import numpy as np
import onnxruntime as ort
from abc import ABC, abstractmethod
from pathlib import Path

from bdx_api.config import load_policy_config, RobotPolicyConfig
from bdx_api.logger import RuntimeLogger


# ==========================================
# Utils & Keyboard ... (Kept the same)
# ==========================================

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

class KeyboardController:
    def __init__(self, lin_vel_step=0.1, ang_vel_step=0.1,
                 lin_vel_x_range=(-1.0, 1.0),
                 lin_vel_y_range=(-0.4, 0.4),
                 ang_vel_z_range=(-1.0, 1.0)):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range
        self._quit = False
        self._lock = threading.Lock()
        self._old_term_settings = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        import tty
        import termios
        import select
        fd = sys.stdin.fileno()
        self._old_term_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._quit:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                if ch3 == 'A': ch = 'w'
                                elif ch3 == 'B': ch = 's'
                                elif ch3 == 'D': ch = 'a'
                                elif ch3 == 'C': ch = 'd'
                                else: continue
                            else: continue
                        else:
                            self._quit = True
                            break
                    elif ch == '\x03':
                        self._quit = True
                        break
                    self._handle_key(ch.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)

    def _handle_key(self, ch):
        with self._lock:
            if ch == 'w': self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_step, self.lin_vel_x_range[1])
            elif ch == 's': self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_step, self.lin_vel_x_range[0])
            elif ch == 'a': self.lin_vel_y = min(self.lin_vel_y + self.lin_vel_step, self.lin_vel_y_range[1])
            elif ch == 'd': self.lin_vel_y = max(self.lin_vel_y - self.lin_vel_step, self.lin_vel_y_range[0])
            elif ch == 'q': self.ang_vel_yaw = min(self.ang_vel_yaw + self.ang_vel_step, self.ang_vel_z_range[1])
            elif ch == 'e': self.ang_vel_yaw = max(self.ang_vel_yaw - self.ang_vel_step, self.ang_vel_z_range[0])
            elif ch == ' ':
                self.lin_vel_x = 0.0; self.lin_vel_y = 0.0; self.ang_vel_yaw = 0.0

    def restore_terminal(self):
        if self._old_term_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term_settings)
            except Exception: pass

    @property
    def should_quit(self): return self._quit

    def get_command(self):
        with self._lock:
            return np.array([round(self.lin_vel_x, 2), round(self.lin_vel_y, 2), round(self.ang_vel_yaw, 2)], dtype=np.float32)

class RobotBackend(ABC):
    @abstractmethod
    def get_imu_angular_velocity(self) -> np.ndarray: ...
    @abstractmethod
    def get_projected_gravity(self) -> np.ndarray: ...
    @abstractmethod
    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None: ...
    @abstractmethod
    def step(self) -> None: ...
    @abstractmethod
    def is_running(self) -> bool: ...
    @abstractmethod
    def get_joint_map(self, joint_names: list[str]) -> np.ndarray: ...
    @abstractmethod
    def get_default_pos_array(self, num_actuators: int) -> np.ndarray: ...
    @abstractmethod
    def standup(self, duration: float = 2.0) -> None: ...
    @abstractmethod
    def hold_standing_tick(self) -> None: ...
    @abstractmethod
    def activate_policy_gains(self) -> None: ...


# ==========================================
# Policy Runner
# ==========================================
class PolicyRunner:
    def __init__(self, backend: RobotBackend, model_path: Path, decimation: int = 4):
        self.backend = backend
        self.cfg: RobotPolicyConfig = load_policy_config(model_path)
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.decimation = decimation

        self.joint_map = backend.get_joint_map(self.cfg.joint_names)
        self.num_actuators = len(self.joint_map)

        self.default_pos = backend.get_default_pos_array(self.num_actuators)
        for i, name in enumerate(self.cfg.joint_names):
            mj_id = self.joint_map[i]
            self.default_pos[mj_id] = self.cfg.default_joint_pos[i]

        self.action_scale = np.array(self.cfg.action_scale, dtype=np.float32)
        self.actions = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
        self.dof_targets = self.default_pos.copy()

        self.kb = KeyboardController()
        self.logger = RuntimeLogger(joint_names=self.cfg.joint_names)
        self._interrupted = False
        self._sigint_count = 0

    def _handle_sigint(self, signum, frame):
        self._sigint_count += 1
        if self._sigint_count >= 2:
            print("\n\n  Double Ctrl+C — force quitting!")
            self._cleanup()
            os._exit(1)
        print("\n\n  Ctrl+C received — stopping and plotting...")
        self._interrupted = True

    def _cleanup(self):
        try: self.kb.restore_terminal()
        except Exception: pass

    def run(self):
        old_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        atexit.register(self._cleanup)

        print("\n" + "=" * 50)
        print("  READY TO STAND")
        print("=" * 50)
        input(">>> Press ENTER to stand up (safe standing gains)...")
        self.backend.standup(duration=2.0)

        print("\nRobot is standing with safe gains. Place it on the floor.")
        print(">>> Press ENTER to deploy the policy (switches to trained Kp/Kd)...")

        enter_pressed = threading.Event()
        def _wait_for_enter():
            input()
            enter_pressed.set()
        waiter = threading.Thread(target=_wait_for_enter, daemon=True)
        waiter.start()

        while not enter_pressed.is_set() and self.backend.is_running():
            self.backend.hold_standing_tick()

        self.backend.activate_policy_gains()
        
        print("\n" + "=" * 50)
        print("  DEPLOYING POLICY (DEBUG MODE - SAFETY OVERRIDE ACTIVE)")
        print("-" * 50)
        print("  !!! THE ROBOT WILL NOT MOVE - HOLDING POS ONLY !!!")
        print("=" * 50)

        it = 0
        policy_counter = 0
        policy_start_time = time.time()
        actual_policy_freq = 0.0
        
        # --- NEW: Loop timings tracking for Jitter ---
        loop_times =[]

        while self.backend.is_running() and not self.kb.should_quit and not self._interrupted:
            # High-res start time for latency test
            loop_start = time.perf_counter()

            if it % self.decimation == 0:
                cmd = self.kb.get_command()

                ang_vel = self.backend.get_imu_angular_velocity()
                projected_gravity = self.backend.get_projected_gravity()
                dof_pos = self.backend.get_joint_positions(self.joint_map)
                dof_vel = self.backend.get_joint_velocities(self.joint_map)

                obs_parts =[]
                for term in self.cfg.observation_names:
                    term = term.strip()
                    if term == "projected_gravity": obs_parts.append(projected_gravity)
                    elif term == "base_ang_vel": obs_parts.append(ang_vel)
                    elif term in ("joint_pos", "dof_pos"): obs_parts.append(dof_pos - self.default_pos[self.joint_map])
                    elif term in ("joint_vel", "dof_vel"): obs_parts.append(dof_vel)
                    elif term == "actions": obs_parts.append(self.actions)
                    elif term in ("command", "commands"): obs_parts.append(cmd)

                obs = np.concatenate(obs_parts).astype(np.float32)

                # Run policy
                self.actions = self.session.run(None, {self.input_name: obs.reshape(1, -1)})[0][0]
                
                # =================================================================
                # [SAFETY OVERRIDE] HOLD ZERO POS
                # We completely ignore the policy output for the hardware commands.
                # =================================================================
                self.dof_targets[self.joint_map] = self.default_pos[self.joint_map].copy()
                
                self.logger.log(
                    cmd=cmd,
                    target_pos=self.dof_targets[self.joint_map],
                    actual_pos=dof_pos,
                    actual_vel=dof_vel,
                    ang_vel=ang_vel,
                    projected_gravity=projected_gravity,
                    actions=self.actions, # Logs the simulated actions to CSV
                )

                policy_counter += 1
                if policy_counter >= 10:
                    now = time.time()
                    actual_policy_freq = policy_counter / (now - policy_start_time)
                    policy_counter = 0
                    policy_start_time = now

            self.backend.set_joint_targets(self.dof_targets)
            self.backend.step()
            
            # --- NEW: Stop high-res timer ---
            loop_end = time.perf_counter()
            loop_times.append(loop_end - loop_start)

            # --- NEW: Debug Block (Runs every ~0.5 seconds) ---
            if it % 100 == 0 and it > 0:
                l_arr = np.array(loop_times) * 1000 # Convert to ms
                loop_times =[] # reset buffer

                print("\n" + "-"*50)
                print(f"[DEBUG TICK {it}]")
                # 1. Loop Timing (Jitter)
                print(f"  Loop Timing:   Avg {l_arr.mean():.2f}ms | Max {l_arr.max():.2f}ms | Std {l_arr.std():.2f}ms")
                
                # 2. CAN Latency (from backend if hardware)
                if hasattr(self.backend, 'get_latency_stats'):
                    lats = self.backend.get_latency_stats()
                    if len(lats) > 0:
                        first_motor = list(lats.keys())[0]
                        print(f"  CAN Latency:   Motor {first_motor} -> Max: {lats[first_motor]['max']:5.2f}ms | Std: {lats[first_motor]['std']:5.2f}ms")
                
                # 3. Network Outputs (Are they sane?)
                print(f"  Actions Out:   Min: {self.actions.min():5.2f} | Max: {self.actions.max():5.2f}")
                print(f"  Raw Actions[0:3]: {self.actions[0:3]}")
                
                # 4. Sensor Inputs (Is gravity drifting?)
                print(f"  Proj Gravity:  {projected_gravity}")
                print(f"  Commanded Pos: {self.dof_targets[0:3]} (SAFETY LOCKED)")
                print("-" * 50)
                
            it += 1

        self._cleanup()
        signal.signal(signal.SIGINT, old_handler)
        print(f"\n\nRun complete. {len(self.logger.entries)} samples logged.")
        
        if self._interrupted and len(self.logger.entries) > 1:
            print("\nGenerating plots...")
            self.logger.plot(save_dir="logs")
        elif len(self.logger.entries) > 1:
            print("(Press Ctrl+C next time to auto-plot, or run logger.plot() manually)")

        print("Shutting down.")