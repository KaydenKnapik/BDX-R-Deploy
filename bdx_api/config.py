# bdx_api/config.py

import os
import onnx
import yaml  # <--- NEW IMPORT
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict  # <--- ADDED Dict


# ==========================================
# Model Path Resolution
# ==========================================

def resolve_model_path() -> Path:
    env_path = os.getenv("BDX_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise FileNotFoundError(f"BDX_MODEL_PATH={env_path} does not exist")
        return p

    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "models" / "best.onnx"

    if not default_path.exists():
        raise FileNotFoundError(
            f"No model found at {default_path}. "
            f"Set BDX_MODEL_PATH or place your .onnx file there."
        )

    return default_path


MODEL_PATH = resolve_model_path()


# ==========================================
# Dataclass Definition
# ==========================================

@dataclass(frozen=True)
class RobotPolicyConfig:
    obs_dim: int
    action_dim: int
    joint_names: List[str]
    joint_stiffness: List[float]
    joint_damping: List[float]
    default_joint_pos: List[float]
    action_scale: List[float]
    observation_names: List[str]
    command_names: List[str]
    obs_scales: Dict[str, float]  # <--- NEW FIELD


# ==========================================
# Utility Parsers
# ==========================================

def _parse_float_list(value: str) -> List[float]:
    cleaned = value.replace("[", "").replace("]", "").strip()
    return [float(x.strip()) for x in cleaned.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    cleaned = value.replace("[", "").replace("]", "").strip()
    return [x.strip().strip("'\"") for x in cleaned.split(",") if x.strip()]


# ==========================================
# ONNX Metadata Loader
# ==========================================

def load_policy_config(model_path: Path = None) -> RobotPolicyConfig:
    if model_path is None:
        from bdx_api.config import resolve_model_path
        model_path = resolve_model_path()

    # ==========================================
    # 1. PYTORCH / ISAACLAB PATH (Reads YAML)
    # ==========================================
    if model_path.suffix == '.pt':
        yaml_path = model_path.with_suffix('.yaml')
        if not yaml_path.exists():
            raise FileNotFoundError(f"PyTorch models require a sidecar config. Missing: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return RobotPolicyConfig(
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            joint_names=data["joint_names"],
            joint_stiffness=data["joint_stiffness"],
            joint_damping=data["joint_damping"],
            default_joint_pos=data["default_joint_pos"],
            action_scale=data["action_scale"],
            observation_names=data["observation_names"],
            command_names=data.get("command_names", ["lin_vel_x", "lin_vel_y", "ang_vel_z"]),
            obs_scales=data.get("obs_scales", {})  # Load scales if they exist
        )

    # ==========================================
    # 2. ONNX / MJLAB PATH (Reads Metadata)
    # ==========================================
    elif model_path.suffix == '.onnx':
        model = onnx.load(str(model_path))

        if len(model.graph.input) != 1:
            raise ValueError(f"Expected 1 input, got {len(model.graph.input)}")
        if len(model.graph.output) != 1:
            raise ValueError(f"Expected 1 output, got {len(model.graph.output)}")

        input_tensor = model.graph.input[0]
        output_tensor = model.graph.output[0]
        obs_dim = input_tensor.type.tensor_type.shape.dim[1].dim_value
        action_dim = output_tensor.type.tensor_type.shape.dim[1].dim_value

        metadata = {prop.key: prop.value for prop in model.metadata_props}

        required_keys =[
            "joint_names", "joint_stiffness", "joint_damping",
            "default_joint_pos", "action_scale", "observation_names",
        ]
        for key in required_keys:
            if key not in metadata:
                raise KeyError(f"Missing required ONNX metadata key: '{key}'")

        return RobotPolicyConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            joint_names=_parse_str_list(metadata["joint_names"]),
            joint_stiffness=_parse_float_list(metadata["joint_stiffness"]),
            joint_damping=_parse_float_list(metadata["joint_damping"]),
            default_joint_pos=_parse_float_list(metadata["default_joint_pos"]),
            action_scale=_parse_float_list(metadata["action_scale"]),
            observation_names=_parse_str_list(metadata["observation_names"]),
            command_names=_parse_str_list(metadata.get("command_names", "lin_vel_x,lin_vel_y,ang_vel_z")),
            obs_scales={}  # Empty dict for ONNX (since you don't use it there)
        )
    else:
        raise ValueError(f"Unsupported model extension: {model_path.suffix}. Use .onnx or .pt")


# ==========================================
# Safe Standing Gains
# ==========================================

# Higher than policy gains — used to hold the robot stiffly upright
# before the learned policy takes over.
STANDUP_GAINS = {
    "Left_Hip_Yaw":    (78.957, 5.027),
    "Left_Hip_Roll":   (78.957, 5.027),
    "Left_Hip_Pitch":  (78.957, 5.027),
    "Left_Knee":       (78.957, 5.027),
    "Left_Ankle":      (16.581, 1.056),
    "Right_Hip_Yaw":   (78.957, 5.027),
    "Right_Hip_Roll":  (78.957, 5.027),
    "Right_Hip_Pitch": (78.957, 5.027),
    "Right_Knee":      (78.957, 5.027),
    "Right_Ankle":     (16.581, 1.056),
    "Neck_Pitch":      (16.581, 1.056),
    "Head_Pitch":      (2.763,  0.176),
    "Head_Yaw":        (2.763,  0.176),
    "Head_Roll":       (2.763,  0.176),
}

# ==========================================
# Fallback Policy Gains
# ==========================================
# Used if the ONNX exporter accidentally writes Kp=1.0 and Kd=0.0
POLICY_GAINS_FALLBACK = {
    "Left_Hip_Yaw":    (78.957, 5.027),
    "Left_Hip_Roll":   (78.957, 5.027),
    "Left_Hip_Pitch":  (78.957, 5.027),
    "Left_Knee":       (78.957, 5.027),
    "Left_Ankle":      (16.581, 1.056),
    "Right_Hip_Yaw":   (78.957, 5.027),
    "Right_Hip_Roll":  (78.957, 5.027),
    "Right_Hip_Pitch": (78.957, 5.027),
    "Right_Knee":      (78.957, 5.027),
    "Right_Ankle":     (16.581, 1.056),
    "Neck_Pitch":      (16.581, 1.056),
    "Head_Pitch":      (2.763,  0.176),
    "Head_Yaw":        (2.763,  0.176),
    "Head_Roll":       (2.763,  0.176),
}

# ==========================================
# Debug
# ==========================================

if __name__ == "__main__":
    cfg = load_policy_config()
    print(f"Obs dim:    {cfg.obs_dim}")
    print(f"Action dim: {cfg.action_dim}")
    print(f"Joints:     {cfg.joint_names}")
    print(f"Kp:         {cfg.joint_stiffness}")
    print(f"Kd:         {cfg.joint_damping}")
    print(f"Defaults:   {cfg.default_joint_pos}")
    print(f"Scale:      {cfg.action_scale}")
    print(f"Obs names:  {cfg.observation_names}")
    print(f"Cmd names:  {cfg.command_names}")