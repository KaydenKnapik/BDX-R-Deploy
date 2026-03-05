# bdx_api/config.py

import os
import onnx
from pathlib import Path
from dataclasses import dataclass
from typing import List


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
    default_path = project_root / "models" / "walk.onnx"

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

def load_policy_config(model_path: Path = MODEL_PATH) -> RobotPolicyConfig:
    model = onnx.load(str(model_path))

    # Shape extraction
    if len(model.graph.input) != 1:
        raise ValueError(f"Expected 1 input, got {len(model.graph.input)}")
    if len(model.graph.output) != 1:
        raise ValueError(f"Expected 1 output, got {len(model.graph.output)}")

    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]
    obs_dim = input_tensor.type.tensor_type.shape.dim[1].dim_value
    action_dim = output_tensor.type.tensor_type.shape.dim[1].dim_value

    # Metadata extraction
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    required_keys = [
        "joint_names", "joint_stiffness", "joint_damping",
        "default_joint_pos", "action_scale", "observation_names",
    ]
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"Missing required ONNX metadata key: '{key}'")

    joint_names = _parse_str_list(metadata["joint_names"])
    joint_stiffness = _parse_float_list(metadata["joint_stiffness"])
    joint_damping = _parse_float_list(metadata["joint_damping"])
    default_joint_pos = _parse_float_list(metadata["default_joint_pos"])
    action_scale = _parse_float_list(metadata["action_scale"])
    observation_names = _parse_str_list(metadata["observation_names"])
    command_names = _parse_str_list(metadata.get("command_names", "lin_vel_x,lin_vel_y,ang_vel_z"))

    return RobotPolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        joint_names=joint_names,
        joint_stiffness=joint_stiffness,
        joint_damping=joint_damping,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        observation_names=observation_names,
        command_names=command_names,
    )


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