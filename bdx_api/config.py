# bdx_api/config.py

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import onnx


# ==========================================
# Model Path Resolution (Robust Version)
# ==========================================

def resolve_model_path() -> Path:
    """
    Resolution priority:
    1. Environment variable: BDX_MODEL_PATH
    2. Project-root-relative: models/walk.onnx
    """

    # 1️⃣ Environment override
    env_path = os.getenv("BDX_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"BDX_MODEL_PATH not found: {path}")
        return path

    # 2️⃣ Resolve relative to project root
    # config.py -> bdx_api/ -> project root
    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "models" / "walk.onnx"

    if not default_path.exists():
        raise FileNotFoundError(
            f"Default model not found at {default_path}\n"
            "Set BDX_MODEL_PATH environment variable."
        )

    return default_path


MODEL_PATH = resolve_model_path()


# ==========================================
# Dataclass Definition
# ==========================================

@dataclass(frozen=True)
class RobotPolicyConfig:
    joint_names: List[str]
    joint_stiffness: List[float]
    joint_damping: List[float]
    default_joint_pos: List[float]
    action_scale: List[float]
    command_names: List[str]
    observation_names: List[str]
    obs_dim: int
    action_dim: int


# ==========================================
# Utility Parsers
# ==========================================

def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


# ==========================================
# ONNX Metadata Loader
# ==========================================

def load_policy_config(model_path: Path = MODEL_PATH) -> RobotPolicyConfig:
    model = onnx.load(str(model_path))

    # ---- Shape extraction ----
    if len(model.graph.input) != 1:
        raise ValueError("Model must have exactly 1 input tensor")

    if len(model.graph.output) != 1:
        raise ValueError("Model must have exactly 1 output tensor")

    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]

    obs_dim = input_tensor.type.tensor_type.shape.dim[1].dim_value
    action_dim = output_tensor.type.tensor_type.shape.dim[1].dim_value

    # ---- Metadata extraction ----
    metadata: Dict[str, str] = {
        prop.key: prop.value for prop in model.metadata_props
    }

    required_keys = [
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "action_scale",
        "command_names",
        "observation_names",
    ]

    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required metadata key: {key}")

    joint_names = _parse_str_list(metadata["joint_names"])
    joint_stiffness = _parse_float_list(metadata["joint_stiffness"])
    joint_damping = _parse_float_list(metadata["joint_damping"])
    default_joint_pos = _parse_float_list(metadata["default_joint_pos"])
    action_scale = _parse_float_list(metadata["action_scale"])
    command_names = _parse_str_list(metadata["command_names"])
    observation_names = _parse_str_list(metadata["observation_names"])

    # ---- Safety validation ----
    if len(joint_names) != action_dim:
        raise ValueError("Joint count != action_dim")

    if len(action_scale) != action_dim:
        raise ValueError("Action scale length mismatch")

    if len(joint_stiffness) != action_dim:
        raise ValueError("joint_stiffness length mismatch")

    if len(joint_damping) != action_dim:
        raise ValueError("joint_damping length mismatch")

    if len(default_joint_pos) != action_dim:
        raise ValueError("default_joint_pos length mismatch")

    return RobotPolicyConfig(
        joint_names=joint_names,
        joint_stiffness=joint_stiffness,
        joint_damping=joint_damping,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        command_names=command_names,
        observation_names=observation_names,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


# ==========================================
# Singleton Config (Load Once)
# ==========================================

CONFIG = load_policy_config()


# ==========================================
# Debug
# ==========================================

if __name__ == "__main__":
    print("\n=== MODEL PATH ===")
    print(MODEL_PATH)
    print("\n=== CONFIG ===")
    print(CONFIG)