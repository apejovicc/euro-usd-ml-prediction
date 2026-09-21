from __future__ import annotations
import os
import random
import yaml
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def resolve_path(*parts: str) -> str:
    path = os.path.join(PROJECT_ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
