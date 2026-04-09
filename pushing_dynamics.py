from __future__ import annotations

import os

import numpy as np

from train_model import load_model

_pushing_model_cache = None
_pushing_model_shape = None


def get_pushing_model(object_shape):
    """Get or load the pushing model (cached)."""
    global _pushing_model_cache, _pushing_model_shape

    if (
        _pushing_model_cache is None
        or _pushing_model_shape is None
        or not np.array_equal(_pushing_model_shape, object_shape)
    ):
        model = load_model("mlp", object_shape)

        model_path = "learned_models/cracker_box_flipped_mlp_0.0_1000_0.pth"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        aura_dir = os.path.dirname(current_dir)
        relative_path = os.path.join(
            aura_dir, "learned_models", "cracker_box_flipped_mlp_0.0_1000_0.pth"
        )

        if os.path.exists(model_path):
            pass
        elif os.path.exists(relative_path):
            model_path = relative_path
        elif os.path.exists(os.path.join("aura", model_path)):
            model_path = os.path.join("aura", model_path)

        print(f"[INFO] Loading pushing model from: {model_path}")
        model.load(model_path)
        model = model.model
        model.eval()
        _pushing_model_cache = model
        _pushing_model_shape = (
            object_shape.copy() if isinstance(object_shape, np.ndarray) else np.array(object_shape)
        )

    return _pushing_model_cache
