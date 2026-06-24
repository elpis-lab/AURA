from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from train_model import load_model

_pushing_model_cache = None
_pushing_model_key = None


def _resolve_model_path(model_name: str, model_path: str | None = None) -> str:
    if model_path:
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(f"Pushing model path does not exist: {model_path}")

    model_names = [str(model_name)]
    if str(model_name) == "real_cracker_box":
        model_names.append("real_cracker_box_flipped")

    candidates = []
    for name in model_names:
        candidates.extend(
            [
                os.path.join("learned_models", f"{name}_mlp_0.0_1000_0.pth"),
                os.path.join("learned_models", f"{name}.pth"),
            ]
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    aura_dir = os.path.dirname(current_dir)
    candidates.extend(os.path.join(aura_dir, candidate) for candidate in list(candidates))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No pushing model found for model_name={model_name!r}. "
        f"Looked for: {', '.join(candidates)}"
    )


def get_pushing_model(
    object_shape,
    model_name: str = "cracker_box_flipped",
    model_path: str | None = None,
):
    """Get or load the pushing model (cached)."""
    global _pushing_model_cache, _pushing_model_key

    object_shape_arr = np.asarray(object_shape, dtype=float)
    resolved_model_path = _resolve_model_path(model_name, model_path)
    model_key = (
        tuple(np.round(object_shape_arr.reshape(-1), 12)),
        str(model_name),
        os.path.abspath(resolved_model_path),
    )
    if _pushing_model_cache is None or _pushing_model_key != model_key:
        model = load_model("mlp", object_shape)
        print(f"[INFO] Loading pushing model from: {resolved_model_path}")
        model.load(resolved_model_path)
        model = model.model
        model.eval()
        _pushing_model_cache = model
        _pushing_model_key = model_key

    return _pushing_model_cache
