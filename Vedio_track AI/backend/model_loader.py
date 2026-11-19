"""Model initialisation: weight download, device selection and process-wide caching.

The cache lives at module level (guarded by a lock) so it is shared by the
Streamlit script thread *and* the worker threads used for video jobs and
WebRTC — ``st.cache_resource`` alone cannot be used from those threads.

Command line helper (downloads the weights ahead of time)::

    python -m backend.model_loader yolo11n.pt
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

import config
from backend.detector import YoloDetector


class ModelLoadError(RuntimeError):
    """Raised with a user-friendly message when a model cannot be loaded."""


_CACHE: dict[tuple[str, str], YoloDetector] = {}
_LOCK = threading.Lock()
_LAST_ERROR: dict[str, str] = {}


def model_path(model_name: str) -> Path:
    return config.MODELS_DIR / model_name


def weights_on_disk(model_name: str) -> bool:
    return model_path(model_name).exists()


def resolve_device(preference: str = "auto") -> str:
    """Map the UI choice (auto / cpu / cuda) to an Ultralytics device string."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # torch missing or broken -> CPU path will report properly
        pass
    if preference == "cuda":
        raise ModelLoadError(
            "GPU inference was requested but no CUDA device is available. "
            "Choose 'CPU' or 'Auto' in Settings."
        )
    return "cpu"


def is_loaded(model_name: str, device: str = "auto") -> bool:
    return (model_name, device) in _CACHE


def last_error(model_name: str) -> str | None:
    return _LAST_ERROR.get(model_name)


def load_detector(model_name: str = config.DEFAULT_MODEL, device: str = "auto") -> YoloDetector:
    """Return a cached detector, loading (and downloading) the weights if needed."""
    key = (model_name, device)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        try:
            detector = _build_detector(model_name, device)
        except ModelLoadError as exc:
            _LAST_ERROR[model_name] = str(exc)
            raise
        _LAST_ERROR.pop(model_name, None)
        _CACHE[key] = detector
        return detector


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _build_detector(model_name: str, device_preference: str) -> YoloDetector:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    device = resolve_device(device_preference)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = model_path(model_name)

    model = None
    errors: list[str] = []
    # 1) preferred: weights inside ./models (Ultralytics downloads them there if missing)
    # 2) fallback: let Ultralytics resolve the bare name (its own weights directory)
    for candidate in (str(target), model_name):
        try:
            model = YOLO(candidate)
            break
        except Exception as exc:  # network failure, corrupt file, unknown name ...
            errors.append(f"{candidate}: {exc}")

    if model is None:
        raise ModelLoadError(
            f"Could not load '{model_name}'. If this is the first run, the weights are "
            f"downloaded automatically, so check your internet connection. You can also "
            f"download the file manually and place it in '{config.MODELS_DIR}'.\n"
            f"Details: {errors[-1] if errors else 'unknown error'}"
        )

    try:  # warm-up: builds the predictor so the first real frame is not slow
        model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device=device, verbose=False)
    except Exception as exc:
        raise ModelLoadError(f"'{model_name}' was loaded but a test inference failed: {exc}") from exc

    return YoloDetector(model, model_name=model_name, device=device)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    name = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_MODEL
    print(f"Loading {name} (downloads the weights on first use) ...")
    try:
        detector = load_detector(name, "cpu")
    except ModelLoadError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    print(f"OK — {len(detector.names)} classes, weights at {model_path(name)}")
