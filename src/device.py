from __future__ import annotations

import torch


def get_device() -> str:
    """Return the best available device in priority order: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def resolve_device(device: str | None) -> str:
    """Normalize an optional user-provided device to a valid runtime device."""
    if device is None:
        return get_device()

    normalized = device.lower()
    if normalized in {"cuda", "mps", "cpu"}:
        return normalized

    return get_device()
