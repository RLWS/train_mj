from __future__ import annotations


def normalize_device_name(device: str | None) -> str:
    """Normalize user-facing device aliases without importing heavy backends."""
    if device is None or device == "auto":
        return "auto"
    normalized = device.lower()
    aliases = {
        "gpu": "cuda",
        "cuda:0": "cuda",
        "p800": "xpu",
    }
    return aliases.get(normalized, normalized)
