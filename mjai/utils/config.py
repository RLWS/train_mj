from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty documents."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data
