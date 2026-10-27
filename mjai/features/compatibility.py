from __future__ import annotations

from mjai.features.state_features import FEATURE_SCHEMA


def require_checkpoint_schema(checkpoint: dict) -> None:
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA}; rebuild data and retrain. "
                         "Use the original checkout to serve legacy checkpoints.")
