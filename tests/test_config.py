from __future__ import annotations

from mjai.utils.config import load_yaml


def test_load_base_config():
    config = load_yaml("configs/base.yaml")

    assert config["seed"] == 20260423
    assert config["exp_name"] == "local_smoke"
