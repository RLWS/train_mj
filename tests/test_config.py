from __future__ import annotations

from mjai.utils.config import load_resolved_yaml, load_yaml


def test_load_base_config():
    config = load_yaml("configs/base.yaml")

    assert config["seed"] == 20260423
    assert config["exp_name"] == "local_smoke"


def test_load_resolved_base_config():
    config = load_resolved_yaml("configs/base.yaml")

    assert config["data"]["ruleset"] == "tenhou_4p_red"
    assert config["task"]["task"] == "turn"
    assert config["output_dir"] == "outputs/checkpoints/local_smoke"
