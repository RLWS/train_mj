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


def test_load_resolved_p800_configs():
    turn_config = load_resolved_yaml("configs/turn_p800.yaml")
    response_config = load_resolved_yaml("configs/response_p800.yaml")

    assert turn_config["task"]["task"] == "turn"
    assert turn_config["train"]["device"] == "xpu"
    assert turn_config["data"]["train_manifest"] == "data/manifests/train.jsonl"
    assert response_config["task"]["task"] == "response"
    assert response_config["train"]["device"] == "xpu"
    assert response_config["data"]["candidate_limit"] == 16
