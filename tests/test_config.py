from __future__ import annotations

from mjai.utils.config import load_resolved_yaml, load_yaml


def test_load_base_config():
    """验证基础配置加载。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    config = load_yaml("configs/base.yaml")

    assert config["seed"] == 20260423
    assert config["exp_name"] == "local_smoke"


def test_load_resolved_base_config():
    """验证基础配置的默认项解析。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    config = load_resolved_yaml("configs/base.yaml")

    assert config["data"]["ruleset"] == "tenhou_4p_red"
    assert config["task"]["task"] == "turn"
    assert config["output_dir"] == "outputs/checkpoints/local_smoke"


def test_load_resolved_p800_configs():
    """验证P800 训练配置的默认项解析。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    turn_config = load_resolved_yaml("configs/turn_p800.yaml")
    response_config = load_resolved_yaml("configs/response_p800.yaml")

    assert turn_config["task"]["task"] == "turn"
    assert turn_config["train"]["device"] == "cuda"
    assert turn_config["data"]["train_manifest"] == "data/manifests/train.jsonl"
    assert response_config["task"]["task"] == "response"
    assert response_config["train"]["device"] == "cuda"
    assert response_config["data"]["candidate_limit"] == 16


def test_load_resolved_multitask_experiment_config():
    config = load_resolved_yaml("configs/experiments/multitask_p800_v1_2_interleave.yaml")

    assert config["task"]["task"] == "multitask"
    assert config["task"]["schedule"] == "batch_interleave"
    assert config["task"]["interleave_batches"] == {"turn": 10150, "response": 1420}
    assert config["train"]["distributed"] is True
    assert config["train"]["find_unused_parameters"] is True
    assert config["output_dir"] == "outputs/checkpoints/multitask_p800_v1_2_interleave"


def test_api_local_targets_2026_06_24_ddp_40w_models():
    config = load_yaml("configs/api_local.yaml")

    assert (
        config["models"]["turn"]["checkpoint"]
        == "runs/2026-06-24_p800_multitask_v1_1_ddp_40w/multitask/checkpoints/best_turn.pt"
    )
    assert (
        config["models"]["response"]["checkpoint"]
        == "runs/2026-06-24_p800_multitask_v1_1_ddp_40w/multitask/checkpoints/best_response.pt"
    )
    assert config["models"]["response"]["pass_bias"] == {
        "default": 2.0,
        "recall": 0.0,
        "conservative": 0.16,
    }
    assert config["models"]["response"]["no_yaku_call_penalty"] == 0.8
