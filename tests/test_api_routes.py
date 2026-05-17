from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from mjai.api.app import create_app
from tests.test_serving_predictor import (
    _api_config,
    _response_state,
    _turn_state,
    _write_checkpoint,
)
from mjai.serving.predictor import build_predictor_from_config


def test_api_health_models_and_predictions(tmp_path: Path):
    client = _client(tmp_path)

    health = client.get("/health")
    models = client.get("/v1/models")
    turn = client.post(
        "/v1/predict/turn",
        json={
            "state": _turn_state(),
            "legal_actions": [
                {"kind": "discard", "tile": "1m"},
                {"kind": "discard", "tile": "2m"},
            ],
            "top_k": 1,
        },
    )
    response = client.post(
        "/v1/predict/response",
        json={
            "state": _response_state(),
            "legal_actions": [
                {"kind": "pass"},
                {"kind": "chi", "tile": "3m", "consumed_tiles": ["1m", "2m"], "source": 0},
            ],
            "top_k": 2,
        },
    )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert models.status_code == 200
    assert set(models.json()["models"]) == {"turn", "response"}
    assert turn.status_code == 200
    assert turn.json()["selected_action"] in [
        {"kind": "discard", "tile": "1m"},
        {"kind": "discard", "tile": "2m"},
    ]
    assert len(turn.json()["predictions"]) == 1
    assert response.status_code == 200
    assert response.json()["selection_mode"] == "calibrated"
    assert response.json()["pass_bias"] == 0.3
    assert response.json()["selected_action"] in [
        {"kind": "pass"},
        {"kind": "chi", "tile": "3m", "consumed_tiles": ["1m", "2m"], "source": 0},
    ]


def test_api_returns_planned_error_codes(tmp_path: Path):
    client = _client(tmp_path, max_legal_actions=1)

    empty_actions = client.post(
        "/v1/predict/turn",
        json={"state": _turn_state(), "legal_actions": [], "top_k": 1},
    )
    too_many_actions = client.post(
        "/v1/predict/turn",
        json={
            "state": _turn_state(),
            "legal_actions": [
                {"kind": "discard", "tile": "1m"},
                {"kind": "discard", "tile": "2m"},
            ],
            "top_k": 1,
        },
    )
    missing_response_target = client.post(
        "/v1/predict/response",
        json={"state": _turn_state(), "legal_actions": [{"kind": "pass"}], "top_k": 1},
    )

    assert empty_actions.status_code == 400
    assert empty_actions.json()["error"]["code"] == "invalid_request"
    assert too_many_actions.status_code == 413
    assert too_many_actions.json()["error"]["code"] == "too_many_candidates"
    assert missing_response_target.status_code == 422
    assert missing_response_target.json()["error"]["code"] == "invalid_state"


def test_api_normalizes_framework_validation_errors(tmp_path: Path):
    client = _client(tmp_path)

    malformed = client.post("/v1/predict/turn", json={"state": _turn_state(), "top_k": 1})
    invalid_top_k = client.post(
        "/v1/predict/turn",
        json={
            "state": _turn_state(),
            "legal_actions": [{"kind": "discard", "tile": "1m"}],
            "top_k": 0,
        },
    )
    invalid_action = client.post(
        "/v1/predict/turn",
        json={
            "state": _turn_state(),
            "legal_actions": [{"kind": "chi", "tile": "1m"}],
            "top_k": 1,
        },
    )

    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_request"
    assert invalid_top_k.status_code == 400
    assert invalid_top_k.json()["error"]["code"] == "invalid_request"
    assert invalid_action.status_code == 422
    assert invalid_action.json()["error"]["code"] == "invalid_state"


def test_api_key_middleware_blocks_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MJAI_API_KEY", "secret")
    client = _client(
        tmp_path,
        security={"api_key_env": "MJAI_API_KEY", "allow_without_api_key": False},
    )

    missing = client.get("/health")
    accepted = client.get("/health", headers={"X-API-Key": "secret"})

    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"
    assert accepted.status_code == 200


def test_api_debug_logging_can_emit_request_and_response_bodies(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    client = _client(
        tmp_path,
        api_overrides={
            "debug_log_requests": True,
            "debug_log_responses": True,
            "debug_log_max_body_chars": 10_000,
            "debug_log_logger": "mjai.api.requests",
        },
    )

    with caplog.at_level("INFO", logger="mjai.api.requests"):
        response = client.post(
            "/v1/predict/turn",
            json={
                "state": _turn_state(),
                "legal_actions": [{"kind": "discard", "tile": "1m"}],
                "top_k": 1,
            },
        )

    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records]
    assert any("path=/v1/predict/turn" in message for message in messages)
    assert any(
        '"legal_actions":[{"kind":"discard","tile":"1m"}]' in message for message in messages
    )
    assert any("API response body" in message for message in messages)
    assert any(
        '"selected_action":{"kind":"discard","tile":"1m"}' in message for message in messages
    )


def _client(
    tmp_path: Path,
    *,
    max_legal_actions: int = 64,
    security: dict | None = None,
    api_overrides: dict | None = None,
) -> TestClient:
    turn_ckpt = _write_checkpoint(tmp_path / "turn.pt", task="turn", hidden_size=16)
    response_ckpt = _write_checkpoint(tmp_path / "response.pt", task="response", hidden_size=16)
    config = _api_config(
        turn_ckpt,
        response_ckpt,
        max_legal_actions=max_legal_actions,
        pass_bias=0.3,
    )
    if api_overrides:
        config["api"].update(api_overrides)
    config["security"] = security or {"allow_without_api_key": True}
    predictor = build_predictor_from_config(config)
    app = create_app(config=config, predictor=predictor)
    return TestClient(app)
