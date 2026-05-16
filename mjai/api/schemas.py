from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfo(BaseModel):
    checkpoint: str
    checkpoint_task: str
    task: str
    device: str
    state_dim: int
    action_dim: int
    candidate_limit: int
    hidden_size: int
    dropout: float
    default_pass_bias: float | None = None


class ModelsResponse(BaseModel):
    models: dict[str, ModelInfo]


class PredictionRequest(BaseModel):
    state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    top_k: int = 5


class TurnPredictionRequest(PredictionRequest):
    pass


class ResponsePredictionRequest(PredictionRequest):
    pass_bias: float | None = None


class PredictionItem(BaseModel):
    rank: int
    index: int
    action: dict[str, Any]
    score: float
    prob: float


class PredictionModelInfo(ModelInfo):
    pass


class TurnPredictionResponse(BaseModel):
    task: str
    selection_mode: str
    selected_action: dict[str, Any]
    predictions: list[PredictionItem]
    model: PredictionModelInfo


class ResponsePredictionResponse(BaseModel):
    task: str
    selection_mode: str
    pass_bias: float
    selected_action: dict[str, Any]
    raw_selected_action: dict[str, Any]
    predictions: list[PredictionItem]
    model: PredictionModelInfo


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
