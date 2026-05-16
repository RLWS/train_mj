from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mjai.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    ResponsePredictionRequest,
    ResponsePredictionResponse,
    TurnPredictionRequest,
    TurnPredictionResponse,
)
from mjai.serving import MahjongPredictor, ServiceUnavailableError, ServingError


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/v1/models", response_model=ModelsResponse)
    def list_models(request: Request) -> dict[str, object]:
        try:
            predictor = _get_predictor(request)
        except ServingError as exc:
            return _error_response(exc)
        return {"models": predictor.model_metadata()}

    @router.post(
        "/v1/predict/turn",
        response_model=TurnPredictionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def predict_turn(request: Request, body: TurnPredictionRequest):
        try:
            predictor = _get_predictor(request)
            result = predictor.predict_turn(body.state, body.legal_actions, top_k=body.top_k)
        except ServingError as exc:
            return _error_response(exc)
        except Exception as exc:
            return _error_response(exc, status_code=500, code="internal_error")
        return asdict(result)

    @router.post(
        "/v1/predict/response",
        response_model=ResponsePredictionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def predict_response(request: Request, body: ResponsePredictionRequest):
        try:
            predictor = _get_predictor(request)
            result = predictor.predict_response(
                body.state,
                body.legal_actions,
                top_k=body.top_k,
                pass_bias=body.pass_bias,
            )
        except ServingError as exc:
            return _error_response(exc)
        except Exception as exc:
            return _error_response(exc, status_code=500, code="internal_error")
        return asdict(result)

    return router


def _get_predictor(request: Request) -> MahjongPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise ServiceUnavailableError("predictor is not ready")
    return predictor


def _error_response(
    exc: Exception,
    *,
    status_code: int | None = None,
    code: str | None = None,
) -> JSONResponse:
    if isinstance(exc, ServingError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc)}},
        )
    return JSONResponse(
        status_code=500 if status_code is None else status_code,
        content={
            "error": {
                "code": "internal_error" if code is None else code,
                "message": str(exc),
            }
        },
    )
