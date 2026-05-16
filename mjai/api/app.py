from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mjai.serving import MahjongPredictor, build_predictor_from_config
from mjai.utils.config import load_resolved_yaml


API_CONFIG_ENV = "MJAI_API_CONFIG"


def create_app(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    predictor: MahjongPredictor | None = None,
):
    try:
        from fastapi import FastAPI
        from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
        from fastapi.responses import JSONResponse
        from mjai.api.deps import ApiKeyMiddleware, security_config
        from mjai.api.routes import build_router
    except ImportError as exc:  # pragma: no cover - exercised when api extra is absent
        raise ImportError("FastAPI is required to run the serving API") from exc

    resolved_config = _load_api_config(config_path=config_path, config=config)
    app_config = resolved_config.get("api", {})
    enable_docs = bool(app_config.get("enable_docs", True))
    app = FastAPI(
        title="MJAI Serving API",
        version="0.1.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    @app.exception_handler(FastAPIRequestValidationError)
    async def _request_validation_error_handler(_request, exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": _format_request_validation_error(exc),
                }
            },
        )

    app.state.api_config = resolved_config
    app.state.predictor = predictor or build_predictor_from_config(
        resolved_config,
        base_dir=Path.cwd(),
    )
    security = security_config(resolved_config)
    app.add_middleware(
        ApiKeyMiddleware,
        api_key_env=str(security.get("api_key_env", "MJAI_API_KEY")),
        allow_without_api_key=bool(security.get("allow_without_api_key", True)),
        header_name=str(security.get("header_name", "X-API-Key")),
    )
    app.include_router(build_router())
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mjai.api.app", description="Serve MJAI inference API")
    parser.add_argument("--config", required=True, help="Path to the API config file.")
    parser.add_argument("--host", default=None, help="Override API host from config.")
    parser.add_argument("--port", type=int, default=None, help="Override API port from config.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    os.environ[API_CONFIG_ENV] = str(config_path)
    app_config = _load_api_config(config_path=config_path, config=None).get("api", {})

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised when api extra is absent
        raise ImportError("uvicorn is required to run the serving API") from exc

    uvicorn.run(
        "mjai.api.app:create_app",
        host=args.host or str(app_config.get("host", "127.0.0.1")),
        port=int(args.port or app_config.get("port", 8000)),
        reload=bool(args.reload),
        factory=True,
        reload_dirs=[str(config_path.parent)],
    )
    return 0


def _load_api_config(
    *,
    config_path: str | Path | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if config is not None:
        return config
    if config_path is None:
        env_path = os.getenv(API_CONFIG_ENV)
        if env_path:
            return load_resolved_yaml(env_path)
        return {}
    return load_resolved_yaml(config_path)


def _format_request_validation_error(exc) -> str:
    errors = exc.errors()
    if not errors:
        return "request validation failed"
    first_error = errors[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()) if part != "body")
    message = str(first_error.get("msg", "request validation failed"))
    if location:
        return f"{location}: {message}"
    return message


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
