from __future__ import annotations

import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        api_key_env: str = "MJAI_API_KEY",
        allow_without_api_key: bool = True,
        header_name: str = "X-API-Key",
    ) -> None:
        super().__init__(app)
        self.api_key_env = api_key_env
        self.allow_without_api_key = allow_without_api_key
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        expected_key = os.getenv(self.api_key_env)
        if self.allow_without_api_key:
            return await call_next(request)
        if not expected_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "api_key_not_configured",
                        "message": f"{self.api_key_env} is not configured",
                    }
                },
            )
        actual_key = request.headers.get(self.header_name)
        if actual_key != expected_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "unauthorized",
                        "message": "missing or invalid API key",
                    }
                },
            )
        return await call_next(request)


def security_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("security", {})
    if not isinstance(raw, dict):
        return {}
    return raw
