from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class RequestBodyLoggingMiddleware:
    """Log selected API request and response bodies without changing ASGI flow."""

    def __init__(
        self,
        app,
        *,
        max_body_chars: int = 20_000,
        logger_name: str = "uvicorn.error",
        log_responses: bool = True,
        paths: tuple[str, ...] = ("/v1/predict/turn", "/v1/predict/response"),
    ) -> None:
        self.app = app
        self.max_body_chars = max(0, int(max_body_chars))
        self.log_responses = bool(log_responses)
        self.paths = tuple(paths)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if not self._should_log(scope):
            await self.app(scope, receive, send)
            return

        messages: list[dict[str, Any]] = []
        body_parts: list[bytes] = []

        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        body = b"".join(body_parts)
        self._log_body(scope, body)

        async def replay_receive() -> dict[str, Any]:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        response_status: int | None = None
        response_parts: list[bytes] = []

        async def logging_send(message: dict[str, Any]) -> None:
            nonlocal response_status
            if self.log_responses and message.get("type") == "http.response.start":
                response_status = int(message.get("status", 0))
            if self.log_responses and message.get("type") == "http.response.body":
                response_parts.append(message.get("body", b""))

            await send(message)

            if (
                self.log_responses
                and message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                self._log_response(scope, response_status, b"".join(response_parts))

        await self.app(scope, replay_receive, logging_send)

    def _should_log(self, scope: dict[str, Any]) -> bool:
        if scope.get("type") != "http":
            return False
        if str(scope.get("method", "")).upper() != "POST":
            return False
        return str(scope.get("path", "")) in self.paths

    def _log_body(self, scope: dict[str, Any], body: bytes) -> None:
        text, truncated = self._format_body(body)
        path = self._path(scope)

        self.logger.info(
            "API request body method=%s path=%s bytes=%d truncated=%s body=%s",
            str(scope.get("method", "")).upper(),
            path,
            len(body),
            truncated,
            text,
        )

    def _log_response(self, scope: dict[str, Any], status_code: int | None, body: bytes) -> None:
        text, truncated = self._format_body(body)
        self.logger.info(
            "API response body method=%s path=%s status=%s bytes=%d truncated=%s body=%s",
            str(scope.get("method", "")).upper(),
            self._path(scope),
            status_code,
            len(body),
            truncated,
            text,
        )

    def _format_body(self, body: bytes) -> tuple[str, bool]:
        text = body.decode("utf-8", errors="replace")
        truncated = len(text) > self.max_body_chars
        if truncated:
            text = text[: self.max_body_chars] + "...<truncated>"
        return text, truncated

    @staticmethod
    def _path(scope: dict[str, Any]) -> str:
        path = str(scope.get("path", ""))
        query_string = scope.get("query_string", b"")
        if query_string:
            path = f"{path}?{query_string.decode('utf-8', errors='replace')}"
        return path


def debug_logging_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("api", {})
    if not isinstance(raw, dict):
        return {
            "enabled": False,
            "max_body_chars": 20_000,
            "logger_name": "uvicorn.error",
            "log_responses": True,
        }
    return {
        "enabled": bool(raw.get("debug_log_requests", False)),
        "max_body_chars": int(raw.get("debug_log_max_body_chars", 20_000)),
        "logger_name": str(raw.get("debug_log_logger", "uvicorn.error")),
        "log_responses": bool(raw.get("debug_log_responses", True)),
    }
