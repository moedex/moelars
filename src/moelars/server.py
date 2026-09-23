"""HTTP server, wire-compatible with the System One API.

- POST /v1/systemone
- GET  /v1/models
- GET  /healthz

Errors use the `{message, error_type}` shape. When MOELARS_API_KEY is set, requests
must carry `Authorization: Bearer <key>`. The response carries both
`x-moelars-request-id` and `x-typesafe-request-id`, since the client SDKs read the
latter for their `request_id` property.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from moelars.engine import Engine
from moelars.schema import ErrorBody, ListModelsResponse, SystemOneRequest


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorBody(message=message, error_type=error_type).model_dump())


def _format_validation(error: RequestValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(x) for x in item.get("loc", ()) if x != "body")
        parts.append(f"{location}: {item.get('msg')}" if location else str(item.get("msg")))
    return "; ".join(parts) or "invalid request"


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="moe-LARS", version=engine.version, docs_url="/docs")
    app.state.engine = engine

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        request_id = str(uuid.uuid4())
        expected = os.environ.get("MOELARS_API_KEY")
        if expected and request.url.path.startswith("/v1/"):
            header = request.headers.get("authorization", "")
            if header != f"Bearer {expected}":
                response = _error(401, "Missing or invalid API key", "authentication_error")
                response.headers["x-moelars-request-id"] = request_id
                return response
        response = await call_next(request)
        response.headers["x-moelars-request-id"] = request_id
        response.headers["x-typesafe-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, error: RequestValidationError):
        return _error(422, _format_validation(error), "invalid_request")

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, error: ValueError):
        return _error(422, str(error), "invalid_request")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "model": engine.model_id}

    @app.get("/v1/models", response_model=ListModelsResponse)
    async def list_models():
        return ListModelsResponse(models=engine.models())

    @app.post("/v1/systemone")
    async def system_one(request: SystemOneRequest):
        response = engine.evaluate(request)
        return JSONResponse(content=response.model_dump(exclude_none=True))

    return app
