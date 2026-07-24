import contextvars
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

CABECERA_REQUEST_ID = "X-Request-ID"
SIN_REQUEST_ID = "-"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=SIN_REQUEST_ID
)


def request_id_actual() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class RequestIdMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        entrante = request.headers.get(CABECERA_REQUEST_ID)
        request_id = entrante or uuid.uuid4().hex
        token = _request_id.set(request_id)
        try:
            respuesta = await call_next(request)
        finally:
            _request_id.reset(token)
        respuesta.headers[CABECERA_REQUEST_ID] = request_id
        return respuesta
