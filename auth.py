import hmac
import logging

from fastapi import Header, HTTPException, status

from config import settings

logger = logging.getLogger(__name__)


async def verificar_api_key(x_api_key: str | None = Header(default=None)) -> bool:
    if x_api_key is None:
        logger.warning("Rechazada petición sin cabecera X-API-Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta la cabecera X-API-Key",
        )

    es_valida = hmac.compare_digest(
        x_api_key.encode("utf-8"),
        settings.ia_service_api_key.encode("utf-8"),
    )
    if not es_valida:
        logger.warning("Rechazada petición con API key inválida")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
        )
    return True
