import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from auth import verificar_api_key
from config import settings
from service.contexto import RequestIdFilter, RequestIdMiddleware
from service.metrics import metrics
from service.ratelimit import RateLimiter
from service.reconocedor import ReconocedorProductos

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s",
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(RequestIdFilter())

logger = logging.getLogger(__name__)

_semaforo_inferencia = asyncio.Semaphore(settings.max_peticiones_concurrentes)

_rate_limiter = RateLimiter(settings.rate_limit_por_minuto)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.reconocedor = None
    app.state.error_arranque = None

    logger.info("Iniciando carga de modelos IA...")
    try:
        app.state.reconocedor = ReconocedorProductos()
        logger.info(
            "Modelos listos. Catálogo: %d productos",
            app.state.reconocedor.total_productos(),
        )
    except Exception as exc:
        app.state.error_arranque = str(exc)
        logger.critical("Fallo al cargar modelos: %s", exc)

    yield
    logger.info("Apagando servicio IA...")


app = FastAPI(
    title="Caja Registradora IA Service",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)


def _get_reconocedor(request: Request) -> ReconocedorProductos:
    reconocedor = request.app.state.reconocedor
    if reconocedor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Servicio no disponible: "
                   f"{request.app.state.error_arranque or 'modelos aún cargando'}",
        )
    return reconocedor


def _aplicar_rate_limit(request: Request) -> None:
    cliente = request.client.host if request.client else "desconocido"
    if not _rate_limiter.permitido(cliente):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiadas peticiones, inténtalo de nuevo en unos segundos",
        )


async def _leer_imagen_validando_tamano(imagen: UploadFile) -> bytes:
    limite_bytes = settings.max_image_size_bytes
    if imagen.size is not None and imagen.size > limite_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Imagen mayor a {settings.max_image_size_mb} MB",
        )

    contenido = await imagen.read()
    if len(contenido) > limite_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Imagen mayor a {settings.max_image_size_mb} MB",
        )
    return contenido


async def _ejecutar_acotado(func, *args):
    async with _semaforo_inferencia:
        return await asyncio.to_thread(func, *args)


@app.get("/health")
async def health(request: Request):
    reconocedor = request.app.state.reconocedor
    listo = reconocedor is not None
    total = await asyncio.to_thread(reconocedor.total_productos) if listo else 0
    error_arranque = request.app.state.error_arranque

    return JSONResponse(
        status_code=200 if listo else 503,
        content={
            "status": "ok" if listo else "error",
            "modelo_cargado": listo,
            "total_catalogo": total,
            **({"detalle": error_arranque} if error_arranque else {}),
        },
    )


@app.post(
    "/reconocer",
    dependencies=[Depends(verificar_api_key), Depends(_aplicar_rate_limit)],
)
async def reconocer(
    imagen: UploadFile = File(...),
    rec: ReconocedorProductos = Depends(_get_reconocedor),
):
    contenido = await _leer_imagen_validando_tamano(imagen)

    inicio = time.perf_counter()
    try:
        resultado = await _ejecutar_acotado(rec.reconocer, contenido)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Error interno al reconocer")
        return JSONResponse(
            status_code=500,
            content={"resultado": "error", "mensaje": "Error interno del servicio"},
        )

    latencia_ms = int((time.perf_counter() - inicio) * 1000)
    resultado["latencia_ms"] = latencia_ms
    metrics.registrar_reconocimiento(resultado.get("resultado", "desconocido"), latencia_ms)
    return resultado


@app.post("/registrar", dependencies=[Depends(verificar_api_key)])
async def registrar_producto(
    imagen: UploadFile = File(...),
    id_chroma: str = Form(...),
    rec: ReconocedorProductos = Depends(_get_reconocedor),
):
    contenido = await _leer_imagen_validando_tamano(imagen)
    try:
        return await _ejecutar_acotado(rec.registrar_producto, contenido, id_chroma)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Error al registrar producto %s", id_chroma)
        raise HTTPException(status_code=500, detail="Error interno del servicio")


@app.delete("/producto/{id_chroma}", dependencies=[Depends(verificar_api_key)])
async def eliminar_producto(
    id_chroma: str,
    rec: ReconocedorProductos = Depends(_get_reconocedor),
):
    try:
        return await asyncio.to_thread(rec.eliminar_producto, id_chroma)
    except Exception:
        logger.exception("Error al eliminar %s", id_chroma)
        raise HTTPException(status_code=500, detail="Error interno del servicio")


@app.get("/catalogo/stats", dependencies=[Depends(verificar_api_key)])
async def stats_catalogo(rec: ReconocedorProductos = Depends(_get_reconocedor)):
    return {"total": await asyncio.to_thread(rec.total_productos)}


@app.get("/metrics", dependencies=[Depends(verificar_api_key)])
async def obtener_metricas():
    return metrics.snapshot()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.ia_service_host,
        port=settings.ia_service_port,
        reload=False,
    )
