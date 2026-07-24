import logging
import os
from io import BytesIO
from typing import NamedTuple

os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

import chromadb
import clip
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from config import settings
from service import matching

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

MENSAJE_MODELO_AUSENTE = (
    "Modelo YOLO no encontrado en '{ruta}'. "
    "Corre scripts/entrenar_yolo.py y copia best.onnx a models_ia/"
)

RESULTADO_NO_DETECTADO = "no_detectado"
RESULTADO_CATALOGO_VACIO = "catalogo_vacio"
RESULTADO_DESCONOCIDO = "producto_desconocido"
RESULTADO_OK = "ok"


class Deteccion(NamedTuple):
    bbox: tuple[int, int, int, int]
    confianza: float


class ReconocedorProductos:

    def __init__(self):
        self._cargar_detector()
        self._cargar_extractor()
        self._conectar_catalogo()

    def _cargar_detector(self) -> None:
        ruta_modelo = settings.ruta_modelo_yolo
        if not ruta_modelo.exists():
            raise FileNotFoundError(MENSAJE_MODELO_AUSENTE.format(ruta=ruta_modelo))

        logger.info("Cargando YOLO desde %s", ruta_modelo)
        self._yolo = YOLO(str(ruta_modelo), task="detect")

    def _cargar_extractor(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Cargando CLIP ViT-B/32 en device=%s", self._device)
        self._clip, self._preprocess = clip.load("ViT-B/32", device=self._device)
        self._clip.eval()

    def _conectar_catalogo(self) -> None:
        ruta_chroma = settings.ruta_chroma
        logger.info("Conectando ChromaDB en %s", ruta_chroma)
        ruta_chroma.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(ruta_chroma))
        self._coleccion = self._chroma.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB listo. Productos en catálogo: %d", self._coleccion.count())

    def reconocer(self, imagen_bytes: bytes) -> dict:
        imagen = self._decodificar_imagen(imagen_bytes)

        deteccion = self._detectar_objeto_principal(imagen)
        if deteccion is None:
            logger.debug("YOLO no detectó ningún objeto")
            return {"resultado": RESULTADO_NO_DETECTADO}

        recorte = imagen.crop(deteccion.bbox)
        vector = self._extraer_vector(recorte)

        if self._coleccion.count() == 0:
            logger.warning("ChromaDB vacío: no hay productos registrados")
            return {"resultado": RESULTADO_CATALOGO_VACIO}

        veredicto = self._buscar_en_catalogo(vector)
        if not veredicto.es_confiable:
            logger.debug(
                "Rechazado por %s: sku=%s dist=%.4f",
                veredicto.motivo_rechazo,
                veredicto.sku,
                veredicto.distancia,
            )
            return {
                "resultado": RESULTADO_DESCONOCIDO,
                "distancia": round(veredicto.distancia, 3),
                "confianza_yolo": deteccion.confianza,
            }

        logger.debug("Aceptado: sku=%s dist=%.4f", veredicto.sku, veredicto.distancia)
        return {
            "resultado": RESULTADO_OK,
            "sku": veredicto.sku,
            "confianza": round(1.0 - veredicto.distancia, 3),
            "bbox": list(deteccion.bbox),
            "confianza_yolo": deteccion.confianza,
        }

    def registrar_producto(self, imagen_bytes: bytes, id_chroma: str) -> dict:
        imagen = self._decodificar_imagen(imagen_bytes)
        vector = self._extraer_vector(imagen)
        self._coleccion.upsert(ids=[id_chroma], embeddings=[vector])
        total = self._coleccion.count()
        logger.info("Producto registrado: %s (total=%d)", id_chroma, total)
        return {"ok": True, "id_chroma": id_chroma, "total_catalogo": total}

    def eliminar_producto(self, id_chroma: str) -> dict:
        self._coleccion.delete(ids=[id_chroma])
        total = self._coleccion.count()
        logger.info("Producto eliminado: %s (total=%d)", id_chroma, total)
        return {"ok": True, "total_catalogo": total}

    def total_productos(self) -> int:
        return self._coleccion.count()

    def _detectar_objeto_principal(self, imagen: Image.Image) -> Deteccion | None:
        cajas = self._yolo(imagen, conf=settings.umbral_yolo_conf, verbose=False)[0].boxes
        if len(cajas) == 0:
            return None

        idx_mejor = int(cajas.conf.argmax().item())
        x1, y1, x2, y2 = (int(coord) for coord in cajas.xyxy[idx_mejor].tolist())
        confianza = round(float(cajas.conf[idx_mejor].item()), 3)
        logger.debug(
            "Detección YOLO: bbox=(%d,%d,%d,%d) conf=%.3f", x1, y1, x2, y2, confianza
        )
        return Deteccion(bbox=(x1, y1, x2, y2), confianza=confianza)

    def _buscar_en_catalogo(self, vector: list[float]) -> matching.Veredicto:
        respuesta = self._coleccion.query(
            query_embeddings=[vector],
            n_results=matching.N_RESULTADOS_BUSQUEDA,
        )
        return matching.evaluar(
            ids=respuesta["ids"][0],
            distancias=respuesta["distances"][0],
            umbral_distancia=settings.umbral_similitud,
        )

    def _decodificar_imagen(self, imagen_bytes: bytes) -> Image.Image:
        try:
            imagen = Image.open(BytesIO(imagen_bytes))
            imagen.load()
            return imagen.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise ValueError("El archivo recibido no es una imagen válida") from exc

    def _extraer_vector(self, pil_image: Image.Image) -> list[float]:
        tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            vector: np.ndarray = self._clip.encode_image(tensor).squeeze().cpu().numpy()
        norma = np.linalg.norm(vector)
        if norma > 0:
            vector = vector / norma
        return vector.tolist()
