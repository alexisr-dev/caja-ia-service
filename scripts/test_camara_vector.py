"""
Flujo completo en cámara tiempo real: YOLO + CLIP + ChromaDB.
Con votación temporal para resultado estable (sin parpadeo).

Uso:
    python scripts/test_camara_vector.py
    python scripts/test_camara_vector.py --conf 0.25
    python scripts/test_camara_vector.py --camara 1

Teclas:
    Q  →  salir
"""

import sys
import argparse
from collections import deque, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import clip
import cv2
import numpy as np
import torch
import chromadb
from PIL import Image
from ultralytics import YOLO

from config import settings
from service import matching

# ── Colores ───────────────────────────────────────────────────────────────────
COLOR_CONOCIDO    = (0, 220, 80)
COLOR_DESCONOCIDO = (0, 120, 255)
COLOR_BUSCANDO    = (180, 180, 0)
COLOR_FONDO       = (20, 20, 20)

# ── Votación temporal ─────────────────────────────────────────────────────────
VENTANA_FRAMES  = 12   # cuántos frames acumular
VOTOS_MINIMOS   = 7    # cuántos deben coincidir para confirmar resultado


class EstabilizadorResultado:
    """Acumula resultados de los últimos N frames y emite el consenso."""

    def __init__(self, ventana: int = VENTANA_FRAMES, minimos: int = VOTOS_MINIMOS):
        self._historial: deque[str] = deque(maxlen=ventana)
        self._minimos = minimos

    def actualizar(self, etiqueta: str) -> tuple[str, str]:
        """
        Agrega etiqueta al historial.
        Retorna (etiqueta_estable, estado) donde estado es:
          'confirmado'  → mayoría alcanzada
          'buscando'    → aún acumulando frames
        """
        self._historial.append(etiqueta)
        if len(self._historial) < self._minimos:
            return "buscando...", "buscando"

        conteo = Counter(self._historial)
        top_etiqueta, top_votos = conteo.most_common(1)[0]
        if top_votos >= self._minimos:
            return top_etiqueta, "confirmado"
        return "buscando...", "buscando"

    def reset(self):
        self._historial.clear()


# ── Modelos ───────────────────────────────────────────────────────────────────

def cargar_clip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def extraer_vector(pil_crop: Image.Image, modelo, preprocess, device) -> list[float]:
    tensor = preprocess(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = modelo.encode_image(tensor).squeeze().cpu().numpy()
    norma = np.linalg.norm(vec)
    if norma > 0:
        vec = vec / norma
    return vec.tolist()


def consultar_chroma(coleccion, vector: list[float]) -> tuple[str, float, bool]:
    """
    Aplica los mismos filtros que el servicio (ver `service.matching`).
    Retorna (etiqueta_raw, similitud, es_conocido).
    """
    if coleccion.count() == 0:
        return "catalogo_vacio", 0.0, False

    res        = coleccion.query(
        query_embeddings=[vector],
        n_results=matching.N_RESULTADOS_BUSQUEDA,
    )
    ids        = res["ids"][0]
    distancias = res["distances"][0]

    veredicto = matching.evaluar(ids, distancias, settings.umbral_similitud)
    similitud = round(1.0 - veredicto.distancia, 3)

    if not veredicto.es_confiable:
        return "desconocido", similitud, False

    partes = ids[0].split("_")
    nombre = partes[1] if len(partes) > 1 else ""
    return f"{veredicto.sku}  {nombre}", similitud, True


# ── Dibujado ──────────────────────────────────────────────────────────────────

def dibujar_bbox(frame, x1, y1, x2, y2, etiqueta, conf_yolo, similitud, estado):
    if estado == "confirmado" and etiqueta != "desconocido":
        color = COLOR_CONOCIDO
    elif estado == "confirmado" and etiqueta == "desconocido":
        color = COLOR_DESCONOCIDO
    else:
        color = COLOR_BUSCANDO

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    fondo_y = max(y1 - 50, 0)
    cv2.rectangle(frame, (x1, fondo_y), (x1 + 260, y1), COLOR_FONDO, -1)

    label_display = etiqueta if estado == "confirmado" else "buscando..."
    cv2.putText(frame, label_display,
                (x1 + 5, fondo_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"YOLO {conf_yolo:.2f}   Sim {similitud:.2f}",
                (x1 + 5, fondo_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)


def dibujar_hud(frame, n_det, n_conocidos):
    cv2.rectangle(frame, (0, 0), (270, 36), COLOR_FONDO, -1)
    cv2.putText(frame, f"Detectados: {n_det}   Reconocidos: {n_conocidos}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ── Cámara ────────────────────────────────────────────────────────────────────

def detectar_camaras(max_idx: int = 6) -> list[int]:
    disponibles = []
    print("Buscando camaras disponibles", end="", flush=True)
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                disponibles.append(i)
        cap.release()
        print(".", end="", flush=True)
    print()
    return disponibles


def seleccionar_camara() -> int:
    camaras = detectar_camaras()
    if not camaras:
        print("[ERROR] No se encontro ninguna camara disponible.")
        sys.exit(1)

    print("\n-- Camaras disponibles --")
    for i, idx in enumerate(camaras):
        print(f"  [{i}] Camara {idx}")
    print("-------------------------")

    if len(camaras) == 1:
        print(f"  -> Usando camara {camaras[0]} (unica disponible)")
        return camaras[0]

    while True:
        try:
            eleccion = int(input(f"Selecciona camara [0-{len(camaras)-1}]: "))
            if 0 <= eleccion < len(camaras):
                return camaras[eleccion]
        except (ValueError, KeyboardInterrupt):
            pass
        print(f"  Opcion invalida.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",   type=float, default=settings.umbral_yolo_conf)
    parser.add_argument("--camara", type=int,   default=None)
    args = parser.parse_args()

    model_path = settings.ruta_modelo_yolo
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        sys.exit(1)

    print(f"Cargando YOLO desde: {model_path}")
    yolo = YOLO(str(model_path), task="detect")

    print("Cargando CLIP ViT-B/32...")
    clip_model, preprocess, device = cargar_clip()
    print(f"CLIP listo en: {device}")

    chroma    = chromadb.PersistentClient(path=str(settings.ruta_chroma))
    coleccion = chroma.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Catalogo: {coleccion.count()} vectores")

    idx_camara = args.camara if args.camara is not None else seleccionar_camara()
    cap = cv2.VideoCapture(idx_camara, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir camara {idx_camara}")
        sys.exit(1)
    print(f"Camara {idx_camara} abierta. conf={args.conf}  |  Q para salir")

    # Un estabilizador por cada detección simultánea (usamos índice 0..N)
    estabilizadores: dict[int, EstabilizadorResultado] = {}
    ultimo_n_det = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] No se pudo leer frame")
            break

        resultados  = yolo(frame, conf=args.conf, verbose=False)[0]
        frame_out   = frame.copy()
        boxes       = resultados.boxes
        n_det       = len(boxes)
        n_conocidos = 0

        # Si cambia el número de detecciones, reiniciar estabilizadores
        if n_det != ultimo_n_det:
            estabilizadores.clear()
        ultimo_n_det = n_det

        if n_det > 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for i in range(n_det):
                if i not in estabilizadores:
                    estabilizadores[i] = EstabilizadorResultado()

                x1, y1, x2, y2 = (int(c) for c in boxes.xyxy[i].tolist())
                conf_yolo = float(boxes.conf[i].item())

                recorte = Image.fromarray(frame_rgb[y1:y2, x1:x2])
                if recorte.width < 4 or recorte.height < 4:
                    continue

                vector = extraer_vector(recorte, clip_model, preprocess, device)
                etiqueta_raw, similitud, conocido = consultar_chroma(coleccion, vector)

                etiqueta_estable, estado = estabilizadores[i].actualizar(etiqueta_raw)

                if estado == "confirmado" and etiqueta_estable != "desconocido" and etiqueta_estable != "buscando...":
                    n_conocidos += 1

                dibujar_bbox(frame_out, x1, y1, x2, y2,
                             etiqueta_estable, conf_yolo, similitud, estado)

        dibujar_hud(frame_out, n_det, n_conocidos)
        cv2.imshow("YOLO + CLIP + ChromaDB   (Q salir)", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
