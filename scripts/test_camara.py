"""
Prueba YOLO en camara en tiempo real. Solo deteccion, sin ResNet ni ChromaDB.

Uso:
    python scripts/test_camara.py
    python scripts/test_camara.py --conf 0.25
    python scripts/test_camara.py --camara 1   (si tienes varias camaras)

Teclas:
    Q  ->  salir
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO

from config import settings


def main():
    parser = argparse.ArgumentParser(description="YOLO en camara tiempo real")
    parser.add_argument("--conf", type=float, default=settings.umbral_yolo_conf)
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()

    model_path = settings.ruta_modelo_yolo
    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        sys.exit(1)

    print(f"Cargando YOLO desde: {model_path}")
    yolo = YOLO(str(model_path), task="detect")

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {args.camara}")
        sys.exit(1)

    print(f"Camara {args.camara} abierta. Umbral conf={args.conf}. Presiona Q para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] No se pudo leer frame")
            break

        resultados = yolo(frame, conf=args.conf, verbose=False)[0]
        frame_anotado = resultados.plot()

        cv2.imshow("YOLO - prueba deteccion (Q para salir)", frame_anotado)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
