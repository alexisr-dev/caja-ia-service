"""
Prueba de deteccion YOLO + extraccion de vector CLIP, sin ChromaDB.

Usa el MISMO extractor que el servicio (CLIP ViT-B/32) para que el diagnostico
refleje lo que ocurre en produccion.

Uso:
    python scripts/test_deteccion.py imagen.jpg
    python scripts/test_deteccion.py --carpeta mis_fotos/
    python scripts/test_deteccion.py imagen.jpg --conf 0.25 --guardar

Salida:
    - Tabla de todas las detecciones con confianza y bbox
    - Imagen anotada guardada en tests_output/ (si --guardar o carpeta)
    - Stats del vector CLIP del mejor recorte
"""

import sys
import argparse
import time
from pathlib import Path

# Para importar config desde la raiz del proyecto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import clip
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from config import settings

OUTPUT_DIR = ROOT / "tests_output"
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_modelos(conf_umbral: float):
    model_path = settings.ruta_modelo_yolo
    if not model_path.exists():
        print(f"[ERROR] Modelo YOLO no encontrado en: {model_path}")
        sys.exit(1)

    print(f"Cargando YOLO desde: {model_path}")
    yolo = YOLO(str(model_path), task="detect")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando CLIP ViT-B/32 en: {device}")
    modelo_clip, preprocess = clip.load("ViT-B/32", device=device)
    modelo_clip.eval()

    print(f"Umbral de confianza YOLO: {conf_umbral}")
    print("-" * 60)
    return yolo, modelo_clip, preprocess, device


def extraer_vector(pil_image: Image.Image, modelo_clip, preprocess, device) -> np.ndarray:
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = modelo_clip.encode_image(tensor).squeeze().cpu().numpy()
    norma = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma
    return vector


def analizar_imagen(
    ruta: Path,
    yolo,
    modelo_clip,
    preprocess,
    device,
    conf_umbral: float,
    guardar: bool,
) -> dict:
    img = Image.open(ruta).convert("RGB")
    print(f"\nImagen: {ruta.name}  ({img.width}x{img.height}px)")

    t0 = time.perf_counter()
    resultados = yolo(img, conf=conf_umbral, verbose=False)[0]
    ms_yolo = (time.perf_counter() - t0) * 1000

    boxes = resultados.boxes
    nombres_clases = resultados.names  # dict id->nombre si el modelo los tiene

    if len(boxes) == 0:
        print(f"  [YOLO] Ninguna deteccion (conf >= {conf_umbral}) en {ms_yolo:.0f} ms")
        return {"imagen": ruta.name, "detecciones": 0}

    print(f"  [YOLO] {len(boxes)} deteccion(es) en {ms_yolo:.0f} ms")
    print(f"  {'#':<4} {'Clase':<20} {'Conf':>6}  {'BBox (x1,y1,x2,y2)'}")
    print(f"  {'-'*4} {'-'*20} {'-'*6}  {'-'*30}")

    detecciones = []
    for i, (xyxy, conf, cls_id) in enumerate(
        zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
    ):
        x1, y1, x2, y2 = (int(c) for c in xyxy)
        nombre = nombres_clases.get(int(cls_id), f"clase_{int(cls_id)}")
        detecciones.append({"idx": i, "clase": nombre, "conf": conf, "bbox": [x1, y1, x2, y2]})
        marcador = " <-- mejor" if i == int(boxes.conf.argmax().item()) else ""
        print(f"  {i:<4} {nombre:<20} {conf:>6.3f}  ({x1},{y1},{x2},{y2}){marcador}")

    # Vector del mejor recorte
    mejor_idx = int(boxes.conf.argmax().item())
    x1, y1, x2, y2 = detecciones[mejor_idx]["bbox"]
    recorte = img.crop((x1, y1, x2, y2))

    t1 = time.perf_counter()
    vector = extraer_vector(recorte, modelo_clip, preprocess, device)
    ms_clip = (time.perf_counter() - t1) * 1000

    print(f"\n  [CLIP] Vector del mejor recorte ({ms_clip:.0f} ms)")
    print(f"    Dimension : {vector.shape[0]}")
    print(f"    Norma L2  : {np.linalg.norm(vector):.6f}  (deberia ser ~1.0 tras normalizacion)")
    print(f"    Min/Max   : {vector.min():.4f} / {vector.max():.4f}")
    print(f"    Media/Std : {vector.mean():.4f} / {vector.std():.4f}")
    print(f"    Primeros 8 valores: {vector[:8].round(4).tolist()}")

    if guardar:
        _guardar_anotada(img, detecciones, mejor_idx, ruta)

    return {
        "imagen": ruta.name,
        "detecciones": len(detecciones),
        "mejor_conf": round(detecciones[mejor_idx]["conf"], 3),
        "mejor_clase": detecciones[mejor_idx]["clase"],
    }


def _guardar_anotada(img: Image.Image, detecciones: list, mejor_idx: int, ruta: Path):
    OUTPUT_DIR.mkdir(exist_ok=True)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    colores = ["#FF4444", "#44AAFF", "#44FF88", "#FFAA00", "#AA44FF"]

    for det in detecciones:
        x1, y1, x2, y2 = det["bbox"]
        color = "#FF4444" if det["idx"] == mejor_idx else colores[det["idx"] % len(colores)]
        grosor = 3 if det["idx"] == mejor_idx else 1
        for t in range(grosor):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        etiqueta = f"{det['clase']} {det['conf']:.2f}"
        bbox_texto = draw.textbbox((x1, y1 - 18), etiqueta, font=font_small)
        draw.rectangle(bbox_texto, fill=color)
        draw.text((x1, y1 - 18), etiqueta, fill="white", font=font_small)

    destino = OUTPUT_DIR / f"det_{ruta.stem}.jpg"
    img.save(destino, quality=92)
    print(f"\n  Imagen anotada guardada en: {destino}")


def main():
    parser = argparse.ArgumentParser(description="Prueba deteccion YOLO sin ChromaDB")
    parser.add_argument("imagen", nargs="?", help="Ruta a imagen individual")
    parser.add_argument("--carpeta", help="Procesa todas las imagenes de una carpeta")
    parser.add_argument(
        "--conf",
        type=float,
        default=settings.umbral_yolo_conf,
        help=f"Umbral de confianza YOLO (default: {settings.umbral_yolo_conf})",
    )
    parser.add_argument(
        "--guardar",
        action="store_true",
        help="Guarda imagen anotada con los recuadros en tests_output/",
    )
    args = parser.parse_args()

    if not args.imagen and not args.carpeta:
        parser.print_help()
        sys.exit(0)

    yolo, modelo_clip, preprocess, device = cargar_modelos(args.conf)

    # Cuando se procesa carpeta, siempre guardar
    guardar = args.guardar or bool(args.carpeta)

    if args.carpeta:
        carpeta = Path(args.carpeta)
        if not carpeta.is_dir():
            print(f"[ERROR] Carpeta no encontrada: {carpeta}")
            sys.exit(1)
        imagenes = [p for p in sorted(carpeta.iterdir()) if p.suffix.lower() in EXTENSIONES_IMAGEN]
        if not imagenes:
            print(f"[ERROR] No hay imagenes en {carpeta}")
            sys.exit(1)
        print(f"Procesando {len(imagenes)} imagen(es) de '{carpeta}'...\n")
        resumen = []
        for img_path in imagenes:
            try:
                r = analizar_imagen(img_path, yolo, modelo_clip, preprocess, device, args.conf, guardar)
                resumen.append(r)
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

        print("\n" + "=" * 60)
        print("RESUMEN")
        print(f"  {'Imagen':<30} {'Dets':>5} {'MejorConf':>10} {'Clase'}")
        print(f"  {'-'*30} {'-'*5} {'-'*10} {'-'*20}")
        for r in resumen:
            dets = r.get("detecciones", 0)
            conf = r.get("mejor_conf", "-")
            clase = r.get("mejor_clase", "-")
            print(f"  {r['imagen']:<30} {dets:>5} {str(conf):>10} {clase}")

    else:
        ruta = Path(args.imagen)
        if not ruta.exists():
            print(f"[ERROR] Imagen no encontrada: {ruta}")
            sys.exit(1)
        analizar_imagen(ruta, yolo, modelo_clip, preprocess, device, args.conf, guardar)

    print("\nListo.")


if __name__ == "__main__":
    main()
