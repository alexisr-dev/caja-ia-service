"""
Entrena YOLO con el dataset en dataset_yolo/ y exporta best.onnx a models_ia/.
Evalúa el modelo actual antes de entrenar y compara al final.
Correr desde cualquier directorio: python scripts/entrenar_yolo.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "dataset_yolo" / "data.yaml"
DESTINO_ONNX = PROJECT_ROOT / "models_ia" / "best.onnx"
RUNS_DIR     = PROJECT_ROOT / "runs" / "detect" / "detector_caja" / "weights"


METRICAS = ["mAP50", "mAP50-95", "Precision", "Recall"]


def detectar_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def extraer_metricas(metrics) -> dict:
    """Normaliza el objeto de métricas de ultralytics a un dict plano."""
    return {
        "mAP50":     metrics.box.map50,
        "mAP50-95":  metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall":    metrics.box.mr,
    }


def evaluar_modelo(model_path: Path, device: str, label: str) -> dict | None:
    if not model_path.exists():
        return None
    print(f"\nEvaluando modelo {label} ({model_path.name})...")
    try:
        m = YOLO(str(model_path))
        return extraer_metricas(m.val(data=str(DATA_YAML), device=device, verbose=False))
    except Exception as e:
        print(f"  ⚠  No se pudo evaluar {label}: {e}")
        return None


def imprimir_tabla(m_actual: dict | None, m_nuevo: dict) -> bool:
    """Imprime comparación y retorna True si el nuevo es mejor o igual."""
    claves = METRICAS

    print("\n" + "=" * 60)
    print(f"{'COMPARACIÓN DE MODELOS':^60}")
    print("=" * 60)

    if m_actual is None:
        print("  (No había modelo previo — se usará el nuevo directamente)")
        print("=" * 60)
        for k in claves:
            print(f"  {k:<12} nuevo={m_nuevo[k]:.4f}")
        print("=" * 60)
        return True

    print(f"  {'Métrica':<12} {'ACTUAL':>10} {'NUEVO':>10}  {'':>8}")
    print("-" * 60)
    gana_nuevo = 0
    gana_actual = 0
    for k in claves:
        va = m_actual[k]
        vn = m_nuevo[k]
        if vn > va:
            tag = "↑ NUEVO"
            gana_nuevo += 1
        elif va > vn:
            tag = "↓ ACTUAL"
            gana_actual += 1
        else:
            tag = "IGUAL"
        print(f"  {k:<12} {va:>10.4f} {vn:>10.4f}  {tag}")
    print("=" * 60)

    nuevo_es_mejor = m_nuevo["mAP50"] >= m_actual["mAP50"]
    delta = m_nuevo["mAP50"] - m_actual["mAP50"]
    if nuevo_es_mejor:
        print(f"\n  ✅ NUEVO modelo gana en mAP50 ({delta:+.4f}) → se reemplaza best.onnx")
    else:
        print(f"\n  ⚠  ACTUAL modelo gana en mAP50 ({delta:+.4f}) → se conserva best.onnx")
    print()
    return nuevo_es_mejor


def entrenar():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_YAML}\n"
            "Asegúrate de tener el dataset en dataset_yolo/ con data.yaml, train/ y valid/"
        )

    device = detectar_device()
    print(f"Dispositivo de entrenamiento: {device}")

    # ── 1. Evaluar modelo actual ─────────────────────────────────────────
    metricas_actual = evaluar_modelo(DESTINO_ONNX, device, "ACTUAL")

    # ── 2. Entrenar nuevo modelo ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  INICIANDO ENTRENAMIENTO")
    print("─" * 60)
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=15,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="detector_caja",
        exist_ok=True,
    )

    # ── 3. Métricas del nuevo ────────────────────────────────────────────
    metricas_nuevo = extraer_metricas(model.val())
    if metricas_nuevo["mAP50"] < 0.85:
        print("⚠  mAP50 < 0.85 — considera más fotos o revisar etiquetado en Roboflow")

    # ── 4. Exportar a ONNX ───────────────────────────────────────────────
    model.export(format="onnx", imgsz=640, simplify=True)
    origen_onnx = RUNS_DIR / "best.onnx"

    # ── 5. Comparar y decidir ─────────────────────────────────────────────
    nuevo_es_mejor = imprimir_tabla(metricas_actual, metricas_nuevo)

    if nuevo_es_mejor:
        if origen_onnx.exists():
            DESTINO_ONNX.parent.mkdir(exist_ok=True)
            shutil.copy2(origen_onnx, DESTINO_ONNX)
            print(f"  ✅ Modelo actualizado → {DESTINO_ONNX}")
        else:
            print(f"  ⚠  No se encontró {origen_onnx}. Cópialo manualmente a models_ia/best.onnx")
    else:
        print(f"  ℹ  Nuevo modelo disponible en: {origen_onnx}")
        print("     Para forzar el reemplazo cópialo manualmente a models_ia/best.onnx")


if __name__ == "__main__":
    entrenar()
