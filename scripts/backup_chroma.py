"""
Copia de seguridad de la base vectorial ChromaDB (db_vectorial/).

Genera una carpeta con marca de tiempo bajo backups/. Correr antes de
reentrenar, migrar o cualquier operación masiva sobre el catálogo.

Uso:
    python scripts/backup_chroma.py
    python scripts/backup_chroma.py --destino D:/backups_caja
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

BACKUPS_DIR_DEFECTO = settings.ruta_chroma.parent / "backups"


def crear_backup(destino_base: Path) -> Path:
    origen = settings.ruta_chroma
    if not origen.exists():
        print(f"[ERROR] No existe la base vectorial en: {origen}")
        sys.exit(1)

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = destino_base / f"chroma_{marca}"
    destino_base.mkdir(parents=True, exist_ok=True)

    print(f"Copiando {origen}  ->  {destino}")
    shutil.copytree(origen, destino)

    n_ficheros = sum(1 for _ in destino.rglob("*") if _.is_file())
    print(f"✅ Backup completo: {n_ficheros} ficheros en {destino}")
    return destino


def main():
    parser = argparse.ArgumentParser(description="Backup de la base vectorial ChromaDB")
    parser.add_argument(
        "--destino",
        type=Path,
        default=BACKUPS_DIR_DEFECTO,
        help=f"Carpeta donde guardar el backup (default: {BACKUPS_DIR_DEFECTO})",
    )
    args = parser.parse_args()
    crear_backup(args.destino)


if __name__ == "__main__":
    main()
