"""
Recorre fotos_catalogo/ y registra cada foto en ChromaDB.
Correr una vez después de entrenar YOLO y antes de levantar el servicio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASE_DIR
from service.reconocedor import ReconocedorProductos

FOTOS_DIR = BASE_DIR / 'fotos_catalogo'
EXTENSIONES_FOTO = ('*.jpg', '*.png')


def listar_fotos() -> list[Path]:
    """Fotos del catálogo en orden estable, para que la salida sea reproducible."""
    fotos: list[Path] = []
    for patron in EXTENSIONES_FOTO:
        fotos.extend(FOTOS_DIR.glob(patron))
    return sorted(fotos)


def main():
    reconocedor = ReconocedorProductos()
    fotos = listar_fotos()
    print(f"Encontradas {len(fotos)} fotos en {FOTOS_DIR}")

    for foto_path in fotos:
        id_chroma = foto_path.stem
        resultado = reconocedor.registrar_producto(foto_path.read_bytes(), id_chroma)
        print(f"  ✓ {id_chroma} → total: {resultado['total_catalogo']}")

    print("\n✅ Inventario registrado.")


if __name__ == '__main__':
    main()
