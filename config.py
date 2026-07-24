from pathlib import Path

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

LONGITUD_MINIMA_API_KEY = 32
BYTES_POR_MB = 1024 * 1024


class Settings(BaseSettings):
    ia_service_host: str = "0.0.0.0"
    ia_service_port: int = 8001

    ia_service_api_key: str

    umbral_similitud: float = 0.23
    umbral_yolo_conf: float = 0.4

    yolo_model_path: str = "models_ia/best.onnx"
    chroma_db_path: str = "db_vectorial"
    chroma_collection: str = "productos_visuales"

    max_image_size_mb: int = 5
    max_image_pixels: int = 40_000_000

    max_peticiones_concurrentes: int = 2
    rate_limit_por_minuto: int = 120

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def ruta_modelo_yolo(self) -> Path:
        return BASE_DIR / self.yolo_model_path

    @property
    def ruta_chroma(self) -> Path:
        return BASE_DIR / self.chroma_db_path

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * BYTES_POR_MB

    @field_validator("umbral_similitud", "umbral_yolo_conf")
    @classmethod
    def debe_ser_fraccion(cls, valor: float, info: ValidationInfo) -> float:
        if not 0.0 < valor < 1.0:
            raise ValueError(f"{info.field_name} debe estar entre 0 y 1, recibido: {valor}")
        return valor

    @field_validator("ia_service_api_key")
    @classmethod
    def api_key_minima_longitud(cls, valor: str) -> str:
        if len(valor) < LONGITUD_MINIMA_API_KEY:
            raise ValueError(
                f"IA_SERVICE_API_KEY debe tener al menos {LONGITUD_MINIMA_API_KEY} caracteres"
            )
        return valor


settings = Settings()
