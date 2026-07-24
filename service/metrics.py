import threading
from collections import deque
from typing import Any

MUESTRAS_LATENCIA = 1000


class Metrics:
    def __init__(self, ventana: int = MUESTRAS_LATENCIA):
        self._lock = threading.Lock()
        self._total = 0
        self._por_resultado: dict[str, int] = {}
        self._latencias: deque[int] = deque(maxlen=ventana)

    def registrar_reconocimiento(self, resultado: str, latencia_ms: int) -> None:
        with self._lock:
            self._total += 1
            self._por_resultado[resultado] = self._por_resultado.get(resultado, 0) + 1
            self._latencias.append(latencia_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencias = sorted(self._latencias)
            por_resultado = dict(self._por_resultado)
            total = self._total

        return {
            "total_reconocimientos": total,
            "por_resultado": por_resultado,
            "latencia_ms": self._resumen_latencia(latencias),
        }

    @staticmethod
    def _percentil(valores_ordenados: list[int], fraccion: float) -> int:
        indice = max(0, round(fraccion * len(valores_ordenados)) - 1)
        return valores_ordenados[indice]

    def _resumen_latencia(self, latencias_ordenadas: list[int]) -> dict[str, int]:
        if not latencias_ordenadas:
            return {"muestras": 0}
        return {
            "muestras": len(latencias_ordenadas),
            "p50": self._percentil(latencias_ordenadas, 0.50),
            "p95": self._percentil(latencias_ordenadas, 0.95),
            "p99": self._percentil(latencias_ordenadas, 0.99),
            "max": latencias_ordenadas[-1],
        }


metrics = Metrics()
