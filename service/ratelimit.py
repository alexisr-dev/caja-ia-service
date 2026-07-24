import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, por_minuto: int, ventana_seg: float = 60.0):
        self._por_minuto = por_minuto
        self._ventana = ventana_seg
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @property
    def activo(self) -> bool:
        return self._por_minuto > 0

    def permitido(self, clave: str) -> bool:
        if not self.activo:
            return True

        ahora = time.monotonic()
        with self._lock:
            marcas = self._hits[clave]
            while marcas and ahora - marcas[0] > self._ventana:
                marcas.popleft()
            if len(marcas) >= self._por_minuto:
                return False
            marcas.append(ahora)
            return True
