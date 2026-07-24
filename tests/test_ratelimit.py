"""Tests del limitador de tasa (lógica pura, sin servidor)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.ratelimit import RateLimiter


class TestRateLimiter:
    def test_cero_desactiva_el_limite(self):
        rl = RateLimiter(por_minuto=0)
        assert rl.activo is False
        for _ in range(1000):
            assert rl.permitido("cliente") is True

    def test_permite_hasta_el_limite_y_luego_bloquea(self):
        rl = RateLimiter(por_minuto=3)
        assert rl.permitido("a") is True
        assert rl.permitido("a") is True
        assert rl.permitido("a") is True
        assert rl.permitido("a") is False

    def test_clientes_distintos_tienen_cubos_independientes(self):
        rl = RateLimiter(por_minuto=1)
        assert rl.permitido("a") is True
        assert rl.permitido("b") is True
        assert rl.permitido("a") is False

    def test_la_ventana_se_desliza_y_libera(self):
        rl = RateLimiter(por_minuto=1, ventana_seg=0.2)
        assert rl.permitido("a") is True
        assert rl.permitido("a") is False
        time.sleep(0.25)
        assert rl.permitido("a") is True


class TestMetrics:
    def test_percentiles_y_conteos(self):
        from service.metrics import Metrics

        m = Metrics()
        for i in range(1, 101):  # latencias 1..100
            m.registrar_reconocimiento("ok" if i % 2 else "no_detectado", i)

        snap = m.snapshot()
        assert snap["total_reconocimientos"] == 100
        assert snap["por_resultado"]["ok"] == 50
        assert snap["por_resultado"]["no_detectado"] == 50
        assert snap["latencia_ms"]["muestras"] == 100
        assert snap["latencia_ms"]["max"] == 100
        assert snap["latencia_ms"]["p50"] == 50
        assert snap["latencia_ms"]["p95"] == 95

    def test_snapshot_vacio_no_falla(self):
        from service.metrics import Metrics

        snap = Metrics().snapshot()
        assert snap["total_reconocimientos"] == 0
        assert snap["latencia_ms"] == {"muestras": 0}
