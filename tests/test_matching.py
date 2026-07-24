"""Tests de las reglas de decisión del catálogo.

Son lógica pura: no cargan YOLO, CLIP ni ChromaDB, así que corren en milisegundos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import matching

UMBRAL = 0.23


class TestExtraerSku:
    def test_toma_el_primer_segmento(self):
        assert matching.extraer_sku("BEB-001_INKA KOLA_03") == "BEB-001"

    def test_id_sin_separador_es_el_sku_completo(self):
        assert matching.extraer_sku("BEB-001") == "BEB-001"


class TestRankearPorSku:
    def test_colapsa_varias_fotos_en_la_mejor_distancia(self):
        ranking = matching.rankear_por_sku(
            ["A_foto_01", "A_foto_02", "B_foto_01"],
            [0.30, 0.10, 0.20],
        )
        assert ranking == [("A", 0.10), ("B", 0.20)]

    def test_ordena_de_menor_a_mayor_distancia(self):
        ranking = matching.rankear_por_sku(["C_x", "A_x", "B_x"], [0.9, 0.1, 0.5])
        assert [sku for sku, _ in ranking] == ["A", "B", "C"]

    def test_sin_resultados_devuelve_lista_vacia(self):
        assert matching.rankear_por_sku([], []) == []


class TestHayMargenSuficiente:
    def test_un_solo_candidato_no_es_ambiguo(self):
        assert matching.hay_margen_suficiente([("A", 0.10)]) is True

    def test_acepta_cuando_el_ganador_destaca(self):
        # margen = (0.40 - 0.10) / 0.40 = 0.75 >= 0.25
        assert matching.hay_margen_suficiente([("A", 0.10), ("B", 0.40)]) is True

    def test_rechaza_cuando_los_dos_primeros_estan_pegados(self):
        # margen = (0.21 - 0.20) / 0.21 = 0.047 < 0.25
        assert matching.hay_margen_suficiente([("A", 0.20), ("B", 0.21)]) is False

    def test_margen_exactamente_en_el_limite_se_acepta(self):
        # margen = (0.40 - 0.30) / 0.40 = 0.25
        assert matching.hay_margen_suficiente([("A", 0.30), ("B", 0.40)]) is True

    def test_segundo_a_distancia_cero_no_divide_entre_cero(self):
        assert matching.hay_margen_suficiente([("A", 0.0), ("B", 0.0)]) is True


class TestEvaluar:
    def test_acepta_un_ganador_claro_y_cercano(self):
        veredicto = matching.evaluar(["A_x", "B_x"], [0.05, 0.60], UMBRAL)
        assert veredicto.es_confiable is True
        assert veredicto.sku == "A"
        assert veredicto.distancia == 0.05
        assert veredicto.motivo_rechazo is None

    def test_rechaza_por_distancia_sobre_el_umbral(self):
        veredicto = matching.evaluar(["A_x"], [0.80], UMBRAL)
        assert veredicto.es_confiable is False
        assert veredicto.motivo_rechazo == matching.MOTIVO_DISTANCIA
        # El SKU y la distancia se conservan para poder reportarlos.
        assert veredicto.sku == "A"
        assert veredicto.distancia == 0.80

    def test_rechaza_por_margen_aunque_este_dentro_del_umbral(self):
        veredicto = matching.evaluar(["A_x", "B_x"], [0.20, 0.21], UMBRAL)
        assert veredicto.es_confiable is False
        assert veredicto.motivo_rechazo == matching.MOTIVO_MARGEN

    def test_el_filtro_de_distancia_tiene_prioridad_sobre_el_de_margen(self):
        # Ambos filtros fallarían; debe reportarse el de distancia.
        veredicto = matching.evaluar(["A_x", "B_x"], [0.90, 0.91], UMBRAL)
        assert veredicto.motivo_rechazo == matching.MOTIVO_DISTANCIA

    def test_el_ganador_se_calcula_por_mejor_foto_no_por_orden(self):
        # B gana porque una de sus fotos está más cerca que la mejor de A.
        veredicto = matching.evaluar(
            ["A_x_01", "B_x_01", "B_x_02"],
            [0.20, 0.90, 0.02],
            UMBRAL,
        )
        assert veredicto.sku == "B"
        assert veredicto.es_confiable is True
