from __future__ import annotations

from typing import NamedTuple, Sequence

N_RESULTADOS_BUSQUEDA = 15

MARGEN_MINIMO_ENTRE_SKUS = 0.25

SEPARADOR_ID_CHROMA = "_"

MOTIVO_DISTANCIA = "distancia"
MOTIVO_MARGEN = "margen insuficiente"


class Veredicto(NamedTuple):
    sku: str
    distancia: float
    es_confiable: bool
    motivo_rechazo: str | None = None


def extraer_sku(id_chroma: str) -> str:
    return id_chroma.split(SEPARADOR_ID_CHROMA)[0]


def rankear_por_sku(
    ids: Sequence[str],
    distancias: Sequence[float],
) -> list[tuple[str, float]]:
    mejor_por_sku: dict[str, float] = {}
    for id_chroma, distancia in zip(ids, distancias):
        sku = extraer_sku(id_chroma)
        if sku not in mejor_por_sku or distancia < mejor_por_sku[sku]:
            mejor_por_sku[sku] = distancia
    return sorted(mejor_por_sku.items(), key=lambda par: par[1])


def hay_margen_suficiente(ranking: Sequence[tuple[str, float]]) -> bool:
    if len(ranking) < 2:
        return True

    dist_ganador = ranking[0][1]
    dist_segundo = ranking[1][1]
    if dist_segundo <= 0:
        return True

    margen = (dist_segundo - dist_ganador) / dist_segundo
    return margen >= MARGEN_MINIMO_ENTRE_SKUS


def evaluar(
    ids: Sequence[str],
    distancias: Sequence[float],
    umbral_distancia: float,
) -> Veredicto:
    ranking = rankear_por_sku(ids, distancias)
    sku_ganador, dist_ganador = ranking[0]

    if dist_ganador > umbral_distancia:
        return Veredicto(sku_ganador, dist_ganador, False, MOTIVO_DISTANCIA)

    if not hay_margen_suficiente(ranking):
        return Veredicto(sku_ganador, dist_ganador, False, MOTIVO_MARGEN)

    return Veredicto(sku_ganador, dist_ganador, True)
