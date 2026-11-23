"""Ejecución de episodios completos (sin interfaz)."""
from __future__ import annotations

import numpy as np

from .config import ConfigMundo
from .mundo import Mundo
from .politicas import Politica


def ejecutar_episodio(cfg: ConfigMundo, pol_presa: Politica, pol_depr: Politica,
                      semilla: int) -> dict:
    """Corre un episodio de principio a fin y devuelve sus métricas.

    La misma semilla produce el mismo mapa, posiciones iniciales y regeneración
    de plantas, así que dos políticas evaluadas con la misma semilla enfrentan
    exactamente las mismas condiciones iniciales.
    """
    mundo = Mundo(cfg, semilla)
    pol_presa.reiniciar(semilla)
    pol_depr.reiniciar(semilla)
    while not mundo.terminado:
        mundo.paso_simulacion(avanzar_presas(mundo, pol_presa), avanzar_depr(mundo, pol_depr))
    return mundo.metricas()


def avanzar_presas(mundo: Mundo, pol: Politica) -> np.ndarray:
    acc = np.zeros(len(mundo.presas_pos), dtype=int)
    vivas = mundo.presas_viva
    if vivas.any():
        acc[vivas] = pol(mundo.observar_presas()[vivas])
    return acc


def avanzar_depr(mundo: Mundo, pol: Politica) -> np.ndarray:
    acc = np.zeros(len(mundo.depr_pos), dtype=int)
    vivos = mundo.depr_viva
    if vivos.any():
        acc[vivos] = pol(mundo.observar_depredadores()[vivos])
    return acc


# ------------------------------------------------------------- aptitud
def aptitud_presa(m: dict) -> float:
    """Sobrevivir es lo principal; comer suma un poco.

    Riesgo conocido: favorece quedarse lejos de todo si eso alarga la vida,
    aunque se coma poco (se discute en la bitácora).
    """
    return m["supervivencia_presas"] + 0.05 * m["comida_por_presa"]


def aptitud_depredador(m: dict) -> float:
    """Capturas por depredador + acercamiento a presas + supervivencia.

    Las capturas son escasas al principio (casi ninguna red aleatoria caza), así
    que sin el término de acercamiento la evolución no tiene gradiente y premia
    quedarse quieto para ahorrar energía (se observó en la versión 1, ver bitácora).
    Riesgo del shaping: puede favorecer "escoltar" presas sin capturarlas.
    """
    return (m["capturas_por_depredador"] + 0.5 * m["acercamiento_depredadores"]
            + 0.2 * m["supervivencia_depredadores"])
