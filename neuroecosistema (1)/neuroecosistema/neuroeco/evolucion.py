"""Neuroevolución (algoritmo genético) para las redes de presas y depredadores.

Cada genoma es el vector plano de pesos de una red. Todas las presas de un
episodio comparten el mismo genoma (población homogénea), así que la aptitud
mide qué tan buena es esa "política de especie".

Ciclo por generación:
  1. evaluar cada genoma en los mismos episodios (mismas semillas = comparación justa)
  2. conservar las élites sin cambios
  3. resto: selección por torneo -> cruce uniforme -> mutación gaussiana
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import ESCENARIOS, ConfigEntrenamiento
from .mundo import ENTRADAS_DEPREDADOR, ENTRADAS_PRESA, N_ACCIONES
from .politicas import PoliticaRed, PoliticaReglasDepredador, PoliticaReglasPresa
from .red import RedMLP
from .simulacion import aptitud_depredador, aptitud_presa, ejecutar_episodio

ESPECIES = {
    "presa": {"entradas": len(ENTRADAS_PRESA), "aptitud": aptitud_presa},
    "depredador": {"entradas": len(ENTRADAS_DEPREDADOR), "aptitud": aptitud_depredador},
}


def crear_red(especie: str, ocultas, activacion="tanh", semilla=0, sigma=0.5) -> RedMLP:
    return RedMLP(ESPECIES[especie]["entradas"], ocultas, N_ACCIONES, activacion, semilla, sigma)


# -------------------------------------------------------------- evaluación
# Los oponentes se describen como ("reglas", None) o ("red", genoma, ocultas, activacion)
# para que se puedan enviar a otros procesos.
def _construir_oponente(especie_oponente, desc):
    if desc[0] == "reglas":
        return PoliticaReglasPresa() if especie_oponente == "presa" else PoliticaReglasDepredador()
    _, genoma, ocultas, act = desc
    return PoliticaRed(crear_red(especie_oponente, ocultas, act).asignar_genoma(genoma))


def _evaluar_genoma(args):
    especie, genoma, ocultas, act, episodios = args
    red = crear_red(especie, ocultas, act).asignar_genoma(genoma)
    propia = PoliticaRed(red)
    otra = "depredador" if especie == "presa" else "presa"
    total = 0.0
    for escenario, semilla, desc_op in episodios:
        oponente = _construir_oponente(otra, desc_op)
        if especie == "presa":
            m = ejecutar_episodio(ESCENARIOS[escenario], propia, oponente, semilla)
        else:
            m = ejecutar_episodio(ESCENARIOS[escenario], oponente, propia, semilla)
        total += ESPECIES[especie]["aptitud"](m)
    return total / len(episodios)


@dataclass
class ResultadoEvolucion:
    mejor_genoma: np.ndarray
    genoma_inicial: np.ndarray
    historial: list = field(default_factory=list)


class Neuroevolucion:
    def __init__(self, especie: str, ocultas, cfg: ConfigEntrenamiento, workers: int = 1):
        self.especie = especie
        self.ocultas = list(ocultas)
        self.cfg = cfg
        self.workers = workers
        self.rng = np.random.default_rng(cfg.semilla + (0 if especie == "presa" else 1))

    def poblacion_inicial(self, genoma_base=None):
        """Pesos aleatorios, o (si hay genoma_base) copias mutadas de la red pre-entrenada.

        El individuo 0 siempre es el genoma sin mutar, para que la evolución
        nunca pueda empezar peor que el punto de partida.
        """
        c = self.cfg
        base = c.semilla * 1000 + (0 if self.especie == "presa" else 500)
        if genoma_base is None:
            return np.stack([crear_red(self.especie, self.ocultas, c.activacion, base + i, c.sigma_inicial).obtener_genoma()
                             for i in range(c.poblacion)])
        rng = np.random.default_rng(base)
        pob = np.repeat(genoma_base[None, :], c.poblacion, axis=0)
        pob[1:] += rng.normal(0, c.sigma_mutacion, pob[1:].shape)
        return pob

    def _episodios(self, generacion, oponentes, base_semilla):
        """Mismos episodios para todos los genomas de una generación (semillas registradas)."""
        eps = []
        for k in range(self.cfg.episodios_por_genoma):
            esc = self.cfg.escenarios_entrenamiento[k % len(self.cfg.escenarios_entrenamiento)]
            op = oponentes[(k + generacion) % len(oponentes)]
            eps.append((esc, base_semilla + generacion * 100 + k, op))
        return eps

    def _evaluar(self, poblacion, episodios, pool):
        tareas = [(self.especie, g, self.ocultas, self.cfg.activacion, episodios) for g in poblacion]
        if pool is None:
            return np.array([_evaluar_genoma(t) for t in tareas])
        return np.array(pool.map(_evaluar_genoma, tareas))

    def _torneo(self, poblacion, aptitud):
        ids = self.rng.choice(len(poblacion), self.cfg.torneo, replace=False)
        return poblacion[ids[np.argmax(aptitud[ids])]]

    def evolucionar(self, oponentes, poblacion=None, ronda=0, log=print) -> ResultadoEvolucion:
        c = self.cfg
        if poblacion is None:
            poblacion = self.poblacion_inicial()
        genoma_inicial = poblacion[0].copy()
        base_semilla = 10_000 * (ronda + 1) + (0 if self.especie == "presa" else 5_000)
        historial, mejor_anterior, mejor_gen0 = [], None, None
        pool = Pool(self.workers) if self.workers > 1 else None
        try:
            for gen in range(c.generaciones):
                t0 = time.time()
                eps = self._episodios(gen, oponentes, base_semilla)
                apt = self._evaluar(poblacion, eps, pool)
                orden = np.argsort(apt)[::-1]
                mejor = poblacion[orden[0]].copy()
                if mejor_gen0 is None:
                    mejor_gen0 = mejor.copy()
                fila = {
                    "ronda": ronda, "especie": self.especie, "generacion": gen,
                    "aptitud_mejor": float(apt[orden[0]]), "aptitud_media": float(apt.mean()),
                    "aptitud_std": float(apt.std()),
                    "norma_pesos_mejor": float(np.linalg.norm(mejor)),
                    "distancia_a_gen0": float(np.linalg.norm(mejor - mejor_gen0)),
                    "cambio_vs_gen_anterior": float(np.linalg.norm(mejor - mejor_anterior)) if mejor_anterior is not None else 0.0,
                    "semillas": "|".join(f"{e}:{s}" for e, s, _ in eps),
                    "segundos": round(time.time() - t0, 2),
                }
                historial.append(fila)
                log(f"  [{self.especie} r{ronda}] gen {gen:3d}  mejor={fila['aptitud_mejor']:.3f}  "
                    f"media={fila['aptitud_media']:.3f}  |Δw|={fila['cambio_vs_gen_anterior']:.2f}  ({fila['segundos']}s)")
                mejor_anterior = mejor

                if gen == c.generaciones - 1:
                    break
                nueva = [poblacion[i].copy() for i in orden[:c.elites]]
                while len(nueva) < c.poblacion:
                    p1 = self._torneo(poblacion, apt)
                    hijo = p1.copy()
                    if self.rng.random() < c.prob_cruce:
                        p2 = self._torneo(poblacion, apt)
                        mascara = self.rng.random(hijo.size) < 0.5
                        hijo[mascara] = p2[mascara]
                    hijo += self.rng.normal(0, c.sigma_mutacion, hijo.size)
                    nueva.append(hijo)
                poblacion = np.stack(nueva)

            # validación: elegir entre las élites finales con semillas que no se usaron al entrenar
            val_eps = [(esc, 90_000 + 37 * i + ronda, oponentes[i % len(oponentes)])
                       for i, esc in enumerate(c.escenarios_entrenamiento * 2)]
            candidatos = poblacion[orden[:max(c.elites, 3)]]
            apt_val = self._evaluar(candidatos, val_eps, pool)
            elegido = candidatos[int(np.argmax(apt_val))]
            log(f"  [{self.especie} r{ronda}] validación élites: {np.round(apt_val, 3).tolist()}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        self.poblacion_final = poblacion
        return ResultadoEvolucion(elegido, genoma_inicial, historial)


def guardar_historial(filas, ruta: Path):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
