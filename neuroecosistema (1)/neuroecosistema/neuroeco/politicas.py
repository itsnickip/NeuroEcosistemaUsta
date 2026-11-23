"""Políticas que convierten observaciones en acciones.

- PoliticaRed: la acción sale del argmax de la red neuronal (la que se evalúa).
- PoliticaReglas: referencia hecha a mano que usa EXACTAMENTE las mismas
  entradas que la red, para que la comparación sea justa.
- PoliticaAleatoria: referencia mínima (acción uniforme).
"""
from __future__ import annotations

import numpy as np

from .mundo import N_ACCIONES, vector_a_accion
from .red import RedMLP


class Politica:
    nombre = "base"

    def __call__(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reiniciar(self, semilla: int = 0):
        """Se llama al inicio de cada episodio (para políticas con azar)."""


class PoliticaAleatoria(Politica):
    nombre = "aleatoria"

    def __init__(self, semilla: int = 0):
        self.rng = np.random.default_rng(semilla)

    def reiniciar(self, semilla: int = 0):
        self.rng = np.random.default_rng(semilla + 7_000)

    def __call__(self, obs):
        return self.rng.integers(0, N_ACCIONES, len(obs))


class PoliticaRed(Politica):
    def __init__(self, red: RedMLP, nombre: str = "red"):
        self.red = red
        self.nombre = nombre
        self.ultimas_probabilidades = None

    def __call__(self, obs):
        if len(obs) == 0:
            return np.zeros(0, dtype=int)
        acciones, p = self.red.decidir(obs)
        self.ultimas_probabilidades = p
        return acciones


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


def _repulsion(obst, umbral=0.6):
    """Empuje contra el obstáculo solo cuando está cerca (magnitud del sensor > umbral)."""
    mag = np.linalg.norm(obst, axis=1, keepdims=True)
    return -_unit(obst) * np.clip((mag - umbral) / (1 - umbral), 0, 1) * 2.0


class _Deambular:
    """Deambular con dirección persistente (cambia cada ~15 pasos) sesgado al centro."""

    def __init__(self, semilla=0):
        self.reiniciar(semilla)

    def reiniciar(self, semilla=0):
        self.rng = np.random.default_rng(semilla + 9_000)
        self.dir = None

    def vector(self, pos_norm):
        n = len(pos_norm)
        if self.dir is None or len(self.dir) != n:
            self.dir = self.rng.normal(size=(n, 2))
        cambia = self.rng.random(n) < 1 / 15
        self.dir[cambia] = self.rng.normal(size=(int(cambia.sum()), 2))
        # pos_norm está en [-1, 1]; -pos empuja hacia el centro cerca de los bordes
        return self.dir + (-pos_norm) * 1.5 * (np.abs(pos_norm) > 0.8)


class PoliticaReglasPresa(Politica):
    """Si ve un depredador cerca huye; si ve una planta va hacia ella; si no, deambula."""
    nombre = "reglas"

    def __init__(self, umbral_huida: float = 0.65, semilla: int = 0):
        self.umbral = umbral_huida
        self.deambular = _Deambular(semilla)

    def reiniciar(self, semilla=0):
        self.deambular.reiniciar(semilla)

    def __call__(self, obs):
        planta, depr = obs[:, 0:2], obs[:, 3:5]
        d_depr, d_planta = obs[:, 5], obs[:, 2]
        pos, obst = obs[:, 7:9], obs[:, 9:11]
        v = self.deambular.vector(pos)
        ve_planta = d_planta < 1
        v = _unit(v)
        v[ve_planta] = _unit(planta[ve_planta])
        huye = d_depr < self.umbral
        # huir en sentido contrario + alejarse de las paredes para no quedar acorraladas
        v[huye] = _unit(-depr[huye]) - 0.8 * pos[huye] * (np.abs(pos[huye]) > 0.7)
        v = v + _repulsion(obst)
        return vector_a_accion(v)


class PoliticaReglasDepredador(Politica):
    """Si ve una presa la persigue en línea recta; si no, deambula."""
    nombre = "reglas"

    def __init__(self, semilla: int = 0):
        self.deambular = _Deambular(semilla)

    def reiniciar(self, semilla=0):
        self.deambular.reiniciar(semilla + 1)

    def __call__(self, obs):
        presa, d_presa = obs[:, 0:2], obs[:, 2]
        pos, obst = obs[:, 5:7], obs[:, 7:9]
        v = self.deambular.vector(pos)
        ve = d_presa < 1
        v = _unit(v)
        v[ve] = _unit(presa[ve])
        v = v + _repulsion(obst)
        return vector_a_accion(v)
