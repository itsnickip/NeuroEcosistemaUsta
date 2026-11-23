"""Perceptrón multicapa en NumPy (sin frameworks de IA).

entradas -> oculta_1 (tanh) -> oculta_2 (tanh) -> 9 salidas (softmax)

Los pesos se pueden leer/escribir como un único vector plano ("genoma"),
que es lo que manipula la neuroevolución.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTIVACIONES = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(0.0, z),
}


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class RedMLP:
    def __init__(self, n_entradas: int, ocultas: list[int], n_salidas: int,
                 activacion: str = "tanh", semilla: int = 0, sigma: float = 0.5):
        if len(ocultas) < 2:
            raise ValueError("El proyecto exige al menos dos capas ocultas")
        self.n_entradas = n_entradas
        self.ocultas = list(ocultas)
        self.n_salidas = n_salidas
        self.activacion = activacion
        self.tamanos = [n_entradas, *self.ocultas, n_salidas]
        rng = np.random.default_rng(semilla)
        self.pesos, self.sesgos = [], []
        for a, b in zip(self.tamanos[:-1], self.tamanos[1:]):
            # escala tipo Xavier multiplicada por sigma
            self.pesos.append(rng.normal(0, sigma * np.sqrt(1.0 / a), (a, b)))
            self.sesgos.append(np.zeros(b))

    # ------------------------------------------------------------ inferencia
    def adelante(self, X, devolver_capas: bool = False):
        f = ACTIVACIONES[self.activacion]
        a = np.atleast_2d(np.asarray(X, dtype=float))
        capas = [a]
        for i, (W, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = a @ W + b
            a = softmax(z) if i == len(self.pesos) - 1 else f(z)
            capas.append(a)
        return (a, capas) if devolver_capas else a

    def decidir(self, X):
        """Acción determinista = argmax de la salida softmax."""
        p = self.adelante(X)
        return p.argmax(axis=1), p

    # ------------------------------------------------- aprendizaje supervisado
    def paso_gradiente(self, X, y, tasa: float) -> float:
        """Un paso de descenso de gradiente (entropía cruzada) sobre un mini-lote.

        Retropropagación genérica para cualquier número de capas ocultas.
        Devuelve la pérdida del lote antes de actualizar.
        """
        p, capas = self.adelante(X, devolver_capas=True)
        n = len(X)
        perdida = -np.mean(np.log(np.clip(p[np.arange(n), y], 1e-10, 1)))
        delta = p.copy()
        delta[np.arange(n), y] -= 1
        delta /= n
        for i in range(len(self.pesos) - 1, -1, -1):
            grad_W = capas[i].T @ delta
            grad_b = delta.sum(axis=0)
            if i > 0:
                a = capas[i]
                deriv = (1 - a ** 2) if self.activacion == "tanh" else (a > 0).astype(float)
                delta = (delta @ self.pesos[i].T) * deriv
            self.pesos[i] = self.pesos[i] - tasa * grad_W
            self.sesgos[i] = self.sesgos[i] - tasa * grad_b
        return float(perdida)

    # --------------------------------------------------------------- genoma
    @property
    def n_parametros(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.pesos, self.sesgos))

    def obtener_genoma(self) -> np.ndarray:
        return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in zip(self.pesos, self.sesgos)])

    def asignar_genoma(self, genoma: np.ndarray):
        genoma = np.asarray(genoma, dtype=float)
        if genoma.size != self.n_parametros:
            raise ValueError(f"Genoma de tamaño {genoma.size}, se esperaban {self.n_parametros}")
        i = 0
        for k, (W, b) in enumerate(zip(self.pesos, self.sesgos)):
            self.pesos[k] = genoma[i:i + W.size].reshape(W.shape); i += W.size
            self.sesgos[k] = genoma[i:i + b.size].copy(); i += b.size
        return self

    def copia_con(self, genoma):
        nueva = RedMLP(self.n_entradas, self.ocultas, self.n_salidas, self.activacion)
        return nueva.asignar_genoma(genoma)

    # ------------------------------------------------------- persistencia
    def guardar(self, ruta, meta: dict | None = None):
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        info = {"n_entradas": self.n_entradas, "ocultas": self.ocultas,
                "n_salidas": self.n_salidas, "activacion": self.activacion,
                "meta": meta or {}}
        np.savez(ruta, genoma=self.obtener_genoma(), info=json.dumps(info))

    @classmethod
    def cargar(cls, ruta):
        datos = np.load(Path(ruta), allow_pickle=False)
        info = json.loads(str(datos["info"]))
        red = cls(info["n_entradas"], info["ocultas"], info["n_salidas"], info["activacion"])
        red.asignar_genoma(datos["genoma"])
        red.meta = info.get("meta", {})
        return red

    def __repr__(self):
        arq = " -> ".join(map(str, self.tamanos))
        return f"RedMLP({arq}, {self.activacion}, {self.n_parametros} parámetros)"
