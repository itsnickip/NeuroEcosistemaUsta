"""Red neuronal multicapa (perceptrón) escrita solo con NumPy.

Arquitectura usada por EcosistemasNeurito:

    entrada (8) -> oculta 1 (ReLU) -> oculta 2 (ReLU) -> oculta 3 (ReLU) -> salida (Softmax, 5)

La salida es una probabilidad por cada acción discreta. Ni el mundo ni ninguna
regla sustituyen la decisión: la acción es el ``argmax`` de esas probabilidades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Funciones de activación
# ---------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU: deja pasar los valores positivos y anula los negativos."""
    return np.maximum(z, 0.0)


def relu_derivada(z: np.ndarray) -> np.ndarray:
    """Derivada de ReLU respecto a su entrada (1 si z > 0, si no 0)."""
    return (z > 0).astype(np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estable: resta el máximo de cada fila antes de exponenciar."""
    desplazados = logits - np.max(logits, axis=1, keepdims=True)
    exponenciales = np.exp(desplazados)
    return exponenciales / np.sum(exponenciales, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Red neuronal
# ---------------------------------------------------------------------------

class RedNeurito:
    """Clasificador multicapa entrenado con mini-lotes (batch) y Adam.

    ``capas_ocultas`` admite dos o más capas, por ejemplo ``(14, 16, 12)``.
    """

    MIN_CAPAS_OCULTAS = 2

    def __init__(
        self,
        entradas: int,
        capas_ocultas: Sequence[int] = (14, 16, 12),
        salidas: int = 5,
        seed: int = 42,
    ) -> None:
        if entradas < 1 or salidas < 2:
            raise ValueError("entradas debe ser positivo y salidas al menos 2")
        capas_ocultas = tuple(int(n) for n in capas_ocultas)
        if len(capas_ocultas) < self.MIN_CAPAS_OCULTAS or any(n < 1 for n in capas_ocultas):
            raise ValueError("se necesitan al menos dos capas ocultas con tamaño positivo")

        self.entradas = int(entradas)
        self.capas_ocultas = capas_ocultas
        self.salidas = int(salidas)
        self.seed = int(seed)

        generador = np.random.default_rng(self.seed)
        tamanos = self.arquitectura
        # Inicialización He: adecuada para capas con ReLU.
        self.pesos = [
            generador.normal(0.0, np.sqrt(2.0 / tamanos[i]), (tamanos[i], tamanos[i + 1]))
            for i in range(len(tamanos) - 1)
        ]
        self.sesgos = [np.zeros(n, dtype=np.float64) for n in tamanos[1:]]

    # -- información de la arquitectura ------------------------------------

    @property
    def arquitectura(self) -> tuple[int, ...]:
        """Tamaño de todas las capas: (entrada, ocultas..., salida)."""
        return (self.entradas, *self.capas_ocultas, self.salidas)

    @property
    def num_capas(self) -> int:
        """Número de matrices de pesos (capas ocultas + capa de salida)."""
        return len(self.pesos)

    def descripcion(self) -> str:
        return " → ".join(str(n) for n in self.arquitectura)

    # -- validación ---------------------------------------------------------

    def _preparar_entrada(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.entradas:
            raise ValueError(f"se esperaba una matriz con {self.entradas} columnas")
        if not np.isfinite(X).all():
            raise ValueError("la entrada contiene valores no finitos")
        return X

    def _preparar_etiquetas(self, y: np.ndarray, n: int) -> np.ndarray:
        y = np.asarray(y)
        if y.shape != (n,) or n == 0:
            raise ValueError("las etiquetas deben ser un arreglo 1D no vacío del tamaño de X")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("las etiquetas deben ser índices enteros de acción")
        if np.any((y < 0) | (y >= self.salidas)):
            raise ValueError("hay etiquetas fuera del rango de salidas")
        return y.astype(np.int64, copy=False)

    # -- propagación hacia adelante ----------------------------------------

    def _propagar(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Devuelve activaciones (incluida la entrada) y preactivaciones z."""
        activaciones = [X]
        preactivaciones = []
        for W, b in zip(self.pesos[:-1], self.sesgos[:-1]):
            z = activaciones[-1] @ W + b
            preactivaciones.append(z)
            activaciones.append(relu(z))
        logits = activaciones[-1] @ self.pesos[-1] + self.sesgos[-1]
        preactivaciones.append(logits)
        activaciones.append(softmax(logits))
        return activaciones, preactivaciones

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Una distribución de probabilidad sobre las acciones por observación."""
        X = self._preparar_entrada(X)
        if len(X) == 0:
            return np.empty((0, self.salidas), dtype=np.float64)
        return self._propagar(X)[0][-1]

    def predict_actions(self, X: np.ndarray) -> np.ndarray:
        """Acción elegida únicamente a partir de las probabilidades de la red."""
        return np.argmax(self.predict_proba(X), axis=1)

    def parametros_planos(self) -> np.ndarray:
        """Copia de todos los pesos y sesgos para auditar si el entrenamiento los cambió."""
        return np.concatenate([p.ravel() for p in (*self.pesos, *self.sesgos)]).copy()

    # -- evaluación ---------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Entropía cruzada y precisión sobre un conjunto de ejemplos."""
        X = self._preparar_entrada(X)
        y = self._preparar_etiquetas(y, len(X))
        probabilidades = self.predict_proba(X)
        correctas = probabilidades[np.arange(len(y)), y]
        perdida = -np.mean(np.log(np.clip(correctas, 1e-12, 1.0)))
        precision = np.mean(np.argmax(probabilidades, axis=1) == y)
        return {"loss": float(perdida), "accuracy": float(precision)}

    # -- entrenamiento: backpropagation + Adam -----------------------------

    def _backpropagation(
        self, xb: np.ndarray, yb: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Gradientes de la entropía cruzada respecto a pesos y sesgos."""
        activaciones, preactivaciones = self._propagar(xb)
        # Softmax + entropía cruzada: dL/dz = p - one_hot(y)
        delta = activaciones[-1].copy()
        delta[np.arange(len(yb)), yb] -= 1.0
        delta /= len(yb)

        grad_W: list[np.ndarray] = [np.empty(0)] * self.num_capas
        grad_b: list[np.ndarray] = [np.empty(0)] * self.num_capas
        for capa in reversed(range(self.num_capas)):
            grad_W[capa] = activaciones[capa].T @ delta
            grad_b[capa] = np.sum(delta, axis=0)
            if capa > 0:
                delta = (delta @ self.pesos[capa].T) * relu_derivada(preactivaciones[capa - 1])
        return grad_W, grad_b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 80,
        batch_size: int = 64,
        learning_rate: float = 0.004,
        seed: int = 42,
    ) -> list[dict[str, float | int]]:
        """Entrena por épocas con mini-lotes barajados de forma reproducible.

        El conjunto de validación debe venir de semillas de simulación distintas
        a las del entrenamiento.
        """
        X = self._preparar_entrada(X)
        y = self._preparar_etiquetas(y, len(X))
        if (X_val is None) != (y_val is None):
            raise ValueError("entrega X_val y y_val juntos, o ninguno")
        if X_val is not None:
            X_val = self._preparar_entrada(X_val)
            y_val = self._preparar_etiquetas(y_val, len(X_val))
        if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("epochs, batch_size y learning_rate deben ser positivos")

        generador = np.random.default_rng(seed)
        parametros = [*self.pesos, *self.sesgos]
        momento_1 = [np.zeros_like(p) for p in parametros]
        momento_2 = [np.zeros_like(p) for p in parametros]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        paso = 0
        historial: list[dict[str, float | int]] = []

        for epoca in range(1, epochs + 1):
            orden = generador.permutation(len(X))
            for inicio in range(0, len(X), batch_size):
                lote = orden[inicio : inicio + batch_size]
                grad_W, grad_b = self._backpropagation(X[lote], y[lote])

                paso += 1
                for i, (param, grad) in enumerate(zip(parametros, [*grad_W, *grad_b])):
                    momento_1[i] = beta1 * momento_1[i] + (1.0 - beta1) * grad
                    momento_2[i] = beta2 * momento_2[i] + (1.0 - beta2) * np.square(grad)
                    m_hat = momento_1[i] / (1.0 - beta1**paso)
                    v_hat = momento_2[i] / (1.0 - beta2**paso)
                    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            metricas = self.evaluate(X, y)
            fila: dict[str, float | int] = {
                "epoch": epoca,
                "loss": metricas["loss"],
                "accuracy": metricas["accuracy"],
            }
            if X_val is not None and y_val is not None:
                validacion = self.evaluate(X_val, y_val)
                fila["val_loss"] = validacion["loss"]
                fila["val_accuracy"] = validacion["accuracy"]
            historial.append(fila)
        return historial

    # -- persistencia -------------------------------------------------------

    def save(self, ruta: str | Path) -> None:
        """Guarda arquitectura y parámetros en un archivo NPZ portable."""
        ruta = Path(ruta)
        if ruta.suffix != ".npz":
            raise ValueError("la ruta del modelo debe terminar en .npz")
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ruta,
            architecture=np.array(self.arquitectura, dtype=np.int64),
            seed=np.array(self.seed, dtype=np.int64),
            **{f"W{i}": W for i, W in enumerate(self.pesos)},
            **{f"b{i}": b for i, b in enumerate(self.sesgos)},
        )

    @classmethod
    def load(cls, ruta: str | Path) -> "RedNeurito":
        """Carga solo arreglos numéricos; pickle queda deshabilitado."""
        with np.load(ruta, allow_pickle=False) as archivo:
            arquitectura = archivo["architecture"].astype(int)
            if arquitectura.ndim != 1 or len(arquitectura) < 2 + cls.MIN_CAPAS_OCULTAS:
                raise ValueError("el modelo debe tener al menos dos capas ocultas")
            modelo = cls(
                int(arquitectura[0]),
                tuple(int(n) for n in arquitectura[1:-1]),
                int(arquitectura[-1]),
                int(archivo["seed"]),
            )
            for i in range(modelo.num_capas):
                W, b = archivo[f"W{i}"], archivo[f"b{i}"]
                if W.shape != modelo.pesos[i].shape or b.shape != modelo.sesgos[i].shape:
                    raise ValueError("la forma de los parámetros no coincide con la arquitectura")
                modelo.pesos[i][...] = W
                modelo.sesgos[i][...] = b
        return modelo
