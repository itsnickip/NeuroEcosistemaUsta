"""Pruebas de la red: ReLU, Softmax, aprendizaje real y persistencia."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ecosistemas_neurito.network import RedNeurito, relu, relu_derivada, softmax


class ActivacionesTests(unittest.TestCase):
    def test_relu_y_su_derivada(self) -> None:
        z = np.array([[-2.0, 0.0, 3.5]])
        np.testing.assert_array_equal(relu(z), [[0.0, 0.0, 3.5]])
        np.testing.assert_array_equal(relu_derivada(z), [[0.0, 0.0, 1.0]])

    def test_softmax_suma_uno_y_es_estable(self) -> None:
        p = softmax(np.array([[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p[1], [1 / 3] * 3)


class RedNeuritoTests(unittest.TestCase):
    def test_arquitectura_8_14_16_12_5(self) -> None:
        red = RedNeurito(8, (14, 16, 12), 5, seed=9)
        probabilidades = red.predict_proba(np.zeros((4, 8)))
        self.assertEqual(probabilidades.shape, (4, 5))
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0)
        self.assertEqual(
            [W.shape for W in red.pesos], [(8, 14), (14, 16), (16, 12), (12, 5)]
        )
        self.assertEqual(red.descripcion(), "8 → 14 → 16 → 12 → 5")

    def test_requiere_al_menos_dos_capas_ocultas(self) -> None:
        with self.assertRaises(ValueError):
            RedNeurito(8, (14,), 5)

    def test_fit_aprende_y_cambia_pesos(self) -> None:
        rng = np.random.default_rng(17)
        X_train = rng.uniform(-1, 1, (900, 3))
        X_val = rng.uniform(-1, 1, (300, 3))
        # Regla multiclase aprendible, evaluada en entradas no vistas.
        etiqueta = lambda X: np.argmax(X, axis=1)
        y_train, y_val = etiqueta(X_train), etiqueta(X_val)
        red = RedNeurito(3, (14, 16, 12), 3, seed=2)
        antes = red.parametros_planos()
        historial = red.fit(
            X_train, y_train, X_val, y_val, epochs=65, batch_size=64,
            learning_rate=0.01, seed=31,
        )
        self.assertGreater(historial[-1]["val_accuracy"], 0.93)
        self.assertLess(historial[-1]["loss"], historial[0]["loss"])
        self.assertGreater(np.linalg.norm(red.parametros_planos() - antes), 0.1)

    def test_semilla_y_guardado_reproducibles(self) -> None:
        a, b = RedNeurito(4, (6, 7, 5), seed=5), RedNeurito(4, (6, 7, 5), seed=5)
        np.testing.assert_array_equal(a.parametros_planos(), b.parametros_planos())
        X = np.array([[0.1, -0.2, 0.5, 0.8], [1.0, 0.0, -1.0, 0.2]])
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "presa.npz"
            a.save(ruta)
            restaurada = RedNeurito.load(ruta)
            self.assertEqual(restaurada.capas_ocultas, (6, 7, 5))
            np.testing.assert_array_equal(restaurada.parametros_planos(), a.parametros_planos())
            np.testing.assert_array_equal(restaurada.predict_proba(X), a.predict_proba(X))

    def test_etiquetas_invalidas(self) -> None:
        red = RedNeurito(2, (3, 3), 3)
        with self.assertRaises(ValueError):
            red.fit(np.ones((2, 2)), np.array([0, 3]), epochs=1)


if __name__ == "__main__":
    unittest.main()
