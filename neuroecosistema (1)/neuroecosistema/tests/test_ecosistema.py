"""Pruebas automáticas: uv run python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuroeco.config import ESCENARIOS, ConfigEntrenamiento, ConfigMundo
from neuroeco.evolucion import Neuroevolucion, crear_red
from neuroeco.mundo import DIRECCIONES, ENTRADAS_DEPREDADOR, ENTRADAS_PRESA, Mundo, vector_a_accion
from neuroeco.politicas import PoliticaAleatoria, PoliticaRed, PoliticaReglasDepredador, PoliticaReglasPresa
from neuroeco.red import RedMLP
from neuroeco.simulacion import ejecutar_episodio

MINI = ConfigMundo(max_pasos=60, n_presas=6, n_depredadores=2, n_plantas=10)


class TestRed(unittest.TestCase):
    def test_exige_dos_capas_ocultas(self):
        with self.assertRaises(ValueError):
            RedMLP(4, [8], 3)

    def test_salida_es_distribucion(self):
        red = RedMLP(11, [16, 12], 9, semilla=3)
        p = red.adelante(np.random.default_rng(0).normal(size=(5, 11)))
        self.assertEqual(p.shape, (5, 9))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_genoma_ida_y_vuelta(self):
        red = RedMLP(9, [16, 8], 9, semilla=1)
        g = red.obtener_genoma()
        self.assertEqual(g.size, red.n_parametros)
        otra = RedMLP(9, [16, 8], 9, semilla=99).asignar_genoma(g)
        np.testing.assert_allclose(otra.obtener_genoma(), g)

    def test_retropropagacion_reduce_perdida(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
        red = RedMLP(4, [12, 8], 4, semilla=0)
        primera = red.paso_gradiente(X, y, 0.5)
        for _ in range(300):
            ultima = red.paso_gradiente(X, y, 0.5)
        self.assertLess(ultima, primera * 0.5)

    def test_guardar_cargar(self):
        red = RedMLP(11, [6, 4], 9, semilla=2)
        with tempfile.TemporaryDirectory() as d:
            red.guardar(Path(d) / "r.npz", {"x": 1})
            c = RedMLP.cargar(Path(d) / "r.npz")
        x = np.ones((1, 11))
        np.testing.assert_allclose(red.adelante(x), c.adelante(x))
        self.assertEqual(c.meta["x"], 1)


class TestMundo(unittest.TestCase):
    def test_dimensiones_de_observacion(self):
        m = Mundo(ESCENARIOS["C"], 1)
        self.assertEqual(m.observar_presas().shape, (m.cfg.n_presas, len(ENTRADAS_PRESA)))
        self.assertEqual(m.observar_depredadores().shape, (m.cfg.n_depredadores, len(ENTRADAS_DEPREDADOR)))

    def test_vector_a_accion(self):
        for k in range(1, 9):
            self.assertEqual(vector_a_accion(DIRECCIONES[k][None, :])[0], k)
        self.assertEqual(vector_a_accion(np.zeros((1, 2)))[0], 0)

    def test_misma_semilla_mismo_episodio(self):
        a = ejecutar_episodio(MINI, PoliticaAleatoria(), PoliticaReglasDepredador(), 5)
        b = ejecutar_episodio(MINI, PoliticaAleatoria(), PoliticaReglasDepredador(), 5)
        self.assertEqual(a, b)

    def test_captura(self):
        m = Mundo(ConfigMundo(n_presas=1, n_depredadores=1, n_plantas=0), 0)
        m.depr_pos[0] = m.presas_pos[0] + [1.0, 0]
        m.paso_simulacion([0], [0])
        self.assertFalse(m.presas_viva[0])
        self.assertEqual(m.metricas()["capturas"], 1)

    def test_comer_y_hambre(self):
        cfg = ConfigMundo(n_presas=1, n_depredadores=0, n_plantas=1, energia_inicial_presa=0.5)
        m = Mundo(cfg, 0)
        m.plantas_pos[0] = m.presas_pos[0]
        m.paso_simulacion([0], [])
        self.assertAlmostEqual(m.presas_energia[0], 0.5 - cfg.costo_quieto_presa + cfg.energia_planta)
        m.presas_energia[0] = 0.001
        m.paso_simulacion([1], [])
        self.assertFalse(m.presas_viva[0])
        self.assertEqual(m.presas_causa[0], "hambre")

    def test_obstaculo_bloquea(self):
        m = Mundo(ESCENARIOS["C"], 3)
        self.assertFalse(m._dentro_obstaculo(m.presas_pos).any())
        for _ in range(100):
            m.paso_simulacion(np.random.default_rng(0).integers(0, 9, len(m.presas_pos)),
                              np.zeros(len(m.depr_pos), int))
            self.assertFalse(m._dentro_obstaculo(m.presas_pos[m.presas_viva]).any())


class TestDecisionesDeLaRed(unittest.TestCase):
    def test_la_accion_sale_de_la_red(self):
        """La acción ejecutada es exactamente el argmax de la red (no una animación fija)."""
        red = crear_red("presa", [16, 12], semilla=4)
        pol = PoliticaRed(red)
        m = Mundo(MINI, 2)
        obs = m.observar_presas()
        np.testing.assert_array_equal(pol(obs), red.adelante(obs).argmax(axis=1))

    def test_cambiar_pesos_cambia_comportamiento(self):
        obs = Mundo(MINI, 2).observar_presas()
        a = PoliticaRed(crear_red("presa", [16, 12], semilla=1))(obs)
        b = PoliticaRed(crear_red("presa", [16, 12], semilla=2))(obs)
        self.assertFalse(np.array_equal(a, b))

    def test_reglas_huyen(self):
        obs = np.zeros((1, len(ENTRADAS_PRESA)))
        obs[0, 2] = 1.0           # sin planta
        obs[0, 3:6] = [0.3, 0, 0.3]  # depredador al este, cerca
        self.assertEqual(PoliticaReglasPresa()(obs)[0], 5)  # huye al oeste


class TestEvolucion(unittest.TestCase):
    def test_entrenamiento_cambia_parametros(self):
        cfg = ConfigEntrenamiento(poblacion=6, generaciones=3, elites=1, episodios_por_genoma=1,
                                  escenarios_entrenamiento=["A"])
        evo = Neuroevolucion("presa", [6, 4], cfg)
        res = evo.evolucionar([("reglas", None)])
        self.assertEqual(len(res.historial), 3)
        self.assertGreater(np.linalg.norm(res.mejor_genoma - res.genoma_inicial), 0)
        # elitismo: la aptitud media final no debe ser un número inválido
        self.assertTrue(np.isfinite(res.historial[-1]["aptitud_media"]))


if __name__ == "__main__":
    unittest.main()
