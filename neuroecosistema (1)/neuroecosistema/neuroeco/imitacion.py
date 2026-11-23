"""Pre-entrenamiento supervisado por imitación (behavioral cloning).

1. Se juegan episodios con las políticas por reglas en los escenarios A, B y C.
2. Cada paso se guarda (observación, acción elegida por la regla) de la especie.
3. La red aprende a predecir esa acción con entropía cruzada + retropropagación.

Así la red parte de un comportamiento razonable (ir hacia la presa, huir...) y
la neuroevolución posterior ajusta a partir de ahí. Es la misma idea del taller
del laberinto (BFS = profesor), aplicada al ecosistema.
"""
from __future__ import annotations

import numpy as np

from .config import ESCENARIOS
from .mundo import Mundo
from .politicas import PoliticaReglasDepredador, PoliticaReglasPresa
from .red import RedMLP


def generar_datos(especie: str, semillas, escenarios=("A", "B", "C"), max_muestras=40_000, semilla=0):
    pres, depr = PoliticaReglasPresa(), PoliticaReglasDepredador()
    X, y = [], []
    for esc in escenarios:
        for s in semillas:
            m = Mundo(ESCENARIOS[esc], s)
            pres.reiniciar(s)
            depr.reiniciar(s)
            while not m.terminado:
                op, od = m.observar_presas(), m.observar_depredadores()
                ap = np.zeros(len(op), int)
                ad = np.zeros(len(od), int)
                if m.presas_viva.any():
                    ap[m.presas_viva] = pres(op[m.presas_viva])
                if m.depr_viva.any():
                    ad[m.depr_viva] = depr(od[m.depr_viva])
                if especie == "presa":
                    X.append(op[m.presas_viva]); y.append(ap[m.presas_viva])
                else:
                    X.append(od[m.depr_viva]); y.append(ad[m.depr_viva])
                m.paso_simulacion(ap, ad)
    X, y = np.concatenate(X), np.concatenate(y)
    rng = np.random.default_rng(semilla)
    idx = rng.permutation(len(X))[:max_muestras]
    return X[idx], y[idx]


def entrenar_imitacion(red: RedMLP, X, y, epocas=60, tasa=0.3, lote=256, semilla=0, fraccion_prueba=0.2, log=print):
    """Entrena `red` en sitio. Devuelve historial por época (pérdida y exactitud en prueba)."""
    rng = np.random.default_rng(semilla)
    corte = int(len(X) * (1 - fraccion_prueba))
    Xtr, ytr, Xte, yte = X[:corte], y[:corte], X[corte:], y[corte:]
    hist = []
    for ep in range(epocas):
        orden = rng.permutation(len(Xtr))
        perdidas = [red.paso_gradiente(Xtr[orden[i:i + lote]], ytr[orden[i:i + lote]], tasa)
                    for i in range(0, len(orden), lote)]
        acc = float(np.mean(red.adelante(Xte).argmax(axis=1) == yte))
        hist.append({"epoca": ep, "perdida": float(np.mean(perdidas)), "exactitud_prueba": acc})
        if ep % 10 == 0 or ep == epocas - 1:
            log(f"  [imitación] época {ep:3d}  pérdida={hist[-1]['perdida']:.3f}  exactitud prueba={acc:.3f}")
    return hist
