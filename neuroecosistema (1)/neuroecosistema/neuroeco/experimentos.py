"""Batería de experimentos reproducibles.

Diseño (lo que cambia y lo que se mantiene en cada comparación):

  E1 Presas      – varía la política de las PRESAS; depredadores fijos = reglas.
  E2 Depredadores – varía la política de los DEPREDADORES; presas fijas = reglas.
  E3 Ecosistema  – red vs red (antes y después de entrenar).

En todos: mismos 3 escenarios, mismas semillas de evaluación (distintas de las
de entrenamiento) y mismos parámetros del mundo. Así la única diferencia entre
filas de una misma tabla es la política evaluada.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .config import DESCRIPCION_ESCENARIOS, ESCENARIOS
from .politicas import (PoliticaAleatoria, PoliticaRed, PoliticaReglasDepredador,
                        PoliticaReglasPresa)
from .red import RedMLP
from .simulacion import ejecutar_episodio

METRICAS = ["supervivencia_presas", "presas_vivas_final", "comida_consumida", "comida_por_presa",
            "capturas", "capturas_por_depredador", "acercamiento_depredadores", "muertes_hambre_presas", "energia_final_presas",
            "energia_final_depredadores", "acercamiento_depredadores", "depredadores_vivos_final", "tiempo_primera_captura",
            "tiempo_medio_captura", "pasos"]

SEMILLAS_EVALUACION = list(range(1000, 1010))  # nunca se usan al entrenar (entrenamiento usa >= 10000)


def cargar_modelos(dir_modelos: Path, arquitecturas=("base", "pequena")):
    """Devuelve {(arquitectura, especie, estado): RedMLP} con lo que exista en disco."""
    modelos = {}
    for arq in arquitecturas:
        for esp in ("presa", "depredador"):
            for estado, archivo in (("entrenada", f"{esp}.npz"), ("sin_entrenar", f"{esp}_sin_entrenar.npz"),
                                    ("imitacion", f"{esp}_imitacion.npz")):
                ruta = Path(dir_modelos) / arq / archivo
                if ruta.exists():
                    modelos[(arq, esp, estado)] = RedMLP.cargar(ruta)
    return modelos


def politicas_experimento(modelos):
    """Construye las tres tablas de comparación a partir de los modelos disponibles."""
    def red(arq, esp, estado):
        m = modelos.get((arq, esp, estado))
        return PoliticaRed(m) if m is not None else None

    e1 = {"aleatoria": PoliticaAleatoria(), "reglas": PoliticaReglasPresa(),
          "red sin entrenar": red("base", "presa", "sin_entrenar"),
          "red solo imitación": red("base", "presa", "imitacion"),
          "red base": red("base", "presa", "entrenada"),
          "red pequeña": red("pequena", "presa", "entrenada")}
    e2 = {"aleatoria": PoliticaAleatoria(), "reglas": PoliticaReglasDepredador(),
          "red sin entrenar": red("base", "depredador", "sin_entrenar"),
          "red solo imitación": red("base", "depredador", "imitacion"),
          "red base": red("base", "depredador", "entrenada"),
          "red pequeña": red("pequena", "depredador", "entrenada")}
    e3 = {}
    if red("base", "presa", "entrenada") is not None:
        e3 = {"redes sin entrenar": (red("base", "presa", "sin_entrenar"), red("base", "depredador", "sin_entrenar")),
              "redes entrenadas": (red("base", "presa", "entrenada"), red("base", "depredador", "entrenada")),
              "reglas vs reglas": (PoliticaReglasPresa(), PoliticaReglasDepredador())}
    e1 = {k: v for k, v in e1.items() if v is not None}
    e2 = {k: v for k, v in e2.items() if v is not None}
    return {
        "E1_presas": {k: (v, PoliticaReglasDepredador()) for k, v in e1.items()},
        "E2_depredadores": {k: (PoliticaReglasPresa(), v) for k, v in e2.items()},
        "E3_ecosistema": e3,
    }


def correr(experimentos: dict, semillas=SEMILLAS_EVALUACION, escenarios=("A", "B", "C"), log=print):
    filas = []
    for nombre_exp, politicas in experimentos.items():
        for esc in escenarios:
            for nombre_pol, (pp, pd) in politicas.items():
                for s in semillas:
                    m = ejecutar_episodio(ESCENARIOS[esc], pp, pd, s)
                    filas.append({"experimento": nombre_exp, "escenario": esc, "politica": nombre_pol,
                                  "semilla": s, **{k: m[k] for k in METRICAS}})
                ult = [f for f in filas[-len(semillas):]]
                log(f"  {nombre_exp:16s} {esc}  {nombre_pol:18s} supervivencia={np.mean([f['supervivencia_presas'] for f in ult]):.3f}  "
                    f"capturas={np.mean([f['capturas'] for f in ult]):.1f}  comida={np.mean([f['comida_consumida'] for f in ult]):.1f}")
    return filas


def resumir(filas):
    """Media, desviación estándar y n por (experimento, escenario, política). NaN se ignora."""
    claves = []
    for f in filas:
        k = (f["experimento"], f["escenario"], f["politica"])
        if k not in claves:
            claves.append(k)
    resumen = []
    for exp, esc, pol in claves:
        g = [f for f in filas if (f["experimento"], f["escenario"], f["politica"]) == (exp, esc, pol)]
        fila = {"experimento": exp, "escenario": esc, "politica": pol, "n": len(g)}
        for m in METRICAS:
            v = np.array([x[m] for x in g], dtype=float)
            v = v[np.isfinite(v)]
            fila[f"{m}_media"] = float(v.mean()) if len(v) else float("nan")
            fila[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            fila[f"{m}_n_validos"] = int(len(v))
        resumen.append(fila)
    return resumen


def prueba_t_welch(a, b):
    """t de Welch y p-valor bilateral aproximado (normal) — sin depender de scipy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p)


def guardar_csv(filas, ruta: Path):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def _fmt(media, std):
    return "—" if not np.isfinite(media) else f"{media:.2f} ± {std:.2f}"


def tabla_md(resumen, experimento, metricas):
    filas = [r for r in resumen if r["experimento"] == experimento]
    if not filas:
        return "_(sin datos)_\n"
    cab = "| Escenario | Política | " + " | ".join(metricas) + " |\n"
    cab += "|---|---|" + "---|" * len(metricas) + "\n"
    for r in filas:
        cab += f"| {r['escenario']} | {r['politica']} | " + " | ".join(
            _fmt(r[f"{m}_media"], r[f"{m}_std"]) for m in metricas) + " |\n"
    return cab


def comparaciones(filas, experimento, metrica, a, b):
    """Tabla de diferencias a − b por escenario con t de Welch."""
    out = f"| Escenario | {a} | {b} | Diferencia | t de Welch | p aprox. |\n|---|---|---|---|---|---|\n"
    for esc in ("A", "B", "C"):
        va = [f[metrica] for f in filas if f["experimento"] == experimento and f["escenario"] == esc and f["politica"] == a]
        vb = [f[metrica] for f in filas if f["experimento"] == experimento and f["escenario"] == esc and f["politica"] == b]
        if not va or not vb:
            continue
        t, p = prueba_t_welch(va, vb)
        out += (f"| {esc} | {np.nanmean(va):.3f} | {np.nanmean(vb):.3f} | {np.nanmean(va) - np.nanmean(vb):+.3f} | "
                f"{t:.2f} | {p:.3g} |\n")
    return out


def escribir_reporte(filas, resumen, ruta: Path, info: dict):
    s = "# Resultados experimentales — NeuroEcosistema\n\n"
    s += "_Archivo generado automáticamente por `evaluar.py`. No editar a mano; volver a ejecutar._\n\n"
    s += f"- Semillas de evaluación: `{info['semillas']}` ({len(info['semillas'])} repeticiones por celda)\n"
    s += f"- Modelos: `{info['modelos']}`\n- Escenarios:\n"
    for k, d in DESCRIPCION_ESCENARIOS.items():
        c = ESCENARIOS[k]
        s += (f"  - **{k}** — {d}: {c.n_presas} presas, {c.n_depredadores} depredadores, {c.n_plantas} plantas "
              f"(regeneran en {c.regeneracion_plantas} pasos), {c.n_obstaculos} obstáculos, {c.max_pasos} pasos máx.\n")
    s += "\nFormato de celdas: media ± desviación estándar entre semillas. `—` = no ocurrió (p. ej. sin capturas).\n"

    s += "\n## E1 — Presas (depredadores fijos = reglas)\n\n"
    s += "Variable independiente: política de las presas. Controladas: escenario, semillas, depredadores por reglas.\n\n"
    s += tabla_md(resumen, "E1_presas", ["supervivencia_presas", "presas_vivas_final", "comida_por_presa",
                                          "muertes_hambre_presas", "capturas", "tiempo_medio_captura"])
    s += "\n![E1](e1_supervivencia_presas.png)\n"
    s += "\n### Antes vs después del entrenamiento (supervivencia de presas)\n\n"
    s += comparaciones(filas, "E1_presas", "supervivencia_presas", "red base", "red sin entrenar")
    s += "\n### Efecto de la neuroevolución tras la imitación (supervivencia de presas)\n\n"
    s += comparaciones(filas, "E1_presas", "supervivencia_presas", "red base", "red solo imitación")
    s += "\n### Red entrenada vs referencia por reglas (supervivencia de presas)\n\n"
    s += comparaciones(filas, "E1_presas", "supervivencia_presas", "red base", "reglas")
    s += "\n### Red entrenada vs referencia aleatoria (supervivencia de presas)\n\n"
    s += comparaciones(filas, "E1_presas", "supervivencia_presas", "red base", "aleatoria")
    s += "\n### Arquitectura base vs pequeña (supervivencia de presas)\n\n"
    s += comparaciones(filas, "E1_presas", "supervivencia_presas", "red base", "red pequeña")

    s += "\n## E2 — Depredadores (presas fijas = reglas)\n\n"
    s += "Variable independiente: política de los depredadores. Controladas: escenario, semillas, presas por reglas.\n\n"
    s += tabla_md(resumen, "E2_depredadores", ["capturas_por_depredador", "energia_final_depredadores",
                                                "acercamiento_depredadores", "depredadores_vivos_final", "tiempo_primera_captura",
                                                "tiempo_medio_captura"])
    s += "\n![E2](e2_capturas.png)\n"
    s += "\n### Antes vs después del entrenamiento (capturas por depredador)\n\n"
    s += comparaciones(filas, "E2_depredadores", "capturas_por_depredador", "red base", "red sin entrenar")
    s += "\n### Efecto de la neuroevolución tras la imitación (capturas por depredador)\n\n"
    s += comparaciones(filas, "E2_depredadores", "capturas_por_depredador", "red base", "red solo imitación")
    s += "\n### Red entrenada vs reglas (capturas por depredador)\n\n"
    s += comparaciones(filas, "E2_depredadores", "capturas_por_depredador", "red base", "reglas")
    s += "\n### Arquitectura base vs pequeña (capturas por depredador)\n\n"
    s += comparaciones(filas, "E2_depredadores", "capturas_por_depredador", "red base", "red pequeña")

    s += "\n## E3 — Ecosistema completo (ambas especies con la misma clase de política)\n\n"
    s += tabla_md(resumen, "E3_ecosistema", ["supervivencia_presas", "comida_consumida", "capturas",
                                              "energia_final_presas", "energia_final_depredadores"])
    s += "\n![E3](e3_ecosistema.png)\n"
    s += ("\n> p aprox. usa la aproximación normal de la t de Welch (n=10 por grupo); tómese como orientativo, "
          "no como prueba formal. Lo importante es el tamaño de la diferencia frente a la desviación estándar.\n")
    Path(ruta).write_text(s, encoding="utf-8")
