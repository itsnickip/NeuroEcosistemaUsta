"""Gráficas de entrenamiento y de experimentos (matplotlib, PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Paleta categórica en orden fijo (validada para daltonismo); el color sigue a la política.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA, TINTA_2, GRILLA, EJE, FONDO = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": TINTA,
    "axes.titlesize": 11, "font.size": 9, "legend.frameon": False,
})


def curvas_entrenamiento(filas: list[dict], ruta: Path, titulo: str):
    """Aptitud (mejor/media) y cambio de pesos por generación, una columna por especie."""
    especies = sorted({f["especie"] for f in filas}, key=lambda e: e != "presa")
    fig, ejes = plt.subplots(2, len(especies), figsize=(5.5 * len(especies), 6.5), squeeze=False)
    for j, esp in enumerate(especies):
        datos = [f for f in filas if f["especie"] == esp]
        rondas = sorted({f["ronda"] for f in datos})
        x0 = 0
        for r in rondas:
            d = [f for f in datos if f["ronda"] == r]
            x = np.arange(x0, x0 + len(d))
            ax = ejes[0, j]
            ax.plot(x, [f["aptitud_mejor"] for f in d], color=COLORES[0], lw=2, label="mejor" if r == 0 else None)
            ax.plot(x, [f["aptitud_media"] for f in d], color=COLORES[1], lw=2, label="media" if r == 0 else None)
            ejes[1, j].plot(x, [f["distancia_a_gen0"] for f in d], color=COLORES[2], lw=2,
                            label="‖w − w₀‖ (vs. mejor gen 0)" if r == 0 else None)
            ejes[1, j].plot(x, [f["cambio_vs_gen_anterior"] for f in d], color=COLORES[3], lw=2,
                            label="‖w_t − w_{t−1}‖" if r == 0 else None)
            if r > 0:
                for ax_ in ejes[:, j]:
                    ax_.axvline(x0 - 0.5, color=EJE, lw=1, ls="--")
                ejes[0, j].text(x0, ejes[0, j].get_ylim()[1], f" ronda {r}", va="top", color=TINTA_2, fontsize=8)
            x0 += len(d)
        ejes[0, j].set_title(f"{esp.capitalize()}: aptitud por generación")
        ejes[0, j].set_ylabel("aptitud")
        ejes[1, j].set_title(f"{esp.capitalize()}: cambio en los pesos")
        ejes[1, j].set_xlabel("generación (acumulada entre rondas)")
        ejes[1, j].set_ylabel("norma L2")
        ejes[0, j].legend(loc="lower right")
        ejes[1, j].legend(loc="upper left")
    fig.suptitle(titulo, color=TINTA, fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def barras_por_escenario(resumen: list[dict], metrica: str, politicas: list[str], ruta: Path,
                         titulo: str, ylabel: str, formato: str = "{:.2f}"):
    """Barras agrupadas: escenario en x, una barra por política, error = desviación estándar."""
    escenarios = sorted({r["escenario"] for r in resumen})
    fig, ax = plt.subplots(figsize=(1.6 + 1.35 * len(escenarios) * max(1, len(politicas)) / 2.2, 4.2))
    ancho = 0.8 / len(politicas)
    for i, pol in enumerate(politicas):
        medias, stds = [], []
        for esc in escenarios:
            fila = next((r for r in resumen if r["escenario"] == esc and r["politica"] == pol), None)
            medias.append(fila[f"{metrica}_media"] if fila else np.nan)
            stds.append(fila[f"{metrica}_std"] if fila else 0)
        x = np.arange(len(escenarios)) + (i - (len(politicas) - 1) / 2) * ancho
        barras = ax.bar(x, medias, ancho * 0.9, yerr=stds, color=COLORES[i % len(COLORES)], label=pol,
                        error_kw={"ecolor": TINTA_2, "elinewidth": 1, "capsize": 2})
        for b, m, sd in zip(barras, medias, stds):
            if np.isfinite(m):
                ax.text(b.get_x() + b.get_width() / 2, m + (sd if np.isfinite(sd) else 0), formato.format(m),
                        ha="center", va="bottom", fontsize=7, color=TINTA_2)
    ax.set_xticks(np.arange(len(escenarios)), [f"Escenario {e}" for e in escenarios])
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.margins(y=0.12)
    ax.legend(ncols=len(politicas), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def curva_imitacion(hist: dict, ruta: Path):
    """Pérdida y exactitud en prueba del pre-entrenamiento supervisado, por especie."""
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (esp, h) in enumerate(hist.items()):
        x = [r["epoca"] for r in h]
        ejes[0].plot(x, [r["perdida"] for r in h], color=COLORES[i], lw=2, label=esp)
        ejes[1].plot(x, [r["exactitud_prueba"] for r in h], color=COLORES[i], lw=2, label=esp)
    ejes[0].set_title("Imitación · pérdida (entropía cruzada)")
    ejes[1].set_title("Imitación · exactitud frente a la regla (datos de prueba)")
    for ax in ejes:
        ax.set_xlabel("época")
        ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
