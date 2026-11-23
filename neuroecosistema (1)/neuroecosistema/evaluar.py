"""Corre los experimentos E1, E2 y E3 en los tres escenarios y genera la evidencia.

    uv run evaluar.py                     # 10 semillas por celda
    uv run evaluar.py --semillas 20       # más repeticiones

Salida en resultados/:
    episodios.csv     una fila por episodio (datos crudos, con semilla)
    resumen.csv       media/desviación por experimento, escenario y política
    RESULTADOS.md     tablas + comparaciones (antes/después, arquitecturas, referencias)
    *.png             gráficas
    config.json       configuración exacta usada
"""
import argparse
from pathlib import Path

from neuroeco.config import ESCENARIOS, guardar_json
from neuroeco.experimentos import (cargar_modelos, correr, escribir_reporte, guardar_csv,
                                   politicas_experimento, resumir)
from neuroeco.graficas import barras_por_escenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="modelos")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--semillas", type=int, default=10)
    a = ap.parse_args()

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    semillas = list(range(1000, 1000 + a.semillas))
    modelos = cargar_modelos(Path(a.modelos))
    if not modelos:
        print("Aviso: no hay modelos entrenados; solo se evaluarán las referencias. Ejecuta antes: uv run entrenar.py")
    for (arq, esp, est), red in modelos.items():
        print(f"  modelo {arq}/{esp}/{est}: {red}")

    exps = politicas_experimento(modelos)
    filas = correr(exps, semillas)
    resumen = resumir(filas)
    guardar_csv(filas, salida / "episodios.csv")
    guardar_csv(resumen, salida / "resumen.csv")

    pol_e1 = list(exps["E1_presas"])
    pol_e2 = list(exps["E2_depredadores"])
    pol_e3 = list(exps["E3_ecosistema"])
    r = lambda exp: [x for x in resumen if x["experimento"] == exp]  # noqa: E731
    barras_por_escenario(r("E1_presas"), "supervivencia_presas", pol_e1, salida / "e1_supervivencia_presas.png",
                         "E1 · Supervivencia media de presas (depredadores = reglas)", "fracción de pasos con vida")
    barras_por_escenario(r("E1_presas"), "comida_por_presa", pol_e1, salida / "e1_comida_presas.png",
                         "E1 · Alimento consumido por presa", "plantas por presa", "{:.1f}")
    barras_por_escenario(r("E2_depredadores"), "capturas_por_depredador", pol_e2, salida / "e2_capturas.png",
                         "E2 · Capturas por depredador (presas = reglas)", "capturas por depredador", "{:.1f}")
    barras_por_escenario(r("E2_depredadores"), "energia_final_depredadores", pol_e2, salida / "e2_energia.png",
                         "E2 · Energía final de depredadores", "energía (0–1)")
    if pol_e3:
        barras_por_escenario(r("E3_ecosistema"), "supervivencia_presas", pol_e3, salida / "e3_ecosistema.png",
                             "E3 · Supervivencia de presas en el ecosistema completo", "fracción de pasos con vida")

    info = {"semillas": semillas, "modelos": str(a.modelos),
            "politicas": {k: list(v) for k, v in exps.items()},
            "escenarios": {k: v.to_dict() for k, v in ESCENARIOS.items()},
            "metadatos_modelos": {f"{k[0]}/{k[1]}/{k[2]}": getattr(v, "meta", {}) for k, v in modelos.items()}}
    guardar_json(info, salida / "config.json")
    escribir_reporte(filas, resumen, salida / "RESULTADOS.md", info)
    print(f"\nListo: {salida}/RESULTADOS.md")


if __name__ == "__main__":
    main()
