"""Entrena las redes de presas y depredadores por neuroevolución.

Ejemplos:
    uv run entrenar.py                              # arquitectura base, configuración por defecto
    uv run entrenar.py --arquitectura pequena       # segunda arquitectura para comparar
    uv run entrenar.py --rapido                     # prueba corta (pocas generaciones)

Salida en modelos/<arquitectura>/:
    presa.npz, depredador.npz                       redes entrenadas
    presa_sin_entrenar.npz, depredador_sin_entrenar.npz   pesos iniciales (comparación antes/después)
    historial.csv, entrenamiento.png                evidencia de cambio de parámetros
    config.json                                     configuración y semillas usadas
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

from neuroeco.config import ARQUITECTURAS, ESCENARIOS, ConfigEntrenamiento, guardar_json
from neuroeco.evolucion import Neuroevolucion, crear_red, guardar_historial
from neuroeco.graficas import curvas_entrenamiento, curva_imitacion
from neuroeco.imitacion import entrenar_imitacion, generar_datos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arquitectura", default="base", choices=list(ARQUITECTURAS))
    ap.add_argument("--generaciones", type=int)
    ap.add_argument("--poblacion", type=int)
    ap.add_argument("--rondas", type=int, help="rondas de co-evolución (1 = solo contra reglas)")
    ap.add_argument("--semilla", type=int)
    ap.add_argument("--workers", type=int, default=2, help="procesos en paralelo")
    ap.add_argument("--salida", default="modelos")
    ap.add_argument("--inicializacion", choices=["imitacion", "aleatoria"],
                    help="imitacion (por defecto) = supervisado + evolución; aleatoria = neuroevolución pura")
    ap.add_argument("--sigma-mutacion", type=float, help="desviación de la mutación gaussiana (defecto 0.10)")
    ap.add_argument("--episodios", type=int, help="episodios por genoma y generación (defecto 3)")
    ap.add_argument("--rapido", action="store_true", help="6 generaciones, población 10, 1 ronda")
    a = ap.parse_args()

    cfg = ConfigEntrenamiento(arquitectura=a.arquitectura)
    if a.rapido:
        cfg = replace(cfg, generaciones=6, poblacion=10, rondas_coevolucion=1)
    cambios = {k: v for k, v in {"generaciones": a.generaciones, "poblacion": a.poblacion,
                                  "rondas_coevolucion": a.rondas, "semilla": a.semilla,
                                  "inicializacion": a.inicializacion, "sigma_mutacion": a.sigma_mutacion,
                                  "episodios_por_genoma": a.episodios}.items() if v is not None}
    cfg = replace(cfg, **cambios)
    capas = ARQUITECTURAS[cfg.arquitectura]
    salida = Path(a.salida) / cfg.arquitectura
    salida.mkdir(parents=True, exist_ok=True)

    print(f"Arquitectura '{cfg.arquitectura}': presa {capas['presa']}, depredador {capas['depredador']}")
    for esp in ("presa", "depredador"):
        print(f"  {esp}: {crear_red(esp, capas[esp], cfg.activacion)}")

    evo_presa = Neuroevolucion("presa", capas["presa"], cfg, a.workers)
    evo_depr = Neuroevolucion("depredador", capas["depredador"], cfg, a.workers)
    poblaciones = {}
    hist_imit = {}
    for esp, evo in (("presa", evo_presa), ("depredador", evo_depr)):
        # "antes del entrenamiento": red con pesos aleatorios (semilla registrada)
        red0 = crear_red(esp, capas[esp], cfg.activacion, cfg.semilla * 1000 + (0 if esp == "presa" else 500), cfg.sigma_inicial)
        red0.guardar(salida / f"{esp}_sin_entrenar.npz", {"estado": "sin entrenar", **cfg.to_dict()})
        if cfg.inicializacion == "imitacion":
            print(f"\n=== Pre-entrenamiento supervisado ({esp}) imitando la política por reglas ===")
            X, y = generar_datos(esp, range(20_000, 20_000 + cfg.episodios_imitacion), semilla=cfg.semilla)
            print(f"  {len(X)} ejemplos (observación, acción de la regla)")
            hist_imit[esp] = entrenar_imitacion(red0, X, y, cfg.epocas_imitacion, cfg.tasa_imitacion, semilla=cfg.semilla)
            red0.guardar(salida / f"{esp}_imitacion.npz", {"estado": "solo imitación", **cfg.to_dict()})
            poblaciones[esp] = evo.poblacion_inicial(red0.obtener_genoma())
        else:
            poblaciones[esp] = evo.poblacion_inicial()
    pob_presa, pob_depr = poblaciones["presa"], poblaciones["depredador"]
    if hist_imit:
        filas = [{"especie": e, **h} for e, hs in hist_imit.items() for h in hs]
        guardar_historial(filas, salida / "historial_imitacion.csv")
        curva_imitacion(hist_imit, salida / "imitacion.png")

    historial = []
    reglas = ("reglas", None)
    mejor_presa = mejor_depr = None
    t0 = time.time()
    for ronda in range(cfg.rondas_coevolucion):
        # Ronda 0: cada especie contra la referencia por reglas.
        # Rondas siguientes: mezcla de reglas + la mejor red actual de la otra especie.
        op_presa = [reglas] if mejor_depr is None else [reglas, ("red", mejor_depr, capas["depredador"], cfg.activacion)]
        print(f"\n=== Ronda {ronda}: presas vs {[o[0] for o in op_presa]} ===")
        res = evo_presa.evolucionar(op_presa, pob_presa, ronda)
        mejor_presa, pob_presa = res.mejor_genoma, evo_presa.poblacion_final
        historial += res.historial

        op_depr = [reglas] if ronda == 0 else [reglas, ("red", mejor_presa, capas["presa"], cfg.activacion)]
        print(f"\n=== Ronda {ronda}: depredadores vs {[o[0] for o in op_depr]} ===")
        res = evo_depr.evolucionar(op_depr, pob_depr, ronda)
        mejor_depr, pob_depr = res.mejor_genoma, evo_depr.poblacion_final
        historial += res.historial

    meta = {"estado": "entrenada", **cfg.to_dict()}
    crear_red("presa", capas["presa"], cfg.activacion).asignar_genoma(mejor_presa).guardar(salida / "presa.npz", meta)
    crear_red("depredador", capas["depredador"], cfg.activacion).asignar_genoma(mejor_depr).guardar(salida / "depredador.npz", meta)
    guardar_historial(historial, salida / "historial.csv")
    guardar_json({"entrenamiento": cfg.to_dict(), "capas": capas,
                  "escenarios": {k: v.to_dict() for k, v in ESCENARIOS.items()},
                  "duracion_s": round(time.time() - t0, 1)}, salida / "config.json")
    curvas_entrenamiento(historial, salida / "entrenamiento.png", f"Neuroevolución — arquitectura '{cfg.arquitectura}'")
    print(f"\nListo en {time.time() - t0:.0f}s. Modelos e historial en {salida}/")


if __name__ == "__main__":
    main()
