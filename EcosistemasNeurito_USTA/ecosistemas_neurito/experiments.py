"""Experimentos emparejados y repetibles con líneas base explícitas."""

from __future__ import annotations

import csv
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import statistics

import numpy as np

from .network import RedNeurito
from .policies import rule_action
from .world import ACTIONS, SCENARIOS, World


CASES = ("inicial", "entrenado", "arquitectura_b", "reglas", "aleatorio")
SPECIES = ("prey", "predator")
METRICS = (
    "prey_survival_mean",
    "prey_food_total",
    "predator_captures_total",
    "prey_energy_mean",
    "predator_energy_mean",
    "prey_alive",
    "predator_alive",
    "time_to_first_capture",
    "capture_occurred",
    "episode_steps",
)


def load_candidates(model_dir: str | Path) -> dict[tuple[str, str], RedNeurito]:
    directory = Path(model_dir)
    if not (directory / "entrenamiento.json").exists():
        raise FileNotFoundError(
            f"Falta {directory / 'entrenamiento.json'}; ejecuta primero 'train'."
        )
    result = {}
    for species in SPECIES:
        for case, filename in (
            ("inicial", f"{species}_base_inicial.npz"),
            ("entrenado", f"{species}_base_entrenado.npz"),
            ("arquitectura_b", f"{species}_alternativa_entrenado.npz"),
        ):
            result[(species, case)] = RedNeurito.load(directory / filename)
    return result


def _actions(
    world: World,
    species: str,
    case: str,
    models: dict[tuple[str, str], RedNeurito],
    rng: np.random.Generator,
) -> list[int]:
    agents = world.prey if species == "prey" else world.predators
    chosen = []
    for index, agent in enumerate(agents):
        if not agent.alive:
            chosen.append(0)
        elif case == "reglas":
            chosen.append(int(rule_action(world, species, index)))
        elif case == "aleatorio":
            chosen.append(int(rng.integers(len(ACTIONS))))
        else:
            observation = world.observe(species, index).reshape(1, -1)
            prediction = models[(species, case)].predict_actions(observation)
            chosen.append(int(prediction[0]))
    return chosen


def run_episode(
    scenario: str,
    seed: int,
    evaluated_species: str,
    case: str,
    models: dict[tuple[str, str], RedNeurito],
) -> dict:
    """Change one policy; keep the other species on its fixed rule baseline."""
    if scenario not in SCENARIOS or evaluated_species not in SPECIES or case not in CASES:
        raise ValueError("Escenario, especie o caso no válido")
    world = World(SCENARIOS[scenario], seed=seed)
    random_rng = np.random.default_rng(seed + 900_001)
    while not world.done:
        prey_case = case if evaluated_species == "prey" else "reglas"
        predator_case = case if evaluated_species == "predator" else "reglas"
        prey_actions = _actions(world, "prey", prey_case, models, random_rng)
        predator_actions = _actions(world, "predator", predator_case, models, random_rng)
        world.step(prey_actions, predator_actions)
    values = world.metrics()
    values["capture_occurred"] = int(values["time_to_first_capture"] is not None)
    return {
        "scenario": scenario,
        "evaluated_species": evaluated_species,
        "case": case,
        "seed": seed,
        **{metric: values.get(metric) for metric in METRICS},
    }


def _summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    for scenario in SCENARIOS:
        for species in SPECIES:
            for case in CASES:
                subset = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["evaluated_species"] == species
                    and row["case"] == case
                ]
                for metric in METRICS:
                    values = [
                        float(row[metric])
                        for row in subset
                        if row[metric] is not None and row[metric] != ""
                    ]
                    summaries.append(
                        {
                            "scenario": scenario,
                            "evaluated_species": species,
                            "case": case,
                            "metric": metric,
                            "n": len(values),
                            "mean": statistics.fmean(values) if values else "",
                            "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else "",
                        }
                    )
    return summaries


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _svg_chart(
    summaries: list[dict], species: str, metric: str, title: str, destination: Path
) -> None:
    relevant = {
        (row["scenario"], row["case"]): float(row["mean"])
        for row in summaries
        if row["evaluated_species"] == species
        and row["metric"] == metric
        and row["mean"] != ""
    }
    if not relevant:
        return
    width, height = 1000, 450
    left, top, bottom = 74, 65, 330
    max_value = max(relevant.values()) * 1.13 or 1
    colors = {
        "inicial": "#8a97af",
        "entrenado": "#ffbf48",
        "arquitectura_b": "#5ccdff",
        "reglas": "#8ade5c",
        "aleatorio": "#ff6880",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#0f131e"/>',
        f'<text x="{left}" y="34" font-family="Verdana" font-size="20" font-weight="bold" fill="#eaeef6">{escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="960" y2="{bottom}" stroke="#8a97af"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = bottom - (bottom - top) * tick / 4
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="960" y2="{y:.1f}" stroke="#25304a"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Verdana" font-size="11" fill="#8a97af">{value:.1f}</text>'
        )
    scenarios = tuple(SCENARIOS)
    for group, scenario in enumerate(scenarios):
        start_x = 107 + group * 285
        for offset, case in enumerate(CASES):
            value = relevant.get((scenario, case), 0.0)
            bar_height = value / max_value * (bottom - top)
            x = start_x + offset * 42
            y = bottom - bar_height
            elements.append(
                f'<rect x="{x}" y="{y:.1f}" width="32" height="{bar_height:.1f}" rx="6" fill="{colors[case]}"/>'
            )
        elements.append(
            f'<text x="{start_x + 100}" y="352" text-anchor="middle" font-family="Verdana" font-size="13" fill="#eaeef6">{escape(scenario)}</text>'
        )
    for index, case in enumerate(CASES):
        x = 95 + index * 177
        elements.append(f'<rect x="{x}" y="391" width="14" height="14" rx="7" fill="{colors[case]}"/>')
        elements.append(
            f'<text x="{x + 20}" y="403" font-family="Verdana" font-size="11" fill="#eaeef6">{escape(case)}</text>'
        )
    elements.append('</svg>')
    destination.write_text("\n".join(elements), encoding="utf-8")


def run_experiments(
    model_dir: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808),
) -> tuple[list[dict], list[dict]]:
    """Run the complete, paired matrix and persist raw + aggregate evidence."""
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Se requieren al menos dos semillas distintas")
    models = load_candidates(model_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for species in SPECIES:
            for case in CASES:
                for seed in seeds:
                    rows.append(run_episode(scenario, seed, species, case, models))
    summaries = _summaries(rows)
    _write_csv(
        destination / "episodios.csv",
        rows,
        ["scenario", "evaluated_species", "case", "seed", *METRICS],
    )
    _write_csv(
        destination / "resumen.csv",
        summaries,
        ["scenario", "evaluated_species", "case", "metric", "n", "mean", "std"],
    )
    metadata = {
        "scenarios": list(SCENARIOS),
        "scenario_parameters": {
            name: asdict(config) for name, config in SCENARIOS.items()
        },
        "evaluated_species": list(SPECIES),
        "cases": list(CASES),
        "seeds": list(seeds),
        "fixed_opponent_policy": "reglas",
        "metrics": list(METRICS),
        "comparison_design": "una politica candidata por especie; oponente basado en reglas fijo; semillas emparejadas",
    }
    (destination / "configuracion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _svg_chart(
        summaries,
        "prey",
        "prey_survival_mean",
        "EcosistemasNeurito · Supervivencia media de presas",
        destination / "supervivencia_presas.svg",
    )
    _svg_chart(
        summaries,
        "predator",
        "predator_captures_total",
        "EcosistemasNeurito · Capturas de depredadores",
        destination / "capturas_depredadores.svg",
    )
    return rows, summaries
