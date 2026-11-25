"""Entrenamiento supervisado reproducible a partir de políticas de referencia.

Las reglas solo etiquetan observaciones; durante la simulación la red nunca las
consulta. La validación usa semillas de mundo distintas a las de entrenamiento.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .network import RedNeurito
from .policies import rule_action
from .world import ACTIONS, INPUT_SIZE, PREDATOR_INPUTS, PREY_INPUTS, SCENARIOS, World


# Capa de entrada: 8 neuronas (INPUT_SIZE) · capa de salida: 5 neuronas (ACTIONS).
# base        -> 8 → 14 → 16 → 12 → 5  (tres capas ocultas ReLU)
# alternativa -> 8 → 16 → 12 → 5       (dos capas ocultas ReLU)
ARCHITECTURES = {"base": (14, 16, 12), "alternativa": (16, 12)}
SPECIES = ("prey", "predator")

# Hiperparámetros del entrenamiento
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.004


def collect_examples(
    seed: int,
    episodes: int,
    limit_per_species: int,
    steps_per_episode: int = 85,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sample observations from rule and exploratory rollouts.

    Random moves broaden the observed states while labels always come from the
    deterministic reference policy. Worlds and exploration use separate RNGs.
    """
    observations: dict[str, list[np.ndarray]] = {name: [] for name in SPECIES}
    labels: dict[str, list[int]] = {name: [] for name in SPECIES}
    exploration_rng = np.random.default_rng(seed + 7_000_003)
    scenario_names = tuple(SCENARIOS)

    for episode in range(episodes):
        scenario = scenario_names[episode % len(scenario_names)]
        world = World(SCENARIOS[scenario], seed=seed + episode * 37)
        # Include valid low-stamina states. Starting every third rollout low is
        # equivalent to observing a predator after a long chase, and lets both
        # train and validation sets cover the rest/pursuit decision boundary.
        if episode % 3 == 0:
            for predator in world.predators:
                predator.energy = float(exploration_rng.uniform(0.12, 0.24))
        for _ in range(steps_per_episode):
            if world.done:
                break
            moves: dict[str, list[int]] = {}
            for species, agents in (("prey", world.prey), ("predator", world.predators)):
                actions = []
                for index, agent in enumerate(agents):
                    if not agent.alive:
                        actions.append(0)
                        continue
                    reference = int(rule_action(world, species, index))
                    if len(labels[species]) < limit_per_species:
                        observations[species].append(world.observe(species, index))
                        labels[species].append(reference)
                    # Some episodes explore states that the rule policy avoids.
                    exploratory = episode % 3 == 0 and exploration_rng.random() < 0.35
                    actions.append(
                        int(exploration_rng.integers(len(ACTIONS))) if exploratory else reference
                    )
                moves[species] = actions
            world.step(moves["prey"], moves["predator"])
        if all(len(labels[name]) >= limit_per_species for name in SPECIES):
            break

    result = {}
    for species in SPECIES:
        input_size = len(PREY_INPUTS if species == "prey" else PREDATOR_INPUTS)
        if not labels[species]:
            raise RuntimeError(f"No se pudieron generar ejemplos para {species}")
        result[species] = (
            np.asarray(observations[species], dtype=np.float64).reshape(-1, input_size),
            np.asarray(labels[species], dtype=np.int64),
        )
    return result


def train_models(
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = EPOCHS,
    train_episodes: int = 75,
    validation_episodes: int = 20,
) -> dict:
    """Entrena ambas especies con las dos arquitecturas definidas en ARCHITECTURES."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training = collect_examples(seed + 1000, train_episodes, 8000)
    validation = collect_examples(seed + 200_000, validation_episodes, 2200)
    manifest: dict = {
        "method": "aprendizaje supervisado por imitacion de politicas de referencia",
        "seed": seed,
        "train_world_seed_start": seed + 1000,
        "validation_world_seed_start": seed + 200_000,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam (beta1=0.9, beta2=0.999)",
        "loss": "entropia cruzada",
        "activations": {"ocultas": "ReLU", "salida": "Softmax"},
        "input_size": INPUT_SIZE,
        "predator_low_energy_augmentation": {
            "every_nth_episode": 3,
            "initial_energy_range": [0.12, 0.24],
            "reason": "cubrir decisiones de descanso y persecucion en entrenamiento y validacion",
        },
        "architectures": {},
    }

    for species in SPECIES:
        x_train, y_train = training[species]
        x_val, y_val = validation[species]
        input_size = x_train.shape[1]
        manifest["architectures"][species] = {}
        for architecture, hidden_sizes in ARCHITECTURES.items():
            # Hold the initialization and batch-order seeds fixed while the
            # hidden-layer widths are the changed experimental variable.
            model_seed = seed + (0 if species == "prey" else 100)
            model = RedNeurito(input_size, hidden_sizes, len(ACTIONS), seed=model_seed)
            initial_parameters = model.parametros_planos().copy()
            initial_name = f"{species}_{architecture}_inicial.npz"
            trained_name = f"{species}_{architecture}_entrenado.npz"
            model.save(destination / initial_name)
            history = model.fit(
                x_train,
                y_train,
                X_val=x_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                seed=model_seed + 10,
            )
            model.save(destination / trained_name)
            delta = float(np.linalg.norm(model.parametros_planos() - initial_parameters))
            record = {
                "hidden_sizes": list(hidden_sizes),
                "layers": list(model.arquitectura),
                "input_size": input_size,
                "output_size": len(ACTIONS),
                "train_examples": len(x_train),
                "validation_examples": len(x_val),
                "train_class_counts": np.bincount(y_train, minlength=len(ACTIONS)).tolist(),
                "validation_class_counts": np.bincount(y_val, minlength=len(ACTIONS)).tolist(),
                "initial_file": initial_name,
                "trained_file": trained_name,
                "parameter_change_l2": delta,
                "final_epoch": history[-1],
            }
            manifest["architectures"][species][architecture] = record
            with (destination / f"{species}_{architecture}_historial.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(history, handle, ensure_ascii=False, indent=2)

    with (destination / "entrenamiento.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
