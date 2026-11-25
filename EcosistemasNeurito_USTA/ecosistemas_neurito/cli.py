"""Línea de comandos: entrenar, evaluar y abrir el visor."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import run_experiments
from .training import EPOCHS, train_models


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecosistemas_neurito",
        description="EcosistemasNeurito: presas y depredadores guiados por redes neuronales",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="entrenar dos modelos por especie")
    train.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--train-episodes", type=int, default=75)
    train.add_argument("--validation-episodes", type=int, default=20)

    evaluate = commands.add_parser("evaluate", help="ejecutar escenarios y comparaciones")
    evaluate.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    evaluate.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    evaluate.add_argument("--repetitions", type=int, default=8)

    view = commands.add_parser("view", help="abrir simulación Pygame")
    view.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    view.add_argument("--scenario", choices=("abundante", "escasez", "presion"), default="abundante")
    view.add_argument("--seed", type=int, default=101)
    view.add_argument("--prey", choices=("neural", "rule", "random"), default="neural")
    view.add_argument("--predator", choices=("neural", "rule", "random"), default="neural")
    view.add_argument("--fps", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        manifest = train_models(
            args.models,
            seed=args.seed,
            epochs=args.epochs,
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
        )
        for species, architectures in manifest["architectures"].items():
            for name, record in architectures.items():
                last = record["final_epoch"]
                capas = " → ".join(str(n) for n in record["layers"])
                print(
                    f"{species:<8} {name:<11} [{capas}]  validación={last['val_accuracy']:.1%}  "
                    f"cambio de pesos L2={record['parameter_change_l2']:.3f}"
                )
        print(f"Modelos y trazas guardados en {args.models}")
    elif args.command == "evaluate":
        if args.repetitions < 2:
            raise SystemExit("--repetitions debe ser al menos 2")
        seeds = tuple(101 * (index + 1) for index in range(args.repetitions))
        rows, _ = run_experiments(args.models, args.output, seeds=seeds)
        print(f"{len(rows)} episodios guardados en {args.output}")
        print("Consulta episodios.csv, resumen.csv y las gráficas SVG.")
    else:
        from .viewer import run_viewer

        prey_model = args.models / "prey_base_entrenado.npz"
        predator_model = args.models / "predator_base_entrenado.npz"
        if args.prey == "neural" and not prey_model.exists():
            raise SystemExit("Falta el modelo de presas; ejecuta primero 'train'.")
        if args.predator == "neural" and not predator_model.exists():
            raise SystemExit("Falta el modelo de depredadores; ejecuta primero 'train'.")
        run_viewer(
            scenario=args.scenario,
            seed=args.seed,
            prey_model=prey_model if prey_model.exists() else None,
            predator_model=predator_model if predator_model.exists() else None,
            prey_policy=args.prey,
            predator_policy=args.predator,
            fps=args.fps,
        )
