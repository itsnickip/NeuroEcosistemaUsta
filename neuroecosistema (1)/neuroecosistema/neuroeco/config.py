"""Configuración del ecosistema y de los tres escenarios obligatorios.

Todos los parámetros del mundo viven aquí para que cada experimento quede
registrado (se guardan en JSON junto a modelos y resultados).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class ConfigMundo:
    # --- espacio
    ancho: float = 100.0
    alto: float = 100.0
    max_pasos: int = 400

    # --- poblaciones
    n_presas: int = 20
    n_depredadores: int = 3
    n_plantas: int = 40
    n_obstaculos: int = 0
    radio_obstaculo: tuple = (5.0, 9.0)

    # --- plantas
    regeneracion_plantas: int = 60      # pasos hasta que una planta comida reaparece
    energia_planta: float = 0.30

    # --- movimiento
    velocidad_presa: float = 1.00
    velocidad_depredador: float = 1.10

    # --- percepción
    vision_presa: float = 22.0
    vision_depredador: float = 28.0

    # --- energía (0..1)
    energia_inicial_presa: float = 0.70
    energia_inicial_depredador: float = 0.80
    costo_mover_presa: float = 0.0040
    costo_quieto_presa: float = 0.0025      # 0.70/0.0025 = 280 pasos < max_pasos: sin comer, muere
    costo_mover_depredador: float = 0.0045
    costo_quieto_depredador: float = 0.0030  # 0.80/0.003 = 266 pasos: quedarse quieto no basta
    energia_captura: float = 0.45

    # --- interacción
    radio_comer: float = 1.8
    radio_captura: float = 1.8

    def to_dict(self):
        return asdict(self)

    def con(self, **cambios):
        return replace(self, **cambios)


# --------------------------------------------------------------- escenarios
# (a) recursos suficientes y pocos depredadores
# (b) escasez de alimento
# (c) mayor presión de depredación + obstáculos
ESCENARIOS: dict[str, ConfigMundo] = {
    "A": ConfigMundo(n_plantas=45, n_depredadores=2, regeneracion_plantas=50),
    "B": ConfigMundo(n_plantas=12, n_depredadores=2, regeneracion_plantas=140),
    "C": ConfigMundo(n_plantas=35, n_depredadores=5, n_obstaculos=6, regeneracion_plantas=60),
}

DESCRIPCION_ESCENARIOS = {
    "A": "Recursos suficientes, pocos depredadores",
    "B": "Escasez de alimento",
    "C": "Alta presión de depredación + obstáculos",
}


# ------------------------------------------------------------ arquitecturas
# Cada especie tiene su propia red; ambas con >= 2 capas ocultas.
ARQUITECTURAS = {
    "base": {"presa": [16, 12], "depredador": [16, 8]},
    "pequena": {"presa": [6, 4], "depredador": [6, 4]},
}


@dataclass
class ConfigEntrenamiento:
    arquitectura: str = "base"
    activacion: str = "tanh"
    poblacion: int = 24
    generaciones: int = 35
    elites: int = 3
    torneo: int = 3
    prob_cruce: float = 0.5
    sigma_mutacion: float = 0.10
    sigma_inicial: float = 0.5
    episodios_por_genoma: int = 3          # uno por escenario de entrenamiento
    escenarios_entrenamiento: list = field(default_factory=lambda: ["A", "B", "C"])
    rondas_coevolucion: int = 2            # presa->depredador alternado
    # "imitacion": pre-entrenamiento supervisado copiando las reglas y luego evolución
    # "aleatoria": neuroevolución pura desde pesos aleatorios (versión 2, ver bitácora)
    inicializacion: str = "imitacion"
    episodios_imitacion: int = 10          # por escenario, semillas 20000+
    epocas_imitacion: int = 80
    tasa_imitacion: float = 0.15
    semilla: int = 42

    def to_dict(self):
        return asdict(self)


def guardar_json(datos: dict, ruta: Path):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=list), encoding="utf-8")


def cargar_json(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))
