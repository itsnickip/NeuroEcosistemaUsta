"""Mundo 2D determinista de presas, depredadores y plantas.

Las políticas deciden *antes* de cada paso. Luego el mundo mueve a los agentes
vivos, resuelve alimento y capturas y registra métricas del episodio. Aquí no
se dibuja ni se usa la red, así que el mismo episodio corre con o sin ventana.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from numbers import Integral
from random import Random
from typing import Sequence

import numpy as np


ACTIONS = ("quieto", "arriba", "abajo", "izquierda", "derecha")
# Ambas especies observan 8 valores normalizados -> capa de entrada de 8 neuronas.
PREY_INPUTS = (
    "energia",
    "alimento_dx",
    "alimento_dy",
    "alimento_distancia",
    "depredador_dx",
    "depredador_dy",
    "depredador_distancia",
    "borde_distancia",
)
PREDATOR_INPUTS = (
    "energia",
    "presa_dx",
    "presa_dy",
    "presa_distancia",
    "posicion_x",
    "posicion_y",
    "presas_vivas",
    "companero_distancia",
)
INPUT_SIZE = 8

_DIRECTIONS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class WorldConfig:
    """All values affecting a run are explicit and serializable."""

    name: str = "abundante"
    width: int = 800
    height: int = 600
    prey_count: int = 6
    predator_count: int = 1
    food_count: int = 28
    max_steps: int = 300
    food_respawn_steps: int = 12
    prey_speed: float = 5.0
    predator_speed: float = 5.2
    prey_energy_cost: float = 0.003
    predator_energy_cost: float = 0.004
    rest_energy_cost: float = 0.001
    food_energy_gain: float = 0.35
    capture_energy_gain: float = 0.55
    food_radius: float = 13.0
    capture_radius: float = 18.0
    danger_radius: float = 130.0

    def __post_init__(self) -> None:
        if self.width <= 40 or self.height <= 40 or self.max_steps < 1:
            raise ValueError("world dimensions and max_steps must be positive")
        if min(self.prey_count, self.predator_count, self.food_count) < 0:
            raise ValueError("population and food counts cannot be negative")
        if self.prey_count == 0:
            raise ValueError("at least one prey is required")
        if self.food_respawn_steps < 1:
            raise ValueError("food_respawn_steps must be at least one")
        if min(self.prey_speed, self.predator_speed, self.food_radius,
               self.capture_radius, self.danger_radius) <= 0:
            raise ValueError("speed and sensing/collision radii must be positive")
        if min(self.prey_energy_cost, self.predator_energy_cost,
               self.rest_energy_cost, self.food_energy_gain,
               self.capture_energy_gain) < 0:
            raise ValueError("energy costs and gains cannot be negative")


SCENARIOS: dict[str, WorldConfig] = {
    "abundante": WorldConfig(),
    "escasez": WorldConfig(
        name="escasez", food_count=7, food_respawn_steps=85,
    ),
    "presion": WorldConfig(
        name="presion", predator_count=3, food_count=18,
        food_respawn_steps=35,
    ),
}


@dataclass
class Agent:
    x: float
    y: float
    energy: float = 1.0
    alive: bool = True
    food_eaten: int = 0
    captures: int = 0
    survival_steps: int = 0


@dataclass
class Plant:
    x: float
    y: float


class World:
    """A complete episode, independent of how actions are selected."""

    def __init__(self, config: WorldConfig, seed: int):
        self.config = config
        self.seed = int(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = Random(self.seed)
        self.time = 0
        self.done = False
        self.first_capture_time: int | None = None
        self.events: list[str] = []
        self._food_queue: list[int] = []
        self.prey = [Agent(*self._position()) for _ in range(self.config.prey_count)]
        self.predators = [
            Agent(*self._position_away_from_prey())
            for _ in range(self.config.predator_count)
        ]
        self.plants = [Plant(*self._position()) for _ in range(self.config.food_count)]

    def _position(self) -> tuple[float, float]:
        return (
            self.rng.uniform(20.0, self.config.width - 20.0),
            self.rng.uniform(20.0, self.config.height - 20.0),
        )

    def _position_away_from_prey(self) -> tuple[float, float]:
        # Avoid an uninformative capture before a policy has acted.
        for _ in range(100):
            x, y = self._position()
            if all(hypot(x - p.x, y - p.y) > 2 * self.config.capture_radius
                   for p in self.prey):
                return x, y
        return self._position()

    def nearest_plant(self, agent: Agent) -> tuple[Plant | None, float]:
        if not self.plants:
            return None, float("inf")
        plant = min(self.plants, key=lambda p: hypot(p.x - agent.x, p.y - agent.y))
        return plant, hypot(plant.x - agent.x, plant.y - agent.y)

    def nearest_agent(self, agent: Agent, species: str) -> tuple[Agent | None, float]:
        if species == "prey":
            population = self.prey
        elif species == "predator":
            population = self.predators
        else:
            raise ValueError("species must be 'prey' or 'predator'")
        living = [other for other in population if other.alive]
        if not living:
            return None, float("inf")
        other = min(living, key=lambda p: hypot(p.x - agent.x, p.y - agent.y))
        return other, hypot(other.x - agent.x, other.y - agent.y)

    def observe(self, species: str, index: int) -> np.ndarray:
        """Vector de 8 observaciones locales y normalizadas.

        Coordenadas relativas en [-1, 1]; energía, distancias y fracciones en
        [0, 1]. Un objetivo inexistente tiene Δx = Δy = 0 y distancia 1.
        """
        if species == "prey":
            agent = self.prey[index]
            food, food_distance = self.nearest_plant(agent)
            danger, danger_distance = self.nearest_agent(agent, "predator")
            return np.array((
                self._unit_energy(agent),
                self._dx(agent, food), self._dy(agent, food),
                self._distance(food_distance),
                self._dx(agent, danger), self._dy(agent, danger),
                self._distance(danger_distance),
                self._wall_distance(agent),
            ), dtype=np.float64)
        if species == "predator":
            agent = self.predators[index]
            prey, distance = self.nearest_agent(agent, "prey")
            partner, partner_distance = self._nearest_partner(index)
            return np.array((
                self._unit_energy(agent),
                self._dx(agent, prey), self._dy(agent, prey),
                self._distance(distance),
                agent.x / self.config.width,
                agent.y / self.config.height,
                sum(p.alive for p in self.prey) / len(self.prey),
                self._distance(partner_distance),
            ), dtype=np.float64)
        raise ValueError("species must be 'prey' or 'predator'")

    def _nearest_partner(self, index: int) -> tuple[Agent | None, float]:
        """Depredador vivo más cercano distinto del agente ``index``."""
        agent = self.predators[index]
        others = [p for i, p in enumerate(self.predators) if i != index and p.alive]
        if not others:
            return None, float("inf")
        other = min(others, key=lambda p: hypot(p.x - agent.x, p.y - agent.y))
        return other, hypot(other.x - agent.x, other.y - agent.y)

    def _wall_distance(self, agent: Agent) -> float:
        """Distancia al borde más cercano, 1 = centro del mapa."""
        nearest = min(agent.x, self.config.width - agent.x,
                      agent.y, self.config.height - agent.y)
        return min(1.0, max(0.0, nearest / (min(self.config.width, self.config.height) / 2)))

    @staticmethod
    def _unit_energy(agent: Agent) -> float:
        return min(1.0, max(0.0, agent.energy))

    def _dx(self, agent: Agent, target: Agent | Plant | None) -> float:
        return 0.0 if target is None else (target.x - agent.x) / self.config.width

    def _dy(self, agent: Agent, target: Agent | Plant | None) -> float:
        return 0.0 if target is None else (target.y - agent.y) / self.config.height

    def _distance(self, distance: float) -> float:
        if distance == float("inf"):
            return 1.0
        return min(1.0, distance / hypot(self.config.width, self.config.height))

    def step(self, prey_actions: Sequence[int], predator_actions: Sequence[int]) -> None:
        """Apply simultaneous decisions and resolve this frame's interactions."""
        if self.done:
            raise RuntimeError("episode ended; call reset() to start a new one")
        self._validate_actions(prey_actions, len(self.prey), "prey")
        self._validate_actions(predator_actions, len(self.predators), "predator")
        self.time += 1
        self.events = []

        # Both species move using actions selected from the same pre-step state.
        for agent, action in zip(self.prey, prey_actions):
            self._move(agent, int(action), self.config.prey_speed,
                       self.config.prey_energy_cost)
        for agent, action in zip(self.predators, predator_actions):
            self._move(agent, int(action), self.config.predator_speed,
                       self.config.predator_energy_cost)

        # A plant can be consumed by at most one prey in a frame. Ordering is
        # stable and therefore does not introduce a hidden source of randomness.
        for agent in self.prey:
            if not agent.alive:
                continue
            for plant in tuple(self.plants):
                if hypot(plant.x - agent.x, plant.y - agent.y) <= self.config.food_radius:
                    self.plants.remove(plant)
                    self._food_queue.append(self.time + self.config.food_respawn_steps)
                    agent.energy = min(1.0, agent.energy + self.config.food_energy_gain)
                    agent.food_eaten += 1
                    self.events.append("alimento")
                    break

        for predator in self.predators:
            if not predator.alive:
                continue
            for prey in self.prey:
                if prey.alive and hypot(prey.x - predator.x, prey.y - predator.y) <= self.config.capture_radius:
                    prey.alive = False
                    prey.energy = 0.0
                    predator.captures += 1
                    predator.energy = min(1.0, predator.energy + self.config.capture_energy_gain)
                    if self.first_capture_time is None:
                        self.first_capture_time = self.time
                    self.events.append("captura")

        for agent in (*self.prey, *self.predators):
            if agent.alive:
                agent.survival_steps += 1
                if agent.energy <= 0:
                    agent.energy = 0.0
                    agent.alive = False
                    self.events.append("agotamiento")

        due = [t for t in self._food_queue if t <= self.time]
        self._food_queue = [t for t in self._food_queue if t > self.time]
        self.plants.extend(Plant(*self._position()) for _ in due)

        self.done = self.time >= self.config.max_steps or not any(p.alive for p in self.prey)

    @staticmethod
    def _validate_actions(actions: Sequence[int], expected: int, species: str) -> None:
        if len(actions) != expected:
            raise ValueError(f"expected {expected} {species} actions, got {len(actions)}")
        if any(not isinstance(action, Integral) or not 0 <= int(action) < len(ACTIONS)
               for action in actions):
            raise ValueError(f"{species} actions must be integers from 0 to 4")

    def _move(self, agent: Agent, action: int, speed: float, move_cost: float) -> None:
        if not agent.alive:
            return
        dx, dy = _DIRECTIONS[action]
        agent.x = min(self.config.width, max(0.0, agent.x + dx * speed))
        agent.y = min(self.config.height, max(0.0, agent.y + dy * speed))
        agent.energy -= self.config.rest_energy_cost if action == 0 else move_cost

    def metrics(self) -> dict[str, float | int | str | None]:
        """Episode metrics; means include dead initial agents with zero energy."""
        return {
            "scenario": self.config.name,
            "seed": self.seed,
            "episode_steps": self.time,
            "prey_survival_mean": sum(a.survival_steps for a in self.prey) / len(self.prey),
            "prey_food_total": sum(a.food_eaten for a in self.prey),
            "predator_captures_total": sum(a.captures for a in self.predators),
            "prey_energy_mean": sum(a.energy for a in self.prey) / len(self.prey),
            "predator_energy_mean": (
                sum(a.energy for a in self.predators) / len(self.predators)
                if self.predators else 0.0
            ),
            "prey_alive": sum(a.alive for a in self.prey),
            "predator_alive": sum(a.alive for a in self.predators),
            "time_to_first_capture": self.first_capture_time,
        }
