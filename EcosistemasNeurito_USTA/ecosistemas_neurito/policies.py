"""Transparent reference policies for training labels and experiments."""

from __future__ import annotations

from random import Random

from .world import ACTIONS, Agent, World


def _axis_action(dx: float, dy: float) -> int:
    """Choose the cardinal action that closes the largest coordinate gap."""
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return 4 if dx > 0 else 3
    return 2 if dy > 0 else 1


def rule_action(world: World, species: str, index: int) -> int:
    """Prey flee nearby threats or seek food; predators balance rest and pursuit.

    This policy reads world state directly and is intentionally a stronger
    information baseline than a random agent. The neural agents receive only
    their documented observation vectors.
    """
    if species == "prey":
        agent = world.prey[index]
        if not agent.alive:
            return 0
        predator, distance = world.nearest_agent(agent, "predator")
        if predator is not None and distance < world.config.danger_radius:
            return _axis_action(agent.x - predator.x, agent.y - predator.y)
        plant, _ = world.nearest_plant(agent)
        if plant is None:
            return 0
        return _axis_action(plant.x - agent.x, plant.y - agent.y)
    if species == "predator":
        agent = world.predators[index]
        if not agent.alive:
            return 0
        prey, distance = world.nearest_agent(agent, "prey")
        if prey is None:
            return 0
        # Wait when energy is low and the prey is far away: resting spends less
        # energy than chasing, but never creates energy. A close capture takes
        # priority because it is the predator's only energy source.
        if agent.energy < 0.30 and distance > 2 * world.config.capture_radius:
            return 0
        return _axis_action(prey.x - agent.x, prey.y - agent.y)
    raise ValueError("species must be 'prey' or 'predator'")


def random_action(world: World, species: str, index: int, rng: Random) -> int:
    """Uniform reference over the five actions, with a separately seeded RNG."""
    if species == "prey":
        agent: Agent = world.prey[index]
    elif species == "predator":
        agent = world.predators[index]
    else:
        raise ValueError("species must be 'prey' or 'predator'")
    return rng.randrange(len(ACTIONS)) if agent.alive else 0
