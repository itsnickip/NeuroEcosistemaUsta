"""Focused checks of the simulation rules and experimental reproducibility."""

import unittest
from dataclasses import replace
from random import Random

import numpy as np

from ecosistemas_neurito.policies import random_action, rule_action
from ecosistemas_neurito.world import Plant, SCENARIOS, World


class WorldTests(unittest.TestCase):
    def test_seed_replays_complete_episode(self):
        config = replace(SCENARIOS["abundante"], max_steps=20)
        a, b = World(config, 73), World(config, 73)
        for _ in range(config.max_steps):
            decisions = []
            for world in (a, b):
                decisions.append((
                    [rule_action(world, "prey", i) for i in range(len(world.prey))],
                    [rule_action(world, "predator", i) for i in range(len(world.predators))],
                ))
            self.assertEqual(decisions[0], decisions[1])
            for world, (prey, predators) in zip((a, b), decisions):
                if not world.done:
                    world.step(prey, predators)
        self.assertEqual(a.metrics(), b.metrics())
        self.assertEqual([(p.x, p.y) for p in a.plants],
                         [(p.x, p.y) for p in b.plants])

    def test_food_is_consumed_and_regrows_on_schedule(self):
        config = replace(SCENARIOS["abundante"], prey_count=1, predator_count=0,
                         food_count=0, food_respawn_steps=2, max_steps=5)
        world = World(config, 1)
        world.prey[0].energy = 0.5
        world.plants = [Plant(world.prey[0].x, world.prey[0].y)]
        world.step([0], [])
        self.assertEqual(world.metrics()["prey_food_total"], 1)
        self.assertGreater(world.prey[0].energy, 0.5)
        self.assertEqual(len(world.plants), 0)
        world.step([0], [])
        self.assertEqual(len(world.plants), 0)
        world.step([0], [])
        self.assertEqual(len(world.plants), 1)

    def test_capture_changes_metrics_and_ends_if_last_prey(self):
        config = replace(SCENARIOS["abundante"], prey_count=1, predator_count=1,
                         food_count=0)
        world = World(config, 2)
        world.predators[0].x = world.prey[0].x
        world.predators[0].y = world.prey[0].y
        world.step([0], [0])
        metrics = world.metrics()
        self.assertTrue(world.done)
        self.assertEqual(metrics["predator_captures_total"], 1)
        self.assertEqual(metrics["prey_alive"], 0)
        self.assertEqual(metrics["time_to_first_capture"], 1)

    def test_observations_and_reference_actions(self):
        config = replace(SCENARIOS["abundante"], prey_count=1, predator_count=1,
                         food_count=0)
        world = World(config, 3)
        prey, predator = world.prey[0], world.predators[0]
        prey.x, prey.y = 300, 300
        predator.x, predator.y = 350, 300
        world.plants = [Plant(400, 300)]
        self.assertEqual(rule_action(world, "prey", 0), 3)  # flee left
        self.assertEqual(rule_action(world, "predator", 0), 3)  # pursue left
        self.assertEqual(world.observe("prey", 0).shape, (8,))
        self.assertEqual(world.observe("predator", 0).shape, (8,))
        self.assertTrue(np.all((world.observe("predator", 0) >= -1) & (world.observe("predator", 0) <= 1)))
        self.assertTrue(np.all(np.abs(world.observe("prey", 0)) <= 1))
        rng_a, rng_b = Random(99), Random(99)
        self.assertEqual([random_action(world, "prey", 0, rng_a) for _ in range(10)],
                         [random_action(world, "prey", 0, rng_b) for _ in range(10)])

    def test_predator_conserves_energy_but_pursues_close_prey(self):
        config = replace(SCENARIOS["abundante"], prey_count=1, predator_count=1,
                         food_count=0, max_steps=5)
        world = World(config, 55)
        prey, predator = world.prey[0], world.predators[0]
        prey.x, prey.y = 600, 300
        predator.x, predator.y = 100, 300
        predator.energy = 0.20
        self.assertEqual(rule_action(world, "predator", 0), 0)
        before = predator.energy
        world.step([0], [0])
        self.assertAlmostEqual(predator.energy, before - config.rest_energy_cost)
        self.assertEqual(predator.x, 100)
        predator.energy = 0.80
        self.assertEqual(rule_action(world, "predator", 0), 4)
        predator.energy = 0.20
        prey.x, prey.y = 125, 300
        self.assertEqual(rule_action(world, "predator", 0), 4)


if __name__ == "__main__":
    unittest.main()
