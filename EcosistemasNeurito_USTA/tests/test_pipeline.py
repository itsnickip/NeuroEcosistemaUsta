"""End-to-end checks for actual learning and repeatable evaluation."""

import tempfile
import unittest

from ecosistemas_neurito.experiments import load_candidates, run_episode
from ecosistemas_neurito.training import collect_examples, train_models


class PipelineTests(unittest.TestCase):
    def test_training_and_validation_include_energy_aware_rest(self):
        for seed in (1042, 200042):
            examples = collect_examples(seed, episodes=6, limit_per_species=1000)
            _, predator_labels = examples["predator"]
            self.assertGreater(sum(predator_labels == 0), 0)
            self.assertGreater(sum(predator_labels != 0), 0)

    def test_trained_models_drive_repeatable_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = train_models(
                directory,
                seed=17,
                epochs=3,
                train_episodes=5,
                validation_episodes=3,
            )
            for species in ("prey", "predator"):
                for architecture in ("base", "alternativa"):
                    record = manifest["architectures"][species][architecture]
                    self.assertGreater(record["parameter_change_l2"], 0.0)
                    self.assertGreater(record["validation_examples"], 0)

            models = load_candidates(directory)
            first = run_episode("abundante", 101, "prey", "entrenado", models)
            repeat = run_episode("abundante", 101, "prey", "entrenado", models)
            self.assertEqual(first, repeat)
            self.assertIn(first["capture_occurred"], (0, 1))
            self.assertGreater(first["episode_steps"], 0)


if __name__ == "__main__":
    unittest.main()
