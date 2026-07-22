import unittest

import torch

from rl_transfer.cifar_models import (
    build_cifar_victim_ensemble,
    build_cifar_victim_population,
    build_cifar_victims,
)


def _same_state(first: torch.nn.Module, second: torch.nn.Module) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[key], second_state[key]) for key in first_state
    )


class CIFARVictimModelTests(unittest.TestCase):
    def test_legacy_builder_shape_ids_and_parameter_counts_are_preserved(self) -> None:
        victims = build_cifar_victims(seed=7)
        expected = {
            "classical_cnn": ("cifar_residual_cnn", 223_146),
            "modern_cnn": ("cifar_depthwise_cnn", 15_906),
            "transformer": ("cifar_patch_transformer", 24_970),
        }
        self.assertEqual(set(victims), set(expected))
        for family, (victim_id, model) in victims.items():
            self.assertEqual(victim_id, expected[family][0])
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), expected[family][1])
            model.eval()
            self.assertEqual(model(torch.rand(2, 3, 32, 32)).shape, (2, 10))

    def test_research_profile_is_stronger_and_has_valid_outputs(self) -> None:
        pilot = build_cifar_victims(seed=7)
        research = build_cifar_victims(seed=7, profile="research")
        for family, (_, model) in research.items():
            pilot_parameters = sum(parameter.numel() for parameter in pilot[family][1].parameters())
            research_parameters = sum(parameter.numel() for parameter in model.parameters())
            self.assertGreater(research_parameters, pilot_parameters)
            model.eval()
            self.assertEqual(model(torch.rand(2, 3, 32, 32)).shape, (2, 10))

    def test_ensemble_is_grouped_reproducible_and_independently_seeded(self) -> None:
        first = build_cifar_victim_ensemble(
            seed=19,
            instances_per_family={"classical_cnn": 2, "modern_cnn": 3},
            families=("classical_cnn", "modern_cnn"),
        )
        second = build_cifar_victim_ensemble(
            seed=19,
            instances_per_family={"classical_cnn": 2, "modern_cnn": 3},
            families=("classical_cnn", "modern_cnn"),
        )
        self.assertEqual(tuple(first), ("classical_cnn", "modern_cnn"))
        self.assertEqual({family: len(models) for family, models in first.items()}, {"classical_cnn": 2, "modern_cnn": 3})
        for family in first:
            self.assertEqual(
                [victim_id for victim_id, _ in first[family]],
                [victim_id for victim_id, _ in second[family]],
            )
            for (_, first_model), (_, second_model) in zip(first[family], second[family]):
                self.assertTrue(_same_state(first_model, second_model))
            self.assertFalse(_same_state(first[family][0][1], first[family][1][1]))

    def test_builders_do_not_consume_the_callers_random_stream(self) -> None:
        torch.manual_seed(101)
        expected = torch.rand(4)
        torch.manual_seed(101)
        build_cifar_victims(seed=7, profile="research")
        self.assertTrue(torch.equal(torch.rand(4), expected))
        torch.manual_seed(101)
        build_cifar_victim_ensemble(seed=7, instances_per_family=2)
        self.assertTrue(torch.equal(torch.rand(4), expected))

    def test_ensemble_rejects_ambiguous_or_invalid_requests(self) -> None:
        with self.assertRaises(ValueError):
            build_cifar_victim_ensemble(1, 0)
        with self.assertRaises(ValueError):
            build_cifar_victim_ensemble(1, 1, families=("classical_cnn", "classical_cnn"))
        with self.assertRaises(ValueError):
            build_cifar_victim_ensemble(1, 1, families=("unknown",))
        with self.assertRaises(ValueError):
            build_cifar_victim_ensemble(
                1,
                {"classical_cnn": 2},
                families=("classical_cnn", "modern_cnn"),
            )

    def test_population_compatibility_name_returns_grouped_instances(self) -> None:
        population = build_cifar_victim_population(
            seed=23,
            instances_per_family=2,
            families=("classical_cnn",),
            profile="pilot",
        )
        self.assertEqual(tuple(population), ("classical_cnn",))
        self.assertEqual(len(population["classical_cnn"]), 2)


if __name__ == "__main__":
    unittest.main()
