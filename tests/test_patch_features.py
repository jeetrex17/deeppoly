import math
import unittest

import numpy as np
import torch

from rl_transfer.config import AttackConfig
from rl_transfer.features import (
    configured_patch_image_features,
    patch_image_feature_dimension,
    patch_image_features,
)


class PatchImageFeatureTests(unittest.TestCase):
    def test_default_means_mode_preserves_the_legacy_output(self) -> None:
        original = torch.tensor(
            (
                ((0.0, 0.0), (1.0, 1.0)),
                ((0.0, 1.0), (0.0, 1.0)),
                ((0.25, 0.25), (0.25, 0.25)),
            ),
            dtype=torch.float32,
        )
        current = torch.tensor(
            (
                ((0.125, 0.125), (0.875, 0.875)),
                ((0.0, 1.0), (0.0, 1.0)),
                ((0.5, 0.5), (0.5, 0.5)),
            ),
            dtype=torch.float32,
        )

        features = patch_image_features(original, current, grid_size=1)

        np.testing.assert_array_equal(
            features,
            np.asarray(
                (0.5, 0.5, 0.25, 0.0, 0.0, 0.25),
                dtype=np.float32,
            ),
        )
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(patch_image_feature_dimension(grid_size=1, mode="means"), 6)

    def test_default_means_mode_is_bitwise_equal_to_legacy_implementation(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(20260728)
        original = torch.rand((3, 8, 8), generator=generator)
        current = (original + 0.02 * torch.randn(
            (3, 8, 8),
            generator=generator,
        )).clamp(0, 1)
        patch_height = original.shape[1] // 4
        patch_width = original.shape[2] // 4

        def legacy_means(image: torch.Tensor) -> torch.Tensor:
            patches = image.unfold(
                1,
                patch_height,
                patch_height,
            ).unfold(
                2,
                patch_width,
                patch_width,
            )
            return patches.mean(dim=(-1, -2)).permute(1, 2, 0).flatten()

        expected = (
            torch.cat(
                (
                    legacy_means(original),
                    legacy_means(current - original),
                ),
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

        actual = patch_image_features(original, current, grid_size=4)

        np.testing.assert_array_equal(actual, expected)

    def test_statistics_mode_has_documented_patch_channel_block_order(self) -> None:
        original = torch.tensor(
            (
                ((0.0, 0.0), (1.0, 1.0)),
                ((0.0, 1.0), (0.0, 1.0)),
                ((0.25, 0.25), (0.25, 0.25)),
            ),
            dtype=torch.float32,
        )
        current = torch.tensor(
            (
                ((0.125, 0.125), (0.875, 0.875)),
                ((0.0, 1.0), (0.0, 1.0)),
                ((0.5, 0.5), (0.5, 0.5)),
            ),
            dtype=torch.float32,
        )

        features = patch_image_features(
            original,
            current,
            grid_size=1,
            mode="statistics",
            epsilon=0.25,
        )

        # Each contiguous block is patch-major, then channel-minor. The blocks
        # are original mean, signed perturbation mean, original population std,
        # original internal-edge magnitude, normalized absolute perturbation,
        # positive attack headroom, and negative attack headroom.
        expected = np.asarray(
            (
                0.5,
                0.5,
                0.25,
                0.0,
                0.0,
                0.25,
                0.5,
                0.5,
                0.0,
                0.5,
                0.5,
                0.0,
                0.5,
                0.0,
                1.0,
                0.25,
                0.25,
                0.0,
                0.25,
                0.25,
                1.0,
            ),
            dtype=np.float32,
        )
        np.testing.assert_allclose(features, expected, rtol=0.0, atol=1e-7)
        self.assertEqual(features.shape, (21,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(
            patch_image_feature_dimension(grid_size=1, mode="statistics"),
            21,
        )

    def test_patch_order_is_row_major_then_channel_minor(self) -> None:
        original = torch.zeros((3, 4, 4), dtype=torch.float32)
        for patch_row in range(2):
            for patch_column in range(2):
                patch_index = patch_row * 2 + patch_column
                for channel in range(3):
                    original[
                        channel,
                        patch_row * 2 : (patch_row + 1) * 2,
                        patch_column * 2 : (patch_column + 1) * 2,
                    ] = (patch_index * 3 + channel) / 20

        features = patch_image_features(
            original,
            original,
            grid_size=2,
            mode="statistics",
            epsilon=0.25,
        )

        expected_original_means = np.arange(12, dtype=np.float32) / 20
        np.testing.assert_allclose(
            features[:12],
            expected_original_means,
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(features.shape, (84,))

    def test_statistics_mode_preserves_inputs_and_returns_finite_float32(self) -> None:
        original = torch.linspace(
            0.0,
            0.8,
            steps=3 * 4 * 4,
            dtype=torch.float16,
        ).reshape(3, 4, 4)
        current = (original + torch.tensor(0.05, dtype=torch.float16)).clamp(0, 1)
        original_before = original.clone()
        current_before = current.clone()

        features = patch_image_features(
            original,
            current,
            grid_size=2,
            mode="statistics",
            epsilon=0.1,
        )

        self.assertTrue(torch.equal(original, original_before))
        self.assertTrue(torch.equal(current, current_before))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_config_declares_exact_legacy_and_statistics_dimensions(self) -> None:
        disabled = AttackConfig(
            grid_size=4,
            image_patch_features=False,
            image_patch_feature_mode="statistics",
        )
        legacy = AttackConfig(
            grid_size=4,
            image_patch_features=True,
        )
        statistics = AttackConfig(
            grid_size=4,
            action_history_features=True,
            image_patch_features=True,
            image_patch_feature_mode="statistics",
        )

        self.assertEqual(disabled.image_patch_feature_dim, 0)
        self.assertEqual(disabled.recurrent_observation_dim, 8)
        self.assertEqual(legacy.image_patch_feature_dim, 96)
        self.assertEqual(legacy.recurrent_observation_dim, 104)
        self.assertEqual(statistics.image_patch_feature_dim, 336)
        self.assertEqual(statistics.recurrent_observation_dim, 536)

    def test_configured_helper_respects_feature_toggle_and_mode(self) -> None:
        original = torch.full((3, 2, 2), 0.5)
        current = torch.full((3, 2, 2), 0.6)

        self.assertIsNone(
            configured_patch_image_features(
                original,
                current,
                AttackConfig(
                    grid_size=1,
                    epsilon=0.2,
                    image_patch_features=False,
                    image_patch_feature_mode="statistics",
                ),
            ),
        )
        configured = configured_patch_image_features(
            original,
            current,
            AttackConfig(
                grid_size=1,
                epsilon=0.2,
                image_patch_features=True,
                image_patch_feature_mode="statistics",
            ),
        )
        direct = patch_image_features(
            original,
            current,
            grid_size=1,
            mode="statistics",
            epsilon=0.2,
        )
        np.testing.assert_array_equal(configured, direct)

    def test_invalid_feature_inputs_fail_closed(self) -> None:
        valid = torch.full((3, 2, 2), 0.5)
        invalid_calls = (
            lambda: patch_image_features(
                valid,
                valid,
                grid_size=1,
                mode="unknown",
            ),
            lambda: patch_image_features(
                valid,
                valid,
                grid_size=1,
                mode=[],
            ),
            lambda: patch_image_features(
                torch.empty((3, 0, 0)),
                torch.empty((3, 0, 0)),
                grid_size=1,
            ),
            lambda: patch_image_features(
                valid,
                valid,
                grid_size=1,
                mode="statistics",
            ),
            lambda: patch_image_features(
                valid,
                valid,
                grid_size=1,
                mode="statistics",
                epsilon=0.0,
            ),
            lambda: patch_image_features(
                valid,
                valid,
                grid_size=1,
                mode="statistics",
                epsilon=math.nan,
            ),
            lambda: patch_image_features(
                valid,
                valid,
                grid_size=1,
                mode="statistics",
                epsilon=True,
            ),
            lambda: patch_image_features(
                valid,
                torch.full_like(valid, 1.1),
                grid_size=1,
                mode="statistics",
                epsilon=0.2,
            ),
            lambda: patch_image_features(
                valid,
                torch.full_like(valid, 0.8),
                grid_size=1,
                mode="statistics",
                epsilon=0.2,
            ),
            lambda: patch_image_feature_dimension(
                grid_size=0,
                mode="statistics",
            ),
            lambda: AttackConfig(image_patch_feature_mode="unknown"),
            lambda: AttackConfig(image_patch_feature_mode=[]),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(ValueError):
                invalid_call()


if __name__ == "__main__":
    unittest.main()
