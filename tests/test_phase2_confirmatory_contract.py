from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from torchvision.datasets import CIFAR10

from rl_transfer.cifar_data import build_cifar_split, indices_digest
from rl_transfer.phase2_confirmatory_contract import (
    CONFIRMATORY_CONTRACT_SHA256,
    DEFAULT_CONFIRMATORY_CONTRACT_PATH,
    allocate_final_source_test_indices,
    assert_final_source_test_access,
    canonical_contract_digest,
    load_confirmatory_contract,
    next_primes,
    validate_confirmatory_contract_payload,
    validate_final_source_test_indices,
)


PHASE1_SEEDS = (17, 29, 41, 53, 67, 79, 97, 113, 131, 149)
STAGE_C_SEEDS = (151, 157)
FINAL_SEEDS = (163, 167, 173, 179, 181, 191, 193, 197, 199, 211)


def _official_cifar_labels() -> tuple[tuple[int, ...], tuple[int, ...]]:
    train = CIFAR10(root="data/cifar10", train=True, download=False)
    test = CIFAR10(root="data/cifar10", train=False, download=False)
    return (
        tuple(int(label) for label in train.targets),
        tuple(int(label) for label in test.targets),
    )


class ConfirmatoryContractTests(unittest.TestCase):
    def test_locked_contract_has_fresh_derived_seed_sequences(self) -> None:
        contract = load_confirmatory_contract()

        self.assertEqual(contract.phase1_seeds, PHASE1_SEEDS)
        self.assertEqual(contract.stage_b_seeds, (17,))
        self.assertEqual(contract.stage_c_seeds, STAGE_C_SEEDS)
        self.assertEqual(contract.final_confirmatory_seeds, FINAL_SEEDS)
        self.assertEqual(next_primes(max(PHASE1_SEEDS), 2), STAGE_C_SEEDS)
        self.assertEqual(next_primes(max(STAGE_C_SEEDS), 10), FINAL_SEEDS)
        development = set(PHASE1_SEEDS) | set(contract.stage_b_seeds)
        self.assertFalse(development.intersection(contract.stage_c_seeds))
        self.assertFalse(development.intersection(contract.final_confirmatory_seeds))
        self.assertFalse(
            set(contract.stage_c_seeds).intersection(
                contract.final_confirmatory_seeds
            )
        )

    def test_contract_digest_is_pinned_and_payload_is_immutable(self) -> None:
        contract = load_confirmatory_contract()

        self.assertEqual(
            canonical_contract_digest(
                json.loads(DEFAULT_CONFIRMATORY_CONTRACT_PATH.read_text())
            ),
            CONFIRMATORY_CONTRACT_SHA256,
        )
        with self.assertRaises(TypeError):
            contract.rules["stage_c"] = {}  # type: ignore[index]

    def test_any_post_lock_payload_change_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_CONFIRMATORY_CONTRACT_PATH.read_text())
        payload["rules"]["stage_c"]["first_rung"]["minimum_macro_asr_gain"] = 0.0

        with self.assertRaisesRegex(ValueError, "locked digest"):
            validate_confirmatory_contract_payload(payload)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text(
                '{"schema_version":1,"schema_version":1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_confirmatory_contract(path)

    def test_locked_stage_c_and_final_rules_match_preregistration(self) -> None:
        rules = load_confirmatory_contract().rules
        stage_c = rules["stage_c"]
        first_rung = stage_c["first_rung"]
        promotion = stage_c["promotion"]
        final = rules["final_confirmatory_source"]

        self.assertEqual(first_rung["initial_ppo_episodes"], 600)
        self.assertEqual(first_rung["maximum_ppo_episodes"], 1200)
        self.assertEqual(first_rung["minimum_macro_asr_gain"], 0.025)
        self.assertEqual(first_rung["minimum_macro_auc_gain"], 0.01)
        self.assertEqual(
            first_rung["minimum_soft_top5_gain_over_validation_oracle"],
            0.01,
        )
        self.assertEqual(
            first_rung[
                "minimum_soft_cross_entropy_improvement_over_validation_oracle"
            ],
            0.02,
        )
        self.assertNotIn(
            "minimum_soft_top5_gain_over_frequency",
            first_rung,
        )
        self.assertNotIn(
            "minimum_soft_cross_entropy_improvement_over_frequency",
            first_rung,
        )
        self.assertEqual(first_rung["no_regression_tail_episodes"], 200)
        self.assertEqual(promotion["source_images"], 200)
        self.assertEqual(promotion["minimum_macro_asr_gain"], 0.04)
        self.assertEqual(promotion["minimum_macro_auc_gain"], 0.015)
        self.assertEqual(final["final_source_test_images"], 1000)
        self.assertEqual(final["minimum_macro_asr_gain"], 0.05)
        self.assertEqual(final["minimum_macro_auc_gain"], 0.02)
        self.assertEqual(final["maximum_exact_sign_flip_pvalue"], 0.05)
        self.assertEqual(final["minimum_hybrid_ablation_asr_gain"], 0.01)
        self.assertTrue(final["run_once"])
        self.assertFalse(final["replace_failed_seeds"])

    def test_final_source_test_access_is_denied_until_final_study(self) -> None:
        contract = load_confirmatory_contract()
        prohibited = (
            "stage_a",
            "stage_b",
            "stage_c_first_rung",
            "stage_c_extended_rung",
            "stage_c_promotion",
        )
        for stage in prohibited:
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(PermissionError, "sealed"):
                    assert_final_source_test_access(contract, stage)
                with self.assertRaisesRegex(PermissionError, "sealed"):
                    allocate_final_source_test_indices(
                        (),
                        (),
                        contract,
                        stage=stage,
                    )
                with self.assertRaisesRegex(PermissionError, "sealed"):
                    validate_final_source_test_indices(
                        (),
                        (),
                        contract,
                        stage=stage,
                    )

        assert_final_source_test_access(contract, "final_confirmatory_source")
        with self.assertRaisesRegex(ValueError, "unknown research stage"):
            assert_final_source_test_access(contract, "target_evaluation")

    def test_allocator_rejects_wrong_dataset_after_access_is_authorized(
        self,
    ) -> None:
        contract = load_confirmatory_contract()
        with self.assertRaisesRegex(ValueError, "50,000"):
            allocate_final_source_test_indices(
                (),
                (),
                contract,
                stage="final_confirmatory_source",
            )

    @unittest.skipUnless(
        Path("data/cifar10/cifar-10-batches-py/data_batch_1").is_file(),
        "local CIFAR-10 labels are unavailable",
    )
    def test_pinned_final_source_test_digest_matches_local_cifar(self) -> None:
        train_labels, test_labels = _official_cifar_labels()
        contract = load_confirmatory_contract()
        first = allocate_final_source_test_indices(
            train_labels,
            test_labels,
            contract,
            stage="final_confirmatory_source",
        )
        selected = validate_final_source_test_indices(
            train_labels,
            test_labels,
            contract,
            stage="final_confirmatory_source",
        )
        payload = json.loads(DEFAULT_CONFIRMATORY_CONTRACT_PATH.read_text())
        split = build_cifar_split(
            train_labels,
            test_labels,
            40000,
            4000,
            1000,
            1000,
            20260727,
        )
        used = set(split.victim_fit) | set(split.policy_train) | set(
            split.source_validation
        )
        complement = set(range(50_000)) - used
        candidate_order: list[int] = []
        for label in range(10):
            class_order = [
                index
                for index, observed in enumerate(train_labels)
                if observed == label
            ]
            random.Random(20260727 + label).shuffle(class_order)
            candidate_order.extend(
                index for index in class_order if index in complement
            )

        self.assertEqual(first, selected)
        self.assertEqual(split.digest, payload["dataset"]["base_split_digest"])
        self.assertEqual(
            indices_digest(split.victim_fit),
            payload["dataset"]["role_digests"]["victim_fit"],
        )
        self.assertEqual(
            indices_digest(split.policy_train),
            payload["dataset"]["role_digests"]["policy_train"],
        )
        self.assertEqual(
            indices_digest(split.source_validation),
            payload["dataset"]["role_digests"]["source_validation"],
        )
        self.assertEqual(
            indices_digest(split.outer_test),
            payload["dataset"]["role_digests"]["outer_test"],
        )
        self.assertEqual(len(complement), 5000)
        self.assertEqual(set(candidate_order), complement)
        self.assertEqual(
            indices_digest(candidate_order),
            payload["final_source_test"]["candidate_indices_digest"],
        )
        self.assertTrue(set(selected).issubset(complement))
        self.assertEqual(
            indices_digest(selected),
            contract.final_source_test_digest,
        )
        self.assertEqual(
            Counter(train_labels[index] for index in selected),
            Counter({label: 100 for label in range(10)}),
        )


if __name__ == "__main__":
    unittest.main()
