import json
import unittest
from pathlib import Path


class MacNotebookTests(unittest.TestCase):
    def test_notebook_is_thin_portable_and_safe_by_default(self) -> None:
        notebook = json.loads(Path("notebooks/cifar10_mac_pilot.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("RUN_TRAINING = False", sources)
        self.assertIn("rl_transfer.cifar_cli", sources)
        self.assertNotIn("/Users/", sources)
        self.assertNotIn("!pip", sources)
        self.assertNotIn("class RecurrentAttackPolicy", sources)

    def test_committed_pilot_snapshot_is_explicitly_not_research_valid(self) -> None:
        snapshot = json.loads(
            Path("docs/research/cifar10_m4_pilot_results.json").read_text()
        )
        self.assertFalse(snapshot["research_valid"])
        self.assertTrue(snapshot["victim_accuracy_gate"]["passed"])
        evaluation = snapshot["evaluation"]
        self.assertEqual(
            set(evaluation),
            {
                "groupdro_recurrent_ppo",
                "groupdro_recurrent_ppo_stochastic",
                "fixed_action",
                "random_action",
            },
        )
        self.assertEqual({metrics["eligible"] for metrics in evaluation.values()}, {99})
        self.assertTrue(all(metrics["frozen"] for metrics in evaluation.values()))

    def test_multi_fold_study_notebook_is_thin_and_package_backed(self) -> None:
        notebook = json.loads(Path("notebooks/cifar10_m4_study.ipynb").read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("rl_transfer.cifar_study_cli", code)
        self.assertIn("promotion_gate", code)
        self.assertNotIn("class RecurrentAttackPolicy", code)


if __name__ == "__main__":
    unittest.main()
