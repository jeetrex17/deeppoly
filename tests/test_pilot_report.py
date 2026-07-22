import json
import tempfile
import unittest
from pathlib import Path

from rl_transfer.pilot_report import PilotReportError, load_pilot_results, write_pilot_report


class PilotReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results_path = Path("docs/research/cifar10_m4_pilot_results.json")

    def test_report_writes_svg_figures_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_pilot_report(self.results_path, Path(directory))
            for path in outputs.values():
                self.assertTrue(path.is_file())
            victim_svg = outputs["victim_accuracy"].read_text()
            asr_svg = outputs["asr_by_query"].read_text()
            pipeline_svg = outputs["pipeline"].read_text()
            summary = outputs["summary"].read_text()
        self.assertIn("<svg", victim_svg)
        self.assertIn("66.3%", victim_svg)
        self.assertIn("Random-action baseline", asr_svg)
        self.assertIn("Freeze policy", pipeline_svg)
        self.assertIn("5/99 successful", summary)
        self.assertIn("does **not** show an RL transfer advantage", summary)

    def test_load_rejects_an_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-fields.json"
            path.write_text(json.dumps({"name": "bad"}))
            with self.assertRaises(PilotReportError):
                load_pilot_results(path)


if __name__ == "__main__":
    unittest.main()
