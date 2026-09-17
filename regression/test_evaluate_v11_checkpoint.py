"""Focused checks for the standalone v11 checkpoint evaluator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_v11_checkpoint as evaluator
from monitored_feature_finetune import REQUIRED_STOPPING_KEYS


class EvaluatorFailureSummaryTests(unittest.TestCase):
    def test_invalid_stopping_config_writes_evaluation_error_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            checkpoint = root / "checkpoint.pt"
            output = root / "evaluation"
            config.write_text(json.dumps({"stopping": {}}), encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")

            with patch.object(evaluator, "run_fixed") as run_fixed:
                with self.assertRaisesRegex(RuntimeError, "Missing required stopping settings"):
                    evaluator.evaluate(config, checkpoint, output)
                run_fixed.assert_not_called()

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "evaluation_error")
            self.assertEqual(summary["config_sha256"], evaluator.file_sha256(config))
            self.assertEqual(summary["checkpoint_sha256"], evaluator.file_sha256(checkpoint))
            self.assertEqual(summary["reports"], {})

    def test_cached_fixed_predictions_must_match_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            checkpoint = root / "checkpoint.pt"
            july_manifest = root / "july.json"
            predictions = root / "fixed.predictions.json"
            output = root / "evaluation"
            stopping = {key: 0 for key in REQUIRED_STOPPING_KEYS}
            july_manifest.write_text(json.dumps({"pages": []}), encoding="utf-8")
            config_path.write_text(json.dumps({
                "stopping": stopping,
                "july_manifest": str(july_manifest),
                "baseline_reports": {},
                "starting_weights_sha256": "baseline",
            }), encoding="utf-8")
            checkpoint.write_bytes(b"selected checkpoint")
            predictions.write_text(json.dumps({"weights_sha256": "different"}), encoding="utf-8")

            with patch.object(evaluator.fixed, "check_candidate") as check_candidate:
                with self.assertRaisesRegex(RuntimeError, "do not belong"):
                    evaluator.evaluate(config_path, checkpoint, output, predictions)
                check_candidate.assert_not_called()

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "evaluation_error")
            self.assertEqual(summary["error"]["type"], "RuntimeError")


class EvaluatorSummaryTests(unittest.TestCase):
    def test_completed_summary_attests_all_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            checkpoint = root / "checkpoint.pt"
            july_manifest = root / "july.json"
            output = root / "evaluation"
            checkpoint.write_bytes(b"selected checkpoint")
            checkpoint_sha256 = evaluator.file_sha256(checkpoint)
            july_manifest.write_text(json.dumps({"pages": []}), encoding="utf-8")

            baseline_reports = {}
            baseline_sha256 = "baseline"
            for suite in ("fixed", "challenge", "july"):
                path = root / f"baseline_{suite}.json"
                report = {"weights_sha256": baseline_sha256} if suite == "july" else {"candidate": {"weights_sha256": baseline_sha256}}
                path.write_text(json.dumps(report), encoding="utf-8")
                baseline_reports[suite] = {"path": str(path), "sha256": evaluator.file_sha256(path)}
            config_path.write_text(json.dumps({
                "stopping": {key: 0 for key in REQUIRED_STOPPING_KEYS},
                "july_manifest": str(july_manifest),
                "fixed_manifest": "fixed.json",
                "challenge_manifest": "challenge.json",
                "baseline_reports": baseline_reports,
                "starting_weights_sha256": baseline_sha256,
            }), encoding="utf-8")

            reports = {
                "fixed": {"candidate": {"weights_sha256": checkpoint_sha256}, "totals": {"valid": 1}},
                "challenge": {"candidate": {"weights_sha256": checkpoint_sha256}, "totals": {"valid": 2}},
                "july": {"weights_sha256": checkpoint_sha256, "totals": {"valid": 3}},
                "target": {"weights_sha256": checkpoint_sha256, "totals": {"valid": 4}, "passed": True},
            }

            def generated(report, output_index):
                def run(*args):
                    evaluator.write(args[output_index], report)
                    return report
                return run

            with patch.object(evaluator, "run_fixed", side_effect=generated(reports["fixed"], 2)), \
                    patch.object(evaluator, "run_challenge", side_effect=generated(reports["challenge"], 2)), \
                    patch.object(evaluator, "evaluate_july", side_effect=generated(reports["july"], 3)), \
                    patch.object(evaluator, "evaluate_target", side_effect=generated(reports["target"], 2)), \
                    patch.object(evaluator, "preservation_failures", return_value=([], {})), \
                    patch.object(evaluator.evaluation_cache, "IssueScopedCache"):
                summary = evaluator.evaluate(config_path, checkpoint, output)

            self.assertEqual(summary["status"], "promotion_passed")
            self.assertEqual(summary["config_sha256"], evaluator.file_sha256(config_path))
            self.assertEqual(set(summary["reports"]), set(reports))
            for suite, artifact in summary["reports"].items():
                report_path = evaluator.project_path(artifact["path"])
                self.assertEqual(artifact["sha256"], evaluator.file_sha256(report_path))
                self.assertEqual(artifact["totals"], reports[suite]["totals"])


if __name__ == "__main__":
    unittest.main()
