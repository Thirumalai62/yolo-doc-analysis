"""Regression checks for confidence-preservation promotion gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import monitored_feature_finetune as monitored
from monitored_feature_finetune import confidence_preservation


def challenge_report(scores):
    return {
        "pages": [{
            "valid_matches": [
                {"reference_id": reference, "prediction": {"confidence": confidence}}
                for reference, confidence in scores.items()
            ]
        }]
    }


class ConfidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.config = {"stopping": {
            "maximum_median_valid_confidence_drop": 0.01,
            "maximum_reference_confidence_drop": 0.05,
        }}
        self.baseline = challenge_report({"a": 0.95, "b": 0.90})

    def test_unchanged_confidences_pass(self):
        result = confidence_preservation(self.config, "challenge", self.baseline, self.baseline)
        self.assertTrue(result["passed"])

    def test_broad_confidence_drop_fails_above_threshold(self):
        current = challenge_report({"a": 0.93, "b": 0.88})
        result = confidence_preservation(self.config, "challenge", self.baseline, current)
        self.assertFalse(result["passed"])

    def test_missing_reference_fails_even_when_remaining_score_is_stable(self):
        current = challenge_report({"a": 0.95})
        result = confidence_preservation(self.config, "challenge", self.baseline, current)
        self.assertFalse(result["passed"])


class EvaluationIntegrityTests(unittest.TestCase):
    def test_required_stopping_settings_are_validated_together(self):
        with self.assertRaisesRegex(RuntimeError, "maximum_median_valid_confidence_drop"):
            monitored.validate_stopping_config({"stopping": {}})

    def test_configured_confidence_drop_thresholds_are_present(self):
        config = monitored.load(monitored.DEFAULT_CONFIG)
        monitored.validate_stopping_config(config)
        self.assertEqual(config["stopping"]["maximum_median_valid_confidence_drop"], 0.01)
        self.assertEqual(config["stopping"]["maximum_reference_confidence_drop"], 0.05)

    def test_report_checkpoint_sha_uses_each_report_schema(self):
        monitored.verify_report_checkpoint_sha("fixed", {"candidate": {"weights_sha256": "pinned"}}, "pinned")
        monitored.verify_report_checkpoint_sha("july", {"weights_sha256": "pinned"}, "pinned")
        with self.assertRaisesRegex(RuntimeError, "pinned checkpoint"):
            monitored.verify_report_checkpoint_sha("challenge", {"candidate": {"weights_sha256": "other"}}, "pinned")

    def test_target_hashes_are_checked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "checkpoint.pt"
            image = root / "page.png"
            label = root / "page.txt"
            manifest_path = root / "manifest.json"
            weights.write_bytes(b"checkpoint")
            image.write_bytes(b"image")
            label.write_text("", encoding="utf-8")
            for changed_key in ("image", "label"):
                page = {
                    "image": str(image),
                    "label": str(label),
                    "image_sha256": monitored.file_sha256(image),
                    "label_sha256": monitored.file_sha256(label),
                }
                page[f"{changed_key}_sha256"] = "0" * 64
                manifest_path.write_text(json.dumps({"pages": [page]}), encoding="utf-8")
                config = {
                    "target_manifest": str(manifest_path),
                    "target_manifest_sha256": monitored.file_sha256(manifest_path),
                }
                with self.subTest(changed_key=changed_key), patch.object(monitored, "YOLO") as yolo:
                    with self.assertRaisesRegex(RuntimeError, changed_key):
                        monitored.evaluate_target(weights, config, root / "report.json")
                    yolo.assert_not_called()

    def test_critical_boundary_count_must_match_requirement(self):
        stopping = {
            "fixed_valid_required": 1,
            "fixed_max_excluded_accepted": 0,
            "critical_boundaries_required": 4,
            "challenge_valid_required": 1,
            "challenge_max_excluded_accepted_regions": 0,
            "july_valid_required": 1,
            "july_max_excluded_accepted_regions": 0,
            "gulf_recovery_targets_required": 1,
        }
        config = {"stopping": stopping, "semantic_extra_reviews": "unused.json"}
        fixed_report = {"totals": {"valid_preserved": 1, "valid_missed": 0, "critical_boundaries": 3, "critical_boundary_failures": 0, "excluded_accepted": 0}, "pages": []}
        challenge_report = {"totals": {"valid_preserved": 1, "valid_missed": 0, "unreviewed_accepted": 0, "excluded_accepted_regions": 0}, "pages": []}
        july_report = {"totals": {"valid_preserved": 1, "excluded_accepted_regions": 0}, "pages": [], "recovery_targets": [{"complete_accepted": True}]}
        baselines = {"fixed": {}, "challenge": {}, "july": {"pages": []}}
        passing_confidence = {"passed": True}
        with patch.object(monitored, "load", return_value={"reviews": []}), patch.object(monitored, "confidence_preservation", return_value=passing_confidence):
            failures, _ = monitored.preservation_failures(config, fixed_report, challenge_report, july_report, baselines)
        self.assertIn("fixed_preservation", failures)


if __name__ == "__main__":
    unittest.main()
