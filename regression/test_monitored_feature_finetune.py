"""Regression checks for confidence-preservation promotion gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

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

    def test_newly_recovered_reference_does_not_fail_preservation(self):
        current = challenge_report({"a": 0.95, "b": 0.90, "recovered": 0.88})
        result = confidence_preservation(self.config, "challenge", self.baseline, current)
        self.assertTrue(result["passed"])
        self.assertEqual(result["current_references"], 3)


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

    def test_reviewed_pilot_config_pins_scope_and_zero_exclusion_gates(self):
        config = monitored.load(monitored.ROOT / "regression/legal_notice_v15_reviewed_cls_pilot_config.json")
        monitored.validate_config(config)
        self.assertEqual(config["trainable_scope"]["head_prefixes"], ["model.23.cv3."])
        self.assertEqual(config["stopping"]["fixed_max_excluded_accepted"], 0)
        self.assertEqual(config["training"]["full_evaluation_period"], 1)

    def test_negative_region_map_contains_all_eight_reviewed_regions(self):
        config = monitored.load(monitored.ROOT / "regression/legal_notice_v15_reviewed_cls_pilot_config.json")
        regions = monitored.negative_region_map(config)
        self.assertEqual(sum(map(len, regions.values())), 8)


class BalancedSamplerTests(unittest.TestCase):
    class Dataset:
        labels = [{"cls": np.ones((1, 1))} for _ in range(4)] + [{"cls": np.empty((0, 1))} for _ in range(6)]

    def test_order_is_reproducible_per_epoch_and_changes_between_epochs(self):
        first = monitored.BalancedPageSampler(self.Dataset(), seed=7)
        second = monitored.BalancedPageSampler(self.Dataset(), seed=7)
        self.assertEqual(first.epoch_order(), second.epoch_order())
        first.set_epoch(1)
        self.assertNotEqual(first.epoch_order(), second.epoch_order())
        self.assertEqual(set(first.epoch_order()), set(range(10)))

    def test_positive_and_background_pages_are_spread(self):
        sampler = monitored.BalancedPageSampler(self.Dataset(), seed=0)
        roles = [index < 4 for index in sampler.epoch_order()]
        self.assertTrue(any(roles[:5]))
        self.assertTrue(any(not role for role in roles[:5]))
        self.assertTrue(any(roles[5:]))
        self.assertTrue(any(not role for role in roles[5:]))


class CheckpointSerializationTests(unittest.TestCase):
    def test_training_criteria_are_detached_before_base_serializer_runs(self):
        trainer = object.__new__(monitored.DifferentialFeatureTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        trainer.model.criterion = object()
        trainer.ema = SimpleNamespace(ema=torch.nn.Linear(2, 1))
        trainer.ema.ema.criterion = object()
        model_criterion = trainer.model.criterion
        ema_criterion = trainer.ema.ema.criterion

        def serializer(instance):
            self.assertIsNone(instance.model.criterion)
            self.assertIsNone(instance.ema.ema.criterion)
            return "saved"

        with patch.object(monitored.ActiveHeadTrainer, "save_model", serializer):
            self.assertEqual(trainer.save_model(), "saved")
        self.assertIs(trainer.model.criterion, model_criterion)
        self.assertIs(trainer.ema.ema.criterion, ema_criterion)


if __name__ == "__main__":
    unittest.main()
