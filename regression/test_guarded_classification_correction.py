"""Focused tests for the V16 guarded classification correction contract."""
from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image
import torch

from guarded_classification_correction import (
    accepted_transaction,
    backward_global_objective,
    diagnostic_gate,
    compare_scans,
    frozen_state_snapshot,
    greedy_matches,
    lower_hinge,
    preprocess,
    restore_training_state,
    snapshot_training_state,
    upper_hinge,
    verify_frozen_state,
)
from analyze_v16_gradient_feasibility import feasibility_report
from guarded_trust_region_correction import adjusted_radius


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 1
        self.cv3 = torch.nn.ModuleList([torch.nn.Conv2d(1, 1, 1, bias=False) for _ in range(3)])
        for layer in self.cv3:
            torch.nn.init.constant_(layer.weight, 0.0)


def scan(recovery: float, negative: float, preservation_failures: int = 0, new_background: int = 0) -> dict:
    return {
        "recoveries": [{"id": "recovery", "baseline": 0.4, "current": recovery}],
        "explicit_regions": [{"id": "negative", "baseline": 0.9, "current": negative}],
        "metrics": {
            "recovery_shortfall": max(0.0, 0.805 - recovery),
            "explicit_negative_excess": max(0.0, negative - 0.49),
            "preservation_failures": preservation_failures,
            "new_background_accepted": new_background,
            "background_confidence_increases": 0,
        },
    }


class GuardedCorrectionTests(unittest.TestCase):
    def test_preprocess_records_exact_odd_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            Image.new("RGB", (617, 1000), "white").save(path)
            tensor, transform = preprocess(path)
        self.assertEqual(list(tensor.shape[-2:]), transform["input_shape"])
        resized_width = round(617 * transform["gain"])
        self.assertEqual(tensor.shape[-1], resized_width + ((1280 - resized_width) % 32))
        self.assertEqual(transform["padding"][0], round(((1280 - resized_width) % 32) / 2 - 0.1))

    def test_matching_is_one_to_one(self):
        references = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32)
        detections = torch.tensor([[0, 0, 10, 10, 0.9, 0]], dtype=torch.float32)
        matches = greedy_matches(references, detections)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(set(matches.values())), 1)

    def test_hinges_push_in_required_directions(self):
        low = torch.tensor([0.0], requires_grad=True)
        lower_hinge(low, 0.8).backward()
        self.assertLess(low.grad.item(), 0)
        high = torch.tensor([3.0], requires_grad=True)
        upper_hinge(high, 0.49).backward()
        self.assertGreater(high.grad.item(), 0)

    def test_global_group_normalization_does_not_average_pages(self):
        head = TinyHead()
        features = [torch.ones(1, 1, 1, 1) for _ in range(3)]
        base = {
            "features": features,
            "explicit_indices": [],
            "background_indices": torch.tensor([1, 2]),
        }
        config = {
            "objectives": {
                "negative_ceiling_probability": 0.49,
                "background_ceiling_probability": 0.79,
                "weights": {"preservation": 1.0, "recovery": 0.0, "explicit_rejection": 0.0, "background_rejection": 0.0},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, group_count in enumerate((1, 3)):
                page = {**base, "references": [{"role": "preservation", "objective_floor_probability": 0.8}] * group_count,
                        "positive_indices": [torch.tensor([0])] * group_count}
                path = Path(directory) / f"{index}.pt"
                torch.save(page, path)
                paths.append(path)
            backward_global_objective(head, paths, config, {"preservation": 4, "recovery": 0, "explicit_rejection": 0, "background_rejection": 2})
        expected = -2 * torch.logit(torch.tensor(0.8)).item()
        self.assertAlmostEqual(head.cv3[0].weight.grad.item(), expected, places=5)

    def test_transaction_restore_includes_optimizer_state(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        snapshot = snapshot_training_state(model, optimizer)
        expected_weight = model.weight.detach().clone()
        expected_state = deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        restore_training_state(model, optimizer, snapshot)
        self.assertTrue(torch.equal(model.weight, expected_weight))
        self.assertEqual(optimizer.state_dict()["state"].keys(), expected_state["state"].keys())
        for key in expected_state["state"]:
            self.assertTrue(torch.equal(optimizer.state_dict()["state"][key]["exp_avg"], expected_state["state"][key]["exp_avg"]))

    def test_frozen_state_guard_includes_parameters_and_buffers(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        model[0].weight.requires_grad_(True)
        model[0].bias.requires_grad_(False)
        snapshot = frozen_state_snapshot(model)
        model[0].weight.data.add_(1)
        verify_frozen_state(model, snapshot)
        model[1].running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, "Frozen parameters or buffers changed"):
            verify_frozen_state(model, snapshot)

    def test_serialized_scan_comparison_checks_continuous_and_discrete_results(self):
        expected = scan(0.5, 0.8)
        self.assertTrue(compare_scans(expected, deepcopy(expected))["passed"])
        changed = deepcopy(expected)
        changed["recoveries"][0]["current"] += 1e-5
        with self.assertRaisesRegex(RuntimeError, "prediction delta"):
            compare_scans(expected, changed)

    def test_transaction_rejects_each_safety_and_direction_failure(self):
        before = scan(0.4, 0.9)
        self.assertTrue(accepted_transaction(before, scan(0.5, 0.8))[0])
        for after in (scan(0.3, 0.8), scan(0.5, 0.95), scan(0.5, 0.8, preservation_failures=1), scan(0.5, 0.8, new_background=1)):
            self.assertFalse(accepted_transaction(before, after)[0])

    def test_diagnostic_gate_requires_progress_in_both_families(self):
        config = {"diagnostic_gate": {
            "minimum_accepted_updates": 5,
            "minimum_recovery_shortfall_reduction": 0.1,
            "minimum_explicit_negative_excess_reduction": 0.1,
            "require_every_recovery_improved": True,
            "require_every_target_negative_not_worse": True,
        }}
        self.assertTrue(diagnostic_gate(config, scan(0.4, 0.9), scan(0.5, 0.8), 5)["passed"])
        self.assertFalse(diagnostic_gate(config, scan(0.4, 0.9), scan(0.5, 0.89), 5)["passed"])

    def test_minimum_norm_direction_identifies_common_descent(self):
        report, weights = feasibility_report(
            [{"id": "a", "kind": "test"}, {"id": "b", "kind": "test"}],
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        )
        self.assertEqual(report["status"], "common_first_order_descent_exists")
        self.assertTrue(torch.allclose(weights, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-6))

    def test_minimum_norm_direction_rejects_opposite_constraints(self):
        report, _ = feasibility_report(
            [{"id": "a", "kind": "test"}, {"id": "b", "kind": "test"}],
            torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
        )
        self.assertEqual(report["status"], "no_common_first_order_descent")

    def test_trust_region_radius_is_bounded_and_directional(self):
        optimizer = {"growth_factor": 1.5, "shrink_factor": 0.5, "minimum_radius": 0.001, "maximum_radius": 0.01}
        self.assertEqual(adjusted_radius(0.008, True, optimizer), 0.01)
        self.assertEqual(adjusted_radius(0.001, False, optimizer), 0.001)
        self.assertEqual(adjusted_radius(0.004, False, optimizer), 0.002)


if __name__ == "__main__":
    unittest.main()
