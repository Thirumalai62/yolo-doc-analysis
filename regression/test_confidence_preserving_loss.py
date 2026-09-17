"""Focused checks for the foreground target floor used by the guarded pilot."""
import unittest

import torch

from confidence_preserving_loss import BaselineForegroundFloorLoss


class ConfidencePreservingLossTests(unittest.TestCase):
    def test_only_foreground_targets_receive_baseline_floor(self):
        native = torch.tensor([[[0.2], [0.0], [0.8]]])
        foreground = torch.tensor([[True, False, True]])
        baseline = torch.tensor([[[0.9], [0.7], [0.6]]])
        protected = BaselineForegroundFloorLoss.protected_targets(native, foreground, baseline)
        torch.testing.assert_close(protected, torch.tensor([[[0.9], [0.0], [0.8]]]))

    def test_initial_foreground_gradient_cannot_lower_baseline_logit(self):
        baseline = torch.tensor([[[0.9], [0.6]]])
        logits = torch.logit(baseline).requires_grad_()
        native = torch.tensor([[[0.3], [0.8]]])
        foreground = torch.tensor([[True, True]])
        targets = BaselineForegroundFloorLoss.protected_targets(native, foreground, baseline)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum").backward()
        self.assertTrue(torch.all(logits.grad <= 0))

    def test_background_gradient_remains_native_bce(self):
        baseline = torch.tensor([[[0.9], [0.6]]])
        logits = torch.tensor([[[1.0], [-1.0]]], requires_grad=True)
        native = torch.zeros_like(logits)
        foreground = torch.tensor([[False, False]])
        targets = BaselineForegroundFloorLoss.protected_targets(native, foreground, baseline)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum").backward()
        torch.testing.assert_close(logits.grad, logits.detach().sigmoid())


if __name__ == "__main__":
    unittest.main()
