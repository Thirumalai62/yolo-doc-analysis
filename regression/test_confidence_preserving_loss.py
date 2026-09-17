"""Focused checks for the foreground target floor used by the guarded pilot."""
import unittest

import torch

from confidence_preserving_loss import (
    BaselineForegroundFloorLoss,
    FrozenTeacherForegroundFloorLoss,
    PinnedTeacherReviewedClassificationLoss,
)


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

    def test_full_teacher_loss_uses_the_same_foreground_floor(self):
        native = torch.tensor([[[0.1], [0.0], [0.9]]])
        foreground = torch.tensor([[True, False, True]])
        teacher = torch.tensor([[[0.8], [0.95], [0.7]]])
        protected = FrozenTeacherForegroundFloorLoss.protected_targets(native, foreground, teacher)
        torch.testing.assert_close(protected, torch.tensor([[[0.8], [0.0], [0.9]]]))

    def test_preservation_and_recovery_hinge_raise_low_group_maximum(self):
        logits = torch.tensor([-2.0, -1.0], requires_grad=True)
        loss = PinnedTeacherReviewedClassificationLoss.lower_hinge(logits, 0.805)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertLess(float(logits.grad[1]), 0)
        self.assertEqual(float(logits.grad[0]), 0)

    def test_rejection_hinge_lowers_high_group_maximum(self):
        logits = torch.tensor([0.2, 2.0], requires_grad=True)
        loss = PinnedTeacherReviewedClassificationLoss.upper_hinge(logits, 0.49)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertGreater(float(logits.grad[1]), 0)
        self.assertEqual(float(logits.grad[0]), 0)

    def test_group_normalization_is_independent_of_anchor_count(self):
        zero = torch.tensor(0.0, requires_grad=True)
        one_anchor = [torch.tensor(4.0), torch.tensor(2.0)]
        many_anchors = [one_anchor[0].repeat(1000).mean(), one_anchor[1].repeat(1000).mean()]
        first = PinnedTeacherReviewedClassificationLoss.separately_normalized(one_anchor, zero)
        second = PinnedTeacherReviewedClassificationLoss.separately_normalized(many_anchors, zero)
        torch.testing.assert_close(first, second)

    def test_empty_group_returns_graph_connected_zero(self):
        source = torch.tensor(2.0, requires_grad=True)
        loss = PinnedTeacherReviewedClassificationLoss.separately_normalized([], source * 0)
        loss.backward()
        self.assertEqual(float(source.grad), 0.0)

    def test_region_selection_uses_prediction_area_fraction(self):
        boxes = torch.tensor([[1.0, 1.0, 5.0, 5.0], [4.0, 4.0, 8.0, 8.0]])
        region = torch.tensor([0.0, 0.0, 6.0, 6.0])
        selected = PinnedTeacherReviewedClassificationLoss.prediction_inside_region(boxes, region, 0.5)
        torch.testing.assert_close(selected, torch.tensor([True, False]))

    def test_region_transform_accepts_scalar_rectangular_gain(self):
        batch = {"ori_shape": ((100, 200),), "ratio_pad": ((2.0, (3, 5)),), "img": torch.zeros(1, 3, 210, 410)}
        region = PinnedTeacherReviewedClassificationLoss.transformed_region(
            batch, 0, [0.1, 0.2, 0.5, 0.8], torch.device("cpu")
        )
        torch.testing.assert_close(region, torch.tensor([43.0, 45.0, 203.0, 165.0]))

    def test_region_transform_accepts_real_collated_gain_pair(self):
        batch = {"ori_shape": ((100, 200),), "ratio_pad": ((0.5, 0.4),), "img": torch.zeros(1, 3, 60, 100)}
        region = PinnedTeacherReviewedClassificationLoss.transformed_region(
            batch, 0, [0.1, 0.2, 0.5, 0.8], torch.device("cpu")
        )
        torch.testing.assert_close(region, torch.tensor([18.0, 15.0, 50.0, 45.0]))


if __name__ == "__main__":
    unittest.main()
