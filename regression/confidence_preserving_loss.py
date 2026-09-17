"""Reviewed classification objectives for confidence-preserving detector training."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.tal import make_anchors


class BaselineForegroundFloorLoss:
    """Prevent native BCE targets from directly lowering baseline foreground logits."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10):
        original_args = model.args
        if isinstance(original_args, dict):
            model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **original_args})
        try:
            self.native = v8DetectionLoss(model, tal_topk=tal_topk)
        finally:
            model.args = original_args
        self.baseline_cv3 = copy.deepcopy(model.model[-1].cv3).to(self.native.device).eval()
        self.baseline_cv3.requires_grad_(False)
        self.last_diagnostics: dict[str, float | int] = {}

    @staticmethod
    def protected_targets(
        native_targets: torch.Tensor,
        foreground: torch.Tensor,
        baseline_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            foreground.unsqueeze(-1),
            torch.maximum(native_targets, baseline_probabilities),
            native_targets,
        )

    def __call__(self, predictions, batch):
        predictions = self.native.parse_output(predictions)["one2many"]
        batch_size = predictions["boxes"].shape[0]
        with torch.no_grad():
            baseline_scores = torch.cat(
                [head(feature).view(batch_size, self.native.nc, -1) for head, feature in zip(self.baseline_cv3, predictions["feats"])],
                dim=-1,
            ).permute(0, 2, 1).contiguous()
            baseline_probabilities = baseline_scores.sigmoid()

        assignments = []

        def use_baseline_scores(_module, args):
            return (baseline_probabilities, *args[1:])

        pre_hook = self.native.assigner.register_forward_pre_hook(use_baseline_scores)
        post_hook = self.native.assigner.register_forward_hook(lambda _module, _args, output: assignments.append(output))
        try:
            _, loss, _ = self.native.get_assigned_targets_and_loss(predictions, batch)
        finally:
            pre_hook.remove()
            post_hook.remove()

        if len(assignments) != 1:
            raise RuntimeError("Expected exactly one baseline task-aligned assignment")
        native_targets, foreground = assignments[0][2], assignments[0][3]
        protected = self.protected_targets(native_targets, foreground, baseline_probabilities)
        student_scores = predictions["scores"].permute(0, 2, 1).contiguous()
        denominator = torch.clamp(native_targets.sum(), min=1)
        native_cls = self.native.bce(student_scores, native_targets.to(student_scores.dtype)).sum() / denominator
        protected_cls = self.native.bce(student_scores, protected.to(student_scores.dtype)).sum() / denominator
        loss[1] += (protected_cls - native_cls) * self.native.hyp.cls
        detached = dict(zip(self.native.loss_names, loss.detach()))

        protected_mask = foreground.unsqueeze(-1) & (protected > native_targets)
        self.last_diagnostics = {
            "foreground_anchors": int(foreground.sum()),
            "protected_logits": int(protected_mask.sum()),
            "maximum_target_floor_increase": float((protected - native_targets).max()) if protected.numel() else 0.0,
        }
        return loss * batch_size, detached


class FrozenTeacherForegroundFloorLoss:
    """Use a complete frozen detector to protect labeled foreground assignments."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10):
        original_args = model.args
        if isinstance(original_args, dict):
            model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **original_args})
        try:
            self.native = v8DetectionLoss(model, tal_topk=tal_topk)
        finally:
            model.args = original_args
        self.teacher = copy.deepcopy(model).to(self.native.device).eval()
        self.teacher.end2end = False
        self.teacher.requires_grad_(False)
        self.teacher.criterion = None
        self.last_diagnostics: dict[str, float | int] = {}

    protected_targets = staticmethod(BaselineForegroundFloorLoss.protected_targets)

    def teacher_assignment_inputs(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen-teacher probabilities and decoded pixel-space boxes."""
        self.teacher.eval()
        with torch.no_grad():
            output = self.native.parse_output(self.teacher(images))["one2many"]
            batch_size = output["boxes"].shape[0]
            scores = output["scores"].view(batch_size, self.native.nc, -1).permute(0, 2, 1).contiguous().sigmoid()
            distributions = output["boxes"].view(batch_size, self.native.reg_max * 4, -1).permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(output["feats"], self.native.stride, 0.5)
            boxes = self.native.bbox_decode(anchor_points, distributions) * stride_tensor
        return scores, boxes

    def __call__(self, predictions, batch):
        predictions = self.native.parse_output(predictions)["one2many"]
        batch_size = predictions["boxes"].shape[0]
        teacher_probabilities, teacher_boxes = self.teacher_assignment_inputs(batch["img"])
        assignments = []

        def use_teacher(_module, args):
            if teacher_probabilities.shape != args[0].shape or teacher_boxes.shape != args[1].shape:
                raise RuntimeError("Frozen teacher and student anchor shapes differ")
            return (teacher_probabilities, teacher_boxes.to(args[1].dtype), *args[2:])

        pre_hook = self.native.assigner.register_forward_pre_hook(use_teacher)
        post_hook = self.native.assigner.register_forward_hook(lambda _module, _args, output: assignments.append(output))
        try:
            _, loss, _ = self.native.get_assigned_targets_and_loss(predictions, batch)
        finally:
            pre_hook.remove()
            post_hook.remove()

        if len(assignments) != 1:
            raise RuntimeError("Expected exactly one frozen-teacher task-aligned assignment")
        native_targets, foreground = assignments[0][2], assignments[0][3]
        protected = self.protected_targets(native_targets, foreground, teacher_probabilities)
        student_scores = predictions["scores"].permute(0, 2, 1).contiguous()
        denominator = torch.clamp(native_targets.sum(), min=1)
        native_cls = self.native.bce(student_scores, native_targets.to(student_scores.dtype)).sum() / denominator
        protected_cls = self.native.bce(student_scores, protected.to(student_scores.dtype)).sum() / denominator
        loss[1] += (protected_cls - native_cls) * self.native.hyp.cls
        detached = dict(zip(self.native.loss_names, loss.detach()))

        protected_mask = foreground.unsqueeze(-1) & (protected > native_targets)
        self.last_diagnostics = {
            "foreground_anchors": int(foreground.sum()),
            "protected_logits": int(protected_mask.sum()),
            "maximum_target_floor_increase": float((protected - native_targets).max()) if protected.numel() else 0.0,
            "maximum_teacher_probability": float(teacher_probabilities.max()) if teacher_probabilities.numel() else 0.0,
        }
        return loss * batch_size, detached


class PinnedTeacherReviewedClassificationLoss:
    """Train reviewed label-level preservation, recovery, and rejection groups."""

    loss_names = ("preservation_loss", "recovery_loss", "rejection_loss")

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        config: dict,
        negative_regions: dict[str, list[list[float]]] | None = None,
    ):
        original_args = student_model.args
        if isinstance(original_args, dict):
            student_model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **original_args})
        try:
            self.native = v8DetectionLoss(student_model, tal_topk=10)
        finally:
            student_model.args = original_args
        self.teacher = teacher_model.to(self.native.device).eval()
        self.teacher.end2end = False
        self.teacher.requires_grad_(False)
        self.teacher.criterion = None
        self.config = config
        self.negative_regions = negative_regions or {}
        self.last_diagnostics: dict[str, float | int] = {}

    @staticmethod
    def separately_normalized(groups: list[torch.Tensor], connected_zero: torch.Tensor) -> torch.Tensor:
        """Average one scalar per reviewed group without cross-family dilution."""
        return torch.stack(groups).mean() if groups else connected_zero

    @staticmethod
    def lower_hinge(logits: torch.Tensor, probability: float | torch.Tensor) -> torch.Tensor:
        target = torch.as_tensor(probability, dtype=logits.dtype, device=logits.device).clamp(1e-6, 1 - 1e-6)
        return torch.relu(torch.logit(target) - logits.max()).square()

    @staticmethod
    def upper_hinge(logits: torch.Tensor, probability: float) -> torch.Tensor:
        target = torch.as_tensor(probability, dtype=logits.dtype, device=logits.device).clamp(1e-6, 1 - 1e-6)
        return torch.relu(logits.max() - torch.logit(target)).square()

    @staticmethod
    def transformed_region(batch: dict, image_index: int, xyxyn: list[float], device: torch.device) -> torch.Tensor:
        """Map an original-image normalized region into the letterboxed training tensor."""
        height, width = batch["ori_shape"][image_index]
        ratio_pad = batch["ratio_pad"][image_index]
        if all(isinstance(value, (int, float)) for value in ratio_pad):
            gain_h, gain_w = ratio_pad
            input_height, input_width = batch["img"].shape[-2:]
            pad_x = (input_width - width * gain_w) / 2
            pad_y = (input_height - height * gain_h) / 2
        else:
            gains, padding = ratio_pad
            if isinstance(gains, (tuple, list)):
                gain_h, gain_w = gains
            else:
                gain_h = gain_w = gains
            pad_x, pad_y = padding
        x1, y1, x2, y2 = xyxyn
        return torch.tensor(
            [x1 * width * gain_w + pad_x, y1 * height * gain_h + pad_y,
             x2 * width * gain_w + pad_x, y2 * height * gain_h + pad_y],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def prediction_inside_region(boxes: torch.Tensor, region: torch.Tensor, minimum_fraction: float) -> torch.Tensor:
        intersection_wh = (torch.minimum(boxes[:, 2:], region[2:]) - torch.maximum(boxes[:, :2], region[:2])).clamp(min=0)
        intersection_area = intersection_wh.prod(1)
        box_area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1).clamp(min=1)
        return intersection_area / box_area >= minimum_fraction

    def decoded_teacher(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.teacher.eval()
        with torch.no_grad():
            output = self.native.parse_output(self.teacher(images))["one2many"]
            batch_size = output["boxes"].shape[0]
            logits = output["scores"].view(batch_size, self.native.nc, -1).permute(0, 2, 1).contiguous()
            distributions = output["boxes"].view(batch_size, self.native.reg_max * 4, -1).permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(output["feats"], self.native.stride, 0.5)
            boxes = self.native.bbox_decode(anchor_points, distributions) * stride_tensor
        return logits, boxes, distributions

    def __call__(self, predictions, batch):
        student = self.native.parse_output(predictions)["one2many"]
        batch_size = student["boxes"].shape[0]
        student_logits = student["scores"].view(batch_size, self.native.nc, -1).permute(0, 2, 1).contiguous()
        student_distributions = student["boxes"].view(batch_size, self.native.reg_max * 4, -1).permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(student["feats"], self.native.stride, 0.5)
        student_boxes = self.native.bbox_decode(anchor_points, student_distributions) * stride_tensor
        teacher_logits, teacher_boxes, teacher_distributions = self.decoded_teacher(batch["img"])
        if student_logits.shape != teacher_logits.shape or student_boxes.shape != teacher_boxes.shape:
            raise RuntimeError("Pinned teacher and student output shapes differ")
        tolerance = self.config["maximum_box_output_delta"]
        maximum_box_delta = float((student_distributions.detach() - teacher_distributions).abs().max())
        if maximum_box_delta > tolerance:
            raise RuntimeError(f"Frozen box outputs drifted by {maximum_box_delta:.8g}")

        image_size = torch.tensor(student["feats"][0].shape[2:], device=self.native.device, dtype=student_logits.dtype)
        image_size = image_size * self.native.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.native.preprocess(targets.to(self.native.device), batch_size, scale_tensor=image_size[[1, 0, 1, 0]])
        _, ground_truth = targets.split((1, 4), 2)
        ground_truth_mask = ground_truth.sum(2) > 0

        preservation, recovery, rejection = [], [], []
        preservation_probabilities, recovery_probabilities = [], []
        explicit_regions = 0
        config = self.config
        for image_index in range(batch_size):
            valid_boxes = ground_truth[image_index][ground_truth_mask[image_index]]
            overlaps = box_iou(teacher_boxes[image_index], valid_boxes) if len(valid_boxes) else torch.zeros(
                (teacher_boxes.shape[1], 0), dtype=teacher_boxes.dtype, device=teacher_boxes.device
            )
            maximum_valid_overlap = overlaps.max(1).values if len(valid_boxes) else torch.zeros(
                teacher_boxes.shape[1], dtype=teacher_boxes.dtype, device=teacher_boxes.device
            )
            probabilities = teacher_logits[image_index].sigmoid().amax(1)
            logits = student_logits[image_index].amax(1)
            for label_index in range(len(valid_boxes)):
                matching = overlaps[:, label_index] >= config["positive_match_iou"]
                if not matching.any():
                    raise RuntimeError(
                        f"No frozen box reaches IoU {config['positive_match_iou']} for "
                        f"{Path(batch['im_file'][image_index]).name} label {label_index + 1}"
                    )
                teacher_probability = probabilities[matching].max()
                if teacher_probability >= config["acceptance_probability"]:
                    lower = torch.maximum(
                        teacher_probability - config["maximum_valid_probability_drop"],
                        torch.as_tensor(config["minimum_valid_probability"], device=teacher_probability.device),
                    )
                    preservation.append(self.lower_hinge(logits[matching], lower))
                    preservation_probabilities.append(float(teacher_probability))
                else:
                    recovery.append(self.lower_hinge(logits[matching], config["minimum_valid_probability"]))
                    recovery_probabilities.append(float(teacher_probability))

            unmatched = maximum_valid_overlap < config["negative_max_valid_iou"]
            noticeable = probabilities >= config["background_candidate_probability"]
            page_background = unmatched & noticeable
            if page_background.any():
                rejection.append(self.upper_hinge(logits[page_background], config["negative_ceiling_probability"]))

            image_name = Path(batch["im_file"][image_index]).name
            for region_xyxyn in self.negative_regions.get(image_name, []):
                region = self.transformed_region(batch, image_index, region_xyxyn, teacher_boxes.device)
                selected = (
                    self.prediction_inside_region(teacher_boxes[image_index], region, config["negative_region_overlap"])
                    & unmatched & noticeable
                )
                if not selected.any():
                    raise RuntimeError(f"No frozen candidate in reviewed negative region for {image_name}")
                rejection.append(self.upper_hinge(logits[selected], config["negative_ceiling_probability"]))
                explicit_regions += 1

        connected_zero = student_logits.sum() * 0
        weights = config["weights"]
        losses = torch.stack((
            weights["preservation"] * self.separately_normalized(preservation, connected_zero),
            weights["recovery"] * self.separately_normalized(recovery, connected_zero),
            weights["rejection"] * self.separately_normalized(rejection, connected_zero),
        ))
        self.last_diagnostics = {
            "preservation_groups": len(preservation),
            "recovery_groups": len(recovery),
            "rejection_groups": len(rejection),
            "explicit_negative_regions": explicit_regions,
            "minimum_preserved_teacher_probability": min(preservation_probabilities, default=0.0),
            "maximum_recovery_teacher_probability": max(recovery_probabilities, default=0.0),
            "maximum_box_output_delta": maximum_box_delta,
        }
        detached = dict(zip(self.loss_names, losses.detach()))
        return losses * batch_size, detached
