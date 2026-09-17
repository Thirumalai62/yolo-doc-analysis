"""Native detection loss with a frozen baseline floor on assigned foreground scores."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils.loss import v8DetectionLoss


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
