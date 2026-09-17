"""Preflight or explicitly launch monitored full-page feature fine-tuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import statistics
import sys

import torch
from ultralytics import YOLO, __version__
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

import fixed080_acceptance as fixed
import full_category_challenge as challenge
import evaluation_cache
from active_head_pilot import ActiveHeadLoss, ActiveHeadTrainer, evaluate_july, probe_scores
from confidence_preserving_loss import (
    BaselineForegroundFloorLoss,
    FrozenTeacherForegroundFloorLoss,
    PinnedTeacherReviewedClassificationLoss,
)
from legal_notice_v10_acceptance import area, intersection, match_references, read_labels
from monitored_clslogit_pilot import run_challenge, run_fixed
import prepare_corrected_dataset as dataset_tools


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v11_feature_finetune_config.json"
DATASET_HASH_ROOTS = ("data.yaml", "train_sampling.txt", "images", "labels")
REQUIRED_STOPPING_KEYS = (
    "fixed_valid_required",
    "fixed_max_excluded_accepted",
    "critical_boundaries_required",
    "challenge_valid_required",
    "challenge_max_excluded_accepted_regions",
    "july_valid_required",
    "july_max_excluded_accepted_regions",
    "gulf_recovery_targets_required",
    "target_valid_required",
    "target_negative_regions_required",
    "target_negative_max_confidence",
    "maximum_median_valid_confidence_drop",
    "maximum_reference_confidence_drop",
    "stop_immediately_on_preservation_failure",
    "stop_when_all_promotion_gates_pass",
)
ACTIVE_CONFIG: dict | None = None
file_sha256 = fixed.file_sha256


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_stopping_config(config: dict) -> None:
    stopping = config.get("stopping")
    if not isinstance(stopping, dict):
        raise RuntimeError("Configuration must contain a stopping object")
    missing = [key for key in REQUIRED_STOPPING_KEYS if key not in stopping]
    require(not missing, f"Missing required stopping settings: {', '.join(missing)}")


def verify_report_checkpoint_sha(suite: str, report: dict, expected_sha256: str) -> None:
    if suite in ("fixed", "challenge"):
        actual = report.get("candidate", {}).get("weights_sha256")
    else:
        actual = report.get("weights_sha256")
    require(actual == expected_sha256, f"{suite} report checkpoint SHA does not match the pinned checkpoint")


def scope_parameters(model: torch.nn.Module, config: dict) -> dict[str, torch.nn.Parameter]:
    scope = config["trainable_scope"]
    prefixes = tuple(scope["feature_prefixes"] + scope["head_prefixes"])
    return {name: parameter for name, parameter in model.named_parameters() if name.startswith(prefixes)}


def set_trainable_scope(model: torch.nn.Module, config: dict) -> dict[str, torch.nn.Parameter]:
    selected = scope_parameters(model, config)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in selected
    return selected


def grouped_parameters(model: torch.nn.Module, config: dict) -> list[dict]:
    selected = scope_parameters(model, config)
    scope = config["trainable_scope"]
    optimizer = config["optimizer"]
    feature_prefixes = tuple(scope["feature_prefixes"])
    head_prefixes = tuple(scope["head_prefixes"])
    groups = {name: [] for name in optimizer["groups"]}
    names = {name: [] for name in optimizer["groups"]}
    norm_types = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm)
    for module_name, module in unwrap_model(model).named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if full_name not in selected:
                continue
            scope_name = "feature" if full_name.startswith(feature_prefixes) else "head" if full_name.startswith(head_prefixes) else None
            require(scope_name is not None, f"Unclassified trainable parameter: {full_name}")
            no_decay = parameter_name == "bias" or isinstance(module, norm_types)
            group_name = f"{scope_name}_{'no_decay' if no_decay else 'decay'}"
            groups[group_name].append(parameter)
            names[group_name].append(full_name)
    require(set().union(*(set(group) for group in names.values())) == set(selected), "Optimizer groups do not cover the trainable scope exactly")
    result = []
    for group_name in optimizer["groups"]:
        require(bool(groups[group_name]), f"Optimizer group is empty: {group_name}")
        learning_rate = optimizer["feature_learning_rate"] if group_name.startswith("feature") else optimizer["head_learning_rate"]
        result.append({
            "params": groups[group_name],
            "lr": learning_rate,
            "weight_decay": 0.0 if group_name.endswith("no_decay") else optimizer["weight_decay"],
            "param_group": group_name,
            "parameter_names": names[group_name],
        })
    return result


class DifferentialFeatureTrainer(ActiveHeadTrainer):
    """Use explicit feature/head AdamW groups while retaining active-head checks."""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        require(ACTIVE_CONFIG is not None, "Trainer configuration was not installed")
        require(name == "AdamW", "Only the frozen AdamW configuration is supported")
        active_config = ACTIVE_CONFIG
        if active_config is None:
            raise RuntimeError("Trainer configuration was not installed")
        groups = grouped_parameters(model, active_config)
        optimizer_config = active_config["optimizer"]
        for group in groups:
            group.pop("parameter_names")
        return torch.optim.AdamW(
            groups,
            betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
            weight_decay=0.0,
        )

    def _build_train_pipeline(self):
        require(ACTIVE_CONFIG is not None, "Trainer configuration was not installed")
        active_config = ACTIVE_CONFIG
        if active_config is None:
            raise RuntimeError("Trainer configuration was not installed")
        set_trainable_scope(unwrap_model(self.model), active_config)
        super()._build_train_pipeline()

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if mode != "train" or ACTIVE_CONFIG is None or ACTIVE_CONFIG.get("experiment") != "pinned_teacher_reviewed_cls_pilot":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        require(batch_size == 1 and rank == -1, "Reviewed balanced sampling supports only single-process physical batch one")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        require(getattr(dataset, "rect", False), "Reviewed training must retain rectangular page geometry")
        sampler = BalancedPageSampler(dataset, ACTIVE_CONFIG["training"]["seed"])
        generator = torch.Generator().manual_seed(ACTIVE_CONFIG["training"]["seed"])
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            generator=generator,
        )

    def validate(self):
        if ACTIVE_CONFIG is not None and ACTIVE_CONFIG.get("experiment") == "pinned_teacher_reviewed_cls_pilot":
            training_loss = sum(float(value) for value in self.tloss.values())
            return {}, -training_loss
        return super().validate()

    def final_eval(self):
        if ACTIVE_CONFIG is not None and ACTIVE_CONFIG.get("experiment") == "pinned_teacher_reviewed_cls_pilot":
            return
        return super().final_eval()

    def save_model(self):
        """Keep the pinned teacher out of checkpoint deep copies and serialized weights."""
        model = unwrap_model(self.model)
        model_criterion = getattr(model, "criterion", None)
        ema_criterion = getattr(self.ema.ema, "criterion", None)
        model.criterion = None
        self.ema.ema.criterion = None
        try:
            return super().save_model()
        finally:
            model.criterion = model_criterion
            self.ema.ema.criterion = ema_criterion

    def optimizer_step(self):
        if getattr(self, "verified_update", False):
            return super(ActiveHeadTrainer, self).optimizer_step()
        active = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        gradients = [parameter.grad for parameter in active.values() if parameter.grad is not None]
        require(bool(gradients) and all(torch.isfinite(gradient).all() for gradient in gradients), "Missing or nonfinite active gradients")
        gradient_norm = sum(float(gradient.detach().square().sum()) for gradient in gradients) ** 0.5
        require(gradient_norm > 0, "Active gradients are zero")
        require(all(parameter.grad is None for parameter in self.model.parameters() if not parameter.requires_grad), "Frozen gradient detected")
        learning_rate = max(group["lr"] for group in self.optimizer.param_groups)
        before = {name: parameter.detach().clone() for name, parameter in active.items()}
        super(ActiveHeadTrainer, self).optimizer_step()
        changed = [name for name, parameter in active.items() if not torch.equal(before[name], parameter.detach())]
        if not changed:
            require(learning_rate == 0.0, "A nonzero-learning-rate optimizer step changed no parameters")
            self.zero_lr_optimizer_steps = getattr(self, "zero_lr_optimizer_steps", 0) + 1
            self._model_train()
            return
        delta = float((probe_scores(self.model, self.probe) - self.initial_probe_scores).abs().max())
        require(delta > 0, "Optimizer update did not affect active inference scores")
        self.update_evidence = {
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "zero_lr_steps_skipped": getattr(self, "zero_lr_optimizer_steps", 0),
            "changed_tensors": changed,
            "raw_score_max_delta": delta,
        }
        self.verified_update = True
        self._model_train()
        write(self.audit_root / "first_update_verification.json", self.update_evidence)


class BalancedPageSampler(torch.utils.data.Sampler[int]):
    """Shuffle positive/background pages and spread both roles across an epoch."""

    def __init__(self, dataset, seed: int):
        self.seed = seed
        self.epoch = 0
        self.positive = [index for index, label in enumerate(dataset.labels) if len(label["cls"])]
        self.background = [index for index, label in enumerate(dataset.labels) if not len(label["cls"])]
        require(bool(self.positive) and bool(self.background), "Balanced sampling requires positive and background pages")

    def __len__(self) -> int:
        return len(self.positive) + len(self.background)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def epoch_order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        def shuffled(values: list[int]) -> list[int]:
            order = torch.randperm(len(values), generator=generator).tolist()
            return [values[index] for index in order]

        positive, background = shuffled(self.positive), shuffled(self.background)
        spread = [((index + 0.5) / len(positive), value) for index, value in enumerate(positive)]
        spread += [((index + 0.5) / len(background), value) for index, value in enumerate(background)]
        return [value for _, value in sorted(spread)]

    def __iter__(self):
        yield from self.epoch_order()


def optimizer_summary(model: torch.nn.Module, config: dict) -> list[dict]:
    summary = []
    for group in grouped_parameters(model, config):
        summary.append({
            "name": group["param_group"],
            "learning_rate": group["lr"],
            "weight_decay": group["weight_decay"],
            "tensors": len(group["params"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "parameter_names": group["parameter_names"],
        })
    return summary


def unresolved_unreviewed(report: dict, suite: str, reviews: list[dict]) -> list[dict]:
    unresolved = []
    for page in report["pages"]:
        for prediction in page.get("unreviewed_accepted", []):
            matches = [
                review for review in reviews
                if review["suite"] == suite and review["issue"] == page["issue"] and review["page"] == page["page"]
                and challenge.iou(review["xyxy"], prediction["xyxy"]) >= 0.85
            ]
            if len(matches) != 1 or matches[0]["classification"] != "valid":
                unresolved.append({"issue": page["issue"], "page": page["page"], "prediction": prediction})
    return unresolved


def validate_baselines(config: dict) -> None:
    expected = config["baseline_expectations"]
    for suite, required in expected.items():
        artifact = config["baseline_reports"][suite]
        path = project_path(artifact["path"])
        require(file_sha256(path) == artifact["sha256"], f"Changed {suite} baseline report")
        report = load(path)
        candidate = report.get("candidate", {})
        if candidate:
            require(candidate.get("weights_sha256") == config["starting_weights_sha256"], f"Wrong {suite} baseline checkpoint")
        for key, value in required.items():
            require(report["totals"].get(key) == value, f"Unexpected baseline {suite} {key}")
        reviews = load(project_path(config["semantic_extra_reviews"]))["reviews"]
        require(not unresolved_unreviewed(report, "fixed080" if suite == "fixed" else suite, reviews), f"Baseline {suite} has unresolved accepted detections")


def validate_config(config: dict) -> None:
    validate_stopping_config(config)
    training = config["training"]
    optimizer = config["optimizer"]
    experiment = config.get("experiment", "feature_finetune")
    expected_epochs = {
        "confidence_preserving_cv3_pilot": 1,
        "teacher_guarded_full_detector": 40,
        "pinned_teacher_reviewed_cls_pilot": 1,
    }.get(experiment, 12)
    expected_warmup = 0.0 if experiment in (
        "confidence_preserving_cv3_pilot", "teacher_guarded_full_detector", "pinned_teacher_reviewed_cls_pilot"
    ) else 1.0
    required = {
        "epochs": expected_epochs,
        "device": "cpu", "image_size": 1280, "batch": 1, "workers": 0,
        "nominal_batch_size": 8, "gradient_accumulation": 8,
        "warmup_epochs": expected_warmup,
        "save_period": 1, "amp": False, "compile": False, "mosaic": 0.0,
    }
    for key, value in required.items():
        require(training.get(key) == value, f"Training setting {key} must be {value!r}")
    require(optimizer["name"] == "AdamW", "Optimizer must remain AdamW")
    if experiment == "confidence_preserving_cv3_pilot":
        require(config.get("loss") == {"name": "baseline_foreground_floor", "tal_topk": 10}, "Confidence-preserving loss changed")
        require(not config["trainable_scope"]["feature_prefixes"] and config["trainable_scope"]["head_prefixes"] == ["model.23.cv3."], "Pilot must train only the active classification tower")
        require(optimizer["head_learning_rate"] == 0.000001 and optimizer["weight_decay"] == 0.0, "Pilot optimizer safeguards changed")
        require(training["rectangular_batches"] is True, "Pilot must use deployment-like rectangular batches")
    elif experiment == "teacher_guarded_full_detector":
        require(config.get("loss") == {"name": "frozen_teacher_foreground_floor", "tal_topk": 10}, "Full-detector guarded loss changed")
        require(config["trainable_scope"]["feature_prefixes"] and config["trainable_scope"]["head_prefixes"] == ["model.23.cv2.", "model.23.cv3."], "Full-detector scope must include features and both active heads")
        require(optimizer["head_learning_rate"] == 0.000005 and optimizer["feature_learning_rate"] == 0.0000005 and optimizer["weight_decay"] == 0.0, "Full-detector optimizer safeguards changed")
        require(training["rectangular_batches"] is True, "Full-detector training must use deployment-like rectangular batches")
        require(training["full_evaluation_period"] == 5 and training["target_evaluation_period"] == 1, "Monitored evaluation cadence changed")
    elif experiment == "pinned_teacher_reviewed_cls_pilot":
        loss = config.get("loss", {})
        require(loss.get("name") == "pinned_teacher_reviewed_classification", "Reviewed classification loss changed")
        require(not config["trainable_scope"]["feature_prefixes"] and config["trainable_scope"]["head_prefixes"] == ["model.23.cv3."], "Reviewed pilot must train only the complete active classification towers")
        require(optimizer["head_learning_rate"] == 0.000001 and optimizer["weight_decay"] == 0.0, "Reviewed pilot optimizer safeguards changed")
        require(training["rectangular_batches"] is True and training["patience"] == 0, "Reviewed pilot geometry or early stopping changed")
        require(training["full_evaluation_period"] == 1 and training["target_evaluation_period"] == 1, "Every pilot checkpoint must receive full fixed-0.80 evaluation")
        require(training.get("native_validation") is False, "Generic mAP validation must not select reviewed pilot checkpoints")
        require(config["deployment"] == {"confidence": 0.8, "image_size": 1280, "device": "cpu", "render_dpi": 200}, "Deployment contract changed")
        for key in ("fixed_max_excluded_accepted", "challenge_max_excluded_accepted_regions", "july_max_excluded_accepted_regions"):
            require(config["stopping"][key] == 0, f"Reviewed exclusion gate {key} must be zero")
    else:
        require(optimizer["head_learning_rate"] == 0.00005 and optimizer["feature_learning_rate"] == 0.000005, "Differential learning rates changed")
    require(config.get("training_authorization_required") is True, "Explicit authorization guard is required")


def gradient_preflight(model: torch.nn.Module, config: dict) -> dict:
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    image_path = project_path(config["dataset"]).parent / "images/train" / config["probe_train_image"]
    image = cv2.imread(str(image_path))
    require(image is not None, f"Could not load gradient probe: {image_path}")
    transformed = LetterBox(new_shape=(config["training"]["image_size"],) * 2, auto=False)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    model.zero_grad(set_to_none=True)
    model.eval()
    output = model(tensor)
    raw = output[1] if isinstance(output, tuple) else output
    branch = raw["one2many"]
    objective = branch["scores"].float().mean() + branch["boxes"].float().square().mean() * 1e-4
    objective.backward()
    selected = scope_parameters(model, config)
    gradients = {name: parameter.grad for name, parameter in selected.items()}
    require(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients.values()), "Missing or nonfinite gradient in selected scope")
    require(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in selected), "Frozen parameter received a gradient")
    norms = {"feature": 0.0, "head": 0.0}
    feature_prefixes = tuple(config["trainable_scope"]["feature_prefixes"])
    for name, gradient in gradients.items():
        key = "feature" if name.startswith(feature_prefixes) else "head"
        norms[key] += float(gradient.detach().square().sum())
    norms = {key: value ** 0.5 for key, value in norms.items()}
    required_norms = [name for name, prefixes in (("feature", config["trainable_scope"]["feature_prefixes"]), ("head", config["trainable_scope"]["head_prefixes"])) if prefixes]
    require(all(norms[name] > 0 for name in required_norms), "Configured gradient path is zero")
    model.zero_grad(set_to_none=True)
    return {"probe": relative(image_path), "objective": float(objective.detach()), "gradient_norms": norms, "selected_tensors_with_gradients": len(gradients)}


def negative_region_map(config: dict) -> dict[str, list[list[float]]]:
    manifest = load(project_path(config["target_manifest"]))
    return {page["image_name"]: page["negative_regions_xyxyn"] for page in manifest["pages"] if page["negative_regions_xyxyn"]}


def confidence_loss_preflight(model: torch.nn.Module, config: dict, teacher_model: torch.nn.Module | None = None) -> dict:
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    dataset_root = project_path(config["dataset"]).parent
    image_path = dataset_root / "images/train" / config["probe_train_image"]
    label_path = dataset_root / "labels/train" / f"{Path(config['probe_train_image']).stem}.txt"
    image = cv2.imread(str(image_path))
    require(image is not None, f"Could not load guarded-loss probe: {image_path}")
    original_height, original_width = image.shape[:2]
    size = config["training"]["image_size"]
    ratio = min(size / original_height, size / original_width)
    resized_width, resized_height = round(original_width * ratio), round(original_height * ratio)
    pad_width, pad_height = size - resized_width, size - resized_height
    if config["training"]["rectangular_batches"]:
        pad_width, pad_height = pad_width % 32, pad_height % 32
    left, top = round(pad_width / 2 - 0.1), round(pad_height / 2 - 0.1)
    transformed = LetterBox(new_shape=(size, size), auto=config["training"]["rectangular_batches"], stride=32)(image=image)
    output_height, output_width = transformed.shape[:2]
    labels = [list(map(float, line.split())) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(labels), "Guarded-loss probe must contain foreground labels")
    boxes = []
    for _, x, y, width, height in labels:
        boxes.append([
            (x * original_width * ratio + left) / output_width,
            (y * original_height * ratio + top) / output_height,
            width * original_width * ratio / output_width,
            height * original_height * ratio / output_height,
        ])
    tensor = torch.from_numpy(np.ascontiguousarray(transformed[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    batch = {
        "img": tensor,
        "im_file": (str(image_path),),
        "ori_shape": ((original_height, original_width),),
        "ratio_pad": ((resized_height / original_height, resized_width / original_width),),
        "batch_idx": torch.zeros(len(boxes)),
        "cls": torch.zeros((len(boxes), 1)),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
    }
    model.zero_grad(set_to_none=True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    loss_name = config["loss"]["name"]
    if loss_name == "pinned_teacher_reviewed_classification":
        require(teacher_model is not None, "Reviewed loss preflight requires the pinned teacher")
        criterion = PinnedTeacherReviewedClassificationLoss(model, teacher_model, config["loss"], negative_region_map(config))
    else:
        loss_class = FrozenTeacherForegroundFloorLoss if loss_name == "frozen_teacher_foreground_floor" else BaselineForegroundFloorLoss
        criterion = loss_class(model, tal_topk=config["loss"]["tal_topk"])
    losses, _ = criterion(model(tensor), batch)
    losses.sum().backward()
    selected = scope_parameters(model, config)
    gradients = [parameter.grad for parameter in selected.values()]
    require(any(gradient is not None for gradient in gradients), "Guarded loss produced no selected gradients")
    require(all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients), "Guarded loss produced nonfinite selected gradients")
    require(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in selected), "Guarded loss reached a frozen parameter")
    diagnostics = criterion.last_diagnostics
    if loss_name == "pinned_teacher_reviewed_classification":
        require(diagnostics["preservation_groups"] + diagnostics["recovery_groups"] == len(labels), "Reviewed loss did not assign every probe label")
        positive_diagnostics = diagnostics
        negative_page = next(page for page in load(project_path(config["target_manifest"]))["pages"] if page["negative_regions_xyxyn"] and page["valid_notice_count"] == 0)
        negative_path = project_path(negative_page["image"])
        negative_image = cv2.imread(str(negative_path))
        require(negative_image is not None, f"Could not load reviewed negative probe: {negative_path}")
        negative_height, negative_width = negative_image.shape[:2]
        negative_ratio = min(size / negative_height, size / negative_width)
        negative_resized_width, negative_resized_height = round(negative_width * negative_ratio), round(negative_height * negative_ratio)
        negative_pad_width, negative_pad_height = (size - negative_resized_width) % 32, (size - negative_resized_height) % 32
        negative_left, negative_top = round(negative_pad_width / 2 - 0.1), round(negative_pad_height / 2 - 0.1)
        negative_transformed = LetterBox(new_shape=(size, size), auto=True, stride=32)(image=negative_image)
        negative_tensor = torch.from_numpy(np.ascontiguousarray(negative_transformed[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
        negative_batch = {
            "img": negative_tensor,
            "im_file": (str(negative_path),),
            "ori_shape": ((negative_height, negative_width),),
            "ratio_pad": ((negative_resized_height / negative_height, negative_resized_width / negative_width),),
            "batch_idx": torch.zeros(0),
            "cls": torch.zeros((0, 1)),
            "bboxes": torch.zeros((0, 4)),
        }
        criterion(model(negative_tensor), negative_batch)
        negative_diagnostics = criterion.last_diagnostics
        require(positive_diagnostics["maximum_box_output_delta"] <= config["loss"]["maximum_box_output_delta"], "Reviewed loss changed frozen box outputs")
        require(negative_diagnostics["rejection_groups"] > 0 and negative_diagnostics["explicit_negative_regions"] == len(negative_page["negative_regions_xyxyn"]), "Reviewed loss did not supervise the explicit negative region")
        diagnostics = {"positive_probe": positive_diagnostics, "negative_probe": negative_diagnostics}
    else:
        require(diagnostics["foreground_anchors"] > 0 and diagnostics["protected_logits"] > 0, "Guarded loss did not protect foreground assignments")
    model.zero_grad(set_to_none=True)
    return {
        "probe": relative(image_path),
        "input_shape": list(tensor.shape),
        "labels": len(labels),
        **diagnostics,
        "selected_tensors_with_gradients": len(gradients),
    }


def preflight(config_path: Path, run_gradient_check: bool = True) -> tuple[dict, YOLO, set[str], dict]:
    config = load(config_path)
    validate_config(config)
    require(__version__ == config["ultralytics_version"], "Ultralytics version changed")
    artifact_keys = ["starting_weights", "dataset_manifest", "target_manifest", "fixed_manifest", "challenge_manifest", "july_review", "july_manifest", "semantic_extra_reviews"]
    artifact_keys.extend(key for key in ("annotation_audit", "acceptance_inventory", "geometry_report") if key in config)
    for key in artifact_keys:
        path = project_path(config[key])
        require(path.is_file() and file_sha256(path) == config[key + "_sha256"], f"Changed {key}")
    data_path = project_path(config["dataset"])
    dataset_root = data_path.parent
    require(file_sha256(data_path) == config["data_yaml_sha256"], "Dataset YAML changed")
    require(dataset_tools.tree_sha256(dataset_root, DATASET_HASH_ROOTS) == config["dataset_sha256"], "Consolidated dataset changed")
    require(dataset_tools.tree_sha256(dataset_root, ("images/test", "labels/test")) == config["historical_test_sha256"], "Historical test split changed")
    manifest = load(project_path(config["dataset_manifest"]))
    require(manifest["training_authorized"] is False and manifest["sampling"]["epoch_samples"] == 109, "Dataset guard or sampling schedule changed")
    validate_baselines(config)
    if config.get("experiment") == "pinned_teacher_reviewed_cls_pilot":
        inventory = load(project_path(config["acceptance_inventory"]))
        geometry = load(project_path(config["geometry_report"]))
        require(inventory["counts"]["geometry_checks"] == 29 and inventory["policy"]["reviewed_exclusions_accepted"] == 0, "Acceptance inventory policy changed")
        require(geometry["status"] == "passed" and geometry["totals"] == {"required": 29, "passed": 29, "failed": 0}, "Frozen-box geometry feasibility failed")
        require(geometry["inventory_sha256"] == config["acceptance_inventory_sha256"] and geometry["weights_sha256"] == config["starting_weights_sha256"], "Geometry report provenance changed")

    split_issues = {}
    train_hashes = set()
    for split in ("train", "val", "test"):
        images = sorted((dataset_root / "images" / split).glob("*.png"))
        split_issues[split] = {image.stem.rsplit("_page_", 1)[0] for image in images}
        if split == "train":
            train_hashes = {file_sha256(image) for image in images}
    require(not split_issues["train"] & split_issues["val"] and not split_issues["train"] & split_issues["test"] and not split_issues["val"] & split_issues["test"], "Issue-level split leakage")
    fixed_manifest = load(project_path(config["fixed_manifest"]))
    challenge_manifest = load(project_path(config["challenge_manifest"]))
    july_review = load(project_path(config["july_review"]))
    july_manifest = load(project_path(config["july_manifest"]))
    forbidden = set(july_review["issues"]) | {page["issue"] for page in fixed_manifest["pages"] + challenge_manifest["pages"]}
    require(not split_issues["train"] & forbidden, "Evaluation issue leaked into training")
    evaluation_pages = fixed_manifest["pages"] + challenge_manifest["pages"] + july_manifest["pages"]
    evaluation_hashes = {page["image_sha256"] for page in evaluation_pages}
    require(not train_hashes & evaluation_hashes, "Preservation image leaked into training")
    cache_inputs = evaluation_cache.validate_inputs(config)

    yolo = YOLO(str(project_path(config["starting_weights"])))
    require(yolo.names == {0: "legal_notice"} and not yolo.model.end2end, "Wrong model class or inference branch")
    selected = set_trainable_scope(yolo.model, config)
    scope = config["trainable_scope"]
    feature_prefixes = tuple(scope["feature_prefixes"])
    feature = {name: parameter for name, parameter in selected.items() if name.startswith(feature_prefixes)}
    head = {name: parameter for name, parameter in selected.items() if name not in feature}
    require(len(selected) == scope["tensors"] and sum(parameter.numel() for parameter in selected.values()) == scope["parameters"], "Trainable scope count changed")
    require(len(feature) == scope["feature_tensors"] and sum(parameter.numel() for parameter in feature.values()) == scope["feature_parameters"], "Feature scope count changed")
    require(len(head) == scope["head_tensors"] and sum(parameter.numel() for parameter in head.values()) == scope["head_parameters"], "Head scope count changed")
    groups = optimizer_summary(yolo.model, config)
    require(sum(group["parameters"] for group in groups) == scope["parameters"], "Optimizer parameter count changed")
    gradient = gradient_preflight(yolo.model, config) if run_gradient_check else {"status": "skipped"}
    teacher_model = YOLO(str(project_path(config["starting_weights"]))).model if config.get("loss", {}).get("name") == "pinned_teacher_reviewed_classification" else None
    guarded_loss = (
        confidence_loss_preflight(yolo.model, config, teacher_model)
        if run_gradient_check and config.get("loss", {}).get("name") in ("baseline_foreground_floor", "frozen_teacher_foreground_floor", "pinned_teacher_reviewed_classification")
        else {"status": "not_applicable" if "loss" not in config else "skipped"}
    )
    summary = {
        "status": "preflight_passed_training_not_authorized" if not config["training_authorized"] else "preflight_passed_training_authorized",
        "training_started": False,
        "config": relative(config_path),
        "config_sha256": file_sha256(config_path),
        "trainer": relative(Path(__file__)),
        "trainer_sha256": file_sha256(Path(__file__)),
        "dataset_sha256": config["dataset_sha256"],
        "physical_train_pages": manifest["counts"]["train"]["images"],
        "epoch_samples": manifest["sampling"]["epoch_samples"],
        "trainable_tensors": len(selected),
        "trainable_parameters": sum(parameter.numel() for parameter in selected.values()),
        "optimizer_groups": groups,
        "training": config["training"],
        "stopping": config["stopping"],
        "gradient_preflight": gradient,
        "guarded_loss_preflight": guarded_loss,
        "effective_batch": config["training"]["gradient_accumulation"],
        "historical_test_policy": config["historical_test_policy"],
        "pinned_teacher": {"path": config["starting_weights"], "sha256": config["starting_weights_sha256"]},
        "evaluation_cache": {**cache_inputs, "mode": "issue_scoped"},
    }
    return config, yolo, train_hashes, summary


def evaluate_target(weights: Path, config: dict, output: Path, expected_weights_sha256: str | None = None) -> dict:
    manifest_value = config.get("target_manifest")
    if not isinstance(manifest_value, str):
        raise RuntimeError("Configuration must identify a target manifest")
    manifest_path = project_path(manifest_value)
    require(
        manifest_path.is_file() and file_sha256(manifest_path) == config.get("target_manifest_sha256"),
        "Target manifest is missing or changed",
    )
    manifest = load(manifest_path)
    weight_hash = expected_weights_sha256 or file_sha256(weights)
    require(file_sha256(weights) == weight_hash, "Target checkpoint does not match the pinned SHA")
    for page in manifest["pages"]:
        for key in ("image", "label"):
            value = page.get(key)
            if not isinstance(value, str):
                raise RuntimeError(f"Target manifest page is missing {key}")
            path = project_path(value)
            require(
                path.is_file() and file_sha256(path) == page.get(f"{key}_sha256"),
                f"Target manifest {key} is missing or changed: {path}",
            )
    model = YOLO(str(weights))
    pages = []
    totals = {"valid_expected": 0, "valid_accepted": 0, "negative_regions": 0, "negative_margin_passed": 0, "excluded_accepted": 0, "unmatched_accepted": 0}
    deployment = config["deployment"]
    for page in manifest["pages"]:
        image = project_path(page["image"])
        label = project_path(page["label"])
        result = model.predict(str(image), conf=0.01, imgsz=deployment["image_size"], device=deployment["device"], verbose=False, save=False)[0]
        predictions = [{"confidence": float(box.conf[0]), "xyxy": [float(value) for value in box.xyxy[0].tolist()]} for box in result.boxes]
        accepted = [prediction for prediction in predictions if prediction["confidence"] >= deployment["confidence"]]
        references = read_labels(label, page["width"], page["height"])
        excluded_ids, regions = set(), []
        for normalized in page["negative_regions_xyxyn"]:
            region = [normalized[0] * page["width"], normalized[1] * page["height"], normalized[2] * page["width"], normalized[3] * page["height"]]
            overlaps = [prediction for prediction in predictions if intersection(prediction["xyxy"], region) / max(1.0, area(prediction["xyxy"])) >= 0.5]
            maximum = max((prediction["confidence"] for prediction in overlaps), default=0.0)
            for index, prediction in enumerate(accepted):
                if intersection(prediction["xyxy"], region) / max(1.0, area(prediction["xyxy"])) >= 0.5:
                    excluded_ids.add(index)
            regions.append({"maximum_confidence": maximum, "margin_passed": maximum <= config["stopping"]["target_negative_max_confidence"]})
        eligible = [prediction for index, prediction in enumerate(accepted) if index not in excluded_ids]
        matches, matched_ids = match_references(references, eligible)
        unmatched = [prediction for index, prediction in enumerate(eligible) if index not in matched_ids]
        totals["valid_expected"] += len(references)
        totals["valid_accepted"] += len(matches)
        totals["negative_regions"] += len(regions)
        totals["negative_margin_passed"] += sum(region["margin_passed"] for region in regions)
        totals["excluded_accepted"] += len(excluded_ids)
        totals["unmatched_accepted"] += len(unmatched)
        pages.append({"issue": page["issue"], "page": page["page"], "matches": matches, "negative_regions": regions, "unmatched_accepted": unmatched})
    stop = config["stopping"]
    passed = totals == {"valid_expected": stop["target_valid_required"], "valid_accepted": stop["target_valid_required"], "negative_regions": stop["target_negative_regions_required"], "negative_margin_passed": stop["target_negative_regions_required"], "excluded_accepted": 0, "unmatched_accepted": 0}
    require(file_sha256(weights) == weight_hash, "Target checkpoint changed during evaluation")
    report = {"suite": "v11_training_target_diagnostic", "passed": passed, "acceptance_confidence": deployment["confidence"], "weights": relative(weights), "weights_sha256": weight_hash, "manifest_sha256": file_sha256(manifest_path), "totals": totals, "pages": pages}
    write(output, report)
    del model
    gc.collect()
    return report


def valid_confidences(report: dict, suite: str) -> dict[str, float]:
    if suite == "fixed":
        return {
            reference["id"]: reference["accepted_match"]["confidence"]
            for page in report["pages"]
            for reference in page["references"]
            if reference["classification"] == "valid" and reference.get("accepted_match")
        }
    return {
        match["reference_id"]: match["prediction"]["confidence"]
        for page in report["pages"]
        for match in page["valid_matches"]
    }


def confidence_preservation(config: dict, suite: str, baseline: dict, current: dict) -> dict:
    baseline_scores = valid_confidences(baseline, suite)
    current_scores = valid_confidences(current, suite)
    common = sorted(set(baseline_scores) & set(current_scores))
    drops = [baseline_scores[reference] - current_scores[reference] for reference in common]
    paired_current = [current_scores[reference] for reference in common]
    result = {
        "paired_references": len(common),
        "baseline_references": len(baseline_scores),
        "current_references": len(current_scores),
        "median_baseline_confidence": statistics.median(baseline_scores.values()),
        "median_current_confidence": statistics.median(paired_current) if paired_current else None,
        "median_confidence_drop": statistics.median(baseline_scores.values()) - statistics.median(paired_current) if paired_current else None,
        "maximum_reference_confidence_drop": max(drops, default=None),
    }
    stop = config["stopping"]
    result["passed"] = (
        len(common) == len(baseline_scores)
        and result["median_confidence_drop"] <= stop["maximum_median_valid_confidence_drop"]
        and result["maximum_reference_confidence_drop"] <= stop["maximum_reference_confidence_drop"]
    )
    return result


def preservation_failures(config: dict, fixed_report: dict, challenge_report: dict, july_report: dict, baseline_reports: dict) -> tuple[list[str], dict]:
    stop = config["stopping"]
    failures = []
    fixed_totals, challenge_totals, july_totals = fixed_report["totals"], challenge_report["totals"], july_report["totals"]
    reviews = load(project_path(config["semantic_extra_reviews"]))["reviews"]
    if fixed_totals["valid_preserved"] != stop["fixed_valid_required"] or fixed_totals["valid_missed"] or fixed_totals["critical_boundaries"] != stop["critical_boundaries_required"] or fixed_totals["critical_boundary_failures"] or unresolved_unreviewed(fixed_report, "fixed080", reviews) or fixed_totals["excluded_accepted"] > stop["fixed_max_excluded_accepted"]:
        failures.append("fixed_preservation")
    if challenge_totals["valid_preserved"] != stop["challenge_valid_required"] or challenge_totals["valid_missed"] or challenge_totals["unreviewed_accepted"] or challenge_totals["excluded_accepted_regions"] > stop["challenge_max_excluded_accepted_regions"]:
        failures.append("challenge_preservation")
    baseline_july = baseline_reports["july"]
    baseline_ids = {record["reference_id"] for page in baseline_july["pages"] for record in page["valid_matches"]}
    current_ids = {record["reference_id"] for page in july_report["pages"] for record in page["valid_matches"]}
    if july_totals["valid_preserved"] != stop["july_valid_required"] or baseline_ids - current_ids or unresolved_unreviewed(july_report, "july", reviews) or july_totals["excluded_accepted_regions"] > stop["july_max_excluded_accepted_regions"]:
        failures.append("july_preservation")
    if sum(record["complete_accepted"] for record in july_report["recovery_targets"]) != stop["gulf_recovery_targets_required"]:
        failures.append("gulf_recovery")
    confidence = {
        "fixed": confidence_preservation(config, "fixed", baseline_reports["fixed"], fixed_report),
        "challenge": confidence_preservation(config, "challenge", baseline_reports["challenge"], challenge_report),
        "july": confidence_preservation(config, "july", baseline_july, july_report),
    }
    failures.extend(f"{suite}_confidence_preservation" for suite, result in confidence.items() if not result["passed"])
    return failures, confidence


def launch(config_path: Path, config: dict, yolo: YOLO, train_hashes: set[str], preflight_summary: dict, resume_training: bool = False) -> None:
    global ACTIVE_CONFIG
    require(config["training_authorized"] is True, "Training remains blocked: set training_authorized=true only after explicit approval")
    run_dir = ROOT / "runs" / config["run_name"]
    audit = project_path(config["monitor_output"])
    resume_checkpoint = run_dir / "weights/last.pt"
    previous_summary = None
    if resume_training:
        require(run_dir.is_dir() and resume_checkpoint.is_file(), "Resume requires the existing run's last.pt")
        require((audit / "monitor_summary.json").is_file(), "Resume requires the existing monitor summary")
        previous_summary = load(audit / "monitor_summary.json")
        require(previous_summary["config_sha256"] == file_sha256(config_path), "Resume configuration differs from the original run")
        require(previous_summary["trainer_sha256"] == file_sha256(Path(__file__)), "Resume trainer differs from the original run")
        require(previous_summary["status"] in ("training", "training_error", "completed_no_passing_checkpoint"), "Run status is not resumable")
        yolo = YOLO(str(resume_checkpoint))
    else:
        require(not run_dir.exists(), "Training output already exists; use --resume-training only for a verified interrupted run")
        require(not audit.exists() or not any(audit.iterdir()), "Monitor output already contains artifacts; use a new reviewed run name")
        audit.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG = config
    july_manifest = load(project_path(config["july_manifest"]))
    write(audit / "july_manifest.json", july_manifest)
    july_manifest_hash = file_sha256(audit / "july_manifest.json")
    require(july_manifest_hash == config["july_manifest_sha256"], "Copied July manifest hash changed")
    baseline_reports = {suite: load(project_path(artifact["path"])) for suite, artifact in config["baseline_reports"].items()}
    baseline_model = YOLO(str(project_path(config["starting_weights"]))).model
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in baseline_model.state_dict().items()}
    pinned_teacher = baseline_model
    pinned_teacher.eval().requires_grad_(False)
    if resume_training:
        assert previous_summary is not None
        summary = previous_summary
        summary["status"] = "training"
        summary.pop("error", None)
        summary.setdefault("resumed_at_utc", []).append(datetime.now(timezone.utc).isoformat())
    else:
        summary = {**preflight_summary, "status": "training", "training_started": True, "started_at_utc": datetime.now(timezone.utc).isoformat(), "epochs": []}
    write(audit / "monitor_summary.json", summary)

    def setup(trainer) -> None:
        trainer.model.end2end = False
        trainer.ema.ema.end2end = False
        loss_config = config.get("loss", {"name": "native_one2many"})
        if loss_config["name"] == "baseline_foreground_floor":
            trainer.model.criterion = BaselineForegroundFloorLoss(trainer.model, tal_topk=loss_config["tal_topk"])
            trainer.ema.ema.criterion = BaselineForegroundFloorLoss(trainer.ema.ema, tal_topk=loss_config["tal_topk"])
        elif loss_config["name"] == "frozen_teacher_foreground_floor":
            trainer.model.criterion = FrozenTeacherForegroundFloorLoss(trainer.model, tal_topk=loss_config["tal_topk"])
            trainer.ema.ema.criterion = ActiveHeadLoss(trainer.ema.ema)
        elif loss_config["name"] == "pinned_teacher_reviewed_classification":
            regions = negative_region_map(config)
            trainer.model.criterion = PinnedTeacherReviewedClassificationLoss(trainer.model, pinned_teacher, loss_config, regions)
            trainer.ema.ema.criterion = PinnedTeacherReviewedClassificationLoss(trainer.ema.ema, pinned_teacher, loss_config, regions)
        else:
            trainer.model.criterion = ActiveHeadLoss(trainer.model)
            trainer.ema.ema.criterion = ActiveHeadLoss(trainer.ema.ema)
        actual = {name: parameter for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
        expected = scope_parameters(trainer.model, config)
        require(set(actual) == set(expected), "Trainer trainable scope differs from preflight")
        require(trainer.accumulate == config["training"]["gradient_accumulation"], "Gradient accumulation changed")
        actual_groups = {group["param_group"]: group["lr"] for group in trainer.optimizer.param_groups}
        expected_groups = {group["name"]: group["learning_rate"] for group in preflight_summary["optimizer_groups"]}
        require(set(actual_groups) == set(expected_groups), "Optimizer groups changed")
        if not resume_training:
            require(actual_groups == expected_groups, "Optimizer learning rates changed")
        trainer.audit_root = audit
        trainer.probe = gradient_probe_tensor(config)
        from active_head_pilot import probe_scores
        trainer.initial_probe_scores = probe_scores(trainer.model, trainer.probe)
        trainer.sampling_audit = []
        trainer.objective_audit = []

    def epoch_start(trainer) -> None:
        sampler = getattr(trainer.train_loader, "sampler", None)
        if isinstance(sampler, BalancedPageSampler):
            sampler.set_epoch(int(trainer.epoch))
            order = sampler.epoch_order()
            files = [Path(trainer.train_loader.dataset.im_files[index]).name for index in order]
            roles = ["positive" if index in sampler.positive else "background" for index in order]
            trainer.sampling_audit.append({"epoch": int(trainer.epoch) + 1, "files": files, "roles": roles})

    def batch_end(trainer) -> None:
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        if isinstance(criterion, PinnedTeacherReviewedClassificationLoss):
            trainer.objective_audit.append(dict(criterion.last_diagnostics))

    def saved(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        checkpoint = Path(trainer.wdir) / f"epoch{epoch - 1}.pt"
        require(checkpoint.is_file() and getattr(trainer, "verified_update", False), "Missing checkpoint or verified optimizer update")
        checkpoint_sha256 = file_sha256(checkpoint)
        current = YOLO(str(checkpoint)).model
        changed = [name for name, value in initial_state.items() if not torch.equal(current.state_dict()[name].cpu().half(), value.half())]
        allowed = set(scope_parameters(yolo.model, config))
        require(bool(changed) and set(changed) <= allowed, "Checkpoint changed a frozen tensor or changed nothing")
        def checkpoint_unchanged() -> None:
            require(file_sha256(checkpoint) == checkpoint_sha256, "Checkpoint changed during monitored evaluation")

        def cached_evaluation(manifest_key: str, callback):
            checkpoint_unchanged()
            cache = evaluation_cache.IssueScopedCache(
                config,
                manifest_key,
                audit / f"epoch_{epoch:02d}_evaluation_cache",
            )
            try:
                return callback(cache.activate)
            finally:
                cache.close()

        training = config["training"]
        target_due = epoch % training.get("target_evaluation_period", 1) == 0
        full_due = epoch % training.get("full_evaluation_period", 1) == 0 or epoch == training["epochs"]
        require(target_due, "Every saved checkpoint must run the targeted diagnostic")
        target_path = audit / f"epoch_{epoch:02d}_target.json"
        target_report = evaluate_target(checkpoint, config, target_path, checkpoint_sha256)
        verify_report_checkpoint_sha("target", target_report, checkpoint_sha256)
        checkpoint_unchanged()
        reports = {"target": target_report}
        report_paths = {"target": target_path}
        failures = None
        confidence = None
        if full_due:
            fixed_report = cached_evaluation(
                "fixed_manifest",
                lambda before_page: run_fixed(checkpoint, project_path(config["fixed_manifest"]), audit / f"epoch_{epoch:02d}_fixed080.json", before_page),
            )
            verify_report_checkpoint_sha("fixed", fixed_report, checkpoint_sha256)
            checkpoint_unchanged()
            challenge_report = cached_evaluation(
                "challenge_manifest",
                lambda before_page: run_challenge(checkpoint, project_path(config["challenge_manifest"]), audit / f"epoch_{epoch:02d}_challenge.json", before_page),
            )
            verify_report_checkpoint_sha("challenge", challenge_report, checkpoint_sha256)
            checkpoint_unchanged()
            july_report = cached_evaluation(
                "july_manifest",
                lambda before_page: evaluate_july(checkpoint, july_manifest, july_manifest_hash, audit / f"epoch_{epoch:02d}_july.json", before_page),
            )
            verify_report_checkpoint_sha("july", july_report, checkpoint_sha256)
            checkpoint_unchanged()
            failures, confidence = preservation_failures(config, fixed_report, challenge_report, july_report, baseline_reports)
            reports.update({"fixed": fixed_report, "challenge": challenge_report, "july": july_report})
            report_paths.update({
                "fixed": audit / f"epoch_{epoch:02d}_fixed080.json",
                "challenge": audit / f"epoch_{epoch:02d}_challenge.json",
                "july": audit / f"epoch_{epoch:02d}_july.json",
            })
        promoted = full_due and not failures and target_report["passed"]
        report_artifacts = {
            suite: {"path": relative(path), "sha256": file_sha256(path), "totals": reports[suite]["totals"]}
            for suite, path in report_paths.items()
        }
        sampler_records = [record for record in getattr(trainer, "sampling_audit", []) if record["epoch"] == epoch]
        objective_records = getattr(trainer, "objective_audit", [])
        objective_totals = {
            key: sum(int(record.get(key, 0)) for record in objective_records)
            for key in ("preservation_groups", "recovery_groups", "rejection_groups", "explicit_negative_regions")
        }
        record = {"epoch": epoch, "checkpoint": relative(checkpoint), "checkpoint_sha256": checkpoint_sha256, "changed_tensors": len(changed), "full_evaluation": full_due, "reports": report_artifacts, "preservation_failures": failures, "confidence_preservation": confidence, "target": target_report["totals"], "objective_groups": objective_totals, "sampling": sampler_records[-1] if sampler_records else None, "promotion_passed": promoted}
        summary["epochs"].append(record)
        if full_due and failures and config["stopping"]["stop_immediately_on_preservation_failure"]:
            trainer.stop = True
            summary["status"] = "stopped_preservation_failure"
        elif promoted and config["stopping"]["stop_when_all_promotion_gates_pass"]:
            candidate = Path(trainer.wdir) / "fixed080_candidate.pt"
            shutil.copy2(checkpoint, candidate)
            require(file_sha256(candidate) == checkpoint_sha256, "Promoted candidate copy changed")
            record["promoted_candidate"] = {"path": relative(candidate), "sha256": checkpoint_sha256}
            trainer.stop = True
            summary["status"] = "passed_all_promotion_gates"
        write(audit / "monitor_summary.json", summary)
        del current
        gc.collect()

    yolo.add_callback("on_pretrain_routine_end", setup)
    yolo.add_callback("on_train_epoch_start", epoch_start)
    yolo.add_callback("on_train_batch_end", batch_end)
    yolo.add_callback("on_model_save", saved)
    training = config["training"]
    try:
        yolo.train(
            trainer=DifferentialFeatureTrainer, data=str(project_path(config["dataset"])), project=str(ROOT / "runs"), name=config["run_name"], exist_ok=False,
            epochs=training["epochs"], device=training["device"], imgsz=training["image_size"], batch=training["batch"], workers=training["workers"],
            pretrained=training["pretrained"], resume=str(resume_checkpoint) if resume_training else training["resume"], optimizer=config["optimizer"]["name"], lr0=config["optimizer"]["head_learning_rate"],
            lrf=training["final_learning_rate_fraction"], momentum=config["optimizer"]["beta1"], weight_decay=config["optimizer"]["weight_decay"], nbs=training["nominal_batch_size"],
            warmup_epochs=training["warmup_epochs"], warmup_momentum=training["warmup_momentum"], warmup_bias_lr=training["warmup_bias_lr"], freeze=training["freeze"],
            mosaic=training["mosaic"], close_mosaic=training["close_mosaic"], scale=training["scale"], translate=training["translate"], degrees=training["degrees"],
            shear=training["shear"], perspective=training["perspective"], fliplr=training["horizontal_flip"], flipud=training["vertical_flip"], hsv_h=training["hsv_h"],
            hsv_s=training["hsv_s"], hsv_v=training["hsv_v"], bgr=training["bgr"], mixup=training["mixup"], cutmix=training["cutmix"], copy_paste=training["copy_paste"],
            multi_scale=training["multi_scale"], amp=training["amp"], compile=training["compile"], cache=training["cache"], rect=training["rectangular_batches"],
            val=training.get("native_validation", True), split="val", plots=False, save=True, save_period=training["save_period"], patience=training["patience"], seed=training["seed"], deterministic=training["deterministic"], single_cls=training["single_class"],
        )
        if summary["status"] == "training":
            summary["status"] = "completed_no_passing_checkpoint"
    except BaseException as error:
        summary["status"] = "training_error" if summary["status"] == "training" else summary["status"]
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(audit / "monitor_summary.json", summary)


def gradient_probe_tensor(config: dict) -> torch.Tensor:
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox
    image_path = project_path(config["dataset"]).parent / "images/train" / config["probe_train_image"]
    image = LetterBox(new_shape=(config["training"]["image_size"],) * 2, auto=False)(image=cv2.imread(str(image_path)))
    return torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float()[None] / 255


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--start-training", action="store_true", help="Start only when the reviewed config also authorizes training")
    parser.add_argument("--resume-training", action="store_true", help="Resume the same hash-verified run from its last.pt")
    parser.add_argument("--skip-gradient-check", action="store_true", help="Use only for fast repeated integrity checks")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config, yolo, train_hashes, summary = preflight(config_path, run_gradient_check=not args.skip_gradient_check)
    require(not (args.start_training and args.resume_training), "Choose either --start-training or --resume-training")
    if args.start_training or args.resume_training:
        launch(config_path, config, yolo, train_hashes, summary, resume_training=args.resume_training)
    else:
        write(project_path(config["preflight_output"]), summary)
        print(json.dumps(summary, indent=2))
        if config["training_authorized"]:
            print("Preflight passed. Training is authorized but was not started without --start-training.")
        else:
            print("Preflight passed. Training was not started and remains authorization-blocked.")


if __name__ == "__main__":
    main()
