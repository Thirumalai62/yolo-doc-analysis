"""Preflight or explicitly launch monitored full-page feature fine-tuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import sys

import torch
from ultralytics import YOLO, __version__
from ultralytics.utils.torch_utils import unwrap_model

import fixed080_acceptance as fixed
import full_category_challenge as challenge
import evaluation_cache
from active_head_pilot import ActiveHeadLoss, ActiveHeadTrainer, evaluate_july, probe_scores
from confidence_preserving_loss import BaselineForegroundFloorLoss
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
    expected = {
        "fixed": {"valid_preserved": 271, "valid_missed": 0, "excluded_accepted": 15, "critical_boundary_failures": 0, "unreviewed_accepted": 2},
        "challenge": {"valid_preserved": 31, "valid_missed": 0, "excluded_accepted_regions": 6, "unreviewed_accepted": 0},
        "july": {"valid_preserved": 548, "valid_missed": 0, "excluded_accepted_regions": 1, "unreviewed_accepted": 0},
    }
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
    required = {
        "epochs": 1 if experiment == "confidence_preserving_cv3_pilot" else 12,
        "device": "cpu", "image_size": 1280, "batch": 1, "workers": 0,
        "nominal_batch_size": 8, "gradient_accumulation": 8,
        "warmup_epochs": 0.0 if experiment == "confidence_preserving_cv3_pilot" else 1.0,
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


def confidence_loss_preflight(model: torch.nn.Module, config: dict) -> dict:
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
        "batch_idx": torch.zeros(len(boxes)),
        "cls": torch.zeros((len(boxes), 1)),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
    }
    model.zero_grad(set_to_none=True)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    criterion = BaselineForegroundFloorLoss(model, tal_topk=config["loss"]["tal_topk"])
    losses, _ = criterion(model(tensor), batch)
    losses.sum().backward()
    selected = scope_parameters(model, config)
    gradients = [parameter.grad for parameter in selected.values()]
    require(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), "Guarded loss produced missing or nonfinite selected gradients")
    require(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in selected), "Guarded loss reached a frozen parameter")
    diagnostics = criterion.last_diagnostics
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
    for key in ("starting_weights", "dataset_manifest", "target_manifest", "fixed_manifest", "challenge_manifest", "july_review", "july_manifest", "semantic_extra_reviews"):
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
    guarded_loss = (
        confidence_loss_preflight(yolo.model, config)
        if run_gradient_check and config.get("loss", {}).get("name") == "baseline_foreground_floor"
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
    for page in manifest["pages"]:
        image = project_path(page["image"])
        label = project_path(page["label"])
        result = model.predict(str(image), conf=0.01, imgsz=1280, device="cpu", verbose=False, save=False)[0]
        predictions = [{"confidence": float(box.conf[0]), "xyxy": [float(value) for value in box.xyxy[0].tolist()]} for box in result.boxes]
        accepted = [prediction for prediction in predictions if prediction["confidence"] >= 0.8]
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
    report = {"suite": "v11_training_target_diagnostic", "passed": passed, "weights": relative(weights), "weights_sha256": weight_hash, "manifest_sha256": file_sha256(manifest_path), "totals": totals, "pages": pages}
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
    result = {
        "paired_references": len(common),
        "baseline_references": len(baseline_scores),
        "current_references": len(current_scores),
        "median_baseline_confidence": statistics.median(baseline_scores.values()),
        "median_current_confidence": statistics.median(current_scores.values()) if current_scores else None,
        "median_confidence_drop": statistics.median(baseline_scores.values()) - statistics.median(current_scores.values()) if current_scores else None,
        "maximum_reference_confidence_drop": max(drops, default=None),
    }
    stop = config["stopping"]
    result["passed"] = (
        len(common) == len(baseline_scores) == len(current_scores)
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


def launch(config_path: Path, config: dict, yolo: YOLO, train_hashes: set[str], preflight_summary: dict) -> None:
    global ACTIVE_CONFIG
    require(config["training_authorized"] is True, "Training remains blocked: set training_authorized=true only after explicit approval")
    run_dir = ROOT / "runs" / config["run_name"]
    audit = project_path(config["monitor_output"])
    require(not run_dir.exists(), "Training output already exists; use a new reviewed run name")
    require(not audit.exists() or not any(audit.iterdir()), "Monitor output already contains artifacts; use a new reviewed run name")
    audit.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG = config
    july_manifest = load(project_path(config["july_manifest"]))
    write(audit / "july_manifest.json", july_manifest)
    july_manifest_hash = file_sha256(audit / "july_manifest.json")
    require(july_manifest_hash == config["july_manifest_sha256"], "Copied July manifest hash changed")
    baseline_reports = {suite: load(project_path(artifact["path"])) for suite, artifact in config["baseline_reports"].items()}
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in yolo.model.state_dict().items()}
    summary = {**preflight_summary, "status": "training", "training_started": True, "started_at_utc": datetime.now(timezone.utc).isoformat(), "epochs": []}
    write(audit / "monitor_summary.json", summary)

    def setup(trainer) -> None:
        trainer.model.end2end = False
        trainer.ema.ema.end2end = False
        loss_config = config.get("loss", {"name": "native_one2many"})
        if loss_config["name"] == "baseline_foreground_floor":
            trainer.model.criterion = BaselineForegroundFloorLoss(trainer.model, tal_topk=loss_config["tal_topk"])
            trainer.ema.ema.criterion = BaselineForegroundFloorLoss(trainer.ema.ema, tal_topk=loss_config["tal_topk"])
        else:
            trainer.model.criterion = ActiveHeadLoss(trainer.model)
            trainer.ema.ema.criterion = ActiveHeadLoss(trainer.ema.ema)
        actual = {name: parameter for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}
        expected = scope_parameters(trainer.model, config)
        require(set(actual) == set(expected), "Trainer trainable scope differs from preflight")
        require(trainer.accumulate == config["training"]["gradient_accumulation"], "Gradient accumulation changed")
        actual_groups = {group["param_group"]: group["lr"] for group in trainer.optimizer.param_groups}
        require(actual_groups == {group["name"]: group["learning_rate"] for group in preflight_summary["optimizer_groups"]}, "Optimizer groups or learning rates changed")
        trainer.audit_root = audit
        trainer.probe = gradient_probe_tensor(config)
        from active_head_pilot import probe_scores
        trainer.initial_probe_scores = probe_scores(trainer.model, trainer.probe)

    def saved(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        checkpoint = Path(trainer.wdir) / f"epoch{epoch - 1}.pt"
        require(checkpoint.is_file() and getattr(trainer, "verified_update", False), "Missing checkpoint or verified optimizer update")
        checkpoint_sha256 = file_sha256(checkpoint)
        current = YOLO(str(checkpoint)).model
        changed = [name for name, value in initial_state.items() if not torch.equal(current.state_dict()[name].cpu().half(), value.half())]
        prefixes = tuple(config["trainable_scope"]["feature_prefixes"] + config["trainable_scope"]["head_prefixes"])
        require(bool(changed) and all(name.startswith(prefixes) for name in changed), "Checkpoint changed a frozen tensor or changed nothing")
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
        target_report = evaluate_target(checkpoint, config, audit / f"epoch_{epoch:02d}_target.json", checkpoint_sha256)
        verify_report_checkpoint_sha("target", target_report, checkpoint_sha256)
        checkpoint_unchanged()
        failures, confidence = preservation_failures(config, fixed_report, challenge_report, july_report, baseline_reports)
        promoted = not failures and target_report["passed"]
        report_paths = {
            "fixed": audit / f"epoch_{epoch:02d}_fixed080.json",
            "challenge": audit / f"epoch_{epoch:02d}_challenge.json",
            "july": audit / f"epoch_{epoch:02d}_july.json",
            "target": audit / f"epoch_{epoch:02d}_target.json",
        }
        reports = {"fixed": fixed_report, "challenge": challenge_report, "july": july_report, "target": target_report}
        report_artifacts = {
            suite: {"path": relative(path), "sha256": file_sha256(path), "totals": reports[suite]["totals"]}
            for suite, path in report_paths.items()
        }
        record = {"epoch": epoch, "checkpoint": relative(checkpoint), "checkpoint_sha256": checkpoint_sha256, "changed_tensors": len(changed), "reports": report_artifacts, "preservation_failures": failures, "confidence_preservation": confidence, "target": target_report["totals"], "promotion_passed": promoted}
        summary["epochs"].append(record)
        if failures and config["stopping"]["stop_immediately_on_preservation_failure"]:
            trainer.stop = True
            summary["status"] = "stopped_preservation_failure"
        elif promoted and config["stopping"]["stop_when_all_promotion_gates_pass"]:
            trainer.stop = True
            summary["status"] = "passed_all_promotion_gates"
        write(audit / "monitor_summary.json", summary)
        del current
        gc.collect()

    yolo.add_callback("on_pretrain_routine_end", setup)
    yolo.add_callback("on_model_save", saved)
    training = config["training"]
    try:
        yolo.train(
            trainer=DifferentialFeatureTrainer, data=str(project_path(config["dataset"])), project=str(ROOT / "runs"), name=config["run_name"], exist_ok=False,
            epochs=training["epochs"], device=training["device"], imgsz=training["image_size"], batch=training["batch"], workers=training["workers"],
            pretrained=training["pretrained"], resume=training["resume"], optimizer=config["optimizer"]["name"], lr0=config["optimizer"]["head_learning_rate"],
            lrf=training["final_learning_rate_fraction"], momentum=config["optimizer"]["beta1"], weight_decay=config["optimizer"]["weight_decay"], nbs=training["nominal_batch_size"],
            warmup_epochs=training["warmup_epochs"], warmup_momentum=training["warmup_momentum"], warmup_bias_lr=training["warmup_bias_lr"], freeze=training["freeze"],
            mosaic=training["mosaic"], close_mosaic=training["close_mosaic"], scale=training["scale"], translate=training["translate"], degrees=training["degrees"],
            shear=training["shear"], perspective=training["perspective"], fliplr=training["horizontal_flip"], flipud=training["vertical_flip"], hsv_h=training["hsv_h"],
            hsv_s=training["hsv_s"], hsv_v=training["hsv_v"], bgr=training["bgr"], mixup=training["mixup"], cutmix=training["cutmix"], copy_paste=training["copy_paste"],
            multi_scale=training["multi_scale"], amp=training["amp"], compile=training["compile"], cache=training["cache"], rect=training["rectangular_batches"],
            val=True, split="val", plots=False, save=True, save_period=training["save_period"], patience=training["patience"], seed=training["seed"], deterministic=training["deterministic"], single_cls=training["single_class"],
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
    parser.add_argument("--skip-gradient-check", action="store_true", help="Use only for fast repeated integrity checks")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config, yolo, train_hashes, summary = preflight(config_path, run_gradient_check=not args.skip_gradient_check)
    if args.start_training:
        launch(config_path, config, yolo, train_hashes, summary)
    else:
        write(project_path(config["preflight_output"]), summary)
        print(json.dumps(summary, indent=2))
        if config["training_authorized"]:
            print("Preflight passed. Training is authorized but was not started without --start-training.")
        else:
            print("Preflight passed. Training was not started and remains authorization-blocked.")


if __name__ == "__main__":
    main()
