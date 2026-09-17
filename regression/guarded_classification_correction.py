"""Fit one bounded, transactional classification correction from the R7 detector."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO, __version__
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes, xywh2xyxy

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v16_guarded_config.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
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


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def preprocess(image_path: Path, image_size: int = 1280) -> tuple[torch.Tensor, dict]:
    image = cv2.imread(str(image_path))
    require(image is not None, f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    gain = min(image_size / height, image_size / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    pad_width = (image_size - resized_width) % 32
    pad_height = (image_size - resized_height) % 32
    left = round(pad_width / 2 - 0.1)
    top = round(pad_height / 2 - 0.1)
    padded = LetterBox(new_shape=(image_size, image_size), auto=True, stride=32)(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    transform = {
        "original_shape": [height, width],
        "input_shape": list(tensor.shape[-2:]),
        "gain": gain,
        "padding": [left, top],
    }
    require(tensor.shape[-1] == resized_width + pad_width and tensor.shape[-2] == resized_height + pad_height, "LetterBox transform metadata differs from the produced tensor")
    return tensor, transform


def read_reference_boxes(label_path: Path, width: int, height: int) -> torch.Tensor:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, box_width, box_height = map(float, line.split())
        require(cls == 0.0, f"Unexpected class in {label_path}")
        boxes.append([
            (x - box_width / 2) * width,
            (y - box_height / 2) * height,
            (x + box_width / 2) * width,
            (y + box_height / 2) * height,
        ])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def transform_boxes(boxes: torch.Tensor, transform: dict) -> torch.Tensor:
    if not len(boxes):
        return boxes.clone()
    gain = transform["gain"]
    left, top = transform["padding"]
    result = boxes.clone()
    result[:, [0, 2]] = result[:, [0, 2]] * gain + left
    result[:, [1, 3]] = result[:, [1, 3]] * gain + top
    return result


def transform_region(xyxyn: list[float], width: int, height: int, transform: dict) -> torch.Tensor:
    x1, y1, x2, y2 = xyxyn
    original = torch.tensor([[x1 * width, y1 * height, x2 * width, y2 * height]], dtype=torch.float32)
    return transform_boxes(original, transform)[0]


def classification_logits(head: torch.nn.Module, features: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([head.cv3[index](feature).view(feature.shape[0], head.nc, -1) for index, feature in enumerate(features)], 2)


def head_prediction(head: torch.nn.Module, features: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    output = head([feature.clone() for feature in features])
    require(isinstance(output, tuple) and isinstance(output[1], dict), "Detection head returned an unexpected inference structure")
    return output


def nms_detections(
    head: torch.nn.Module,
    features: list[torch.Tensor],
    input_shape: list[int],
    original_shape: list[int],
    config: dict,
    confidence: float = 0.01,
) -> torch.Tensor:
    decoded, _ = head_prediction(head, features)
    return decoded_detections(decoded, input_shape, original_shape, config, confidence)


def decoded_detections(
    decoded: torch.Tensor,
    input_shape: list[int],
    original_shape: list[int],
    config: dict,
    confidence: float,
) -> torch.Tensor:
    deployment = config["deployment"]
    detections = non_max_suppression(
        decoded,
        conf_thres=confidence,
        iou_thres=deployment["nms_iou"],
        max_det=deployment["max_detections"],
        nc=1,
    )[0]
    if len(detections):
        detections = detections.clone()
        detections[:, :4] = scale_boxes(input_shape, detections[:, :4], original_shape)
    return detections.cpu()


def greedy_matches(references: torch.Tensor, detections: torch.Tensor, minimum_iou: float = 0.5) -> dict[int, int]:
    if not len(references) or not len(detections):
        return {}
    overlaps = box_iou(references, detections[:, :4])
    candidates = []
    for reference_index, detection_index in torch.nonzero(overlaps >= minimum_iou, as_tuple=False).tolist():
        candidates.append((float(overlaps[reference_index, detection_index]), float(detections[detection_index, 4]), reference_index, detection_index))
    matches = {}
    used_detections = set()
    for _, _, reference_index, detection_index in sorted(candidates, reverse=True):
        if reference_index not in matches and detection_index not in used_detections:
            matches[reference_index] = detection_index
            used_detections.add(detection_index)
    return matches


def unmatched_detections(references: torch.Tensor, detections: torch.Tensor, minimum_iou: float = 0.5) -> torch.Tensor:
    matches = greedy_matches(references, detections, minimum_iou)
    used = set(matches.values())
    indices = [index for index in range(len(detections)) if index not in used]
    return detections[indices] if indices else torch.empty((0, 6), dtype=detections.dtype)


def verify_config(config_path: Path, config: dict) -> dict:
    require(__version__ == config["ultralytics_version"], "Ultralytics version changed")
    require(config["training_authorized"] is True, "V16 fitting is not authorized")
    for value_key, hash_key, description in (
        ("starting_weights", "starting_weights_sha256", "starting weights"),
        ("dataset_manifest", "dataset_manifest_sha256", "dataset manifest"),
        ("target_manifest", "target_manifest_sha256", "target manifest"),
        ("release_inventory", "release_inventory_sha256", "release inventory"),
    ):
        path = project_path(config[value_key])
        require(path.is_file() and file_sha256(path) == config[hash_key], f"Changed {description}: {path}")
    for family in ("evaluation_manifests", "baseline_reports"):
        for name, artifact in config[family].items():
            path = project_path(artifact["path"])
            require(path.is_file() and file_sha256(path) == artifact["sha256"], f"Changed {family} artifact {name}")
    for name, artifact in config.get("pinned_artifacts", {}).items():
        path = project_path(artifact["path"])
        require(path.is_file() and file_sha256(path) == artifact["sha256"], f"Changed pinned artifact {name}")
    manifest_path = project_path(config["prepared_manifest"])
    require(manifest_path.is_file(), "Run prepare_v16_guarded_data.py first")
    manifest = load_json(manifest_path)
    require(manifest["config_sha256"] == file_sha256(config_path), "Prepared manifest belongs to a different configuration")
    expected_counts = config.get("inventory_counts", {"unique_pages": 96, "valid_notices": 352, "explicit_negative_regions": 8})
    for key, expected in expected_counts.items():
        require(manifest["counts"][key] == expected, f"Prepared inventory has unexpected {key}: {manifest['counts'][key]} != {expected}")
    deployment = config["deployment"]
    require(deployment == {
        "confidence": 0.8, "image_size": 1280, "device": "cpu", "render_dpi": 200,
        "nms_iou": 0.7, "max_detections": 300, "precision": "fp32", "batch": 1, "end2end": False,
    }, "Deployment contract changed")
    return manifest


def verify_baseline_reports(config: dict) -> dict:
    reports = {name: load_json(project_path(artifact["path"])) for name, artifact in config["baseline_reports"].items()}
    expected = config["starting_weights_sha256"]
    require(reports["fixed"]["candidate"]["weights_sha256"] == expected, "Fixed baseline uses different weights")
    require(reports["challenge"]["candidate"]["weights_sha256"] == expected, "Challenge baseline uses different weights")
    require(reports["july"]["weights_sha256"] == expected, "July baseline uses different weights")
    require(reports["reviewed"]["candidate_sha256"] == expected, "Reviewed baseline uses different weights")
    fixed, challenge, july = reports["fixed"]["totals"], reports["challenge"]["totals"], reports["july"]["totals"]
    require((fixed["valid_preserved"], fixed["critical_boundary_failures"]) == (271, 0), "R7 fixed baseline no longer has its verified result")
    require(challenge["valid_preserved"] == 31 and challenge["valid_missed"] == 0, "R7 challenge baseline no longer has its verified result")
    require(july["valid_preserved"] == 548 and july["valid_missed"] == 0, "R7 July baseline no longer has its verified result")
    require(reports["reviewed"]["all_three_gulf_targets_recovered"] is True, "R7 recovery verification changed")
    return {
        "checkpoint_sha256": expected,
        "fixed": fixed,
        "challenge": challenge,
        "july": july,
        "reviewed_status": reports["reviewed"]["status"],
    }


def compare_detection_sets(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    require(len(expected) == len(actual), f"Prediction count differs: {len(expected)} != {len(actual)}")
    if not len(expected):
        return {"detections": 0, "minimum_iou": 1.0, "maximum_confidence_delta": 0.0}
    overlaps = box_iou(expected[:, :4], actual[:, :4])
    used = set()
    ious, confidence_deltas = [], []
    for expected_index in range(len(expected)):
        ranked = overlaps[expected_index].argsort(descending=True).tolist()
        actual_index = next(index for index in ranked if index not in used)
        used.add(actual_index)
        ious.append(float(overlaps[expected_index, actual_index]))
        confidence_deltas.append(abs(float(expected[expected_index, 4]) - float(actual[actual_index, 4])))
    return {"detections": len(expected), "minimum_iou": min(ious), "maximum_confidence_delta": max(confidence_deltas)}


def frozen_state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name not in trainable}


def verify_frozen_state(model: torch.nn.Module, expected: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    require(expected.keys() <= current.keys(), "Frozen model state keys changed")
    changed = [name for name, value in expected.items() if not torch.equal(current[name].detach().cpu(), value)]
    require(not changed, f"Frozen parameters or buffers changed: {changed[:5]}")


def compare_scans(expected: dict, actual: dict, tolerance: float = 1e-7) -> dict:
    require(expected["metrics"].keys() == actual["metrics"].keys(), "Serialized scan metric keys changed")
    require(expected["metrics"] == actual["metrics"], "Serialized checkpoint changed discrete guard metrics")
    maximum_delta = 0.0
    for family in ("recoveries", "explicit_regions"):
        expected_records = {record["id"]: record for record in expected[family]}
        actual_records = {record["id"]: record for record in actual[family]}
        require(expected_records.keys() == actual_records.keys(), f"Serialized checkpoint changed {family} identities")
        for identity, record in expected_records.items():
            maximum_delta = max(maximum_delta, abs(record["current"] - actual_records[identity]["current"]))
    require(maximum_delta <= tolerance, f"Serialized checkpoint prediction delta exceeds tolerance: {maximum_delta}")
    return {"passed": True, "maximum_confidence_delta": maximum_delta, "tolerance": tolerance}


def prepare_cache(config_path: Path, config: dict) -> None:
    torch.set_num_threads(4)
    manifest = verify_config(config_path, config)
    baseline_ledger = verify_baseline_reports(config)
    cache_root = project_path(config["cache"])
    require(not cache_root.exists(), "V16 cache already exists; use the pinned cache or a new run name")
    cache_root.mkdir(parents=True)
    wrapper = YOLO(str(project_path(config["starting_weights"])))
    model = wrapper.model.float().eval()
    require(model.end2end is False, "R7 uses the wrong inference branch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    captured: dict[str, list[torch.Tensor]] = {}

    def capture(_module, args):
        captured["features"] = [feature.detach().cpu().clone() for feature in args[0]]

    handle = head.register_forward_pre_hook(capture)
    page_summaries = []
    maximum_cache_delta = 0.0
    predictor_equivalence = []
    checked_shapes = set()
    family_counts = {"preservation": 0, "recovery": 0, "explicit_rejection": 0, "background_rejection": 0}
    try:
        for page_number, page in enumerate(manifest["pages"], 1):
            image_path = project_path(page["image"])
            label_path = project_path(page["label"])
            require(file_sha256(image_path) == page["image_sha256"] and file_sha256(label_path) == page["label_sha256"], f"Changed source page: {page['id']}")
            tensor, transform = preprocess(image_path, config["deployment"]["image_size"])
            captured.clear()
            with torch.no_grad():
                decoded, raw = model(tensor)
            features = captured["features"]
            with torch.no_grad():
                cached_decoded, cached_raw = head_prediction(head, features)
            cache_delta = max(float((decoded - cached_decoded).abs().max()), float((raw["one2many"]["scores"] - cached_raw["one2many"]["scores"]).abs().max()))
            maximum_cache_delta = max(maximum_cache_delta, cache_delta)
            require(cache_delta <= 1e-6, f"Cached head differs from native model on {page['id']}: {cache_delta}")

            references_original = read_reference_boxes(label_path, page["width"], page["height"])
            references_input = transform_boxes(references_original, transform)
            raw_boxes = xywh2xyxy(decoded[0, :4].T.detach().cpu())
            raw_probabilities = raw["one2many"]["scores"][0, 0].sigmoid().detach().cpu()
            overlaps = box_iou(raw_boxes, references_input) if len(references_input) else torch.empty((len(raw_boxes), 0))
            maximum_valid_overlap = overlaps.max(1).values if len(references_input) else torch.zeros(len(raw_boxes))
            detections = nms_detections(head, features, transform["input_shape"], transform["original_shape"], config, confidence=0.01)
            accepted = detections[detections[:, 4] >= config["deployment"]["confidence"]]
            accepted_matches = greedy_matches(references_original, accepted, config["assignment"]["positive_minimum_iou"])
            diagnostic_matches = greedy_matches(references_original, detections, config["assignment"]["positive_minimum_iou"])
            references = []
            positive_indices = []
            for reference_index in range(len(references_original)):
                eligible = torch.where(overlaps[:, reference_index] >= config["assignment"]["positive_minimum_iou"])[0]
                require(bool(len(eligible)), f"No frozen candidate reaches the positive IoU requirement for {page['id']} label {reference_index + 1}")
                baseline_raw = float(raw_probabilities[eligible].max())
                accepted_confidence = float(accepted[accepted_matches[reference_index], 4]) if reference_index in accepted_matches else 0.0
                diagnostic_confidence = float(detections[diagnostic_matches[reference_index], 4]) if reference_index in diagnostic_matches else 0.0
                role = "preservation" if accepted_confidence >= config["deployment"]["confidence"] else "recovery"
                if role == "preservation":
                    requested = max(config["objectives"]["preservation_probability"], baseline_raw - config["objectives"]["maximum_preservation_drop"])
                    floor = min(baseline_raw, requested)
                else:
                    floor = config["objectives"]["recovery_probability"]
                references.append({
                    "id": f"{page['id']}:label_{reference_index + 1}",
                    "role": role,
                    "baseline_raw_probability": baseline_raw,
                    "baseline_accepted_confidence": accepted_confidence,
                    "baseline_diagnostic_confidence": diagnostic_confidence,
                    "objective_floor_probability": floor,
                })
                positive_indices.append(eligible)
                family_counts[role] += 1

            explicit_indices = []
            explicit_regions = []
            explicit_union = torch.zeros(len(raw_boxes), dtype=torch.bool)
            for region_index, region_xyxyn in enumerate(page["negative_regions_xyxyn"], 1):
                region = transform_region(region_xyxyn, page["width"], page["height"], transform)
                intersection_wh = (torch.minimum(raw_boxes[:, 2:], region[2:]) - torch.maximum(raw_boxes[:, :2], region[:2])).clamp(min=0)
                prediction_area = (raw_boxes[:, 2:] - raw_boxes[:, :2]).clamp(min=0).prod(1).clamp(min=1)
                selected = (intersection_wh.prod(1) / prediction_area >= config["assignment"]["negative_region_overlap"]) & (maximum_valid_overlap < config["assignment"]["negative_maximum_valid_iou"])
                indices = torch.where(selected)[0]
                require(bool(len(indices)), f"No frozen candidate exists in explicit negative region {page['id']}:{region_index}")
                explicit_union[indices] = True
                explicit_indices.append(indices)
                explicit_regions.append({
                    "id": f"{page['id']}:negative_{region_index}",
                    "xyxyn": region_xyxyn,
                    "baseline_raw_probability": float(raw_probabilities[indices].max()),
                })
                family_counts["explicit_rejection"] += 1
            background_indices = torch.where((maximum_valid_overlap < config["assignment"]["negative_maximum_valid_iou"]) & ~explicit_union)[0]
            require(bool(len(background_indices)), f"No background anchors on {page['id']}")
            family_counts["background_rejection"] += 1
            baseline_unmatched = unmatched_detections(references_original, accepted, config["assignment"]["positive_minimum_iou"])

            cache_record = {
                "id": page["id"],
                "image": page["image"],
                "features": features,
                "transform": transform,
                "reference_boxes": references_original,
                "references": references,
                "positive_indices": positive_indices,
                "explicit_regions": explicit_regions,
                "explicit_indices": explicit_indices,
                "background_indices": background_indices,
                "baseline_unmatched_accepted": baseline_unmatched,
            }
            cache_path = cache_root / f"{page_number:03d}_{page['issue']}_p{page['page']:04d}.pt"
            torch.save(cache_record, cache_path)
            page_summaries.append({
                "id": page["id"], "cache": relative(cache_path), "cache_sha256": file_sha256(cache_path),
                "input_shape": transform["input_shape"], "valid_notices": len(references),
                "preservation": sum(record["role"] == "preservation" for record in references),
                "recovery": sum(record["role"] == "recovery" for record in references),
                "explicit_negative_regions": len(explicit_indices),
                "baseline_unmatched_accepted": len(baseline_unmatched),
            })
            shape_signature = (tuple(transform["input_shape"]), tuple(transform["original_shape"]))
            if shape_signature not in checked_shapes:
                direct_wrapper = YOLO(str(project_path(config["starting_weights"])))
                direct_result = direct_wrapper.predict(
                    str(image_path), imgsz=config["deployment"]["image_size"], conf=config["deployment"]["confidence"],
                    iou=config["deployment"]["nms_iou"], max_det=config["deployment"]["max_detections"],
                    device="cpu", half=False, verbose=False, save=False,
                )[0]
                direct = torch.cat((direct_result.boxes.xyxy.cpu(), direct_result.boxes.conf[:, None].cpu(), direct_result.boxes.cls[:, None].cpu()), 1)
                comparison = compare_detection_sets(direct, accepted)
                require(comparison["minimum_iou"] >= 0.9999 and comparison["maximum_confidence_delta"] <= 1e-4, "Cached path does not reproduce deployment prediction")
                predictor_equivalence.append({"page": page["id"], "input_shape": transform["input_shape"], "original_shape": transform["original_shape"], **comparison})
                checked_shapes.add(shape_signature)
            print(f"cached {page_number:03d}/{len(manifest['pages']):03d} {page['id']} preserve={page_summaries[-1]['preservation']} recover={page_summaries[-1]['recovery']} negatives={len(explicit_indices)}", flush=True)
    finally:
        handle.remove()

    cache_manifest = {
        "status": "prepared_not_fitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": file_sha256(config_path),
        "data_manifest": config["prepared_manifest"],
        "data_manifest_sha256": file_sha256(project_path(config["prepared_manifest"])),
        "starting_weights_sha256": config["starting_weights_sha256"],
        "family_counts": family_counts,
        "maximum_cache_output_delta": maximum_cache_delta,
        "predictor_equivalence": predictor_equivalence,
        "pages": page_summaries,
    }
    write_json(cache_root / "cache_manifest.json", cache_manifest)
    baseline_ledger["training_inventory"] = {
        "pages": len(page_summaries),
        "preservation_references": family_counts["preservation"],
        "recovery_references": family_counts["recovery"],
        "explicit_negative_regions": family_counts["explicit_rejection"],
        "baseline_unmatched_accepted": sum(page["baseline_unmatched_accepted"] for page in page_summaries),
    }
    baseline_ledger["cache_manifest"] = relative(cache_root / "cache_manifest.json")
    baseline_ledger["cache_manifest_sha256"] = file_sha256(cache_root / "cache_manifest.json")
    write_json(project_path(config["audit_root"]) / "baseline_ledger.json", baseline_ledger)
    print(json.dumps({"status": cache_manifest["status"], "family_counts": family_counts, "predictor_equivalence": predictor_equivalence, "cache_manifest_sha256": baseline_ledger["cache_manifest_sha256"]}, indent=2))


def cache_files(config_path: Path, config: dict) -> tuple[dict, list[Path]]:
    cache_root = project_path(config["cache"])
    manifest_path = cache_root / "cache_manifest.json"
    require(manifest_path.is_file(), "Run guarded_classification_correction.py prepare first")
    manifest = load_json(manifest_path)
    require(manifest["config_sha256"] == file_sha256(config_path), "Cache configuration hash changed")
    paths = [project_path(page["cache"]) for page in manifest["pages"]]
    for page, path in zip(manifest["pages"], paths):
        require(path.is_file() and file_sha256(path) == page["cache_sha256"], f"Changed cache page: {path}")
    return manifest, paths


def load_cache(path: Path) -> dict:
    return torch.load(path, weights_only=True, map_location="cpu")


def set_trainable_scope(model: torch.nn.Module, config: dict) -> list[torch.nn.Parameter]:
    prefix = config["trainable_scope"]["prefix"]
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefix))
        if parameter.requires_grad:
            selected.append(parameter)
    require(len(selected) == config["trainable_scope"]["tensors"], "Trainable tensor count changed")
    require(sum(parameter.numel() for parameter in selected) == config["trainable_scope"]["parameters"], "Trainable parameter count changed")
    model.eval()
    return selected


def lower_hinge(logits: torch.Tensor, probability: float) -> torch.Tensor:
    return torch.relu(torch.as_tensor(logit(probability), dtype=logits.dtype) - logits.max()).square()


def upper_hinge(logits: torch.Tensor, probability: float) -> torch.Tensor:
    return torch.relu(logits.max() - torch.as_tensor(logit(probability), dtype=logits.dtype)).square()


def snapshot_training_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    return parameters, deepcopy(optimizer.state_dict())


def restore_training_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, snapshot: tuple[dict, dict]) -> None:
    parameters, optimizer_state = snapshot
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(parameters[name])
    optimizer.load_state_dict(optimizer_state)


def gradient_probe(
    model: torch.nn.Module,
    head: torch.nn.Module,
    selected: list[torch.nn.Parameter],
    paths: list[Path],
    config: dict,
) -> dict:
    pages = [load_cache(path) for path in paths]
    recovery_page = next(page for page in pages if any(reference["role"] == "recovery" for reference in page["references"]))
    negative_page = next(page for page in pages if page["explicit_indices"])
    background_page = next(page for page in pages if len(page["baseline_unmatched_accepted"]))
    probes = list({page["id"]: page for page in (recovery_page, negative_page, background_page)}.values())
    objectives = config["objectives"]
    gradients = {}
    group_counts = {}
    for family in objectives["weights"]:
        model.zero_grad(set_to_none=True)
        groups = 0
        for page in probes:
            logits = classification_logits(head, page["features"])[0, 0]
            loss = logits.sum() * 0
            if family in ("preservation", "recovery"):
                for reference, indices in zip(page["references"], page["positive_indices"]):
                    if reference["role"] == family:
                        loss = loss + lower_hinge(logits[indices], reference["objective_floor_probability"])
                        groups += 1
            elif family == "explicit_rejection":
                for indices in page["explicit_indices"]:
                    loss = loss + upper_hinge(logits[indices], objectives["negative_ceiling_probability"])
                    groups += 1
            else:
                loss = loss + upper_hinge(logits[page["background_indices"]], objectives["background_ceiling_probability"])
                groups += 1
            loss.backward()
        require(groups > 0, f"Gradient probe has no {family} group")
        vector = torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().flatten().cpu() for parameter in selected]) / groups
        norm = float(vector.norm())
        require(math.isfinite(norm), f"Nonfinite {family} probe gradient")
        gradients[family] = vector
        group_counts[family] = groups
    cosines = {}
    families = list(gradients)
    for left_index, left in enumerate(families):
        for right in families[left_index + 1:]:
            denominator = float(gradients[left].norm() * gradients[right].norm())
            cosines[f"{left}:{right}"] = float(torch.dot(gradients[left], gradients[right]) / denominator) if denominator else None
    model.zero_grad(set_to_none=True)
    return {
        "pages": [page["id"] for page in probes],
        "group_counts": group_counts,
        "gradient_norms": {family: float(vector.norm()) for family, vector in gradients.items()},
        "gradient_cosines": cosines,
    }


def backward_global_objective(head: torch.nn.Module, paths: list[Path], config: dict, counts: dict) -> dict:
    objectives = config["objectives"]
    weights = objectives["weights"]
    totals = {name: 0.0 for name in weights}
    for path in paths:
        page = load_cache(path)
        logits = classification_logits(head, page["features"])[0, 0]
        page_losses = {name: logits.sum() * 0 for name in weights}
        for reference, indices in zip(page["references"], page["positive_indices"]):
            family = reference["role"]
            page_losses[family] = page_losses[family] + lower_hinge(logits[indices], reference["objective_floor_probability"])
        for indices in page["explicit_indices"]:
            page_losses["explicit_rejection"] = page_losses["explicit_rejection"] + upper_hinge(logits[indices], objectives["negative_ceiling_probability"])
        page_losses["background_rejection"] = upper_hinge(logits[page["background_indices"]], objectives["background_ceiling_probability"])
        loss = logits.sum() * 0
        for family in weights:
            if counts[family]:
                component = page_losses[family] / counts[family]
                loss = loss + weights[family] * component
                totals[family] += float(component.detach())
        loss.backward()
    return totals


def scan_model(head: torch.nn.Module, paths: list[Path], config: dict) -> dict:
    objective = config["objectives"]
    preserve_failures, recoveries, explicit_regions = [], [], []
    new_background, background_increases = [], []
    with torch.no_grad():
        for path in paths:
            page = load_cache(path)
            decoded, raw = head_prediction(head, page["features"])
            logits = raw["one2many"]["scores"][0, 0]
            probabilities = logits.sigmoid()
            detections = decoded_detections(decoded, page["transform"]["input_shape"], page["transform"]["original_shape"], config, confidence=0.01)
            accepted = detections[detections[:, 4] >= config["deployment"]["confidence"]]
            accepted_matches = greedy_matches(page["reference_boxes"], accepted, config["assignment"]["positive_minimum_iou"])
            diagnostic_matches = greedy_matches(page["reference_boxes"], detections, config["assignment"]["positive_minimum_iou"])
            for index, reference in enumerate(page["references"]):
                accepted_confidence = float(accepted[accepted_matches[index], 4]) if index in accepted_matches else 0.0
                diagnostic_confidence = float(detections[diagnostic_matches[index], 4]) if index in diagnostic_matches else 0.0
                if reference["role"] == "preservation":
                    required = min(reference["baseline_accepted_confidence"], max(objective["preservation_probability"], reference["baseline_accepted_confidence"] - objective["maximum_preservation_drop"]))
                    if accepted_confidence + 1e-7 < required:
                        preserve_failures.append({"id": reference["id"], "baseline": reference["baseline_accepted_confidence"], "current": accepted_confidence, "required": required})
                else:
                    recoveries.append({"id": reference["id"], "baseline": reference["baseline_diagnostic_confidence"], "current": diagnostic_confidence})
            for region, indices in zip(page["explicit_regions"], page["explicit_indices"]):
                explicit_regions.append({"id": region["id"], "baseline": region["baseline_raw_probability"], "current": float(probabilities[indices].max())})
            unmatched = unmatched_detections(page["reference_boxes"], accepted, config["assignment"]["positive_minimum_iou"])
            baseline_unmatched = page["baseline_unmatched_accepted"]
            for detection in unmatched:
                if not len(baseline_unmatched):
                    new_background.append({"page": page["id"], "confidence": float(detection[4])})
                    continue
                overlaps = box_iou(detection[None, :4], baseline_unmatched[:, :4])[0]
                best = int(overlaps.argmax())
                if float(overlaps[best]) < 0.5:
                    new_background.append({"page": page["id"], "confidence": float(detection[4])})
                elif float(detection[4]) > float(baseline_unmatched[best, 4]) + 0.001:
                    background_increases.append({"page": page["id"], "baseline": float(baseline_unmatched[best, 4]), "current": float(detection[4])})
    recovery_shortfall = sum(max(0.0, objective["recovery_probability"] - record["current"]) for record in recoveries)
    explicit_excess = sum(max(0.0, record["current"] - objective["negative_ceiling_probability"]) for record in explicit_regions)
    return {
        "preservation_failures": preserve_failures,
        "recoveries": recoveries,
        "explicit_regions": explicit_regions,
        "new_background_accepted": new_background,
        "background_confidence_increases": background_increases,
        "metrics": {
            "recovery_shortfall": recovery_shortfall,
            "explicit_negative_excess": explicit_excess,
            "recoveries_accepted": sum(record["current"] >= config["deployment"]["confidence"] for record in recoveries),
            "recoveries_total": len(recoveries),
            "explicit_regions_at_margin": sum(record["current"] <= objective["negative_ceiling_probability"] for record in explicit_regions),
            "explicit_regions_total": len(explicit_regions),
            "preservation_failures": len(preserve_failures),
            "new_background_accepted": len(new_background),
            "background_confidence_increases": len(background_increases),
        },
    }


def accepted_transaction(before: dict, after: dict) -> tuple[bool, list[str]]:
    reasons = []
    metrics = after["metrics"]
    if metrics["preservation_failures"]:
        reasons.append("preservation_regression")
    if metrics["new_background_accepted"]:
        reasons.append("new_background_detection")
    if metrics["background_confidence_increases"]:
        reasons.append("background_confidence_increase")
    if metrics["recovery_shortfall"] > before["metrics"]["recovery_shortfall"] + 1e-7:
        reasons.append("recovery_shortfall_increased")
    if metrics["explicit_negative_excess"] > before["metrics"]["explicit_negative_excess"] + 1e-7:
        reasons.append("explicit_negative_excess_increased")
    return not reasons, reasons


def save_fp32_checkpoint(
    model: torch.nn.Module,
    wrapper: YOLO,
    path: Path,
    config: dict,
    progress: dict,
    cache_paths: list[Path] | None = None,
    expected_scan: dict | None = None,
    expected_frozen_state: dict[str, torch.Tensor] | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_model = deepcopy(model).float().eval()
    checkpoint_model.criterion = None
    torch.save({
        "model": checkpoint_model,
        "ema": None,
        "optimizer": None,
        "epoch": -1,
        "train_args": {**wrapper.overrides, "device": "cpu", "half": False},
        "v16": {"config": config, "progress": progress},
    }, path)
    reloaded = YOLO(str(path)).model.float().eval()
    original_state, reloaded_state = model.state_dict(), reloaded.state_dict()
    require(original_state.keys() == reloaded_state.keys(), "Reloaded checkpoint state keys changed")
    require(all(torch.equal(original_state[name].cpu(), reloaded_state[name].cpu()) for name in original_state), "FP32 checkpoint round trip changed model state")
    if expected_frozen_state is not None:
        verify_frozen_state(reloaded, expected_frozen_state)
    if cache_paths is not None and expected_scan is not None:
        progress["serialized_prediction_equivalence"] = compare_scans(expected_scan, scan_model(reloaded.model[-1], cache_paths, config))
    return file_sha256(path)


def diagnostic_gate(config: dict, initial: dict, current: dict, accepted_updates: int) -> dict:
    gate = config["diagnostic_gate"]
    initial_recovery = initial["metrics"]["recovery_shortfall"]
    initial_negative = initial["metrics"]["explicit_negative_excess"]
    recovery_reduction = (initial_recovery - current["metrics"]["recovery_shortfall"]) / initial_recovery if initial_recovery else 1.0
    negative_reduction = (initial_negative - current["metrics"]["explicit_negative_excess"]) / initial_negative if initial_negative else 1.0
    recovery_by_id = {record["id"]: record for record in current["recoveries"]}
    every_recovery_improved = all(recovery_by_id[record["id"]]["current"] > record["baseline"] + 1e-6 for record in initial["recoveries"])
    negative_by_id = {record["id"]: record for record in current["explicit_regions"]}
    every_negative_not_worse = all(negative_by_id[record["id"]]["current"] <= record["baseline"] + 1e-7 for record in initial["explicit_regions"])
    checks = {
        "minimum_accepted_updates": accepted_updates >= gate["minimum_accepted_updates"],
        "no_safety_failures": current["metrics"]["preservation_failures"] == 0 and current["metrics"]["new_background_accepted"] == 0 and current["metrics"]["background_confidence_increases"] == 0,
        "recovery_shortfall_reduction": recovery_reduction >= gate["minimum_recovery_shortfall_reduction"],
        "explicit_negative_excess_reduction": negative_reduction >= gate["minimum_explicit_negative_excess_reduction"],
        "every_recovery_improved": every_recovery_improved,
        "every_target_negative_not_worse": every_negative_not_worse,
    }
    return {"passed": all(checks.values()), "checks": checks, "recovery_shortfall_reduction": recovery_reduction, "explicit_negative_excess_reduction": negative_reduction}


def run_fit(config_path: Path, config: dict, stage: str) -> None:
    torch.set_num_threads(4)
    verify_config(config_path, config)
    cache_manifest, paths = cache_files(config_path, config)
    run_root = project_path(config["run_root"])
    audit_root = project_path(config["audit_root"])
    state_path = audit_root / "fit_state.pt"
    progress_path = audit_root / "fit_progress.json"
    wrapper = YOLO(str(project_path(config["starting_weights"])))
    model = wrapper.model.float().eval()
    require(model.end2end is False, "Starting model uses the wrong inference branch")
    selected = set_trainable_scope(model, config)
    frozen_state = frozen_state_snapshot(model)
    head = model.model[-1]
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(selected, lr=optimizer_config["learning_rate"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]), weight_decay=0.0)
    if stage == "diagnostic":
        require(not run_root.exists(), "Diagnostic run already exists")
        run_root.mkdir(parents=True)
        initial = scan_model(head, paths, config)
        require(initial["metrics"]["preservation_failures"] == 0, "R7 fails the initial training safety guard")
        probe = gradient_probe(model, head, selected, paths, config)
        write_json(audit_root / "gradient_probe.json", probe)
        require(probe["gradient_norms"]["recovery"] > 0 and probe["gradient_norms"]["explicit_rejection"] > 0, "An active correction family has zero gradient in the real-model probe")
        progress = {
            "status": "diagnostic_running",
            "config_sha256": file_sha256(config_path),
            "cache_manifest_sha256": file_sha256(project_path(config["cache"]) / "cache_manifest.json"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "proposals": 0,
            "accepted_updates": 0,
            "consecutive_rejections": 0,
            "elapsed_model_compute_seconds": 0.0,
            "initial_scan": initial,
            "gradient_probe": probe,
            "history": [],
        }
        before = initial
    else:
        require(state_path.is_file() and progress_path.is_file(), "Full fitting requires a completed diagnostic state")
        progress = load_json(progress_path)
        require(progress["status"] == "diagnostic_passed", "Diagnostic gate did not authorize full fitting")
        saved = torch.load(state_path, weights_only=True, map_location="cpu")
        require(saved["config_sha256"] == file_sha256(config_path), "Saved fit state uses a different configuration")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        progress["status"] = "fit_running"
        progress["stage"] = stage
        before = scan_model(head, paths, config)
    counts = cache_manifest["family_counts"]
    proposal_limit = config["budget"]["diagnostic_proposals"] if stage == "diagnostic" else config["budget"]["fit_proposals"]
    for stage_proposal in range(1, proposal_limit + 1):
        if progress["elapsed_model_compute_seconds"] >= config["budget"]["maximum_model_compute_seconds"]:
            progress["status"] = "stopped_compute_budget"
            break
        proposal_started = time.perf_counter()
        transaction_snapshot = snapshot_training_state(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        losses = backward_global_objective(head, paths, config, counts)
        gradients = [parameter.grad for parameter in selected]
        require(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), "Missing or nonfinite trainable gradient")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(selected, optimizer_config["maximum_gradient_norm"]))
        require(math.isfinite(gradient_norm) and gradient_norm > 0, "Invalid global gradient norm")
        optimizer.step()
        verify_frozen_state(model, frozen_state)
        after = scan_model(head, paths, config)
        accepted, rejection_reasons = accepted_transaction(before, after)
        if accepted:
            before = after
            progress["accepted_updates"] += 1
            progress["consecutive_rejections"] = 0
        else:
            restore_training_state(model, optimizer, transaction_snapshot)
            expected_parameters, _ = transaction_snapshot
            require(all(torch.equal(parameter.detach(), expected_parameters[name]) for name, parameter in model.named_parameters() if parameter.requires_grad), "Transactional rollback did not restore trainable parameters")
            verify_frozen_state(model, frozen_state)
            progress["consecutive_rejections"] += 1
            if optimizer_config["halve_learning_rate_on_rejection"]:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
        elapsed = time.perf_counter() - proposal_started
        progress["elapsed_model_compute_seconds"] += elapsed
        progress["proposals"] += 1
        record = {
            "stage": stage,
            "stage_proposal": stage_proposal,
            "proposal": progress["proposals"],
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm_before_clipping": gradient_norm,
            "losses": losses,
            "metrics": before["metrics"],
            "elapsed_seconds": elapsed,
        }
        progress["history"].append(record)
        print(json.dumps(record), flush=True)
        if progress["consecutive_rejections"] >= optimizer_config["maximum_consecutive_rejections"] or optimizer.param_groups[0]["lr"] < optimizer_config["minimum_learning_rate"]:
            progress["status"] = "stopped_repeated_rejection"
            break
    current = before
    if stage == "diagnostic" and progress["status"] == "diagnostic_running":
        gate = diagnostic_gate(config, progress["initial_scan"], current, progress["accepted_updates"])
        progress["diagnostic_gate"] = gate
        progress["status"] = "diagnostic_passed" if gate["passed"] else "diagnostic_failed"
    elif stage == "full" and progress["status"] == "fit_running":
        progress["status"] = "fit_completed"
    progress["current_scan"] = current
    progress["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint_path = run_root / "weights" / ("diagnostic.pt" if stage == "diagnostic" else "candidate.pt")
    verify_frozen_state(model, frozen_state)
    checkpoint_sha = save_fp32_checkpoint(model, wrapper, checkpoint_path, config, progress, paths, current, frozen_state)
    progress["checkpoint"] = relative(checkpoint_path)
    progress["checkpoint_sha256"] = checkpoint_sha
    torch.save({
        "config_sha256": file_sha256(config_path),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, state_path)
    write_json(progress_path, progress)
    print(json.dumps({"status": progress["status"], "accepted_updates": progress["accepted_updates"], "proposals": progress["proposals"], "checkpoint": progress["checkpoint"], "checkpoint_sha256": checkpoint_sha, "metrics": current["metrics"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=("diagnostic", "full"), default="diagnostic")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_json(config_path)
    if args.command == "prepare":
        prepare_cache(config_path, config)
    else:
        run_fit(config_path, config, args.stage)


if __name__ == "__main__":
    main()
