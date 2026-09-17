"""Fit a bounded final-classifier correction with complete-page anchor guards."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from ultralytics import YOLO, __version__

from diagnose_table_targets import page_tensor
import evaluation_cache
import fixed080_acceptance as fixed


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCALES = (1, 2)


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


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def verify_config(config: dict) -> None:
    require(__version__ == config["ultralytics_version"], "Ultralytics version changed")
    for key in (
        "starting_weights",
        "warm_start_weights",
        "acceptance_inventory",
        "fixed_manifest",
        "challenge_manifest",
        "july_manifest",
        "target_manifest",
        "semantic_extra_reviews",
    ):
        path = project_path(config[key])
        require(path.is_file() and fixed.file_sha256(path) == config[f"{key}_sha256"], f"Changed {key}")
    require(
        config["deployment"] == {"confidence": 0.8, "image_size": 1280, "render_dpi": 200, "device": "cpu"},
        "Deployment contract changed",
    )
    scope = config["trainable_scope"]
    require(len(scope["tensors"]) == 4 and scope["parameters"] == 514, "Unexpected trainable scope")
    constraints = config["constraints"]
    require(constraints["negative_ceiling_probability"] < 0.5, "Negative ceiling lacks margin")
    require(constraints["background_ceiling_probability"] < 0.8, "Background ceiling is not below deployment")


def all_review_pages(config: dict) -> list[dict]:
    pages: dict[tuple[str, str, int], dict] = {}
    manifests = {
        "fixed": load(project_path(config["fixed_manifest"])),
        "challenge": load(project_path(config["challenge_manifest"])),
        "july": load(project_path(config["july_manifest"])),
    }
    for suite, manifest in manifests.items():
        for source in manifest["pages"]:
            if suite == "fixed":
                records = [
                    {"id": reference["id"], "kind": reference["classification"], "xyxy": reference["xyxy"]}
                    for reference in source["references"]
                    if reference["classification"] in {"valid", "excluded"}
                ]
            else:
                records = [
                    {"id": reference["id"], "kind": "valid", "xyxy": reference["xyxy"]}
                    for reference in source["valid"]
                ] + [
                    {"id": reference["id"], "kind": "excluded", "xyxy": reference["xyxy"]}
                    for reference in source["excluded"]
                ]
            key = (suite, source["issue"], source["page"])
            require(key not in pages, f"Duplicate review page: {key}")
            pages[key] = {
                "suite": suite,
                "issue": source["issue"],
                "page": source["page"],
                "image": source["image"],
                "image_sha256": source["image_sha256"],
                "records": records,
            }

    for number, review in enumerate(load(project_path(config["semantic_extra_reviews"]))["reviews"], 1):
        suite = "fixed" if review["suite"] == "fixed080" else review["suite"]
        key = (suite, review["issue"], review["page"])
        require(key in pages, f"Semantic review page is absent: {key}")
        pages[key]["records"].append({
            "id": f"semantic:{suite}:{review['issue']}:p{review['page']:04d}:{number}",
            "kind": review["classification"],
            "xyxy": review["xyxy"],
        })

    target = load(project_path(config["target_manifest"]))
    for source in target["pages"]:
        label = project_path(source["label"])
        require(label.is_file() and fixed.file_sha256(label) == source["label_sha256"], f"Changed target label: {label}")
        records = []
        for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            class_id, center_x, center_y, width, height = map(float, line.split())
            require(class_id == 0, f"Unexpected target class in {label}")
            records.append({
                "id": f"target:{source['issue']}:p{source['page']:04d}:valid:{number}",
                "kind": "valid",
                "xyxy": [
                    (center_x - width / 2) * source["width"],
                    (center_y - height / 2) * source["height"],
                    (center_x + width / 2) * source["width"],
                    (center_y + height / 2) * source["height"],
                ],
            })
        for number, normalized in enumerate(source["negative_regions_xyxyn"], 1):
            records.append({
                "id": f"target:{source['issue']}:p{source['page']:04d}:negative:{number}",
                "kind": "target_negative",
                "xyxy": [
                    normalized[0] * source["width"],
                    normalized[1] * source["height"],
                    normalized[2] * source["width"],
                    normalized[3] * source["height"],
                ],
            })
        key = ("target", source["issue"], source["page"])
        require(key not in pages, f"Duplicate target page: {key}")
        pages[key] = {
            "suite": "target",
            "issue": source["issue"],
            "page": source["page"],
            "image": source["image"],
            "image_sha256": source["image_sha256"],
            "records": records,
        }
    return list(pages.values())


def transformed_box(xyxy: list[float], transform: tuple) -> torch.Tensor:
    _, _, ratio, left, top = transform
    return torch.tensor([
        xyxy[0] * ratio + left,
        xyxy[1] * ratio + top,
        xyxy[2] * ratio + left,
        xyxy[3] * ratio + top,
    ], dtype=torch.float32)


def xywh_to_xyxy(output: torch.Tensor) -> torch.Tensor:
    center = output[0, :2].T
    size = output[0, 2:4].T
    return torch.cat((center - size / 2, center + size / 2), dim=1)


def background_upper(teacher_logits: torch.Tensor, config: dict) -> torch.Tensor:
    constraints = config["constraints"]
    probabilities = teacher_logits.sigmoid()
    low = torch.full_like(probabilities, constraints["background_ceiling_probability"])
    high = (probabilities + constraints["maximum_background_probability_increase"]).clamp(max=1.0 - 1e-6)
    upper_probability = torch.where(probabilities < constraints["background_teacher_threshold"], low, high)
    return torch.logit(upper_probability)


def constraint_violations(logits: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.relu(lower - logits), torch.relu(logits - upper))


def final_logits(features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.linear(features.to(weight.dtype), weight.reshape(1, -1), bias).flatten()


def prepare(config: dict, config_path: Path) -> None:
    verify_config(config)
    audit = project_path(config["audit_root"])
    require(not audit.exists(), "Audit directory already exists; use a new immutable run name")
    audit.mkdir(parents=True)
    feature_root = audit / "feature_cache"
    feature_root.mkdir()
    write(audit / "config.json", config)

    baseline_wrapper = YOLO(str(project_path(config["starting_weights"])))
    warm_wrapper = YOLO(str(project_path(config["warm_start_weights"])))
    baseline = baseline_wrapper.model.eval().float()
    warm = warm_wrapper.model.eval().float()
    require(not baseline.end2end and not warm.end2end, "Checkpoint uses the wrong inference branch")
    baseline_state, warm_state = baseline.state_dict(), warm.state_dict()
    inherited_prefixes = (
        "model.23.cv3.1.1.1.", "model.23.cv3.1.2.",
        "model.23.cv3.2.1.1.", "model.23.cv3.2.2.",
    )
    inherited_changes = [name for name in baseline_state if not torch.equal(baseline_state[name], warm_state[name])]
    require(bool(inherited_changes) and all(name.startswith(inherited_prefixes) for name in inherited_changes), "Warm start changed an unexpected tensor")

    warm_head = warm.model[-1]
    captured: dict[int, torch.Tensor] = {}
    handles = [
        warm_head.cv3[scale][-1].register_forward_pre_hook(
            lambda _module, args, index=scale: captured.__setitem__(index, args[0].detach())
        )
        for scale in TRAIN_SCALES
    ]
    pages = all_review_pages(config)
    sessions = {
        "fixed": evaluation_cache.IssueScopedCache(config, "fixed_manifest", audit / "preparation_cache"),
        "july": evaluation_cache.IssueScopedCache(config, "july_manifest", audit / "preparation_cache"),
    }
    cache_files = []
    record_audit = []
    inactive_negatives = []
    inactive_semantic_valid = []
    immutable_scale0_valid = []
    counts = {"pages": 0, "anchors": 0, "positive_anchors": 0, "negative_anchors": 0, "background_anchors": 0}
    required_ids = set(config["required_guard_ids"])
    active_ids = set()
    try:
        for page_index, page in enumerate(pages):
            session = sessions.get(page["suite"])
            if session is not None:
                session.activate(page)
            image = project_path(page["image"])
            require(image.is_file() and fixed.file_sha256(image) == page["image_sha256"], f"Changed review image: {image}")
            tensor, transform = page_tensor(image, auto=True)
            captured.clear()
            with torch.no_grad():
                baseline_output = baseline(tensor)
                warm_output = warm(tensor)
                baseline_raw = baseline_output[1]["one2many"]["scores"][0, 0].float()
                warm_raw = warm_output[1]["one2many"]["scores"][0, 0].float()
                baseline_boxes = xywh_to_xyxy(baseline_output[0])
                warm_boxes = xywh_to_xyxy(warm_output[0])
            require(torch.allclose(baseline_boxes, warm_boxes, atol=1e-4, rtol=0), "Warm start changed the frozen box branch")
            features = {scale: captured[scale][0].flatten(1).T.float().clone() for scale in TRAIN_SCALES}
            lengths = {scale: len(features[scale]) for scale in TRAIN_SCALES}
            scale0_length = len(baseline_raw) - lengths[1] - lengths[2]
            offsets = {0: (0, scale0_length), 1: (scale0_length, scale0_length + lengths[1]), 2: (scale0_length + lengths[1], len(baseline_raw))}
            for scale in TRAIN_SCALES:
                start, end = offsets[scale]
                reconstructed = final_logits(
                    features[scale],
                    warm_state[f"model.23.cv3.{scale}.2.weight"],
                    warm_state[f"model.23.cv3.{scale}.2.bias"],
                )
                require(torch.allclose(reconstructed, warm_raw[start:end], atol=1e-5, rtol=1e-5), "Captured feature does not reconstruct warm logits")

            valid_boxes = torch.stack([
                transformed_box(record["xyxy"], transform)
                for record in page["records"]
                if record["kind"] == "valid"
            ]) if any(record["kind"] == "valid" for record in page["records"]) else torch.empty((0, 4))
            valid_overlap = box_iou(baseline_boxes, valid_boxes).max(dim=1).values if len(valid_boxes) else torch.zeros(len(baseline_boxes))
            scale_cache = {}
            for scale in TRAIN_SCALES:
                start, end = offsets[scale]
                teacher = baseline_raw[start:end]
                scale_cache[scale] = {
                    "features": features[scale],
                    "lower": torch.full((lengths[scale],), -torch.inf),
                    "upper": background_upper(teacher, config),
                    "role": torch.zeros(lengths[scale], dtype=torch.uint8),
                    "teacher": teacher.clone(),
                    "warm": warm_raw[start:end].clone(),
                }

            transformed_records = []
            for record in page["records"]:
                box = transformed_box(record["xyxy"], transform)
                intersection = (torch.minimum(baseline_boxes[:, 2:], box[2:]) - torch.maximum(baseline_boxes[:, :2], box[:2])).clamp(min=0).prod(1)
                prediction_area = (baseline_boxes[:, 2:] - baseline_boxes[:, :2]).clamp(min=0).prod(1).clamp(min=1)
                reference_area = (box[2:] - box[:2]).prod().clamp(min=1)
                overlap = intersection / (prediction_area + reference_area - intersection)
                if record["kind"] == "valid":
                    candidates = torch.where(overlap >= 0.5)[0]
                    require(candidates.numel() > 0, f"No baseline anchor for valid reference {record['id']}")
                    best = int(candidates[baseline_raw[candidates].argmax()])
                    probability = float(baseline_raw[best].sigmoid())
                    if probability < 0.8 and record["id"].startswith("semantic:"):
                        inactive_semantic_valid.append({"id": record["id"], "baseline_probability": probability})
                        continue
                    require(probability >= 0.8, f"Baseline misses valid reference {record['id']}: {probability}")
                    lower_probability = max(config["constraints"]["minimum_valid_probability"], probability - config["constraints"]["maximum_valid_probability_drop"])
                    upper_probability = min(1.0 - 1e-6, probability + config["constraints"]["maximum_valid_probability_increase"])
                    if best < scale0_length:
                        immutable_scale0_valid.append({"id": record["id"], "baseline_probability": probability})
                    else:
                        scale = 1 if best < offsets[1][1] else 2
                        local = best - offsets[scale][0]
                        require(scale_cache[scale]["role"][local] != 2, f"Positive/negative anchor conflict: {record['id']}")
                        scale_cache[scale]["lower"][local] = max(float(scale_cache[scale]["lower"][local]), logit(lower_probability))
                        scale_cache[scale]["upper"][local] = min(float(scale_cache[scale]["upper"][local]), logit(upper_probability))
                        scale_cache[scale]["role"][local] = 1
                        active_ids.add(record["id"])
                    transformed_records.append({"id": record["id"], "kind": "valid", "baseline_probability": probability, "best_anchor": best})
                    record_audit.append({"id": record["id"], "kind": "valid", "baseline_probability": probability})
                    continue

                if page["suite"] in {"fixed", "july"}:
                    spatial = overlap >= 0.5
                else:
                    spatial = (intersection / prediction_area >= 0.5) & (valid_overlap < 0.1)
                candidates = torch.where(spatial)[0]
                if not candidates.numel():
                    inactive_negatives.append({"id": record["id"], "reason": "No frozen-box anchor meets the suite overlap rule"})
                    continue
                scale0_candidates = candidates[candidates < scale0_length]
                if scale0_candidates.numel():
                    scale0_max = float(warm_raw[scale0_candidates].sigmoid().max())
                    require(scale0_max <= config["constraints"]["negative_ceiling_probability"], f"Immutable stride-8 anchor violates negative ceiling for {record['id']}: {scale0_max}")
                assigned = 0
                for scale in TRAIN_SCALES:
                    start, end = offsets[scale]
                    selected = candidates[(candidates >= start) & (candidates < end)] - start
                    if selected.numel():
                        require(not (scale_cache[scale]["role"][selected] == 1).any(), f"Positive/negative anchor conflict: {record['id']}")
                        scale_cache[scale]["upper"][selected] = torch.minimum(
                            scale_cache[scale]["upper"][selected],
                            torch.full_like(scale_cache[scale]["upper"][selected], logit(config["constraints"]["negative_ceiling_probability"])),
                        )
                        scale_cache[scale]["role"][selected] = 2
                        assigned += int(selected.numel())
                if assigned:
                    active_ids.add(record["id"])
                    baseline_probability = float(baseline_raw[candidates].sigmoid().max())
                    transformed_records.append({"id": record["id"], "kind": "negative", "anchors": assigned, "baseline_probability": baseline_probability})
                    record_audit.append({"id": record["id"], "kind": "negative", "baseline_probability": baseline_probability, "anchors": assigned})
                else:
                    inactive_negatives.append({"id": record["id"], "reason": "Only immutable stride-8 anchors meet the suite overlap rule"})

            cache_path = feature_root / f"page_{page_index:04d}.pt"
            torch.save({
                "suite": page["suite"],
                "issue": page["issue"],
                "page": page["page"],
                "image": page["image"],
                "image_sha256": page["image_sha256"],
                "scales": scale_cache,
                "records": transformed_records,
            }, cache_path)
            role_counts = {role: sum(int((bucket["role"] == role).sum()) for bucket in scale_cache.values()) for role in (0, 1, 2)}
            counts["pages"] += 1
            counts["anchors"] += sum(lengths.values())
            counts["background_anchors"] += role_counts[0]
            counts["positive_anchors"] += role_counts[1]
            counts["negative_anchors"] += role_counts[2]
            cache_files.append({"path": relative(cache_path), "sha256": fixed.file_sha256(cache_path), "anchors": sum(lengths.values())})
            print(f"feature cache: {page['suite']} {page['issue']} p{page['page']:04d} ({sum(lengths.values())} anchors)", flush=True)
    finally:
        for session in sessions.values():
            session.close()
        for handle in handles:
            handle.remove()

    require(required_ids <= active_ids, f"Required weak-reference guards are missing: {sorted(required_ids - active_ids)}")
    digest = hashlib.sha256("".join(record["sha256"] for record in cache_files).encode("ascii")).hexdigest()
    manifest = {
        "status": "prepared_not_fitted",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": fixed.file_sha256(config_path),
        "counts": {**counts, "records": len(record_audit), "inactive_negatives": len(inactive_negatives), "inactive_semantic_valid": len(inactive_semantic_valid), "immutable_scale0_valid": len(immutable_scale0_valid)},
        "feature_cache_digest": digest,
        "feature_files": cache_files,
        "records": record_audit,
        "inactive_negatives": inactive_negatives,
        "inactive_semantic_valid": inactive_semantic_valid,
        "immutable_scale0_valid": immutable_scale0_valid,
        "inherited_warm_changes": inherited_changes,
    }
    write(audit / "guard_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "feature_cache_digest": digest}, indent=2))


def verify_cache(config: dict, config_path: Path) -> tuple[Path, dict]:
    audit = project_path(config["audit_root"])
    manifest_path = audit / "guard_manifest.json"
    require(manifest_path.is_file(), "Guard manifest is missing")
    manifest = load(manifest_path)
    require(manifest["config_sha256"] == fixed.file_sha256(config_path), "Config changed after cache preparation")
    hashes = []
    for record in manifest["feature_files"]:
        path = project_path(record["path"])
        require(path.is_file() and fixed.file_sha256(path) == record["sha256"], f"Changed feature cache: {path}")
        hashes.append(record["sha256"])
    digest = hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()
    require(digest == manifest["feature_cache_digest"], "Feature-cache digest changed")
    return audit, manifest


def scope_parameters(model: torch.nn.Module, config: dict) -> dict[str, torch.nn.Parameter]:
    expected = set(config["trainable_scope"]["tensors"])
    parameters = {name: parameter for name, parameter in model.named_parameters() if name in expected}
    require(set(parameters) == expected, "Trainable tensor names changed")
    require(sum(parameter.numel() for parameter in parameters.values()) == config["trainable_scope"]["parameters"], "Trainable parameter count changed")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in expected)
    return parameters


def parameter_vectors(model: torch.nn.Module) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    state = model.state_dict()
    return {
        scale: (state[f"model.23.cv3.{scale}.2.weight"].reshape(-1).clone(), state[f"model.23.cv3.{scale}.2.bias"].reshape(-1).clone())
        for scale in TRAIN_SCALES
    }


def scan_constraints(manifest: dict, vectors: dict[int, tuple[torch.Tensor, torch.Tensor]], tolerance: float, active_keys: set[tuple], maximum_new: int) -> dict:
    heap = []
    serial = 0
    violations = 0
    maximum = 0.0
    maximum_logit = -math.inf
    exact_one = 0
    for file_record in manifest["feature_files"]:
        cache = torch.load(project_path(file_record["path"]), weights_only=True, map_location="cpu")
        for scale in TRAIN_SCALES:
            bucket = cache["scales"][scale]
            weight, bias = vectors[scale]
            logits = final_logits(bucket["features"], weight, bias)
            current = constraint_violations(logits, bucket["lower"], bucket["upper"])
            maximum = max(maximum, float(current.max()))
            maximum_logit = max(maximum_logit, float(logits.max()))
            exact_one += int((logits.sigmoid() == 1).sum())
            failing = torch.where(current > tolerance)[0]
            violations += int(failing.numel())
            for index in failing.tolist():
                key = (file_record["path"], scale, index)
                if key in active_keys:
                    continue
                serial += 1
                entry = (
                    float(current[index]), serial, key,
                    bucket["features"][index].clone(),
                    float(bucket["lower"][index]),
                    float(bucket["upper"][index]),
                    int(bucket["role"][index]),
                )
                if maximum_new <= 0:
                    continue
                if len(heap) < maximum_new:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
    return {
        "violations": violations,
        "maximum_violation": maximum,
        "maximum_logit": maximum_logit,
        "exact_one": exact_one,
        "new": sorted(heap, reverse=True),
    }


def fit(config: dict, config_path: Path) -> None:
    verify_config(config)
    require(config.get("training_authorized") is True, "Linear closure fitting is not authorized")
    audit, manifest = verify_cache(config, config_path)
    run = project_path(config["run_root"])
    require(not run.exists(), "Run directory already exists")

    warm_wrapper = YOLO(str(project_path(config["warm_start_weights"])))
    model = warm_wrapper.model.eval().float()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    scope = scope_parameters(model, config)
    trainable = {}
    warm_vectors = parameter_vectors(model)
    for scale in TRAIN_SCALES:
        trainable[scale] = (
            torch.nn.Parameter(warm_vectors[scale][0].double()),
            torch.nn.Parameter(warm_vectors[scale][1].double()),
        )

    active: dict[int, dict[str, list]] = {scale: {"x": [], "lower": [], "upper": [], "keys": []} for scale in TRAIN_SCALES}
    active_keys: set[tuple] = set()

    def add_constraint(scale: int, key: tuple, feature: torch.Tensor, lower: float, upper: float) -> None:
        if key in active_keys:
            return
        active_keys.add(key)
        active[scale]["x"].append(feature)
        active[scale]["lower"].append(lower)
        active[scale]["upper"].append(upper)
        active[scale]["keys"].append(key)

    tolerance = config["constraints"]["logit_feasibility_tolerance"]
    maximum_new = config["optimizer"]["maximum_new_constraints_per_round"]
    seed_scan = scan_constraints(manifest, warm_vectors, tolerance, active_keys, maximum_new)
    for file_record in manifest["feature_files"]:
        cache = torch.load(project_path(file_record["path"]), weights_only=True, map_location="cpu")
        for scale in TRAIN_SCALES:
            bucket = cache["scales"][scale]
            explicit = torch.where(bucket["role"] > 0)[0]
            for index in explicit.tolist():
                add_constraint(scale, (file_record["path"], scale, index), bucket["features"][index].clone(), float(bucket["lower"][index]), float(bucket["upper"][index]))
    for _, _, key, feature, lower, upper, _ in seed_scan["new"]:
        add_constraint(key[1], key, feature, lower, upper)

    history = []
    feasible = False
    no_new_rounds = 0
    for cutting_round in range(1, config["optimizer"]["maximum_cutting_rounds"] + 1):
        tensors = []
        for scale in TRAIN_SCALES:
            require(bool(active[scale]["x"]), f"Scale {scale} has no active constraints")
            tensors.append((
                scale,
                torch.stack(active[scale]["x"]).double(),
                torch.tensor(active[scale]["lower"], dtype=torch.float64),
                torch.tensor(active[scale]["upper"], dtype=torch.float64),
            ))
        parameters = [parameter for pair in trainable.values() for parameter in pair]
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=config["optimizer"]["learning_rate"],
            max_iter=config["optimizer"]["max_iterations_per_round"],
            max_eval=config["optimizer"]["max_evaluations_per_round"],
            history_size=config["optimizer"]["history_size"],
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )
        evaluations = 0
        margin = config["constraints"]["optimization_safety_margin"]

        def closure() -> torch.Tensor:
            nonlocal evaluations
            optimizer.zero_grad()
            penalty = torch.zeros(())
            for scale, x, lower, upper in tensors:
                logits = final_logits(x, *trainable[scale])
                penalty = penalty + torch.relu(lower - logits + margin).square().sum() + torch.relu(logits - upper + margin).square().sum()
            ridge = sum((parameter - warm.to(parameter.dtype)).square().sum() for scale in TRAIN_SCALES for parameter, warm in zip(trainable[scale], warm_vectors[scale]))
            radius = config["constraints"]["maximum_l2_displacement_from_warm_start"]
            trust = torch.relu(torch.sqrt(ridge + 1e-24) - (radius - margin)).square()
            active_count = sum(len(x) for _, x, _, _ in tensors)
            loss = (
                config["optimizer"]["constraint_weight"] * (penalty + active_count * trust)
                + config["optimizer"]["ridge_weight"] * ridge
            )
            require(torch.isfinite(loss), "Nonfinite linear closure objective")
            loss.backward()
            require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters), "Invalid linear closure gradient")
            evaluations += 1
            return loss

        optimizer.step(closure)
        vectors = {scale: (trainable[scale][0].detach(), trainable[scale][1].detach()) for scale in TRAIN_SCALES}
        displacement = math.sqrt(sum(float((current - warm).square().sum()) for scale in TRAIN_SCALES for current, warm in zip(vectors[scale], warm_vectors[scale])))
        scan = scan_constraints(manifest, vectors, tolerance, active_keys, maximum_new)
        entry = {
            "round": cutting_round,
            "closure_evaluations": evaluations,
            "active_constraints": len(active_keys),
            "violations": scan["violations"],
            "maximum_violation": scan["maximum_violation"],
            "maximum_logit": scan["maximum_logit"],
            "exact_one": scan["exact_one"],
            "l2_displacement": displacement,
            "new_constraints": len(scan["new"]),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if scan["violations"] == 0 and displacement <= config["constraints"]["maximum_l2_displacement_from_warm_start"]:
            feasible = True
            break
        added = 0
        for _, _, key, feature, lower, upper, _ in scan["new"]:
            before = len(active_keys)
            add_constraint(key[1], key, feature, lower, upper)
            added += len(active_keys) - before
        no_new_rounds = no_new_rounds + 1 if added == 0 else 0
        if no_new_rounds >= 2:
            break

    report = {
        "status": "infeasible_no_candidate_saved",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": fixed.file_sha256(config_path),
        "guard_manifest_sha256": fixed.file_sha256(audit / "guard_manifest.json"),
        "history": history,
        "active_constraints": len(active_keys),
    }
    if feasible:
        with torch.no_grad():
            for scale in TRAIN_SCALES:
                scope[f"model.23.cv3.{scale}.2.weight"].copy_(trainable[scale][0].reshape_as(scope[f"model.23.cv3.{scale}.2.weight"]))
                scope[f"model.23.cv3.{scale}.2.bias"].copy_(trainable[scale][1].reshape_as(scope[f"model.23.cv3.{scale}.2.bias"]))
        changed = [name for name, value in model.state_dict().items() if not torch.equal(value, original[name])]
        require(bool(changed) and set(changed) <= set(config["trainable_scope"]["tensors"]), "Candidate changed a frozen tensor or changed nothing")
        model.end2end = False
        model.criterion = None
        checkpoint = {
            "model": deepcopy(model).float().eval(),
            "ema": None,
            "epoch": -1,
            "train_args": {**warm_wrapper.overrides, "device": "cpu"},
            "correction": {"config": config, "guard_manifest_sha256": report["guard_manifest_sha256"]},
        }
        temporary = audit / "candidate.verify.pt"
        torch.save(checkpoint, temporary)
        reloaded = YOLO(str(temporary)).model.eval().float()
        require(not reloaded.end2end, "Reloaded candidate changed inference branch")
        reloaded_vectors = parameter_vectors(reloaded)
        reload_scan = scan_constraints(manifest, reloaded_vectors, tolerance, set(), 0)
        require(reload_scan["violations"] == 0, "Reloaded candidate violates cached constraints")
        require(reload_scan["exact_one"] == 0, "Reloaded candidate saturates an anchor to probability one")
        require(reload_scan["maximum_logit"] <= config["constraints"]["maximum_logit"], "Reloaded candidate exceeds maximum logit")
        (run / "weights").mkdir(parents=True)
        candidate = run / "weights/candidate.pt"
        shutil.move(str(temporary), candidate)
        report.update({
            "status": "feasible_candidate_saved",
            "candidate": relative(candidate),
            "candidate_sha256": fixed.file_sha256(candidate),
            "changed_tensors": changed,
            "reload_scan": {key: value for key, value in reload_scan.items() if key != "new"},
        })
    write(audit / "fit_report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "active_constraints")}, indent=2))


def preflight(config: dict, config_path: Path) -> None:
    verify_config(config)
    audit, manifest = verify_cache(config, config_path)
    baseline = YOLO(str(project_path(config["starting_weights"]))).model.eval().float()
    warm = YOLO(str(project_path(config["warm_start_weights"]))).model.eval().float()
    baseline_vectors = parameter_vectors(baseline)
    warm_vectors = parameter_vectors(warm)
    tolerance = config["constraints"]["logit_feasibility_tolerance"]
    result = {
        "status": "preflight_passed_training_authorized" if config.get("training_authorized") else "preflight_passed_training_not_authorized",
        "training_started": False,
        "config_sha256": fixed.file_sha256(config_path),
        "guard_manifest_sha256": fixed.file_sha256(audit / "guard_manifest.json"),
        "counts": manifest["counts"],
        "checkpoints": {},
    }
    for name, vectors in (("baseline_final_layers_on_warm_features", baseline_vectors), ("warm_start", warm_vectors)):
        scan = scan_constraints(manifest, vectors, tolerance, set(), 0)
        result["checkpoints"][name] = {key: value for key, value in scan.items() if key != "new"}
    write(audit / "fit_preflight.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "fit"))
    parser.add_argument("--config", default="regression/legal_notice_v13_linear_guarded_config.json")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load(config_path)
    if args.command == "prepare":
        prepare(config, config_path)
    elif args.command == "preflight":
        preflight(config, config_path)
    else:
        fit(config, config_path)


if __name__ == "__main__":
    main()
