"""Build guard constraints and fit one bounded closure candidate from R7."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from torchvision.ops import box_iou
from ultralytics import YOLO, __version__

from active_head_pilot import evaluate_july
from diagnose_table_targets import page_tensor
import evaluation_cache
import fixed080_acceptance as fixed
import full_category_challenge as challenge
from monitored_clslogit_pilot import run_challenge, run_fixed
from monitored_feature_finetune import evaluate_target
from protected_correction import nonlinear_logits


ROOT = Path(__file__).resolve().parents[1]


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


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_config(config: dict) -> None:
    require(__version__ == config["ultralytics_version"], "Ultralytics version changed")
    for key in ("starting_weights", "warm_start_weights", "fixed_manifest", "challenge_manifest", "july_manifest", "target_manifest", "semantic_extra_reviews"):
        path = project_path(config[key])
        require(path.is_file() and fixed.file_sha256(path) == config[key + "_sha256"], f"Changed {key}")
    require(config["deployment"] == {"confidence": 0.8, "image_size": 1280, "render_dpi": 200, "device": "cpu"}, "Deployment contract changed")
    require(config["constraints"]["negative_ceiling_probability"] < 0.5, "Negative ceiling needs an inference margin")


def add_page(pages: dict[tuple[str, int], dict], suite: str, page: dict, records: list[dict]) -> None:
    if not records:
        return
    key = (suite, len(pages))
    pages[key] = {
        "suite": suite,
        "issue": page["issue"],
        "page": page["page"],
        "image": page["image"],
        "image_sha256": page["image_sha256"],
        "records": records,
    }


def review_pages(config: dict) -> list[dict]:
    pages: dict[tuple[str, int], dict] = {}
    fixed_manifest = load(project_path(config["fixed_manifest"]))
    challenge_manifest = load(project_path(config["challenge_manifest"]))
    july_manifest = load(project_path(config["july_manifest"]))
    target_manifest = load(project_path(config["target_manifest"]))

    indexed: dict[tuple[str, str, int], dict] = {}
    for suite, manifest in (("fixed", fixed_manifest), ("challenge", challenge_manifest), ("july", july_manifest)):
        for page in manifest["pages"]:
            indexed[(suite, page["issue"], page["page"])] = page

    for page in fixed_manifest["pages"]:
        records = [
            {"id": reference["id"], "kind": reference["classification"], "xyxy": reference["xyxy"]}
            for reference in page["references"]
            if reference["classification"] in {"valid", "excluded"}
        ]
        add_page(pages, "fixed", page, records)

    for page in challenge_manifest["pages"]:
        records = [
            {"id": reference["id"], "kind": "valid", "xyxy": reference["xyxy"]}
            for reference in page["valid"]
        ] + [
            {"id": reference["id"], "kind": "excluded", "xyxy": reference["xyxy"]}
            for reference in page["excluded"]
        ]
        add_page(pages, "challenge", page, records)

    for page in july_manifest["pages"]:
        records = [
            {"id": reference["id"], "kind": "valid", "xyxy": reference["xyxy"]}
            for reference in page["valid"]
        ] + [
            {"id": reference["id"], "kind": "excluded", "xyxy": reference["xyxy"]}
            for reference in page["excluded"]
        ]
        add_page(pages, "july", page, records)

    for review in load(project_path(config["semantic_extra_reviews"]))["reviews"]:
        suite = "fixed" if review["suite"] == "fixed080" else review["suite"]
        source = indexed[(suite, review["issue"], review["page"])]
        record = {
            "id": f"semantic:{suite}:{review['issue']}:p{review['page']:04d}:{len(pages)}",
            "kind": review["classification"],
            "xyxy": review["xyxy"],
        }
        add_page(pages, suite, source, [record])

    for page in target_manifest["pages"]:
        label_path = project_path(page["label"])
        valid = []
        for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            _, x, y, width, height = map(float, line.split())
            valid.append({
                "id": f"target:{page['issue']}:p{page['page']:04d}:valid:{number}",
                "kind": "valid",
                "xyxy": [
                    (x - width / 2) * page["width"],
                    (y - height / 2) * page["height"],
                    (x + width / 2) * page["width"],
                    (y + height / 2) * page["height"],
                ],
            })
        negatives = [
            {
                "id": f"target:{page['issue']}:p{page['page']:04d}:negative:{number}",
                "kind": "target_negative",
                "xyxy": [
                    normalized[0] * page["width"],
                    normalized[1] * page["height"],
                    normalized[2] * page["width"],
                    normalized[3] * page["height"],
                ],
            }
            for number, normalized in enumerate(page["negative_regions_xyxyn"], 1)
        ]
        add_page(pages, "target", page, valid + negatives)
    return list(pages.values())


def transformed_box(xyxy: list[float], transform: tuple) -> torch.Tensor:
    _, _, ratio, left, top = transform
    return torch.tensor([
        xyxy[0] * ratio + left,
        xyxy[1] * ratio + top,
        xyxy[2] * ratio + left,
        xyxy[3] * ratio + top,
    ], dtype=torch.float32)


def prepare(config: dict, config_path: Path) -> None:
    verify_config(config)
    audit = project_path(config["audit_root"])
    require(not audit.exists(), "Closure audit already exists; use a new immutable run name")
    audit.mkdir(parents=True)
    write(audit / "config.json", config)

    wrapper = YOLO(str(project_path(config["starting_weights"])))
    model = wrapper.model.eval()
    require(not model.end2end, "Starting checkpoint uses the wrong inference branch")
    head = model.model[-1]
    captured: dict[int, torch.Tensor] = {}
    handles = [
        head.cv3[scale][1][1].register_forward_pre_hook(
            lambda _module, args, index=scale: captured.__setitem__(index, args[0].detach())
        )
        for scale in range(3)
    ]
    records: list[dict] = []
    buckets = [{"x": [], "record": []} for _ in range(3)]
    pages = review_pages(config)
    cache_sessions = {
        "fixed": evaluation_cache.IssueScopedCache(config, "fixed_manifest", audit / "preparation_cache"),
        "july": evaluation_cache.IssueScopedCache(config, "july_manifest", audit / "preparation_cache"),
    }
    page_audit = []
    inactive_negatives = []
    inactive_semantic_valid = []
    try:
        for page in pages:
            session = cache_sessions.get(page["suite"])
            if session is not None:
                session.activate(page)
            image_path = project_path(page["image"])
            require(image_path.is_file() and fixed.file_sha256(image_path) == page["image_sha256"], f"Changed review image: {image_path}")
            tensor, transform = page_tensor(image_path, auto=True)
            captured.clear()
            with torch.no_grad():
                output = model(tensor)
                raw = output[1]["one2many"]
                decoded = torch.cat((output[0][0, :2].T - output[0][0, 2:4].T / 2, output[0][0, :2].T + output[0][0, 2:4].T / 2), dim=1)
                probabilities = raw["scores"][0, 0].sigmoid()
                features = [captured[scale][0].flatten(1).T.float() for scale in range(3)]
            offsets = [0]
            for feature in features:
                offsets.append(offsets[-1] + len(feature))
            valid_boxes = torch.stack([
                transformed_box(reference["xyxy"], transform)
                for reference in page["records"]
                if reference["kind"] == "valid"
            ]) if any(reference["kind"] == "valid" for reference in page["records"]) else torch.empty((0, 4))
            valid_overlap = box_iou(decoded, valid_boxes).max(dim=1).values if len(valid_boxes) else torch.zeros(len(decoded))
            page_record_ids = []
            for reference in page["records"]:
                box = transformed_box(reference["xyxy"], transform)
                intersection = (torch.minimum(decoded[:, 2:], box[2:]) - torch.maximum(decoded[:, :2], box[:2])).clamp(min=0).prod(1)
                prediction_area = (decoded[:, 2:] - decoded[:, :2]).clamp(min=0).prod(1).clamp(min=1)
                if reference["kind"] == "valid" or page["suite"] in {"fixed", "july"}:
                    reference_area = (box[2:] - box[:2]).prod().clamp(min=1)
                    union = prediction_area + reference_area - intersection
                    spatial = intersection / union >= 0.5
                else:
                    spatial = (intersection / prediction_area >= 0.5) & (valid_overlap < 0.1)
                candidates = torch.where(spatial & (probabilities >= 0.001))[0]
                if not candidates.numel() and reference["kind"] != "valid":
                    inactive_negatives.append({
                        "id": reference["id"],
                        "suite": page["suite"],
                        "issue": page["issue"],
                        "page": page["page"],
                        "reason": "Frozen box branch has no raw anchor meeting the evaluation overlap rule",
                    })
                    print(f"inactive negative: {reference['id']}", flush=True)
                    continue
                require(candidates.numel() > 0, f"No baseline anchor for valid reference {reference['id']}")
                candidates = candidates[probabilities[candidates].argsort(descending=True)[:64]]
                baseline_probability = float(probabilities[candidates].max())
                if reference["kind"] == "valid":
                    if baseline_probability < 0.8 and reference["id"].startswith("semantic:"):
                        inactive_semantic_valid.append({
                            "id": reference["id"],
                            "suite": page["suite"],
                            "issue": page["issue"],
                            "page": page["page"],
                            "baseline_probability": baseline_probability,
                            "reason": "Semantic review box is not an active baseline detection",
                        })
                        print(f"inactive semantic valid: {reference['id']}", flush=True)
                        continue
                    require(baseline_probability >= 0.8, f"Baseline does not preserve {reference['id']}: {baseline_probability}")
                    required_probability = max(
                        config["constraints"]["minimum_valid_probability"],
                        baseline_probability - config["constraints"]["maximum_valid_probability_drop"],
                    )
                else:
                    required_probability = config["constraints"]["negative_ceiling_probability"]
                record_index = len(records)
                records.append({
                    "id": reference["id"],
                    "kind": "valid" if reference["kind"] == "valid" else "negative",
                    "suite": page["suite"],
                    "issue": page["issue"],
                    "page": page["page"],
                    "baseline_probability": baseline_probability,
                    "required_probability": required_probability,
                    "anchors": int(candidates.numel()),
                })
                page_record_ids.append(reference["id"])
                for scale in range(3):
                    selected = candidates[(candidates >= offsets[scale]) & (candidates < offsets[scale + 1])] - offsets[scale]
                    if selected.numel():
                        buckets[scale]["x"].append(features[scale][selected].clone())
                        buckets[scale]["record"].append(torch.full((len(selected),), record_index, dtype=torch.long))
            page_audit.append({
                "suite": page["suite"],
                "issue": page["issue"],
                "page": page["page"],
                "image": relative(image_path),
                "image_sha256": page["image_sha256"],
                "records": page_record_ids,
            })
            print(f"guard cache: {page['suite']} {page['issue']} p{page['page']:04d} ({len(page_record_ids)} records)", flush=True)
    finally:
        for session in cache_sessions.values():
            session.close()
        for handle in handles:
            handle.remove()

    require(len({record["id"] for record in records}) == len(records), "Duplicate closure constraint ID")
    cache = []
    for scale, bucket in enumerate(buckets):
        x = torch.cat(bucket["x"]) if bucket["x"] else torch.empty((0, 256))
        record_ids = torch.cat(bucket["record"]) if bucket["record"] else torch.empty(0, dtype=torch.long)
        with torch.no_grad():
            logits = nonlinear_logits(x, head.cv3[scale][1][1], head.cv3[scale][-1]) if len(x) else torch.empty(0)
        cache.append({"x": x, "record": record_ids, "teacher": logits})
    cache_path = audit / "guard_features.pt"
    torch.save({"records": records, "buckets": cache}, cache_path)
    counts = {
        "records": len(records),
        "valid": sum(record["kind"] == "valid" for record in records),
        "negative": sum(record["kind"] == "negative" for record in records),
        "inactive_negative": len(inactive_negatives),
        "inactive_semantic_valid": len(inactive_semantic_valid),
        "anchors": sum(len(bucket["record"]) for bucket in cache),
        "pages": len(page_audit),
    }
    required_ids = set(config["required_guard_ids"])
    require(required_ids <= {record["id"] for record in records}, "A required weak-reference guard is missing")
    report = {
        "status": "prepared_not_fitted",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": fixed.file_sha256(config_path),
        "starting_weights_sha256": config["starting_weights_sha256"],
        "cache": relative(cache_path),
        "cache_sha256": fixed.file_sha256(cache_path),
        "counts": counts,
        "records": records,
        "inactive_negatives": inactive_negatives,
        "inactive_semantic_valid": inactive_semantic_valid,
        "pages": page_audit,
    }
    write(audit / "guard_manifest.json", report)
    print(json.dumps({"status": report["status"], "counts": counts, "cache_sha256": report["cache_sha256"]}, indent=2))


def active_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    active = {}
    for name, parameter in model.named_parameters():
        learn = any(
            name.startswith(f"model.23.cv3.{scale}.{suffix}")
            for scale in range(3)
            for suffix in ("1.1.", "2.")
        )
        parameter.requires_grad_(learn)
        if learn:
            active[name] = parameter
    require(len(active) == 15 and sum(parameter.numel() for parameter in active.values()) == 198915, "Unexpected closure scope")
    return active


def constrained_scores(model: torch.nn.Module, cache: dict) -> torch.Tensor:
    records = cache["records"]
    result = torch.full((len(records),), -torch.inf)
    head = model.model[-1]
    for scale, bucket in enumerate(cache["buckets"]):
        if not len(bucket["x"]):
            continue
        logits = nonlinear_logits(bucket["x"], head.cv3[scale][1][1], head.cv3[scale][-1])
        result = result.scatter_reduce(0, bucket["record"], logits, reduce="amax", include_self=True)
    require(torch.isfinite(result).all(), "A closure constraint has no scored anchors")
    return result


def teacher_scores(cache: dict) -> torch.Tensor:
    result = torch.full((len(cache["records"]),), -torch.inf)
    for bucket in cache["buckets"]:
        if len(bucket["teacher"]):
            result = result.scatter_reduce(0, bucket["record"], bucket["teacher"], reduce="amax", include_self=True)
    return result


def prune_positive_anchors(cache: dict, maximum: int) -> dict:
    valid = torch.tensor([record["kind"] == "valid" for record in cache["records"]], dtype=torch.bool)
    keep_by_scale = [torch.ones(len(bucket["record"]), dtype=torch.bool) for bucket in cache["buckets"]]
    for record_index in torch.where(valid)[0].tolist():
        locations = []
        for scale, bucket in enumerate(cache["buckets"]):
            for index in torch.where(bucket["record"] == record_index)[0].tolist():
                locations.append((float(bucket["teacher"][index]), scale, index))
        for _, scale, index in sorted(locations, reverse=True)[maximum:]:
            keep_by_scale[scale][index] = False
    return {
        "records": cache["records"],
        "buckets": [
            {key: value[keep] for key, value in bucket.items()}
            for bucket, keep in zip(cache["buckets"], keep_by_scale)
        ],
    }


def constraint_state(scores: torch.Tensor, cache: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kinds = torch.tensor([record["kind"] == "valid" for record in cache["records"]], dtype=torch.bool)
    requirements = torch.tensor([logit(record["required_probability"]) for record in cache["records"]])
    violations = torch.where(kinds, torch.relu(requirements - scores), torch.relu(scores - requirements))
    return violations, kinds, requirements


def preflight(config: dict, config_path: Path) -> None:
    verify_config(config)
    audit = project_path(config["audit_root"])
    manifest = load(audit / "guard_manifest.json")
    require(manifest["config_sha256"] == fixed.file_sha256(config_path), "Config changed after guard preparation")
    cache_path = project_path(manifest["cache"])
    require(fixed.file_sha256(cache_path) == manifest["cache_sha256"], "Guard cache changed")
    cache = torch.load(cache_path, weights_only=True, map_location="cpu")
    before_anchors = sum(len(bucket["record"]) for bucket in cache["buckets"])
    cache = prune_positive_anchors(cache, 8)
    after_anchors = sum(len(bucket["record"]) for bucket in cache["buckets"])
    baseline = YOLO(str(project_path(config["starting_weights"]))).model.eval().float()
    warm = YOLO(str(project_path(config["warm_start_weights"]))).model.eval().float()
    cached = teacher_scores(cache)
    baseline_scores = constrained_scores(baseline, cache)
    require(torch.allclose(cached, baseline_scores, atol=1e-4, rtol=1e-5), "Cached features do not reconstruct baseline logits")
    result = {"status": "preflight_passed_training_authorized" if config["training_authorized"] else "preflight_passed_training_not_authorized", "training_started": False, "config_sha256": fixed.file_sha256(config_path), "guard_manifest_sha256": fixed.file_sha256(audit / "guard_manifest.json"), "anchors_before_pruning": before_anchors, "anchors_after_pruning": after_anchors, "checkpoints": {}}
    for name, model in (("baseline", baseline), ("warm_start", warm)):
        scores = constrained_scores(model, cache)
        violations, valid, _ = constraint_state(scores, cache)
        result["checkpoints"][name] = {
            "positive_violations": int((violations[valid] > config["constraints"]["logit_feasibility_tolerance"]).sum()),
            "negative_violations": int((violations[~valid] > config["constraints"]["logit_feasibility_tolerance"]).sum()),
            "positive_violation_max": float(violations[valid].max()),
            "negative_violation_max": float(violations[~valid].max()),
        }
    write(audit / "fit_preflight.json", result)
    print(json.dumps(result, indent=2))


def fit(config: dict, config_path: Path) -> None:
    verify_config(config)
    require(config.get("training_authorized") is True, "Closure fitting is not authorized")
    audit = project_path(config["audit_root"])
    manifest = load(audit / "guard_manifest.json")
    require(manifest["config_sha256"] == fixed.file_sha256(config_path), "Config changed after guard preparation")
    cache_path = project_path(manifest["cache"])
    require(fixed.file_sha256(cache_path) == manifest["cache_sha256"], "Guard cache changed")
    cache = torch.load(cache_path, weights_only=True, map_location="cpu")
    cache = prune_positive_anchors(cache, 8)
    run = project_path(config["run_root"])
    require(not run.exists(), "Closure run already exists")

    baseline_wrapper = YOLO(str(project_path(config["starting_weights"])))
    baseline = baseline_wrapper.model.eval().float()
    original = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
    warm = YOLO(str(project_path(config["warm_start_weights"]))).model.eval().float()
    active = active_parameters(baseline)
    warm_state = warm.state_dict()
    require(all(torch.equal(warm_state[name], value) for name, value in original.items() if name not in active), "Warm start changed a frozen tensor")
    with torch.no_grad():
        for name, parameter in active.items():
            parameter.copy_(warm_state[name])

    positive_weight = config["optimizer"]["positive_constraint_weight"]
    negative_weight = config["optimizer"]["negative_constraint_weight"]
    ridge_weight = config["optimizer"]["ridge_weight"]
    history = []
    closure_evaluations = 0
    optimizer = torch.optim.LBFGS(
        list(active.values()),
        lr=config["optimizer"]["learning_rate"],
        max_iter=config["optimizer"]["max_iterations"],
        max_eval=config["optimizer"]["max_evaluations"],
        history_size=config["optimizer"]["history_size"],
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad()
        scores = constrained_scores(baseline, cache)
        violations, valid, _ = constraint_state(scores, cache)
        positive = violations[valid].square().sum()
        negative = violations[~valid].square().sum()
        ridge = sum((parameter - original[name]).square().mean() for name, parameter in active.items())
        loss = positive_weight * positive + negative_weight * negative + ridge_weight * ridge
        require(torch.isfinite(loss), "Nonfinite closure objective")
        loss.backward()
        require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in active.values()), "Invalid closure gradient")
        closure_evaluations += 1
        if closure_evaluations == 1 or closure_evaluations % 20 == 0:
            entry = {
                "evaluation": closure_evaluations,
                "loss": float(loss.detach()),
                "positive_violation_max": float(violations[valid].max()),
                "negative_violation_max": float(violations[~valid].max()),
                "positive_violations": int((violations[valid] > 0).sum()),
                "negative_violations": int((violations[~valid] > 0).sum()),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
        return loss

    optimizer.step(closure)
    require(closure_evaluations <= config["optimizer"]["max_evaluations"], "LBFGS exceeded its evaluation budget")
    with torch.no_grad():
        scores = constrained_scores(baseline, cache)
        violations, valid, _ = constraint_state(scores, cache)
        probabilities = scores.sigmoid()
    tolerance = config["constraints"]["logit_feasibility_tolerance"]
    feasible = bool((violations <= tolerance).all())
    record_results = []
    for index, record in enumerate(cache["records"]):
        record_results.append({
            **record,
            "candidate_probability": float(probabilities[index]),
            "logit_violation": float(violations[index]),
            "passed": bool(violations[index] <= tolerance),
        })
    changed = [name for name, value in baseline.state_dict().items() if not torch.equal(value, original[name])]
    require(bool(changed) and set(changed) <= set(active), "Closure fit changed a frozen tensor or changed nothing")
    report = {
        "status": "feasible_candidate_saved" if feasible else "infeasible_no_candidate_saved",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": fixed.file_sha256(config_path),
        "guard_manifest_sha256": fixed.file_sha256(audit / "guard_manifest.json"),
        "closure_evaluations": closure_evaluations,
        "changed_tensors": changed,
        "positive_violation_max": float(violations[valid].max()),
        "negative_violation_max": float(violations[~valid].max()),
        "positive_violations": int((violations[valid] > tolerance).sum()),
        "negative_violations": int((violations[~valid] > tolerance).sum()),
        "history": history,
        "records": record_results,
    }
    if feasible:
        (run / "weights").mkdir(parents=True)
        baseline.end2end = False
        baseline.criterion = None
        checkpoint = {
            "model": deepcopy(baseline).float().eval(),
            "ema": None,
            "epoch": -1,
            "train_args": {**baseline_wrapper.overrides, "device": "cpu"},
            "correction": {"config": config, "guard_manifest_sha256": report["guard_manifest_sha256"]},
        }
        candidate = run / "weights/candidate.pt"
        torch.save(checkpoint, candidate)
        reloaded = YOLO(str(candidate)).model
        require(not reloaded.end2end, "Saved candidate changed the inference branch")
        require(all(torch.equal(reloaded.state_dict()[name], baseline.state_dict()[name]) for name in original), "Saved candidate state changed")
        report["candidate"] = relative(candidate)
        report["candidate_sha256"] = fixed.file_sha256(candidate)
    write(audit / "fit_report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "closure_evaluations", "positive_violation_max", "negative_violation_max", "positive_violations", "negative_violations")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "fit"))
    parser.add_argument("--config", default="regression/legal_notice_v12_closure_config.json")
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
