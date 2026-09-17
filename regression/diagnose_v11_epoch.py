"""Diagnose state and fixed-threshold confidence drift in the rejected v11 epoch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import torch
from ultralytics import YOLO

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    return {
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def reference_scores(report: dict, classification: str) -> dict[str, float]:
    scores = {}
    for page in report["pages"]:
        for reference in page["references"]:
            if reference["classification"] != classification:
                continue
            match = reference.get("accepted_match") or reference.get("closest_candidate_at_or_above_0.05")
            scores[reference["id"]] = float(match["confidence"]) if match else 0.0
    return scores


def group_name(name: str, config: dict) -> str:
    scope = config["trainable_scope"]
    if name.startswith(tuple(scope["feature_prefixes"])):
        return "feature"
    if name.startswith(tuple(scope["head_prefixes"])):
        return "head"
    return "frozen"


def state_drift(baseline: torch.nn.Module, candidate: torch.nn.Module, config: dict) -> dict:
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if baseline_state.keys() != candidate_state.keys():
        raise RuntimeError("Checkpoint state dictionaries have different keys")
    parameter_names = {name for name, _ in baseline.named_parameters()}
    groups = {
        key: {"changed_tensors": 0, "delta_l2_squared": 0.0, "source_l2_squared": 0.0, "maximum_absolute_delta": 0.0, "names": []}
        for key in ("feature", "head", "frozen_parameters", "buffers")
    }
    changed = []
    for name, source in baseline_state.items():
        current = candidate_state[name]
        source_float = source.detach().float()
        current_float = current.detach().float()
        delta = current_float - source_float
        if not torch.equal(source, current):
            changed.append(name)
        if name in parameter_names:
            scope = group_name(name, config)
            key = scope if scope != "frozen" else "frozen_parameters"
        else:
            key = "buffers"
        if torch.count_nonzero(delta).item():
            groups[key]["changed_tensors"] += 1
            groups[key]["names"].append(name)
        groups[key]["delta_l2_squared"] += float(delta.square().sum())
        groups[key]["source_l2_squared"] += float(source_float.square().sum())
        groups[key]["maximum_absolute_delta"] = max(groups[key]["maximum_absolute_delta"], float(delta.abs().max()))
    for group in groups.values():
        group["delta_l2"] = group.pop("delta_l2_squared") ** 0.5
        source_l2 = group.pop("source_l2_squared") ** 0.5
        group["relative_delta_l2"] = group["delta_l2"] / source_l2 if source_l2 else 0.0
    tracked = {}
    for scale in range(3):
        for branch in ("cv2", "cv3"):
            for suffix in ("weight", "bias"):
                name = f"model.23.{branch}.{scale}.2.{suffix}"
                source = baseline_state[name].detach().float()
                delta = candidate_state[name].detach().float() - source
                tracked[name] = {
                    "changed": bool(torch.count_nonzero(delta).item()),
                    "mean_delta": float(delta.mean()),
                    "maximum_absolute_delta": float(delta.abs().max()),
                    "delta_l2": float(delta.square().sum().sqrt()),
                }
    return {"changed_tensors": len(changed), "groups": groups, "output_layer_drift": tracked}


def sampling_composition(config: dict) -> dict:
    dataset = (ROOT / config["dataset"]).parent
    schedule = (dataset / "train_sampling.txt").read_text(encoding="utf-8").splitlines()
    pages = []
    for entry in schedule:
        image = dataset / entry.removeprefix("./")
        label = dataset / "labels/train" / image.with_suffix(".txt").name
        boxes = len(label.read_text(encoding="utf-8").splitlines())
        pages.append({"image": image.name, "boxes": boxes})
    return {
        "epoch_samples": len(pages),
        "background_samples": sum(page["boxes"] == 0 for page in pages),
        "positive_samples": sum(page["boxes"] > 0 for page in pages),
        "boxes_presented": sum(page["boxes"] for page in pages),
        "background_fraction": sum(page["boxes"] == 0 for page in pages) / len(pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/legal_notice_v11_feature_finetune_config.json")
    parser.add_argument("--baseline", default="runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt")
    parser.add_argument("--candidate", default="runs/legal_notice_v11_feature_finetune_cpu_r2/weights/epoch0.pt")
    parser.add_argument("--baseline-fixed", default="output/regression_audit/legal_notice_v8_protected_head_cpu_r7_paa_analogue/fixed080.json")
    parser.add_argument("--candidate-fixed", default="output/regression_audit/legal_notice_v11_feature_finetune_cpu_r2_epoch01_recovery_eval/fixed080.json")
    parser.add_argument("--output", default="output/regression_audit/legal_notice_v11_feature_finetune_cpu_r2_epoch01_diagnosis.json")
    args = parser.parse_args()
    paths = {key: ROOT / value for key, value in vars(args).items() if key != "output"}
    output = ROOT / args.output
    config = load(paths["config"])
    baseline_yolo = YOLO(str(paths["baseline"]))
    candidate_yolo = YOLO(str(paths["candidate"]))
    baseline_model = baseline_yolo.model.float()
    candidate_model = candidate_yolo.model.float()
    baseline_scores = reference_scores(load(paths["baseline_fixed"]), "valid")
    candidate_scores = reference_scores(load(paths["candidate_fixed"]), "valid")
    if baseline_scores.keys() != candidate_scores.keys():
        raise RuntimeError("Fixed-suite valid reference identities differ")
    deltas = [candidate_scores[key] - baseline_scores[key] for key in baseline_scores]
    report = {
        "status": "rejected_epoch_diagnosed",
        "baseline": {"path": args.baseline, "sha256": file_sha256(paths["baseline"])},
        "candidate": {"path": args.candidate, "sha256": file_sha256(paths["candidate"])},
        "inference": {
            "baseline_end2end": bool(baseline_model.end2end),
            "candidate_end2end": bool(candidate_model.end2end),
            "baseline_names": baseline_yolo.names,
            "candidate_names": candidate_yolo.names,
        },
        "state_drift": state_drift(baseline_model, candidate_model, config),
        "sampling_composition": sampling_composition(config),
        "fixed_valid_confidence": {
            "references": len(baseline_scores),
            "baseline": quantiles(list(baseline_scores.values())),
            "candidate": quantiles(list(candidate_scores.values())),
            "delta": quantiles(deltas),
            "fell_below_0_80": sum(baseline_scores[key] >= 0.8 > candidate_scores[key] for key in baseline_scores),
            "candidate_at_or_above_0_80": sum(value >= 0.8 for value in candidate_scores.values()),
            "candidate_at_or_above_0_50": sum(value >= 0.5 for value in candidate_scores.values()),
            "candidate_absent_below_0_05": sum(value == 0.0 for value in candidate_scores.values()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
