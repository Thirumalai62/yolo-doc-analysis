"""Run the remaining V16 budget using a common guarded correction direction."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from ultralytics import YOLO

from analyze_v16_gradient_feasibility import constraint_gradients, feasibility_report
import guarded_classification_correction as guarded


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v16_guarded_directional_config.json"


def load_config(path: Path) -> tuple[dict, dict, Path]:
    directional = guarded.load_json(path)
    base_path = guarded.project_path(directional["base_config"])
    guarded.require(guarded.file_sha256(base_path) == directional["base_config_sha256"], "Base V16 configuration changed")
    base = guarded.load_json(base_path)
    guarded.verify_config(base_path, base)
    cache_manifest = guarded.project_path(directional["cache_manifest"])
    guarded.require(guarded.file_sha256(cache_manifest) == directional["cache_manifest_sha256"], "V16 cache changed")
    prior = guarded.load_json(guarded.project_path(directional["prior_diagnostic"]))
    guarded.require(prior["status"] == directional["prior_diagnostic_status"] and prior["proposals"] == directional["prior_proposals_consumed"], "Prior diagnostic evidence changed")
    guarded.require(directional["training_authorized"] is True, "Directional correction is not authorized")
    return directional, base, base_path


def scan_individual_progress(before: dict, after: dict) -> list[str]:
    failures = []
    before_recovery = {record["id"]: record["current"] for record in before["recoveries"]}
    for record in after["recoveries"]:
        if record["current"] + 1e-7 < before_recovery[record["id"]]:
            failures.append(f"recovery_worsened:{record['id']}")
    before_negative = {record["id"]: record["current"] for record in before["explicit_regions"]}
    for record in after["explicit_regions"]:
        if record["current"] > before_negative[record["id"]] + 1e-7:
            failures.append(f"negative_worsened:{record['id']}")
    return failures


def apply_direction(selected: list[torch.nn.Parameter], direction: torch.Tensor, learning_rate: float) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in selected:
            count = parameter.numel()
            parameter.add_(direction[offset:offset + count].view_as(parameter).to(parameter.dtype), alpha=-learning_rate)
            offset += count
    guarded.require(offset == len(direction), "Guarded direction does not cover the trainable scope")


def snapshot(selected: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in selected]


def restore(selected: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(selected, values):
            parameter.copy_(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=guarded.relative(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=("diagnostic", "full"), default="diagnostic")
    args = parser.parse_args()
    config_path = guarded.project_path(args.config)
    directional, base, base_path = load_config(config_path)
    cache_manifest, paths = guarded.cache_files(base_path, base)
    run_root = guarded.project_path(directional["run_root"])
    audit_root = guarded.project_path(directional["audit_root"])
    progress_path = audit_root / "fit_progress.json"
    state_path = audit_root / "fit_state.pt"
    wrapper = YOLO(str(guarded.project_path(base["starting_weights"])))
    model = wrapper.model.float().eval()
    selected = guarded.set_trainable_scope(model, base)
    frozen_state = guarded.frozen_state_snapshot(model)
    head = model.model[-1]
    prior = guarded.load_json(guarded.project_path(directional["prior_diagnostic"]))

    if args.stage == "diagnostic":
        guarded.require(not run_root.exists() and not audit_root.exists(), "Directional diagnostic already exists")
        run_root.mkdir(parents=True)
        audit_root.mkdir(parents=True)
        initial = guarded.scan_model(head, paths, base)
        progress = {
            "status": "diagnostic_running",
            "config": guarded.relative(config_path),
            "config_sha256": guarded.file_sha256(config_path),
            "base_config_sha256": directional["base_config_sha256"],
            "cache_manifest_sha256": directional["cache_manifest_sha256"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "prior_proposals_consumed": directional["prior_proposals_consumed"],
            "proposals": 0,
            "accepted_updates": 0,
            "consecutive_rejections": 0,
            "elapsed_model_compute_seconds": prior["elapsed_model_compute_seconds"],
            "learning_rate": directional["optimizer"]["learning_rate"],
            "initial_scan": initial,
            "history": [],
        }
        before = initial
    else:
        guarded.require(progress_path.is_file() and state_path.is_file(), "Full stage requires a passed directional diagnostic")
        progress = guarded.load_json(progress_path)
        guarded.require(progress["status"] == "diagnostic_passed", "Directional diagnostic gate did not pass")
        saved = torch.load(state_path, weights_only=True, map_location="cpu")
        guarded.require(saved["config_sha256"] == guarded.file_sha256(config_path), "Directional fit state belongs to another configuration")
        model.load_state_dict(saved["model_state"])
        progress["status"] = "fit_running"
        before = guarded.scan_model(head, paths, base)

    proposal_limit = directional["budget"]["remaining_diagnostic_proposals"] if args.stage == "diagnostic" else directional["budget"]["fit_proposals"]
    for stage_proposal in range(1, proposal_limit + 1):
        if progress["elapsed_model_compute_seconds"] >= directional["budget"]["maximum_model_compute_seconds_including_prior"]:
            progress["status"] = "stopped_compute_budget"
            break
        started = time.perf_counter()
        parameter_snapshot = snapshot(selected)
        records, matrix = constraint_gradients(model, head, selected, paths, base)
        feasibility, weights = feasibility_report(records, matrix)
        guarded.require(feasibility["status"] == "common_first_order_descent_exists", "Active constraints no longer have a common first-order descent direction")
        direction = weights @ matrix
        guarded.require(torch.isfinite(direction).all() and float(direction.norm()) > 0, "Invalid guarded direction")
        apply_direction(selected, direction, progress["learning_rate"])
        guarded.verify_frozen_state(model, frozen_state)
        after = guarded.scan_model(head, paths, base)
        accepted, reasons = guarded.accepted_transaction(before, after)
        reasons.extend(scan_individual_progress(before, after))
        accepted = accepted and not reasons
        if accepted:
            before = after
            progress["accepted_updates"] += 1
            progress["consecutive_rejections"] = 0
        else:
            restore(selected, parameter_snapshot)
            guarded.require(all(torch.equal(parameter, value) for parameter, value in zip(selected, parameter_snapshot)), "Directional rollback failed")
            guarded.verify_frozen_state(model, frozen_state)
            progress["consecutive_rejections"] += 1
            if directional["optimizer"]["halve_learning_rate_on_rejection"]:
                progress["learning_rate"] *= 0.5
        elapsed = time.perf_counter() - started
        progress["elapsed_model_compute_seconds"] += elapsed
        progress["proposals"] += 1
        entry = {
            "stage": args.stage,
            "stage_proposal": stage_proposal,
            "total_proposal_including_prior": directional["prior_proposals_consumed"] + progress["proposals"],
            "accepted": accepted,
            "rejection_reasons": reasons,
            "learning_rate": progress["learning_rate"],
            "active_constraints": feasibility["constraints"],
            "minimum_directional_improvement": feasibility["minimum_directional_improvement"],
            "direction_norm": float(direction.norm()),
            "metrics": before["metrics"],
            "elapsed_seconds": elapsed,
        }
        progress["history"].append(entry)
        print(json.dumps(entry), flush=True)
        if progress["consecutive_rejections"] >= directional["optimizer"]["maximum_consecutive_rejections"] or progress["learning_rate"] < directional["optimizer"]["minimum_learning_rate"]:
            progress["status"] = "stopped_repeated_rejection"
            break

    current = before
    if args.stage == "diagnostic" and progress["status"] == "diagnostic_running":
        gate = guarded.diagnostic_gate(base, progress["initial_scan"], current, progress["accepted_updates"])
        progress["diagnostic_gate"] = gate
        progress["status"] = "diagnostic_passed" if gate["passed"] else "diagnostic_failed"
    elif args.stage == "full" and progress["status"] == "fit_running":
        progress["status"] = "fit_completed"
    progress["current_scan"] = current
    progress["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint_path = run_root / "weights" / ("diagnostic.pt" if args.stage == "diagnostic" else "candidate.pt")
    merged_config = {**base, **directional, "base_config": directional["base_config"]}
    guarded.verify_frozen_state(model, frozen_state)
    checkpoint_sha = guarded.save_fp32_checkpoint(model, wrapper, checkpoint_path, merged_config, progress, paths, current, frozen_state)
    progress["checkpoint"] = guarded.relative(checkpoint_path)
    progress["checkpoint_sha256"] = checkpoint_sha
    torch.save({"config_sha256": guarded.file_sha256(config_path), "model_state": model.state_dict()}, state_path)
    guarded.write_json(progress_path, progress)
    print(json.dumps({"status": progress["status"], "accepted_updates": progress["accepted_updates"], "total_proposals_including_prior": directional["prior_proposals_consumed"] + progress["proposals"], "checkpoint": progress["checkpoint"], "checkpoint_sha256": checkpoint_sha, "metrics": current["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
