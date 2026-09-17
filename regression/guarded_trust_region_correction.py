"""Run the bounded V17 minimum-norm trust-region correction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from ultralytics import YOLO

from analyze_v16_gradient_feasibility import constraint_gradients, feasibility_report
import guarded_classification_correction as guarded


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v17_guarded_config.json"


def adjusted_radius(radius: float, accepted: bool, optimizer: dict) -> float:
    factor = optimizer["growth_factor"] if accepted else optimizer["shrink_factor"]
    return min(optimizer["maximum_radius"], max(optimizer["minimum_radius"], radius * factor))


def apply_direction(selected: list[torch.nn.Parameter], direction: torch.Tensor, radius: float) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in selected:
            count = parameter.numel()
            parameter.add_(direction[offset:offset + count].view_as(parameter).to(parameter.dtype), alpha=-radius)
            offset += count
    guarded.require(offset == len(direction), "Trust-region direction does not cover the trainable scope")


def snapshot(selected: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in selected]


def restore(selected: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(selected, values):
            parameter.copy_(value)


def individual_failures(before: dict, after: dict) -> list[str]:
    failures = []
    prior_recoveries = {record["id"]: record["current"] for record in before["recoveries"]}
    for record in after["recoveries"]:
        if record["current"] + 1e-7 < prior_recoveries[record["id"]]:
            failures.append(f"recovery_worsened:{record['id']}")
    prior_negatives = {record["id"]: record["current"] for record in before["explicit_regions"]}
    for record in after["explicit_regions"]:
        if record["current"] > prior_negatives[record["id"]] + 1e-7:
            failures.append(f"negative_worsened:{record['id']}")
    return failures


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    guarded.write_json(temporary, value)
    temporary.replace(path)


def atomic_state(path: Path, config_sha256: str, model: torch.nn.Module) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config_sha256": config_sha256, "model_state": model.state_dict()}, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=guarded.relative(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=("diagnostic", "full"), default="diagnostic")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = guarded.project_path(args.config)
    config = guarded.load_json(config_path)
    guarded.verify_config(config_path, config)
    cache_manifest, paths = guarded.cache_files(config_path, config)
    config_sha256 = guarded.file_sha256(config_path)
    run_root = guarded.project_path(config["run_root"])
    audit_root = guarded.project_path(config["audit_root"])
    progress_path = audit_root / "fit_progress.json"
    state_path = audit_root / "fit_state.pt"
    wrapper = YOLO(str(guarded.project_path(config["starting_weights"])))
    model = wrapper.model.float().eval()
    selected = guarded.set_trainable_scope(model, config)
    frozen_state = guarded.frozen_state_snapshot(model)
    head = model.model[-1]

    if args.resume:
        guarded.require(progress_path.is_file() and state_path.is_file(), "No interrupted V17 state exists")
        progress = guarded.load_json(progress_path)
        guarded.require(progress["config_sha256"] == config_sha256 and progress["stage"] == args.stage, "Interrupted state belongs to another configuration or stage")
        guarded.require(progress["status"] in ("diagnostic_running", "fit_running", "interrupted"), "Stored V17 state is not resumable")
        saved = torch.load(state_path, weights_only=True, map_location="cpu")
        guarded.require(saved["config_sha256"] == config_sha256, "Stored V17 model state uses another configuration")
        model.load_state_dict(saved["model_state"])
        if progress.get("active_proposal"):
            progress["history"].append({"stage": args.stage, "proposal": progress["active_proposal"]["proposal"], "accepted": False, "status": "interrupted_reserved_budget_consumed"})
            progress["active_proposal"] = None
        progress["status"] = f"{args.stage}_running"
        before = guarded.scan_model(head, paths, config)
    elif args.stage == "diagnostic":
        guarded.require(not run_root.exists() and not progress_path.exists() and not state_path.exists(), "V17 diagnostic already exists")
        run_root.mkdir(parents=True)
        initial = guarded.scan_model(head, paths, config)
        guarded.require(initial["metrics"]["preservation_failures"] == 0, "R7 fails the complete initial preservation guard")
        progress = {
            "status": "diagnostic_running", "stage": "diagnostic", "config": guarded.relative(config_path),
            "config_sha256": config_sha256, "cache_manifest_sha256": guarded.file_sha256(guarded.project_path(config["cache"]) / "cache_manifest.json"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "proposals": 0, "accepted_updates": 0,
            "consecutive_rejections": 0, "elapsed_model_compute_seconds": 0.0,
            "radius": config["optimizer"]["initial_radius"], "initial_scan": initial, "active_proposal": None, "history": [],
        }
        before = initial
    else:
        guarded.require(progress_path.is_file() and state_path.is_file(), "Full V17 fit requires a passed diagnostic")
        progress = guarded.load_json(progress_path)
        guarded.require(progress["status"] == "diagnostic_passed", "V17 diagnostic did not authorize full fitting")
        saved = torch.load(state_path, weights_only=True, map_location="cpu")
        guarded.require(saved["config_sha256"] == config_sha256, "V17 diagnostic state uses another configuration")
        model.load_state_dict(saved["model_state"])
        progress["status"], progress["stage"] = "fit_running", "full"
        progress["stage_proposals"] = 0
        before = guarded.scan_model(head, paths, config)

    optimizer = config["optimizer"]
    budget = config["budget"]
    proposal_limit = budget["diagnostic_proposals"] if args.stage == "diagnostic" else budget["fit_proposals"]
    completed_stage = sum(record.get("stage") == args.stage and "direction_norm" in record for record in progress["history"])
    for stage_proposal in range(completed_stage + 1, proposal_limit + 1):
        reservation = budget["reserved_seconds_per_proposal"]
        if progress["elapsed_model_compute_seconds"] + reservation > budget["maximum_model_compute_seconds"]:
            progress["status"] = "stopped_compute_budget"
            break
        progress["proposals"] += 1
        progress["elapsed_model_compute_seconds"] += reservation
        progress["active_proposal"] = {"proposal": progress["proposals"], "stage_proposal": stage_proposal, "reserved_seconds": reservation, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_state(state_path, config_sha256, model)
        atomic_json(progress_path, progress)
        started = time.perf_counter()
        parameter_snapshot = snapshot(selected)
        try:
            directional_config = {**config, "direction_constraints": {"include_background": False}}
            records, matrix = constraint_gradients(model, head, selected, paths, directional_config)
            feasibility, weights = feasibility_report(records, matrix)
            guarded.require(feasibility["status"] == "common_first_order_descent_exists", "Active V17 constraints have no common first-order descent direction")
            direction = weights @ matrix
            guarded.require(torch.isfinite(direction).all() and float(direction.norm()) > 0, "Invalid V17 trust-region direction")
            applied_radius = progress["radius"]
            apply_direction(selected, direction, applied_radius)
            guarded.verify_frozen_state(model, frozen_state)
            after = guarded.scan_model(head, paths, config)
            accepted, reasons = guarded.accepted_transaction(before, after)
            reasons.extend(individual_failures(before, after))
            accepted = accepted and not reasons
            if accepted:
                before = after
                progress["accepted_updates"] += 1
                progress["consecutive_rejections"] = 0
            else:
                restore(selected, parameter_snapshot)
                guarded.verify_frozen_state(model, frozen_state)
                progress["consecutive_rejections"] += 1
            progress["radius"] = adjusted_radius(applied_radius, accepted, optimizer)
            elapsed = time.perf_counter() - started
            progress["elapsed_model_compute_seconds"] += elapsed - reservation
            entry = {
                "stage": args.stage, "stage_proposal": stage_proposal, "proposal": progress["proposals"], "accepted": accepted,
                "rejection_reasons": reasons, "applied_radius": applied_radius, "next_radius": progress["radius"],
                "active_constraints": feasibility["constraints"], "constraints_by_kind": feasibility["by_kind"],
                "minimum_directional_improvement": feasibility["minimum_directional_improvement"], "direction_norm": float(direction.norm()),
                "metrics": before["metrics"], "elapsed_seconds": elapsed,
            }
            progress["history"].append(entry)
            progress["active_proposal"] = None
            atomic_state(state_path, config_sha256, model)
            atomic_json(progress_path, progress)
            print(json.dumps(entry), flush=True)
        except Exception:
            restore(selected, parameter_snapshot)
            guarded.verify_frozen_state(model, frozen_state)
            progress["status"] = "interrupted"
            atomic_state(state_path, config_sha256, model)
            atomic_json(progress_path, progress)
            raise
        if progress["consecutive_rejections"] >= optimizer["maximum_consecutive_rejections"]:
            progress["status"] = "stopped_repeated_rejection"
            break

    current = before
    if args.stage == "diagnostic" and progress["status"] == "diagnostic_running":
        gate = guarded.diagnostic_gate(config, progress["initial_scan"], current, progress["accepted_updates"])
        progress["diagnostic_gate"] = gate
        progress["status"] = "diagnostic_passed" if gate["passed"] else "diagnostic_failed"
    elif args.stage == "full" and progress["status"] == "fit_running":
        progress["status"] = "fit_completed"
    progress["current_scan"] = current
    progress["active_proposal"] = None
    progress["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint_path = run_root / "weights" / ("diagnostic.pt" if args.stage == "diagnostic" else "candidate.pt")
    guarded.verify_frozen_state(model, frozen_state)
    checkpoint_sha256 = guarded.save_fp32_checkpoint(model, wrapper, checkpoint_path, config, progress, paths, current, frozen_state)
    progress["checkpoint"], progress["checkpoint_sha256"] = guarded.relative(checkpoint_path), checkpoint_sha256
    atomic_state(state_path, config_sha256, model)
    atomic_json(progress_path, progress)
    print(json.dumps({"status": progress["status"], "accepted_updates": progress["accepted_updates"], "proposals": progress["proposals"], "checkpoint": progress["checkpoint"], "checkpoint_sha256": checkpoint_sha256, "metrics": current["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
