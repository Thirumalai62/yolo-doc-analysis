"""Test whether V16's individual correction and safety gradients share a descent direction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from ultralytics import YOLO

import guarded_classification_correction as guarded


ROOT = Path(__file__).resolve().parents[1]


def project_simplex(vector: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto nonnegative weights summing to one."""
    sorted_values = torch.sort(vector, descending=True).values
    cumulative = torch.cumsum(sorted_values, 0) - 1
    indices = torch.arange(1, len(vector) + 1, dtype=vector.dtype)
    condition = sorted_values - cumulative / indices > 0
    rho = int(torch.where(condition)[0][-1])
    threshold = cumulative[rho] / (rho + 1)
    return torch.clamp(vector - threshold, min=0)


def minimum_norm_weights(gram: torch.Tensor, iterations: int = 20000) -> torch.Tensor:
    weights = torch.full((len(gram),), 1.0 / len(gram), dtype=torch.float64)
    largest_eigenvalue = float(torch.linalg.eigvalsh(gram).max().clamp(min=1e-6))
    step = 0.9 / largest_eigenvalue
    for _ in range(iterations):
        updated = project_simplex(weights - step * (gram @ weights))
        if float((updated - weights).abs().max()) < 1e-12:
            weights = updated
            break
        weights = updated
    return weights


def gradient_vector(model: torch.nn.Module, selected: list[torch.nn.Parameter], loss: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss.backward()
    vector = torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().flatten().cpu() for parameter in selected]).double()
    norm = vector.norm()
    guarded.require(float(norm) > 0, "Constraint gradient is zero")
    return vector / norm


def constraint_gradients(
    model: torch.nn.Module,
    head: torch.nn.Module,
    selected: list[torch.nn.Parameter],
    paths: list[Path],
    config: dict,
) -> tuple[list[dict], torch.Tensor]:
    objectives = config["objectives"]
    records, vectors = [], []

    for path in paths:
        page = guarded.load_cache(path)
        for reference, indices in zip(page["references"], page["positive_indices"]):
            include_guard = reference["role"] == "preservation" and reference["baseline_accepted_confidence"] < 0.805
            if reference["role"] != "recovery" and not include_guard:
                continue
            logits = guarded.classification_logits(head, page["features"])[0, 0]
            if reference["role"] == "recovery":
                loss = guarded.lower_hinge(logits[indices], reference["objective_floor_probability"])
                kind = "recovery"
            else:
                loss = -logits[indices].max()
                kind = "preservation_guard"
            vectors.append(gradient_vector(model, selected, loss))
            records.append({"id": reference["id"], "kind": kind})
        for region, indices in zip(page["explicit_regions"], page["explicit_indices"]):
            logits = guarded.classification_logits(head, page["features"])[0, 0]
            loss = guarded.upper_hinge(logits[indices], objectives["negative_ceiling_probability"])
            if float(loss.detach()) == 0:
                continue
            vectors.append(gradient_vector(model, selected, loss))
            records.append({"id": region["id"], "kind": "explicit_rejection"})
        if config.get("direction_constraints", {}).get("include_background", True):
            logits = guarded.classification_logits(head, page["features"])[0, 0]
            background_loss = guarded.upper_hinge(logits[page["background_indices"]], objectives["background_ceiling_probability"])
            if float(background_loss.detach()) > 0:
                vectors.append(gradient_vector(model, selected, background_loss))
                records.append({"id": f"{page['id']}:background", "kind": "background_guard"})

    return records, torch.stack(vectors)


def feasibility_report(records: list[dict], matrix: torch.Tensor) -> tuple[dict, torch.Tensor]:
    gram = matrix @ matrix.T
    weights = minimum_norm_weights(gram)
    directional_improvements = gram @ weights
    minimum_improvement = float(directional_improvements.min())
    combined_norm = float(torch.sqrt(weights @ gram @ weights))
    for index, record in enumerate(records):
        record["minimum_norm_weight"] = float(weights[index])
        record["directional_improvement"] = float(directional_improvements[index])
    off_diagonal = gram[~torch.eye(len(gram), dtype=torch.bool)]
    return {
        "status": "common_first_order_descent_exists" if minimum_improvement > 1e-8 else "no_common_first_order_descent",
        "constraints": len(records),
        "by_kind": {kind: sum(record["kind"] == kind for record in records) for kind in sorted({record["kind"] for record in records})},
        "minimum_pairwise_cosine": float(off_diagonal.min()),
        "maximum_pairwise_cosine": float(off_diagonal.max()),
        "minimum_directional_improvement": minimum_improvement,
        "combined_direction_norm": combined_norm,
        "records": records,
    }, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/legal_notice_v16_guarded_config.json")
    parser.add_argument("--correction-only", action="store_true", help="Use recovery and explicit-rejection gradients; enforce preservation/background transactionally")
    args = parser.parse_args()
    config_path = guarded.project_path(args.config)
    config = guarded.load_json(config_path)
    guarded.verify_config(config_path, config)
    _, paths = guarded.cache_files(config_path, config)
    wrapper = YOLO(str(guarded.project_path(config["starting_weights"])))
    model = wrapper.model.float().eval()
    selected = guarded.set_trainable_scope(model, config)
    directional_config = {**config, "direction_constraints": {"include_background": False}} if args.correction_only else config
    records, matrix = constraint_gradients(model, model.model[-1], selected, paths, directional_config)
    report, _ = feasibility_report(records, matrix)
    report.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": guarded.file_sha256(config_path),
        "checkpoint_sha256": config["starting_weights_sha256"],
        "correction_only": args.correction_only,
    })
    output = guarded.project_path(config["audit_root"]) / "gradient_feasibility.json"
    guarded.write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
