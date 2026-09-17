"""Record the immutable promotion decision for the single v10 candidate."""
from __future__ import annotations

import json
from pathlib import Path

from fixed080_acceptance import file_sha256
from protected_correction import ROOT, load, resolve_config


CONFIG_PATH = ROOT / "regression/legal_notice_v10_targeted_correction_config.json"


def main() -> None:
    config = resolve_config(CONFIG_PATH)
    audit = ROOT / "output/regression_audit" / config["run_name"]
    candidate = ROOT / "runs" / config["run_name"] / "weights/candidate.pt"
    fit = load(audit / "fit_report.json")
    targeted = load(audit / "targeted_acceptance.json")
    preservation = load(audit / "acceptance.json")
    fixed = load(audit / "fixed080.json")
    assert file_sha256(candidate) == fit["candidate_sha256"] == targeted["weights_sha256"]

    margin_failures = []
    for page in targeted["pages"]:
        for region in page["negative_regions"]:
            if not region["margin_passed"]:
                margin_failures.append({
                    "issue": page["issue"], "page": page["page"], "category": page["category"],
                    "maximum_confidence": region["maximum_confidence"], "required_ceiling": 0.5,
                })
    missed_valid = []
    for page in fixed["pages"]:
        for reference in page["references"]:
            if reference["classification"] == "valid" and reference.get("accepted_match") is None:
                diagnostic = reference.get("closest_candidate_at_or_above_0.05")
                missed_valid.append({
                    "id": reference["id"], "issue": page["issue"], "page": page["page"],
                    "confidence": diagnostic["confidence"] if diagnostic else None,
                    "required_confidence": 0.8,
                })

    closure_evaluations = len(fit["history"])
    report = {
        "status": "rejected_not_for_production",
        "candidate": str(candidate.relative_to(ROOT)),
        "candidate_sha256": file_sha256(candidate),
        "starting_checkpoint": config["starting_weights"],
        "starting_checkpoint_sha256": config["starting_weights_sha256"],
        "optimizer_audit": {
            "invocations": 1,
            "configured_max_iterations": config["max_iterations"],
            "configured_max_evaluations": config["max_evaluations"],
            "closure_evaluations": closure_evaluations,
            "evaluation_budget_passed": closure_evaluations <= config["max_evaluations"],
            "note": "The completed run predates max_eval enforcement in protected_nonlinear_correction.py; it was not rerun.",
        },
        "targeted_correction": {
            "operational_threshold_passed": targeted["totals"]["excluded_accepted"] == 0,
            "valid_preserved": f"{targeted['totals']['valid_accepted']}/{targeted['totals']['valid_expected']}",
            "excluded_accepted_at_0_80": targeted["totals"]["excluded_accepted"],
            "negative_margin_passed": f"{targeted['totals']['negative_margin_passed']}/{targeted['totals']['negative_regions']}",
            "margin_failures": margin_failures,
        },
        "preservation": {
            "july_valid": f"{preservation['july']['valid_preserved']}/{preservation['july']['valid_expected']}",
            "gulf_recovery_targets": preservation["targets"],
            "fixed_valid": f"{preservation['fixed']['valid_preserved']}/{preservation['fixed']['valid_expected']}",
            "fixed_missed": missed_valid,
            "critical_boundaries": f"{preservation['fixed']['critical_boundaries'] - preservation['fixed']['critical_boundary_failures']}/{preservation['fixed']['critical_boundaries']}",
            "challenge_valid": f"{preservation['challenge']['valid_preserved']}/{preservation['challenge']['valid_expected']}",
        },
        "rejection_reasons": [
            "negative_margin_failed",
            "fixed_valid_references_lost",
            "optimizer_evaluation_budget_exceeded",
        ],
        "not_run_after_rejection": [
            "all 23 complete September 1-8 issues",
            "normal main.py detect equivalence check",
        ],
        "recommendation": {
            "retain": config["starting_weights"],
            "candidate_use": "diagnostic testing only; do not promote as the robust fixed detector",
        },
    }
    output = audit / "verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "candidate": report["candidate"],
                      "rejection_reasons": report["rejection_reasons"], "output": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
