"""Verify and summarize the saved active-head pilot without retraining."""
from pathlib import Path
import json
import torch
from ultralytics import YOLO
import fixed080_acceptance as fixed
import prepare_corrected_dataset as data
from active_head_pilot import ROOT, load, write
from monitored_clslogit_pilot import DEVELOPMENT_ROOTS


def main():
    config = load(ROOT / "regression/active_head_pilot_config.json")
    audit = ROOT / "output/regression_audit" / (config["run_name"] + "_monitor")
    summary = load(audit / "summary.json")
    epoch = summary["epochs"][-1]
    checkpoint = ROOT / epoch["checkpoint"]
    assert fixed.file_sha256(checkpoint) == epoch["checkpoint_sha256"]
    assert fixed.file_sha256(ROOT / config["starting_weights"]) == config["starting_weights_sha256"]
    assert data.tree_sha256((ROOT / config["dataset"]).parent, DEVELOPMENT_ROOTS) == config["development_sha256"]
    original = YOLO(str(ROOT / config["starting_weights"])).model
    candidate = YOLO(str(checkpoint)).model
    best = YOLO(str(checkpoint.parent / "best.pt")).model
    assert not original.end2end and not candidate.end2end and not best.end2end
    a, b, c = original.state_dict(), candidate.state_dict(), best.state_dict()
    assert a.keys() == b.keys() == c.keys()
    changed = [n for n in a if not torch.equal(a[n], b[n])]
    assert changed and all(n.startswith(config["trainable_prefix"]) for n in changed)
    assert all(torch.equal(b[n], c[n]) for n in b)
    f = load(audit / "epoch_01_fixed080.json")
    j = load(audit / "epoch_01_july.json")
    baseline = load(audit / "baseline_july.json")
    initial_ids = {r["reference_id"] for p in baseline["pages"] for r in p["valid_matches"]}
    final_ids = {r["reference_id"] for p in j["pages"] for r in p["valid_matches"]}
    details = {
        "checkpoint_sha256_verified": True, "development_data_unchanged": True,
        "starting_checkpoint_unchanged": True, "best_equals_evaluated_epoch_tensors": True,
        "changed_tensors": changed, "inference_end2end": False,
        "july_previously_valid_lost": sorted(initial_ids - final_ids),
        "july_exclusions": [{"image": p["image"], **r} for p in j["pages"] for r in p["excluded"]],
        "fixed_misses": [{"image": p["image"], "reference": r} for p in f["pages"] for r in p["references"]
                         if r["classification"] == "valid" and r["accepted_match"] is None],
        "boundary_failures": [{"image": p["image"], **b} for p in f["pages"] for b in p["critical_boundaries"] if not b["passed"]],
        "baseline_targets": baseline["recovery_targets"], "candidate_targets": j["recovery_targets"],
        "disposition": "rejected_keep_starting_v6_best",
        "execution_note": "All epoch-1 acceptance reports were saved and the trainer stop flag was set. The shell timeout interrupted the subsequent redundant final validation. No Python training process remained at verification."
    }
    write(audit / "post_run_verification.json", details)
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
