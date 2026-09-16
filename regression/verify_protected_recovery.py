"""Verify R2 recovery and review extra detections without weakening old gates."""
from pathlib import Path
import json
import torch
import argparse
from ultralytics import YOLO
import fixed080_acceptance as fixed
import full_category_challenge as challenge
from active_head_pilot import stop_reasons
from protected_correction import ROOT, verify, load, write, OLD_AUDIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/protected_correction_r5_config.json")
    args = parser.parse_args()
    config = load(ROOT / args.config)
    if "base_config" in config:
        config = {**load(ROOT / config["base_config"]), **config}
    verify(config)
    audit = ROOT / "output/regression_audit" / config["run_name"]
    candidate = ROOT / "runs" / config["run_name"] / "weights/candidate.pt"
    expected_hash = load(audit / "fit_report.json")["candidate_sha256"]
    assert fixed.file_sha256(candidate) == expected_hash
    reviews_path = ROOT / "regression/protected_correction_extra_reviews.json"
    reviews = load(reviews_path)["reviews"]
    reviewed, unresolved, excluded = [], [], []
    reports = {suite: load(audit / f"{suite}.json") for suite in ("july", "fixed080", "challenge")}
    for suite in ("july", "fixed080"):
        for page in reports[suite]["pages"]:
            for prediction in page["unreviewed_accepted"]:
                matches = [r for r in reviews if r["suite"] == suite and r["issue"] == page["issue"]
                           and r["page"] == page["page"] and challenge.iou(r["xyxy"], prediction["xyxy"]) >= 0.85]
                record = {"suite": suite, "issue": page["issue"], "page": page["page"], "prediction": prediction}
                if len(matches) != 1:
                    unresolved.append(record)
                elif matches[0]["classification"] == "excluded":
                    excluded.append({**record, "review": matches[0]})
                else:
                    reviewed.append({**record, "review": matches[0]})
    f, c, j = (reports[k] for k in ("fixed080", "challenge", "july"))
    reasons = [r for r in stop_reasons(f, c, j, load(OLD_AUDIT / "baseline_july.json"))
               if r not in ("fixed_unreviewed_accepted", "july_unreviewed_accepted")]
    assert not reasons and not unresolved and not excluded, "Recovery checkpoint failed reviewed preservation"
    assert all(r["complete_accepted"] for r in j["recovery_targets"])
    initial = YOLO(str(ROOT / config["starting_weights"])).model
    current = YOLO(str(candidate)).model
    a, b = initial.state_dict(), current.state_dict()
    assert a.keys() == b.keys() and not current.end2end
    changed = [n for n in a if not torch.equal(a[n], b[n])]
    allowed = {f"model.23.cv3.{scale}.2.{kind}" for scale in range(3) for kind in ("weight", "bias")}
    if config.get("feature_layer") == "pre_pointwise":
        allowed |= {f"model.23.cv3.{scale}.1.1.{suffix}" for scale in range(3) for suffix in ("conv.weight", "bn.weight", "bn.bias")}
    assert set(changed) <= allowed and changed
    from main import render_pdf
    render_dir = audit / "cli_render_equivalence" / "gulftoday_2026-07-03"
    rendered = render_pdf(ROOT / "input/gulftoday_2026-07-03.pdf", render_dir, 200, {12})[0]
    manifest = load(audit / "july_manifest.json")
    page = next(p for p in manifest["pages"] if p["issue"] == "gulftoday_2026-07-03" and p["page"] == 12)
    assert fixed.file_sha256(rendered) == page["image_sha256"]
    model = YOLO(str(candidate))
    result = model.predict(str(rendered), conf=0.8, imgsz=1280, device="cpu", verbose=False, save=False)[0]
    actual = [fixed.prediction_record(box) for box in result.boxes]
    saved = load(audit / "july.predictions.json")
    expected = next(p["predictions"] for p in saved["pages"] if p["image"] == page["image"])
    assert actual == expected, "Normal detect rendering/inference differs from audit"
    result.save(filename=str(audit / "gulftoday_july03_page12_annotated.png"))
    remaining = [{"issue": p["issue"], "page": p["page"], **r} for p in j["pages"] for r in p["excluded"] if r["accepted"]]
    report = {
        "status": "verified_partial_recovery_not_full_scope_pass",
        "candidate": str(candidate.relative_to(ROOT)), "candidate_sha256": expected_hash,
        "all_three_gulf_targets_recovered": True, "july_previously_valid_preserved": 545,
        "july_added_recovery_targets": 3,
        "july_additional_reviewed_valid": sum(r["suite"] == "july" for r in reviewed),
        "fixed_valid_preserved": 271,
        "fixed_additional_reviewed_valid": sum(r["suite"] == "fixed080" for r in reviewed),
        "critical_boundaries_preserved": 4, "challenge_valid_preserved": 31,
        "unreviewed_accepted_after_review": len(unresolved), "new_excluded_after_review": len(excluded),
        "july_remaining_exclusions": remaining,
        "full_promotion_pass": False,
        "original_fixed_exclusions_still_accepted": f["totals"]["excluded_accepted"],
        "challenge_exclusions_still_accepted": c["totals"]["excluded_accepted_regions"],
        "reviews_sha256": fixed.file_sha256(reviews_path), "reviewed_extras": reviewed,
        "normal_detect_page12_equivalence": True,
        "changed_tensors": changed
    }
    write(audit / "reviewed_recovery_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
