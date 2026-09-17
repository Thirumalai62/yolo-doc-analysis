"""Compare v15 pilot movement on explicit training and release objectives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ultralytics import YOLO

from fixed080_acceptance import file_sha256
from full_category_challenge import iou
from legal_notice_v10_acceptance import read_labels


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def reference_scores(weights: Path, cases: list[dict]) -> dict[str, float]:
    model = YOLO(str(weights))
    scores = {}
    for case in cases:
        result = model.predict(str(case["image"]), conf=0.01, imgsz=1280, device="cpu", verbose=False, save=False)[0]
        predictions = [
            {"confidence": float(box.conf[0]), "xyxy": [float(value) for value in box.xyxy[0].tolist()]}
            for box in result.boxes
        ]
        references = read_labels(case["label"], case["width"], case["height"])
        for line in case["lines"]:
            matching = [prediction["confidence"] for prediction in predictions if iou(prediction["xyxy"], references[line - 1]) >= 0.5]
            scores[f"{case['image'].name}:label_{line}"] = max(matching, default=0.0)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="runs/legal_notice_v15_reviewed_cls_cpu_r4/weights/epoch0.pt")
    parser.add_argument("--output", default="output/regression_audit/legal_notice_v15_reviewed_cls_cpu_r4_monitor/pilot_diagnosis.json")
    args = parser.parse_args()
    baseline = ROOT / "runs/legal_notice_v6_clslogit_cpu_2_r1/weights/best.pt"
    candidate = ROOT / args.candidate
    output = ROOT / args.output
    dataset = ROOT / "dataset_v11_consolidated"
    corrections = load(ROOT / "regression/protected_correction_r2_config.json")["positive_corrections"]
    cases = []
    for image_name, lines in corrections.items():
        image = dataset / "images/train" / image_name
        label = dataset / "labels/train" / f"{image.stem}.txt"
        from PIL import Image
        with Image.open(image) as opened:
            width, height = opened.size
        cases.append({"image": image, "label": label, "width": width, "height": height, "lines": lines})
    baseline_scores = reference_scores(baseline, cases)
    candidate_scores = reference_scores(candidate, cases)

    baseline_target = load(output.parent / "baseline_target.json")
    candidate_target = load(output.parent / "epoch_01_target.json")
    negative_deltas = []
    for baseline_page, candidate_page in zip(baseline_target["pages"], candidate_target["pages"]):
        for baseline_region, candidate_region in zip(baseline_page["negative_regions"], candidate_page["negative_regions"]):
            negative_deltas.append({
                "issue": baseline_page["issue"], "page": baseline_page["page"],
                "baseline": baseline_region["maximum_confidence"],
                "candidate": candidate_region["maximum_confidence"],
                "delta": candidate_region["maximum_confidence"] - baseline_region["maximum_confidence"],
            })
    baseline_july = load(ROOT / "output/regression_audit/legal_notice_v7_active_head_cpu_3_monitor/baseline_july.json")
    candidate_july = load(output.parent / "epoch_01_july.json")
    july_baseline = {record["id"]: record for record in baseline_july["recovery_targets"]}
    july_recovery = [{
        "id": record["id"], "baseline": july_baseline[record["id"]]["confidence"],
        "candidate": record["confidence"], "delta": record["confidence"] - july_baseline[record["id"]]["confidence"],
    } for record in candidate_july["recovery_targets"]]
    training_recovery = [{
        "id": case_id, "baseline": baseline_scores[case_id], "candidate": candidate_scores[case_id],
        "delta": candidate_scores[case_id] - baseline_scores[case_id],
    } for case_id in baseline_scores]
    report = {
        "status": "pilot_rejected_no_continuation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {"path": relative(baseline), "sha256": file_sha256(baseline)},
        "candidate": {"path": relative(candidate), "sha256": file_sha256(candidate)},
        "training_recovery": training_recovery,
        "target_negative_regions": negative_deltas,
        "july_recovery": july_recovery,
        "summary": {
            "training_recovery_improved": sum(record["delta"] > 0 for record in training_recovery),
            "training_recovery_total": len(training_recovery),
            "target_negatives_reduced": sum(record["delta"] < 0 for record in negative_deltas),
            "target_negatives_total": len(negative_deltas),
            "july_recovery_improved": sum(record["delta"] > 0 for record in july_recovery),
            "july_recovery_total": len(july_recovery),
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
