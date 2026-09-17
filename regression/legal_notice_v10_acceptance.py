"""Evaluate the reviewed v10 correction pages at the fixed deployment settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "output/correction_data/legal_notice_v10_targeted_correction/manifest.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def iou(first: list[float], second: list[float]) -> float:
    overlap = intersection(first, second)
    return overlap / max(1.0, area(first) + area(second) - overlap)


def read_labels(path: Path, width: int, height: int) -> list[list[float]]:
    references = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, w, h = line.split()
        assert class_id == "0"
        x, y, w, h = (float(value) for value in (x, y, w, h))
        references.append([
            (x - w / 2) * width,
            (y - h / 2) * height,
            (x + w / 2) * width,
            (y + h / 2) * height,
        ])
    return references


def match_references(references: list[list[float]], predictions: list[dict]) -> tuple[list[dict], list[int]]:
    candidates = []
    for reference_id, reference in enumerate(references, start=1):
        for prediction_id, prediction in enumerate(predictions):
            overlap = iou(reference, prediction["xyxy"])
            if overlap >= 0.5:
                candidates.append((overlap, prediction["confidence"], reference_id, prediction_id))
    matched_references, matched_predictions, matches = set(), set(), []
    for overlap, _, reference_id, prediction_id in sorted(candidates, reverse=True):
        if reference_id in matched_references or prediction_id in matched_predictions:
            continue
        matched_references.add(reference_id)
        matched_predictions.add(prediction_id)
        matches.append({
            "reference_id": reference_id,
            "prediction_id": prediction_id + 1,
            "iou": overlap,
            "confidence": predictions[prediction_id]["confidence"],
        })
    return sorted(matches, key=lambda match: match["reference_id"]), sorted(matched_predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--output", required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    weights = ROOT / args.weights
    manifest_path = ROOT / args.manifest
    output = ROOT / args.output
    assert weights.is_file() and manifest_path.is_file()
    assert not output.exists(), f"Refusing to overwrite {output}"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = load(manifest_path)
    model = YOLO(str(weights))
    pages = []
    totals = {"valid_expected": 0, "valid_accepted": 0, "negative_regions": 0,
              "negative_margin_passed": 0, "excluded_accepted": 0, "unmatched_accepted": 0}

    for page in manifest["pages"]:
        image = ROOT / page["image"]
        label = ROOT / page["label"]
        assert file_sha256(image) == page["image_sha256"]
        assert file_sha256(label) == page["label_sha256"]
        references = read_labels(label, page["width"], page["height"])
        result = model.predict(str(image), conf=0.01, imgsz=1280, device="cpu", verbose=False, save=False)[0]
        predictions = [{
            "confidence": float(box.conf[0]),
            "xyxy": [float(value) for value in box.xyxy[0].tolist()],
        } for box in result.boxes]
        accepted = [prediction for prediction in predictions if prediction["confidence"] >= 0.8]
        regions = [[region[0] * page["width"], region[1] * page["height"],
                    region[2] * page["width"], region[3] * page["height"]]
                   for region in page["negative_regions_xyxyn"]]
        region_results = []
        excluded_prediction_ids = set()
        for region_id, region in enumerate(regions, start=1):
            overlapping = [(index, prediction) for index, prediction in enumerate(predictions)
                           if intersection(prediction["xyxy"], region) / max(1.0, area(prediction["xyxy"])) >= 0.5]
            maximum = max((prediction["confidence"] for _, prediction in overlapping), default=0.0)
            for index, prediction in enumerate(accepted):
                if intersection(prediction["xyxy"], region) / max(1.0, area(prediction["xyxy"])) >= 0.5:
                    excluded_prediction_ids.add(index)
            region_results.append({"region_id": region_id, "xyxy": region, "maximum_confidence": maximum,
                                   "accepted": maximum >= 0.8, "margin_passed": maximum <= 0.5})

        eligible = [prediction for index, prediction in enumerate(accepted) if index not in excluded_prediction_ids]
        matches, matched_ids = match_references(references, eligible)
        unmatched = [prediction for index, prediction in enumerate(eligible) if index not in matched_ids]
        totals["valid_expected"] += len(references)
        totals["valid_accepted"] += len(matches)
        totals["negative_regions"] += len(regions)
        totals["negative_margin_passed"] += sum(region["margin_passed"] for region in region_results)
        totals["excluded_accepted"] += len(excluded_prediction_ids)
        totals["unmatched_accepted"] += len(unmatched)
        pages.append({
            "issue": page["issue"], "page": page["page"], "category": page["category"],
            "valid_expected": len(references), "valid_accepted": len(matches), "matches": matches,
            "negative_regions": region_results, "excluded_accepted": len(excluded_prediction_ids),
            "unmatched_accepted": unmatched, "predictions_at_0_80": len(accepted),
        })

    passed = (
        totals["valid_accepted"] == totals["valid_expected"]
        and totals["negative_margin_passed"] == totals["negative_regions"]
        and totals["excluded_accepted"] == 0
        and totals["unmatched_accepted"] == 0
    )
    report = {
        "suite": "legal_notice_v10_targeted_correction",
        "passed": passed,
        "weights": str(weights.relative_to(ROOT)),
        "weights_sha256": file_sha256(weights),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "settings": {"confidence": 0.8, "diagnostic_confidence": 0.01, "image_size": 1280, "device": "cpu"},
        "totals": totals,
        "pages": pages,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, **totals, "output": str(output.relative_to(ROOT))}, indent=2))
    if args.enforce and not passed:
        raise SystemExit("Targeted v10 acceptance failed")


if __name__ == "__main__":
    main()
