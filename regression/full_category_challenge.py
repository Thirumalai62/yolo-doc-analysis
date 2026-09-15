"""Build and run the reviewed full-category development challenge."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ISSUES = {
    "albayan_2026-09-04",
    "albayan_2026-09-07",
    "alfajr_2026-09-04",
    "alfajr_2026-09-05",
    "gulftoday_2026-09-04",
    "khaleejtimes_2026-09-04",
}
EXPECTED_DATASET_SHA256 = "639a294efc01ef5b360a92f9aef55d3c41d6bb469507dab48cecf895b5225c5b"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(value: str) -> str:
    return Path(value.replace("\\", "/")).as_posix().lower()


def xyxy_to_xywhn(box: list[float], width: int, height: int) -> list[float]:
    left, top, right, bottom = box
    return [
        round((left + right) / 2 / width, 6),
        round((top + bottom) / 2 / height, 6),
        round((right - left) / width, 6),
        round((bottom - top) / height, 6),
    ]


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    center_x, center_y, box_width, box_height = box
    return [
        round((center_x - box_width / 2) * width, 2),
        round((center_y - box_height / 2) * height, 2),
        round((center_x + box_width / 2) * width, 2),
        round((center_y + box_height / 2) * height, 2),
    ]


def intersection_area(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(first: list[float], second: list[float]) -> float:
    intersection = intersection_area(first, second)
    union = area(first) + area(second) - intersection
    return intersection / union if union else 0.0


def read_valid_references(page: dict, label_path: Path) -> list[dict]:
    references = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise SystemExit(f"Invalid class-0 YOLO label at {label_path}:{number}")
        coordinates = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
            raise SystemExit(f"Invalid YOLO coordinates at {label_path}:{number}")
        references.append({
            "id": f"{page['issue']}_p{page['page']:04d}_valid_{number:03d}",
            "classification": "valid",
            "category": "valid_legal_notice",
            "reason": "reviewed dataset legal-notice label",
            "xywhn": coordinates,
            "xyxy": xywhn_to_xyxy(coordinates, page["width"], page["height"]),
        })
    return references


def validate_page_guardrails(page: dict) -> None:
    if page["issue"] in FORBIDDEN_ISSUES:
        raise SystemExit(f"Held-out issue is forbidden: {page['issue']}")
    if "_2026-09-09" in page["issue"] or "_2026-09-10" in page["issue"]:
        raise SystemExit(f"Preservation issue is forbidden: {page['issue']}")
    expected_image_prefix = "dataset_v5_corrected/images/val/"
    expected_label_prefix = "dataset_v5_corrected/labels/val/"
    if not page["image"].startswith(expected_image_prefix) or not page["label"].startswith(expected_label_prefix):
        raise SystemExit(f"Challenge page must remain in dataset validation: {page['image']}")


def build_manifest(decisions_path: Path, output_path: Path) -> dict:
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    dataset_release_path = project_path(decisions["dataset_release"])
    dataset_release = json.loads(dataset_release_path.read_text(encoding="utf-8"))
    if dataset_release.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise SystemExit("Dataset release does not attest to the frozen corrected dataset.")

    pages = []
    page_keys = set()
    category_counts = Counter()
    for source_page in decisions["pages"]:
        validate_page_guardrails(source_page)
        key = (source_page["issue"], source_page["page"])
        if key in page_keys:
            raise SystemExit(f"Duplicate challenge page: {key}")
        page_keys.add(key)
        image_path = project_path(source_page["image"])
        label_path = project_path(source_page["label"])
        if not image_path.is_file() or not label_path.is_file():
            raise SystemExit(f"Challenge image or label is missing: {source_page['image']}")

        valid = read_valid_references(source_page, label_path)
        excluded = []
        for reference in source_page["excluded"]:
            box = reference["xyxy"]
            if (
                len(box) != 4
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
                or box[0] < 0
                or box[1] < 0
                or box[2] > source_page["width"]
                or box[3] > source_page["height"]
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise SystemExit(f"Invalid excluded box: {reference['id']}")
            excluded.append({
                **reference,
                "classification": "excluded",
                "xywhn": xyxy_to_xywhn(box, source_page["width"], source_page["height"]),
            })
            category_counts[reference["category"]] += 1

        page = {
            **source_page,
            "image_sha256": file_sha256(image_path),
            "label_sha256": file_sha256(label_path),
            "valid": valid,
            "excluded": excluded,
        }
        category_counts["valid_legal_notice"] += len(valid)
        pages.append(page)

    manifest = {
        "suite": decisions["suite"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "acceptance_confidence": decisions["acceptance_confidence"],
        "image_size": decisions["image_size"],
        "inference": "one page per predict call on CPU",
        "valid_match_iou": decisions["valid_match_iou"],
        "excluded_prediction_overlap": decisions["excluded_prediction_overlap"],
        "dataset_release": relative_path(dataset_release_path),
        "dataset_release_sha256": file_sha256(dataset_release_path),
        "dataset_sha256": dataset_release["dataset_sha256"],
        "decisions": relative_path(decisions_path),
        "decisions_sha256": file_sha256(decisions_path),
        "strict_gates": {
            "valid_missed": 0,
            "excluded_accepted_regions": 0,
            "unreviewed_accepted": 0,
        },
        "reference_counts": dict(sorted(category_counts.items())),
        "unavailable_categories": decisions["unavailable_categories"],
        "notes": [
            "These validation pages are outside training but are not an unbiased final test.",
            "Valid references come from the frozen corrected YOLO labels.",
            "Excluded regions were visually reviewed over complete pages.",
            "A detection is attributed to an exclusion when at least half of the prediction lies inside that reviewed region.",
        ],
        "pages": pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": len(pages), "reference_counts": manifest["reference_counts"]}, indent=2))
    print(f"Challenge manifest created: {relative_path(output_path)}")
    return manifest


def validate_manifest(manifest_path: Path, manifest: dict) -> None:
    if manifest.get("development_only") is not True:
        raise SystemExit("Challenge must be marked development-only.")
    if manifest.get("acceptance_confidence") != 0.8 or manifest.get("image_size") != 1280:
        raise SystemExit("Challenge confidence and image size are not pinned to 0.80 and 1280.")
    dataset_release_path = project_path(manifest["dataset_release"])
    decisions_path = project_path(manifest["decisions"])
    if file_sha256(dataset_release_path) != manifest["dataset_release_sha256"]:
        raise SystemExit("Dataset release manifest changed; rebuild the challenge.")
    if file_sha256(decisions_path) != manifest["decisions_sha256"]:
        raise SystemExit("Challenge decisions changed; rebuild the challenge.")
    if manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise SystemExit("Challenge references the wrong dataset release.")
    for page in manifest["pages"]:
        validate_page_guardrails(page)
        if file_sha256(project_path(page["image"])) != page["image_sha256"]:
            raise SystemExit(f"Challenge image changed: {page['image']}")
        if file_sha256(project_path(page["label"])) != page["label_sha256"]:
            raise SystemExit(f"Challenge label changed: {page['label']}")


def prediction_record(box) -> dict:
    return {
        "confidence": round(float(box.conf[0]), 6),
        "xywhn": [round(float(value), 6) for value in box.xywhn[0].tolist()],
        "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
    }


def run_predictions(weights: Path, manifest: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    import main as application

    weights_sha256 = file_sha256(weights)
    model = application.load_yolo(weights)
    application.require_legal_notice_model(model)
    pages = []
    for index, page in enumerate(manifest["pages"], start=1):
        result = model.predict(
            str(project_path(page["image"])),
            conf=manifest["acceptance_confidence"],
            imgsz=manifest["image_size"],
            device="cpu",
            verbose=False,
        )[0]
        pages.append({
            "issue": page["issue"],
            "page": page["page"],
            "image": page["image"],
            "predictions": [prediction_record(box) for box in result.boxes],
        })
        print(f"baseline: {index}/{len(manifest['pages'])} pages")
    if file_sha256(weights) != weights_sha256:
        raise SystemExit("Checkpoint changed during challenge inference.")
    return {
        "suite": manifest["suite"],
        "manifest_sha256": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": relative_path(weights),
        "weights_sha256": weights_sha256,
        "image_size": manifest["image_size"],
        "minimum_confidence": manifest["acceptance_confidence"],
        "pages": pages,
    }


def greedy_valid_matches(predictions: list[dict], references: list[dict], minimum_iou: float):
    candidates = sorted(
        (
            (iou(prediction["xyxy"], reference["xyxy"]), prediction_index, reference_index)
            for prediction_index, prediction in enumerate(predictions)
            for reference_index, reference in enumerate(references)
        ),
        reverse=True,
    )
    matched_predictions = set()
    matched_references = set()
    matches = []
    for overlap, prediction_index, reference_index in candidates:
        if overlap < minimum_iou:
            break
        if prediction_index not in matched_predictions and reference_index not in matched_references:
            matches.append((prediction_index, reference_index, overlap))
            matched_predictions.add(prediction_index)
            matched_references.add(reference_index)
    return matches, matched_predictions, matched_references


def score_predictions(manifest: dict, predictions: dict) -> dict:
    prediction_pages = {normalized_path(page["image"]): page for page in predictions["pages"]}
    expected_paths = {normalized_path(page["image"]) for page in manifest["pages"]}
    if set(prediction_pages) != expected_paths:
        raise SystemExit("Prediction pages do not match the challenge manifest.")

    totals = Counter()
    categories = defaultdict(Counter)
    page_results = []
    for page in manifest["pages"]:
        page_predictions = prediction_pages[normalized_path(page["image"])]["predictions"]
        accepted = [
            prediction
            for prediction in page_predictions
            if prediction["confidence"] >= manifest["acceptance_confidence"]
        ]
        matches, matched_prediction_ids, matched_reference_ids = greedy_valid_matches(
            accepted, page["valid"], manifest["valid_match_iou"]
        )
        totals["valid_expected"] += len(page["valid"])
        totals["valid_preserved"] += len(matched_reference_ids)
        totals["valid_missed"] += len(page["valid"]) - len(matched_reference_ids)
        categories["valid_legal_notice"]["expected"] += len(page["valid"])
        categories["valid_legal_notice"]["preserved"] += len(matched_reference_ids)
        categories["valid_legal_notice"]["missed"] += len(page["valid"]) - len(matched_reference_ids)

        excluded_hits = defaultdict(list)
        unreviewed = []
        for prediction_index, prediction in enumerate(accepted):
            if prediction_index in matched_prediction_ids:
                continue
            candidates = []
            prediction_area = area(prediction["xyxy"])
            for reference_index, reference in enumerate(page["excluded"]):
                overlap = intersection_area(prediction["xyxy"], reference["xyxy"])
                overlap = overlap / prediction_area if prediction_area else 0.0
                candidates.append((overlap, reference_index))
            best_overlap, best_reference = max(candidates, default=(0.0, -1))
            if best_overlap >= manifest["excluded_prediction_overlap"]:
                excluded_hits[best_reference].append({"prediction": prediction, "overlap": round(best_overlap, 4)})
            else:
                unreviewed.append(prediction)

        excluded_results = []
        for reference_index, reference in enumerate(page["excluded"]):
            hits = excluded_hits[reference_index]
            accepted_region = bool(hits)
            totals["excluded_expected"] += 1
            totals["excluded_accepted_regions"] += accepted_region
            totals["excluded_rejected_regions"] += not accepted_region
            totals["excluded_accepted_predictions"] += len(hits)
            category = categories[reference["category"]]
            category["expected"] += 1
            category["accepted_regions"] += accepted_region
            category["rejected_regions"] += not accepted_region
            category["accepted_predictions"] += len(hits)
            excluded_results.append({
                "id": reference["id"],
                "category": reference["category"],
                "accepted": accepted_region,
                "matches": hits,
            })

        totals["unreviewed_accepted"] += len(unreviewed)
        page_results.append({
            "issue": page["issue"],
            "page": page["page"],
            "image": page["image"],
            "accepted_count": len(accepted),
            "valid_expected": len(page["valid"]),
            "valid_preserved": len(matched_reference_ids),
            "valid_missed_ids": [
                reference["id"]
                for index, reference in enumerate(page["valid"])
                if index not in matched_reference_ids
            ],
            "valid_matches": [
                {
                    "reference_id": page["valid"][reference_index]["id"],
                    "iou": round(overlap, 4),
                    "prediction": accepted[prediction_index],
                }
                for prediction_index, reference_index, overlap in matches
            ],
            "excluded": excluded_results,
            "unreviewed_accepted": unreviewed,
        })

    gate_results = {
        key: totals[key] <= maximum
        for key, maximum in manifest["strict_gates"].items()
    }
    return {
        "passed": all(gate_results.values()),
        "gate_results": gate_results,
        "totals": dict(totals),
        "categories": {key: dict(value) for key, value in sorted(categories.items())},
        "pages": page_results,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Full-Category Development Challenge",
        "",
        f"- Passed strict gates: `{str(report['passed']).lower()}`",
        f"- Checkpoint: `{report['candidate']['weights']}`",
        f"- Confidence: `{report['acceptance_confidence']:.2f}`",
        f"- Image size: `{report['image_size']}`",
        "- Scope: development validation pages only; not an unbiased final test",
        "",
        "## Totals",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for key, value in report["totals"].items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.extend(["", "## Category Scorecard", "", "| Category | Expected | Rejected/Preserved | Accepted/Missed |", "|---|---:|---:|---:|"])
    for category, values in report["categories"].items():
        if category == "valid_legal_notice":
            lines.append(
                f"| {category} | {values.get('expected', 0)} | {values.get('preserved', 0)} | {values.get('missed', 0)} |"
            )
        else:
            lines.append(
                f"| {category} | {values.get('expected', 0)} | {values.get('rejected_regions', 0)} | {values.get('accepted_regions', 0)} |"
            )
    lines.extend(["", "## Unavailable Coverage", ""])
    for item in report["unavailable_categories"]:
        lines.append(f"- `{item['category']}`: {item['reason']}")
    lines.extend(["", "## Interpretation", ""])
    lines.append("This baseline is descriptive. Do not tune from this challenge and report it as independent test performance.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--decisions", default="regression/full_category_challenge_decisions.json")
    build.add_argument("--output", default="regression/full_category_challenge_manifest.json")
    run = commands.add_parser("run")
    run.add_argument("--manifest", default="regression/full_category_challenge_manifest.json")
    run.add_argument("--weights", default="runs/legal_notice_v3_controlled_50/weights/best.pt")
    run.add_argument("--output", default="output/regression_audit/full_category_challenge_v1/v3_baseline.json")
    args = parser.parse_args()

    if args.command == "build":
        build_manifest(project_path(args.decisions), project_path(args.output))
        return

    manifest_path = project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest_path, manifest)
    weights_path = project_path(args.weights)
    predictions = run_predictions(weights_path, manifest)
    predictions["manifest_sha256"] = file_sha256(manifest_path)
    scored = score_predictions(manifest, predictions)
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path.with_suffix(".predictions.json")
    prediction_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    report = {
        "suite": manifest["suite"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": relative_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "candidate": {
            "weights": predictions["weights"],
            "weights_sha256": predictions["weights_sha256"],
            "predictions": relative_path(prediction_path),
            "predictions_sha256": file_sha256(prediction_path),
        },
        "acceptance_confidence": manifest["acceptance_confidence"],
        "image_size": manifest["image_size"],
        "unavailable_categories": manifest["unavailable_categories"],
        **scored,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    write_markdown(report, markdown_path)
    print(json.dumps({"passed": report["passed"], "gate_results": report["gate_results"], "totals": report["totals"], "categories": report["categories"]}, indent=2))
    print(f"Challenge report: {relative_path(output_path)}")
    print(f"Challenge scorecard: {relative_path(markdown_path)}")


if __name__ == "__main__":
    main()
