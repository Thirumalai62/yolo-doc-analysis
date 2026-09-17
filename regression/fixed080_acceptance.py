"""Build and run the reviewed fixed-confidence preservation acceptance suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MATCH_IOU = 0.50
DECISION_MATCH_IOU = 0.995
ACCEPTANCE_CONFIDENCE = 0.80
DIAGNOSTIC_CONFIDENCE = 0.05
CLASSIFICATIONS = {"valid", "excluded", "uncertain"}
STRICT_GATES = {
    "valid_missed": 0,
    "excluded_accepted": 0,
    "critical_boundary_failures": 0,
    "unreviewed_accepted": 0,
}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def normalized_path(value: str) -> str:
    return Path(value.replace("\\", "/")).as_posix().lower()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_manifest_sha256(manifest: dict) -> str:
    digest = hashlib.sha256()
    for page in manifest["pages"]:
        digest.update(normalized_path(page["image"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(page["image_sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def pixel_iou(first: list[float], second: list[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def normalized_iou(first: list[float], second: list[float]) -> float:
    def corners(box: list[float]) -> list[float]:
        x, y, width, height = box
        return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]

    return pixel_iou(corners(first), corners(second))


def greedy_matches(
    predictions: list[dict], references: list[dict], minimum_iou: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = sorted(
        (
            (normalized_iou(prediction["xywhn"], reference["xywhn"]), prediction_index, reference_index)
            for prediction_index, prediction in enumerate(predictions)
            for reference_index, reference in enumerate(references)
        ),
        reverse=True,
    )
    matches = []
    matched_predictions = set()
    matched_references = set()
    for iou, prediction_index, reference_index in candidates:
        if iou < minimum_iou:
            break
        if prediction_index not in matched_predictions and reference_index not in matched_references:
            matches.append((prediction_index, reference_index, iou))
            matched_predictions.add(prediction_index)
            matched_references.add(reference_index)
    return (
        matches,
        [index for index in range(len(predictions)) if index not in matched_predictions],
        [index for index in range(len(references)) if index not in matched_references],
    )


def find_reference(page: dict, reference_xyxy: list[float], used_ids: set[str]) -> dict:
    candidates = sorted(
        (
            (pixel_iou(reference_xyxy, reference["xyxy"]), reference)
            for reference in page["references"]
            if reference["id"] not in used_ids
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not candidates or candidates[0][0] < DECISION_MATCH_IOU:
        best_iou = candidates[0][0] if candidates else 0.0
        raise SystemExit(
            f"Reviewed box does not uniquely match {page['issue']} page {page['page']}: "
            f"best IoU {best_iou:.4f}"
        )
    return candidates[0][1]


def build_manifest(decisions_path: Path, output_path: Path) -> None:
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    baseline_path = project_path(decisions["baseline_predictions"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("minimum_confidence") != DIAGNOSTIC_CONFIDENCE:
        raise SystemExit(f"Baseline predictions must start at confidence {DIAGNOSTIC_CONFIDENCE:.2f}.")
    weights_path = project_path(baseline["weights"])
    if not weights_path.is_file():
        raise SystemExit(f"Baseline checkpoint is missing: {weights_path}")
    weights_sha256 = file_sha256(weights_path)
    if baseline.get("weights_sha256") != weights_sha256:
        raise SystemExit(
            "Baseline predictions are not attested to the current checkpoint. "
            "Regenerate them with this acceptance tool."
        )

    pages = []
    pages_by_key = {}
    for baseline_page in baseline["pages"]:
        image_path = project_path(baseline_page["image"])
        if not image_path.is_file():
            raise SystemExit(f"Regression image is missing: {image_path}")
        accepted = [
            prediction
            for prediction in baseline_page["predictions"]
            if prediction["confidence"] >= ACCEPTANCE_CONFIDENCE
        ]
        references = [
            {
                "id": f"{baseline_page['issue']}_p{baseline_page['page']:04d}_v3_{number:03d}",
                "classification": decisions["default_classification"],
                "reason": "reviewed valid core notice",
                "baseline_confidence": prediction["confidence"],
                "xywhn": prediction["xywhn"],
                "xyxy": prediction["xyxy"],
                "critical_boundary": None,
            }
            for number, prediction in enumerate(accepted, start=1)
        ]
        page = {
            "issue": baseline_page["issue"],
            "newspaper": baseline_page["newspaper"],
            "date": baseline_page["issue"].split("_", 1)[1],
            "page": baseline_page["page"],
            "image": Path(baseline_page["image"]).as_posix(),
            "image_sha256": file_sha256(image_path),
            "references": references,
        }
        key = (page["issue"], page["page"])
        if key in pages_by_key:
            raise SystemExit(f"Duplicate baseline page: {key}")
        pages_by_key[key] = page
        pages.append(page)

    classified_ids = set()
    for decision in decisions["classification_overrides"]:
        key = (decision["issue"], decision["page"])
        if key not in pages_by_key:
            raise SystemExit(f"Reviewed classification page is missing from baseline: {key}")
        reference = find_reference(pages_by_key[key], decision["reference_xyxy"], classified_ids)
        reference["classification"] = decision["classification"]
        reference["reason"] = decision["reason"]
        classified_ids.add(reference["id"])

    boundary_ids = set()
    for decision in decisions["critical_boundaries"]:
        key = (decision["issue"], decision["page"])
        if key not in pages_by_key:
            raise SystemExit(f"Reviewed boundary page is missing from baseline: {key}")
        reference = find_reference(pages_by_key[key], decision["reference_xyxy"], boundary_ids)
        if reference["classification"] != "valid":
            raise SystemExit(f"Critical boundary reference must be valid: {reference['id']}")
        reference["critical_boundary"] = {
            "description": decision["description"],
            **decisions["boundary_requirements"],
        }
        boundary_ids.add(reference["id"])

    review_counts = {"valid": 0, "excluded": 0, "uncertain": 0, "critical_boundaries": 0}
    for page in pages:
        for reference in page["references"]:
            review_counts[reference["classification"]] += 1
            review_counts["critical_boundaries"] += reference["critical_boundary"] is not None
    if review_counts != decisions["expected_review_counts"]:
        raise SystemExit(
            "Reviewed baseline count mismatch:\n"
            f"Expected: {json.dumps(decisions['expected_review_counts'], sort_keys=True)}\n"
            f"Actual:   {json.dumps(review_counts, sort_keys=True)}"
        )

    suite = "fixed080_sep09_10_preservation"
    expected_image_manifest_sha256 = image_manifest_sha256({"pages": pages})
    if (
        baseline.get("suite") != suite
        or baseline.get("image_manifest_sha256") != expected_image_manifest_sha256
    ):
        raise SystemExit("Baseline predictions are not attested to the reviewed image manifest.")
    manifest = {
        "suite": suite,
        "acceptance_confidence": ACCEPTANCE_CONFIDENCE,
        "diagnostic_confidence": DIAGNOSTIC_CONFIDENCE,
        "match_iou": MATCH_IOU,
        "image_size": baseline["image_size"],
        "inference": "one page per predict call",
        "baseline_weights": baseline["weights"],
        "baseline_weights_sha256": weights_sha256,
        "baseline_predictions": relative_path(baseline_path),
        "baseline_predictions_sha256": file_sha256(baseline_path),
        "review_decisions": relative_path(decisions_path),
        "review_decisions_sha256": file_sha256(decisions_path),
        "review_counts": review_counts,
        "strict_gates": STRICT_GATES,
        "notes": [
            "This is a reviewed v3 preservation suite, not exhaustive page ground truth.",
            "Uncertain references are reported but excluded from pass/fail gates.",
            "Any unmatched accepted candidate requires manual semantic review.",
        ],
        "pages": pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(review_counts, indent=2))
    print(f"Regression manifest created: {relative_path(output_path)}")


def prediction_record(box) -> dict:
    return {
        "confidence": round(float(box.conf[0]), 6),
        "xywhn": [round(float(value), 6) for value in box.xywhn[0].tolist()],
        "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
    }


def run_candidate(weights: Path, manifest: dict, before_page=None) -> dict:
    sys.path.insert(0, str(ROOT))
    import main as application

    weights_sha256 = file_sha256(weights)
    model = application.load_yolo(weights)
    application.require_legal_notice_model(model)
    pages = []
    for index, page in enumerate(manifest["pages"], start=1):
        if before_page is not None:
            before_page(page)
        image_path = project_path(page["image"])
        if not image_path.is_file() or file_sha256(image_path) != page["image_sha256"]:
            raise SystemExit(f"Regression image is missing or changed: {image_path}")
        result = model.predict(
            str(image_path),
            conf=DIAGNOSTIC_CONFIDENCE,
            imgsz=manifest["image_size"],
            device="cpu",
            verbose=False,
        )[0]
        pages.append({
            "issue": page["issue"],
            "newspaper": page["newspaper"],
            "date": page["date"],
            "page": page["page"],
            "image": page["image"],
            "predictions": [prediction_record(box) for box in result.boxes],
        })
        if index % 20 == 0 or index == len(manifest["pages"]):
            print(f"candidate: {index}/{len(manifest['pages'])} pages")
    if file_sha256(weights) != weights_sha256:
        raise SystemExit(f"Candidate checkpoint changed during inference: {weights}")
    return {
        "weights": relative_path(weights),
        "weights_sha256": weights_sha256,
        "suite": manifest["suite"],
        "image_manifest_sha256": image_manifest_sha256(manifest),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": manifest["image_size"],
        "minimum_confidence": DIAGNOSTIC_CONFIDENCE,
        "pages": pages,
    }


def validate_manifest(manifest: dict) -> None:
    expected_constants = {
        "acceptance_confidence": ACCEPTANCE_CONFIDENCE,
        "diagnostic_confidence": DIAGNOSTIC_CONFIDENCE,
        "match_iou": MATCH_IOU,
        "image_size": 1280,
        "inference": "one page per predict call",
    }
    for key, expected in expected_constants.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"Manifest {key} must be {expected!r}; found {manifest.get(key)!r}.")
    if manifest.get("strict_gates") != STRICT_GATES:
        raise SystemExit(f"Manifest strict gates must be exactly: {STRICT_GATES}")

    decisions_path = project_path(manifest.get("review_decisions", ""))
    if not decisions_path.is_file() or file_sha256(decisions_path) != manifest.get("review_decisions_sha256"):
        raise SystemExit("Manifest review decisions are missing or have changed. Rebuild the manifest.")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))

    baseline_path = project_path(manifest.get("baseline_predictions", ""))
    if not baseline_path.is_file() or file_sha256(baseline_path) != manifest.get("baseline_predictions_sha256"):
        raise SystemExit("Manifest baseline predictions are missing or have changed. Rebuild the manifest.")
    baseline_weights = project_path(manifest.get("baseline_weights", ""))
    if not baseline_weights.is_file() or file_sha256(baseline_weights) != manifest.get("baseline_weights_sha256"):
        raise SystemExit("Manifest baseline checkpoint is missing or has changed. Rebuild the manifest.")

    review_counts = {"valid": 0, "excluded": 0, "uncertain": 0, "critical_boundaries": 0}
    page_keys = set()
    image_paths = set()
    reference_ids = set()
    for page in manifest.get("pages", []):
        page_key = (page.get("issue"), page.get("page"))
        image_path = normalized_path(page.get("image", ""))
        if page_key in page_keys or image_path in image_paths:
            raise SystemExit(f"Manifest contains a duplicate page or image: {page_key}")
        page_keys.add(page_key)
        image_paths.add(image_path)
        for reference in page.get("references", []):
            reference_id = reference.get("id")
            classification = reference.get("classification")
            coordinates = reference.get("xywhn")
            if reference_id in reference_ids:
                raise SystemExit(f"Manifest contains duplicate reference ID: {reference_id}")
            if classification not in CLASSIFICATIONS:
                raise SystemExit(f"Manifest reference has invalid classification: {reference_id}")
            if not isinstance(coordinates, list) or len(coordinates) != 4 or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in coordinates
            ):
                raise SystemExit(f"Manifest reference has invalid coordinates: {reference_id}")
            if reference.get("critical_boundary") and classification != "valid":
                raise SystemExit(f"Critical boundary reference is not valid: {reference_id}")
            reference_ids.add(reference_id)
            review_counts[classification] += 1
            review_counts["critical_boundaries"] += reference.get("critical_boundary") is not None
    if len(page_keys) != 172:
        raise SystemExit(f"Manifest must contain 172 reviewed pages; found {len(page_keys)}.")
    if review_counts != manifest.get("review_counts") or review_counts != decisions.get("expected_review_counts"):
        raise SystemExit(f"Manifest review counts are inconsistent: {review_counts}")


def validate_candidate_predictions(
    candidate: dict, manifest: dict, allow_unverified: bool
) -> tuple[dict[str, dict], bool]:
    if candidate.get("image_size") != manifest["image_size"]:
        raise SystemExit(
            f"Candidate image size is {candidate.get('image_size')}; expected {manifest['image_size']}."
        )
    if candidate.get("minimum_confidence", 1.0) > DIAGNOSTIC_CONFIDENCE:
        raise SystemExit(
            f"Candidate predictions must start at confidence {DIAGNOSTIC_CONFIDENCE:.2f} or lower."
        )
    weights_path = project_path(candidate.get("weights", ""))
    stored_weights_sha256 = candidate.get("weights_sha256")
    provenance_verified = (
        weights_path.is_file()
        and isinstance(stored_weights_sha256, str)
        and file_sha256(weights_path) == stored_weights_sha256
        and candidate.get("suite") == manifest["suite"]
        and candidate.get("image_manifest_sha256") == image_manifest_sha256(manifest)
    )
    if not provenance_verified and not allow_unverified:
        raise SystemExit(
            "Cached predictions are not attested to the current checkpoint. "
            "Use --weights to regenerate them."
        )
    candidate_page_list = candidate.get("pages", [])
    candidate_pages = {normalized_path(page["image"]): page for page in candidate_page_list}
    if len(candidate_pages) != len(candidate_page_list):
        raise SystemExit("Candidate predictions contain duplicate page paths.")
    expected_paths = {normalized_path(page["image"]) for page in manifest["pages"]}
    if set(candidate_pages) != expected_paths:
        missing = sorted(expected_paths - set(candidate_pages))
        extra = sorted(set(candidate_pages) - expected_paths)
        raise SystemExit(
            f"Candidate page set does not match the manifest. Missing: {missing[:3]}; extra: {extra[:3]}"
        )
    for page in candidate_page_list:
        for prediction in page.get("predictions", []):
            confidence = prediction.get("confidence")
            coordinates = prediction.get("xywhn")
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
                raise SystemExit(f"Candidate prediction has invalid confidence: {page['image']}")
            if not isinstance(coordinates, list) or len(coordinates) != 4 or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in coordinates
            ):
                raise SystemExit(f"Candidate prediction has invalid coordinates: {page['image']}")
    return candidate_pages, provenance_verified


def closest_diagnostic(reference: dict, predictions: list[dict]) -> dict | None:
    if not predictions:
        return None
    iou, prediction = max(
        (
            (normalized_iou(reference["xywhn"], prediction["xywhn"]), prediction)
            for prediction in predictions
        ),
        key=lambda pair: pair[0],
    )
    return {"iou": round(iou, 4), **prediction}


def boundary_result(reference: dict, prediction: dict, iou: float) -> dict:
    requirements = reference["critical_boundary"]
    reference_width = reference["xywhn"][2]
    reference_height = reference["xywhn"][3]
    prediction_width = prediction["xywhn"][2]
    prediction_height = prediction["xywhn"][3]
    area_ratio = prediction_width * prediction_height / (reference_width * reference_height)
    reference_top = reference["xywhn"][1] - reference_height / 2
    reference_bottom = reference["xywhn"][1] + reference_height / 2
    prediction_top = prediction["xywhn"][1] - prediction_height / 2
    prediction_bottom = prediction["xywhn"][1] + prediction_height / 2
    top_inset = max(0.0, prediction_top - reference_top)
    bottom_inset = max(0.0, reference_bottom - prediction_bottom)
    checks = {
        "iou": iou >= requirements["minimum_iou"],
        "area_ratio": requirements["minimum_area_ratio"] <= area_ratio <= requirements["maximum_area_ratio"],
        "top_inset": top_inset <= requirements["maximum_top_inset_normalized"],
        "bottom_inset": bottom_inset <= requirements["maximum_bottom_inset_normalized"],
    }
    return {
        "reference_id": reference["id"],
        "description": requirements["description"],
        "passed": all(checks.values()),
        "checks": checks,
        "iou": round(iou, 4),
        "area_ratio": round(area_ratio, 4),
        "top_inset_normalized": round(top_inset, 6),
        "bottom_inset_normalized": round(bottom_inset, 6),
    }


def check_candidate(
    manifest_path: Path,
    candidate: dict,
    candidate_artifact: dict,
    output_path: Path,
    enforce: bool,
    allow_unverified: bool,
    verify_images: bool = True,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    candidate_pages, provenance_verified = validate_candidate_predictions(
        candidate, manifest, allow_unverified
    )

    totals = {
        "valid_expected": 0,
        "valid_preserved": 0,
        "valid_missed": 0,
        "excluded_expected": 0,
        "excluded_accepted": 0,
        "excluded_rejected": 0,
        "uncertain_expected": 0,
        "uncertain_accepted": 0,
        "uncertain_absent": 0,
        "unreviewed_accepted": 0,
        "critical_boundaries": 0,
        "critical_boundary_failures": 0,
    }
    newspapers = {}
    page_results = []
    for page in manifest["pages"]:
        image_path = project_path(page["image"])
        if verify_images and (not image_path.is_file() or file_sha256(image_path) != page["image_sha256"]):
            raise SystemExit(f"Regression image is missing or changed: {image_path}")
        predictions = candidate_pages[normalized_path(page["image"])]["predictions"]
        accepted = [prediction for prediction in predictions if prediction["confidence"] >= ACCEPTANCE_CONFIDENCE]
        matches, unmatched_predictions, unmatched_references = greedy_matches(
            accepted, page["references"], manifest["match_iou"]
        )
        matched_by_reference = {
            reference_index: (accepted[prediction_index], iou)
            for prediction_index, reference_index, iou in matches
        }
        unmatched_reference_set = set(unmatched_references)
        newspaper = newspapers.setdefault(page["newspaper"], {key: 0 for key in totals})
        reference_results = []
        boundary_results = []
        for reference_index, reference in enumerate(page["references"]):
            classification = reference["classification"]
            is_matched = reference_index not in unmatched_reference_set
            result = {
                "id": reference["id"],
                "classification": classification,
                "reason": reference["reason"],
                "accepted_match": None,
            }
            if is_matched:
                prediction, iou = matched_by_reference[reference_index]
                result["accepted_match"] = {"iou": round(iou, 4), **prediction}
            elif classification == "valid":
                result["closest_candidate_at_or_above_0.05"] = closest_diagnostic(reference, predictions)

            if classification == "valid":
                totals["valid_expected"] += 1
                newspaper["valid_expected"] += 1
                key = "valid_preserved" if is_matched else "valid_missed"
                totals[key] += 1
                newspaper[key] += 1
            elif classification == "excluded":
                totals["excluded_expected"] += 1
                newspaper["excluded_expected"] += 1
                key = "excluded_accepted" if is_matched else "excluded_rejected"
                totals[key] += 1
                newspaper[key] += 1
            else:
                totals["uncertain_expected"] += 1
                newspaper["uncertain_expected"] += 1
                key = "uncertain_accepted" if is_matched else "uncertain_absent"
                totals[key] += 1
                newspaper[key] += 1

            if reference["critical_boundary"]:
                totals["critical_boundaries"] += 1
                newspaper["critical_boundaries"] += 1
                if is_matched:
                    prediction, iou = matched_by_reference[reference_index]
                    boundary = boundary_result(reference, prediction, iou)
                else:
                    boundary = {
                        "reference_id": reference["id"],
                        "description": reference["critical_boundary"]["description"],
                        "passed": False,
                        "reason": "reference was not accepted at confidence 0.80",
                    }
                boundary_results.append(boundary)
                if not boundary["passed"]:
                    totals["critical_boundary_failures"] += 1
                    newspaper["critical_boundary_failures"] += 1
            reference_results.append(result)

        totals["unreviewed_accepted"] += len(unmatched_predictions)
        newspaper["unreviewed_accepted"] += len(unmatched_predictions)
        if reference_results or accepted:
            page_results.append({
                "issue": page["issue"],
                "page": page["page"],
                "image": page["image"],
                "accepted_count": len(accepted),
                "references": reference_results,
                "unreviewed_accepted": [accepted[index] for index in unmatched_predictions],
                "critical_boundaries": boundary_results,
            })

    gate_results = {
        key: totals[key] <= maximum
        for key, maximum in manifest["strict_gates"].items()
    }
    gate_results["prediction_provenance_verified"] = provenance_verified
    report = {
        "suite": manifest["suite"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": relative_path(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "candidate": {
            "weights": candidate.get("weights"),
            "weights_sha256": candidate.get("weights_sha256"),
            "prediction_provenance_verified": provenance_verified,
            "image_size": candidate["image_size"],
            "acceptance_confidence": ACCEPTANCE_CONFIDENCE,
            "diagnostic_confidence": candidate["minimum_confidence"],
            "predictions": candidate_artifact,
        },
        "passed": all(gate_results.values()),
        "gate_results": gate_results,
        "totals": totals,
        "newspapers": newspapers,
        "pages": page_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gate_results": gate_results, "totals": totals}, indent=2))
    print(f"Acceptance report: {relative_path(output_path)}")
    if enforce and not report["passed"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build the reviewed baseline manifest")
    build.add_argument("--decisions", default="regression/fixed080_review_decisions.json")
    build.add_argument("--output", default="regression/fixed080_manifest.json")

    check = commands.add_parser("check", help="Check a candidate against the reviewed manifest")
    check.add_argument("--manifest", default="regression/fixed080_manifest.json")
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", help="Candidate legal_notice checkpoint")
    source.add_argument("--predictions", help="Cached diagnostic predictions JSON")
    check.add_argument("--output", required=True, help="Acceptance report JSON")
    check.add_argument("--enforce", action="store_true", help="Exit nonzero when a strict gate fails")
    check.add_argument(
        "--allow-unverified-predictions",
        action="store_true",
        help="Allow legacy cached predictions without checkpoint attestation; never use for acceptance",
    )

    args = parser.parse_args()
    if args.command == "build":
        build_manifest(project_path(args.decisions), project_path(args.output))
        return

    manifest_path = project_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.weights:
        candidate = run_candidate(project_path(args.weights), manifest)
        prediction_path = project_path(args.output).with_suffix(".predictions.json")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        print(f"Candidate predictions: {relative_path(prediction_path)}")
    else:
        prediction_path = project_path(args.predictions)
        candidate = json.loads(prediction_path.read_text(encoding="utf-8"))
    candidate_artifact = {
        "path": relative_path(prediction_path),
        "sha256": file_sha256(prediction_path),
    }
    check_candidate(
        manifest_path,
        candidate,
        candidate_artifact,
        project_path(args.output),
        args.enforce,
        args.allow_unverified_predictions,
    )


if __name__ == "__main__":
    main()
