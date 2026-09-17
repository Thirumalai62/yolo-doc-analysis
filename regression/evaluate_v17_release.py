"""Evaluate one immutable checkpoint against the complete reviewed release contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
from ultralytics import YOLO

from active_head_pilot import evaluate_july
import fixed080_acceptance as fixed
import full_category_challenge as challenge
from legal_notice_v10_acceptance import area, intersection, read_labels
from monitored_clslogit_pilot import run_challenge, run_fixed
from monitored_feature_finetune import evaluate_target


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v17_release_config.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_artifact(artifact: dict, description: str) -> Path:
    path = project_path(artifact["path"])
    require(path.is_file() and fixed.file_sha256(path) == artifact["sha256"], f"Missing or changed {description}: {path}")
    return path


def restore_manifest_images(manifest: dict) -> None:
    from main import render_pdf
    for page in manifest["pages"]:
        image = project_path(page["image"])
        if image.is_file():
            require(fixed.file_sha256(image) == page["image_sha256"], f"Changed evaluation image: {image}")
            continue
        issue = page["issue"]
        pdf = ROOT / "input" / f"{issue}.pdf"
        require(pdf.is_file(), f"Missing source PDF needed to restore {image}")
        rendered = render_pdf(pdf, image.parent, 200, {int(page["page"])})
        require(rendered == [image] and fixed.file_sha256(image) == page["image_sha256"], f"Restored image failed provenance: {image}")


def prediction_records(model: YOLO, image: Path, config: dict) -> list[dict]:
    deployment = config["deployment"]
    result = model.predict(
        str(image), conf=deployment["diagnostic_confidence"], imgsz=deployment["image_size"],
        iou=deployment["nms_iou"], max_det=deployment["max_detections"], device=deployment["device"],
        half=False, verbose=False, save=False,
    )[0]
    return [fixed.prediction_record(box) for box in result.boxes]


def greedy_matches(references: list[dict], predictions: list[dict], minimum_iou: float) -> dict[int, tuple[int, float]]:
    candidates = []
    for reference_index, reference in enumerate(references):
        for prediction_index, prediction in enumerate(predictions):
            overlap = challenge.iou(reference["xyxy"], prediction["xyxy"])
            if overlap >= minimum_iou:
                candidates.append((overlap, prediction["confidence"], reference_index, prediction_index))
    matches, used = {}, set()
    for overlap, _, reference_index, prediction_index in sorted(candidates, reverse=True):
        if reference_index not in matches and prediction_index not in used:
            matches[reference_index] = (prediction_index, overlap)
            used.add(prediction_index)
    return matches


def evaluate_historical(weights: Path, config: dict, manifest: dict, baseline: dict, output: Path) -> dict:
    model = YOLO(str(weights))
    baseline_scores = {
        reference["id"]: reference.get("confidence", 0.0)
        for page in baseline["pages"]
        for reference in page["references"]
        if reference["classification"] == "valid"
    }
    valid_records, excluded_records, pages = [], [], []
    for page in manifest["pages"]:
        image = project_path(page["image"])
        require(fixed.file_sha256(image) == page["image_sha256"], f"Historical image changed: {image}")
        predictions = prediction_records(model, image, config)
        valid = [reference for reference in page["references"] if reference["classification"] == "valid"]
        matches = greedy_matches(valid, predictions, manifest["runtime"]["valid_match_iou"])
        page_valid = []
        for index, reference in enumerate(valid):
            prediction_index, overlap = matches.get(index, (None, 0.0))
            confidence = predictions[prediction_index]["confidence"] if prediction_index is not None else 0.0
            baseline_confidence = baseline_scores[reference["id"]]
            record = {
                "id": reference["id"], "confidence": confidence, "iou": overlap,
                "baseline_confidence": baseline_confidence,
                "score_drop": baseline_confidence - confidence,
                "accepted": confidence >= config["deployment"]["confidence"],
                "drop_passed": baseline_confidence - confidence <= config["requirements"]["historical_maximum_valid_score_drop"],
            }
            page_valid.append(record)
            valid_records.append(record)
        page_excluded = []
        for reference in (record for record in page["references"] if record["classification"] == "excluded"):
            matching = [
                prediction for prediction in predictions
                if intersection(prediction["xyxy"], reference["xyxy"]) / max(1.0, area(prediction["xyxy"])) >= manifest["runtime"]["negative_prediction_overlap"]
            ]
            maximum = max((prediction["confidence"] for prediction in matching), default=0.0)
            record = {"id": reference["id"], "maximum_confidence": maximum, "margin_passed": maximum <= config["deployment"]["negative_ceiling"]}
            page_excluded.append(record)
            excluded_records.append(record)
        pages.append({"issue": page["issue"], "page": page["page"], "valid": page_valid, "excluded": page_excluded})
    totals = {
        "valid_expected": len(valid_records),
        "valid_accepted": sum(record["accepted"] for record in valid_records),
        "valid_drop_passed": sum(record["drop_passed"] for record in valid_records),
        "excluded_expected": len(excluded_records),
        "excluded_margin_passed": sum(record["margin_passed"] for record in excluded_records),
    }
    requirements = config["requirements"]
    passed = totals == {
        "valid_expected": requirements["historical_valid"],
        "valid_accepted": requirements["historical_valid"],
        "valid_drop_passed": requirements["historical_valid"],
        "excluded_expected": requirements["historical_excluded"],
        "excluded_margin_passed": requirements["historical_excluded"],
    }
    report = {"suite": manifest["suite"], "passed": passed, "weights_sha256": fixed.file_sha256(weights), "manifest_sha256": fixed.file_sha256(project_path(config["manifests"]["historical"]["path"])), "totals": totals, "pages": pages}
    write(output, report)
    del model
    gc.collect()
    return report


def evaluate_september15(weights: Path, config: dict, manifest: dict, output: Path) -> dict:
    model = YOLO(str(weights))
    records, pages, unmatched = [], [], []
    for page in manifest["pages"]:
        image = project_path(page["image"])
        label = project_path(page["labels"])
        require(fixed.file_sha256(image) == page["image_sha256"] and fixed.file_sha256(label) == page["labels_sha256"], "September 15 replay changed")
        predictions = prediction_records(model, image, config)
        with Image.open(image) as opened:
            width, height = opened.size
        references = [
            {"id": f"gulftoday_2026-09-15:p{page['page']:04d}:label_{index + 1}", "xyxy": box}
            for index, box in enumerate(read_labels(label, width, height))
        ]
        matches = greedy_matches(references, predictions, 0.5)
        used_accepted = set()
        page_records = []
        for index, reference in enumerate(references):
            prediction_index, overlap = matches.get(index, (None, 0.0))
            confidence = predictions[prediction_index]["confidence"] if prediction_index is not None else 0.0
            critical = page["page"] == 13 and index + 1 in (10, 11)
            required_confidence = config["deployment"]["confidence"]
            required_iou = 0.5
            accepted = confidence >= required_confidence and overlap >= required_iou
            if prediction_index is not None and confidence >= config["deployment"]["confidence"]:
                used_accepted.add(prediction_index)
            record = {"id": reference["id"], "critical": critical, "confidence": confidence, "iou": overlap, "required_confidence": required_confidence, "required_iou": required_iou, "accepted": accepted}
            page_records.append(record)
            records.append(record)
        page_unmatched = [prediction for index, prediction in enumerate(predictions) if prediction["confidence"] >= config["deployment"]["confidence"] and index not in used_accepted]
        unmatched.extend({"page": page["page"], **prediction} for prediction in page_unmatched)
        pages.append({"page": page["page"], "references": page_records, "unmatched_accepted": page_unmatched})
    totals = {
        "valid_expected": len(records), "valid_accepted": sum(record["accepted"] for record in records),
        "critical_expected": sum(record["critical"] for record in records),
        "critical_accepted": sum(record["critical"] and record["accepted"] for record in records),
        "unmatched_accepted": len(unmatched),
    }
    requirements = config["requirements"]
    passed = totals == {"valid_expected": requirements["september15_valid"], "valid_accepted": requirements["september15_valid"], "critical_expected": requirements["september15_critical"], "critical_accepted": requirements["september15_critical"], "unmatched_accepted": 0}
    report = {"suite": "gulftoday_2026-09-15_complete_replay", "passed": passed, "weights_sha256": fixed.file_sha256(weights), "totals": totals, "pages": pages}
    write(output, report)
    del model
    gc.collect()
    return report


def evaluate_semantic_reviews(weights: Path, config: dict, reviews: dict, manifests: dict, output: Path) -> dict:
    model = YOLO(str(weights))
    page_index = {}
    for suite, manifest in manifests.items():
        for page in manifest["pages"]:
            page_index[(suite, page["issue"], page["page"])] = page
    records = []
    for review in reviews["reviews"]:
        suite = "fixed080" if review["suite"] == "fixed080" else review["suite"]
        page = page_index[(suite, review["issue"], review["page"])]
        predictions = prediction_records(model, project_path(page["image"]), config)
        if review["classification"] == "valid":
            matching = [prediction for prediction in predictions if challenge.iou(review["xyxy"], prediction["xyxy"]) >= 0.5]
        else:
            matching = [prediction for prediction in predictions if intersection(prediction["xyxy"], review["xyxy"]) / max(1.0, area(prediction["xyxy"])) >= 0.5]
        maximum = max((prediction["confidence"] for prediction in matching), default=0.0)
        passed = maximum >= config["deployment"]["confidence"] if review["classification"] == "valid" else maximum <= config["deployment"]["negative_ceiling"]
        records.append({"suite": review["suite"], "issue": review["issue"], "page": review["page"], "classification": review["classification"], "maximum_confidence": maximum, "passed": passed})
    totals = {
        "valid_expected": sum(record["classification"] == "valid" for record in records),
        "valid_passed": sum(record["classification"] == "valid" and record["passed"] for record in records),
        "excluded_expected": sum(record["classification"] == "excluded" for record in records),
        "excluded_passed": sum(record["classification"] == "excluded" and record["passed"] for record in records),
    }
    requirements = config["requirements"]
    passed = totals == {"valid_expected": requirements["semantic_valid"], "valid_passed": requirements["semantic_valid"], "excluded_expected": requirements["semantic_excluded"], "excluded_passed": requirements["semantic_excluded"]}
    report = {"suite": "reviewed_additional_detections", "passed": passed, "weights_sha256": fixed.file_sha256(weights), "totals": totals, "records": records}
    write(output, report)
    del model
    gc.collect()
    return report


def evaluate_september11(weights: Path, config: dict, manifest: dict, output: Path) -> dict:
    model = YOLO(str(weights))
    image = project_path(manifest["image"])
    require(fixed.file_sha256(image) == manifest["image_sha256"], "September 11 auction image changed")
    predictions = prediction_records(model, image, config)
    records = []
    for region in manifest["regions"]:
        x1, y1, x2, y2 = region["xyxyn"]
        xyxy = [x1 * manifest["width"], y1 * manifest["height"], x2 * manifest["width"], y2 * manifest["height"]]
        matching = [prediction for prediction in predictions if intersection(prediction["xyxy"], xyxy) / max(1.0, area(prediction["xyxy"])) >= 0.5]
        maximum = max((prediction["confidence"] for prediction in matching), default=0.0)
        records.append({"id": region["id"], "maximum_confidence": maximum, "margin_passed": maximum <= config["deployment"]["negative_ceiling"]})
    totals = {"excluded_expected": len(records), "excluded_margin_passed": sum(record["margin_passed"] for record in records)}
    passed = totals == {"excluded_expected": config["requirements"]["september11_excluded"], "excluded_margin_passed": config["requirements"]["september11_excluded"]}
    report = {"suite": manifest["suite"], "passed": passed, "weights_sha256": fixed.file_sha256(weights), "totals": totals, "records": records}
    write(output, report)
    del model
    gc.collect()
    return report


def unresolved_additions(report: dict, semantic_reviews: dict, suite: str) -> list[dict]:
    unresolved = []
    reviews = [review for review in semantic_reviews["reviews"] if review["suite"] == suite]
    for page in report["pages"]:
        for prediction in page.get("unreviewed_accepted", []):
            resolved = any(
                review["issue"] == page["issue"] and review["page"] == page["page"]
                and challenge.iou(review["xyxy"], prediction["xyxy"]) >= 0.5
                for review in reviews
            )
            if not resolved:
                unresolved.append({"suite": suite, "issue": page["issue"], "page": page["page"], "prediction": prediction})
    return unresolved


def production_equivalence(weights: Path, config: dict, sept15_report: dict, output: Path) -> dict:
    run_name = f"v17_release_{fixed.file_sha256(weights)[:12]}"
    destination = ROOT / "output/legal_notices" / run_name
    require(not destination.exists(), f"Production equivalence output already exists: {destination}")
    command = [sys.executable, str(ROOT / "main.py"), "detect", "--input", str(ROOT / "input/gulftoday_2026-09-15.pdf"), "--weights", str(weights), "--dpi", "200", "--image-size", "1280", "--confidence", "0.80", "--name", run_name]
    subprocess.run(command, check=True, cwd=ROOT)
    detections = load(destination / "detections.json")
    by_page = {int(Path(record["page"]).stem.split("_")[-1]): record["detections"] for record in detections}
    expected = {page["page"]: sum(reference["accepted"] for reference in page["references"]) for page in sept15_report["pages"]}
    counts_match = all(len(by_page.get(page, [])) == count for page, count in expected.items())
    other_pages_zero = all(not records for page, records in by_page.items() if page not in expected)
    report = {"passed": counts_match and other_pages_zero, "command": command, "output": relative(destination / "detections.json"), "page_counts": {str(page): len(records) for page, records in by_page.items()}, "expected_counts": expected, "other_pages_zero": other_pages_zero}
    write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--production-equivalence", action="store_true")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load(config_path)
    manifests = {name: load(verify_artifact(artifact, f"{name} manifest")) for name, artifact in config["manifests"].items() if name != "semantic_reviews"}
    semantic_reviews = load(verify_artifact(config["manifests"]["semantic_reviews"], "semantic reviews"))
    baseline_path = project_path(config["baselines"]["historical_r7"])
    require(fixed.file_sha256(baseline_path) == config["baselines"]["historical_r7_sha256"], "Historical R7 baseline changed")
    historical_baseline = load(baseline_path)
    weights = project_path(args.weights)
    require(weights.is_file(), f"Missing checkpoint: {weights}")
    weights_sha = fixed.file_sha256(weights)
    output = project_path(args.output)
    require(not output.exists(), "Release output already exists; use a new immutable output path")
    output.mkdir(parents=True)
    for name in ("fixed", "challenge", "july", "historical"):
        restore_manifest_images(manifests[name])

    fixed_report = run_fixed(weights, verify_artifact(config["manifests"]["fixed"], "fixed manifest"), output / "fixed.json")
    challenge_report = run_challenge(weights, verify_artifact(config["manifests"]["challenge"], "challenge manifest"), output / "challenge.json")
    july_path = verify_artifact(config["manifests"]["july"], "July manifest")
    july_report = evaluate_july(weights, manifests["july"], fixed.file_sha256(july_path), output / "july.json")
    target_config = {
        "target_manifest": config["manifests"]["target"]["path"],
        "target_manifest_sha256": config["manifests"]["target"]["sha256"],
        "deployment": {"confidence": config["deployment"]["confidence"], "image_size": config["deployment"]["image_size"], "device": config["deployment"]["device"]},
        "stopping": {"target_valid_required": config["requirements"]["target_valid"], "target_negative_regions_required": config["requirements"]["target_negative_regions"], "target_negative_max_confidence": config["deployment"]["negative_ceiling"]},
    }
    target_report = evaluate_target(weights, target_config, output / "target.json", weights_sha)
    september15_report = evaluate_september15(weights, config, manifests["september15"], output / "september15.json")
    historical_report = evaluate_historical(weights, config, manifests["historical"], historical_baseline, output / "historical.json")
    semantic_report = evaluate_semantic_reviews(weights, config, semantic_reviews, {"fixed080": manifests["fixed"], "july": manifests["july"]}, output / "semantic.json")
    september11_report = evaluate_september11(weights, config, manifests["september11"], output / "september11.json")

    unresolved = unresolved_additions(fixed_report, semantic_reviews, "fixed080")
    unresolved += unresolved_additions(challenge_report, semantic_reviews, "challenge")
    unresolved += unresolved_additions(july_report, semantic_reviews, "july")
    unresolved += [{"suite": "target", **record} for page in target_report["pages"] for record in page["unmatched_accepted"]]
    unresolved += [{"suite": "september15", **record} for page in september15_report["pages"] for record in page["unmatched_accepted"]]
    requirements = config["requirements"]
    gates = {
        "fixed": fixed_report["totals"]["valid_preserved"] == requirements["fixed_valid"] and fixed_report["totals"]["excluded_accepted"] == 0 and fixed_report["totals"]["critical_boundary_failures"] == 0,
        "challenge": challenge_report["totals"]["valid_preserved"] == requirements["challenge_valid"] and challenge_report["totals"]["excluded_accepted_regions"] == 0,
        "july": july_report["totals"]["valid_preserved"] == requirements["july_valid"] and july_report["totals"]["excluded_accepted_regions"] == 0 and sum(record["complete_accepted"] for record in july_report["recovery_targets"]) == requirements["july_recoveries"],
        "target": target_report["passed"],
        "september15": september15_report["passed"],
        "historical": historical_report["passed"],
        "semantic_reviews": semantic_report["passed"],
        "september11": september11_report["passed"],
        "unresolved_additions": len(unresolved) == requirements["unresolved_additional_detections"],
    }
    core_passed = all(gates.values())
    production_report = None
    if core_passed and args.production_equivalence:
        production_report = production_equivalence(weights, config, september15_report, output / "production_equivalence.json")
        gates["production_equivalence"] = production_report["passed"]
    else:
        gates["production_equivalence"] = None
    coverage = {
        "fixed": {"valid": fixed_report["totals"]["valid_expected"], "excluded": fixed_report["totals"]["excluded_expected"], "critical_boundaries": fixed_report["totals"]["critical_boundaries"]},
        "challenge": {"valid": challenge_report["totals"]["valid_expected"], "excluded": challenge_report["totals"]["excluded_expected"]},
        "july": {"valid": july_report["totals"]["valid_expected"], "excluded": july_report["totals"]["excluded_expected"], "recoveries": len(july_report["recovery_targets"])},
        "target": {"valid": target_report["totals"]["valid_expected"], "negative_regions": target_report["totals"]["negative_regions"]},
        "september15": {"valid": september15_report["totals"]["valid_expected"], "critical": september15_report["totals"]["critical_expected"]},
        "historical": {"valid": historical_report["totals"]["valid_expected"], "excluded": historical_report["totals"]["excluded_expected"]},
        "semantic_reviews": semantic_report["totals"],
        "september11": september11_report["totals"],
    }
    require(coverage["fixed"] == {"valid": requirements["fixed_valid"], "excluded": 17, "critical_boundaries": requirements["critical_boundaries"]}, "Fixed release coverage is incomplete")
    require(coverage["challenge"] == {"valid": requirements["challenge_valid"], "excluded": 13}, "Challenge release coverage is incomplete")
    require(coverage["july"] == {"valid": requirements["july_valid"], "excluded": 2, "recoveries": requirements["july_recoveries"]}, "July release coverage is incomplete")
    require(coverage["target"] == {"valid": requirements["target_valid"], "negative_regions": requirements["target_negative_regions"]}, "Target release coverage is incomplete")
    require(coverage["september15"] == {"valid": requirements["september15_valid"], "critical": requirements["september15_critical"]}, "September 15 release coverage is incomplete")
    require(coverage["historical"] == {"valid": requirements["historical_valid"], "excluded": requirements["historical_excluded"]}, "Historical release coverage is incomplete")
    summary = {
        "status": "passed_all_release_gates" if core_passed and production_report is not None and production_report["passed"] else "passed_core_gates_production_pending" if core_passed else "rejected_not_for_deployment",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path), "config_sha256": fixed.file_sha256(config_path),
        "weights": relative(weights), "weights_sha256": weights_sha,
        "coverage": coverage, "gates": gates, "unresolved_additions": unresolved,
        "production_equivalence": production_report,
    }
    write(output / "release_summary.json", summary)
    print(json.dumps({"status": summary["status"], "gates": gates, "coverage": coverage, "output": relative(output / "release_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
