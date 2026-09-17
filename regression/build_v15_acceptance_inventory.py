"""Build the case-level acceptance inventory for reviewed v15 training."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "regression/legal_notice_v15_acceptance_inventory.json"
TARGET = ROOT / "output/correction_data/legal_notice_v10_targeted_correction/manifest.json"
FIXED = ROOT / "regression/fixed080_manifest.json"
CHALLENGE = ROOT / "regression/full_category_challenge_manifest.json"
JULY = ROOT / "output/regression_audit/legal_notice_v8_protected_head_cpu_r7_paa_analogue/july_manifest.json"
HISTORICAL = ROOT / "output/regression_audit/legal_notice_v9_preflight/historical_manifest.json"
SEPT15 = ROOT / "output/regression_audit/gulftoday_sept15_direct_training/manifest.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source(path: Path) -> dict:
    return {"path": relative(path), "sha256": file_sha256(path)}


def read_yolo_labels(path: Path) -> list[list[float]]:
    return [list(map(float, line.split()))[1:] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_cases(manifest: dict) -> tuple[list[dict], list[dict]]:
    positives, negatives = [], []
    for page in manifest["pages"]:
        common = {
            "issue": page["issue"],
            "page": page["page"],
            "image": page["image"].replace("\\", "/"),
            "image_sha256": page["image_sha256"],
            "width": page["width"],
            "height": page["height"],
            "data_role": "train",
        }
        label_path = ROOT / page["label"]
        for index, xywhn in enumerate(read_yolo_labels(label_path), 1):
            positives.append({
                **common,
                "id": f"{Path(page['image_name']).stem}_label_{index:03d}",
                "classification": "valid",
                "category": "valid_legal_notice",
                "xywhn": xywhn,
                "label": page["label"].replace("\\", "/"),
                "label_sha256": page["label_sha256"],
            })
        for index, xyxyn in enumerate(page["negative_regions_xyxyn"], 1):
            negatives.append({
                **common,
                "id": f"{Path(page['image_name']).stem}_negative_{index:02d}",
                "classification": "excluded",
                "category": page["category"],
                "xyxyn": xyxyn,
            })
    return positives, negatives


def manifest_cases(manifest: dict, valid_key: str, excluded_key: str, role: str) -> tuple[list[dict], list[dict]]:
    valid, excluded = [], []
    for page in manifest["pages"]:
        common = {
            "issue": page["issue"],
            "page": page["page"],
            "image": page["image"].replace("\\", "/"),
            "image_sha256": page["image_sha256"],
            "width": page.get("width"),
            "height": page.get("height"),
            "data_role": role,
        }
        for record in page.get(valid_key, []):
            if record.get("classification", "valid") == "valid":
                valid.append({**common, **record})
        for record in page.get(excluded_key, []):
            excluded.append({**common, **record})
    return valid, excluded


def fixed_cases(manifest: dict) -> tuple[list[dict], list[dict], list[dict]]:
    valid, excluded, uncertain = [], [], []
    for page in manifest["pages"]:
        common = {
            "issue": page["issue"],
            "page": page["page"],
            "image": page["image"],
            "image_sha256": page["image_sha256"],
            "data_role": "locked_evaluation",
        }
        for record in page["references"]:
            item = {**common, **record}
            {"valid": valid, "excluded": excluded, "uncertain": uncertain}[record["classification"]].append(item)
    return valid, excluded, uncertain


def find_case(cases: list[dict], case_id: str) -> dict:
    matches = [case for case in cases if case["id"] == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one inventory case for {case_id}, found {len(matches)}")
    return matches[0]


def september15_cases(manifest: dict) -> list[dict]:
    ids = {
        12: [
            "p12_left_row1", "p12_center_row1", "p12_right_creditors", "p12_left_row2",
            "p12_center_row2", "p12_right_amicable_settlement", "p12_left_court_judgment",
            "p12_center_srti_liquidation",
        ],
        13: [
            "p13_left_row1", "p13_center_row1", "p13_right_row1", "p13_left_row2", "p13_center_row2",
            "p13_right_name_change", "p13_left_row3", "p13_center_row3", "p13_left_tall_notice",
            "p13_samana_default", "p13_dubai_courts_property_sale",
        ],
    }
    cases = []
    for page in manifest["pages"]:
        label_path = ROOT / page["labels"]
        boxes = read_yolo_labels(label_path)
        if len(boxes) != len(ids[page["page"]]):
            raise RuntimeError("September 15 label count changed")
        for case_id, xywhn in zip(ids[page["page"]], boxes):
            cases.append({
                "id": case_id,
                "issue": "gulftoday_2026-09-15",
                "page": page["page"],
                "image": page["image"].replace("\\", "/"),
                "image_sha256": page["image_sha256"],
                "classification": "valid",
                "category": "valid_legal_notice",
                "xywhn": xywhn,
                "data_role": "training_replay",
                "critical_recovery": case_id in {"p13_samana_default", "p13_dubai_courts_property_sale"},
            })
    return cases


def main() -> None:
    target = load(TARGET)
    fixed = load(FIXED)
    challenge = load(CHALLENGE)
    july = load(JULY)
    historical = load(HISTORICAL)
    sept15 = load(SEPT15)

    target_valid, target_excluded = target_cases(target)
    fixed_valid, fixed_excluded, fixed_uncertain = fixed_cases(fixed)
    challenge_valid, challenge_excluded = manifest_cases(
        challenge, "valid", "excluded", "development_validation"
    )
    july_valid, july_excluded = manifest_cases(july, "valid", "excluded", "locked_evaluation")
    sept15_valid = september15_cases(sept15)
    historical_cases = [
        {**case, "data_role": "training_replay", "image": page["image"], "image_sha256": page["image_sha256"],
         "issue": page["issue"], "page": page["page"], "width": page["width"], "height": page["height"]}
        for page in historical["pages"]
        for case in page["references"]
        if case["id"] in {"khaleejtimes_2026-07-06_p0001_d001", "khaleejtimes_2026-07-06_p0004_d001"}
    ]
    required_ids = [
        "albayan_2026-09-09_p0034_v3_013",
        "gulftoday_2026-09-09_p0012_v3_009",
        "khaleejtimes_2026-09-09_p0014_v3_005",
    ]
    geometry_cases = [
        *target_valid,
        *(find_case(july_valid, case_id) for case_id in (
            "srtip_left_liquidation", "srtip_right_liquidation", "hamriyah_termination_shareholders"
        )),
        *(find_case(fixed_valid, case_id) for case_id in required_ids),
        *(find_case(sept15_valid, case_id) for case_id in (
            "p13_samana_default", "p13_dubai_courts_property_sale"
        )),
    ]
    inventory = {
        "name": "legal_notice_v15_acceptance_inventory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment": {"confidence": 0.8, "image_size": 1280, "device": "cpu", "render_dpi": 200},
        "policy": {
            "reviewed_exclusions_accepted": 0,
            "unreviewed_accepted": 0,
            "target_negative_maximum_confidence": 0.5,
            "critical_boundaries_required": 4,
            "pending_cases_are_not_spatial_gates": True,
        },
        "sources": [source(path) for path in (TARGET, FIXED, CHALLENGE, JULY, HISTORICAL, SEPT15)],
        "counts": {
            "target_valid": len(target_valid),
            "target_excluded": len(target_excluded),
            "fixed_valid": len(fixed_valid),
            "fixed_excluded": len(fixed_excluded),
            "fixed_uncertain": len(fixed_uncertain),
            "challenge_valid": len(challenge_valid),
            "challenge_excluded": len(challenge_excluded),
            "july_valid": len(july_valid),
            "july_excluded": len(july_excluded),
            "september15_training_replay_valid": len(sept15_valid),
            "geometry_checks": len(geometry_cases),
        },
        "suites": {
            "target": {"valid": target_valid, "excluded": target_excluded},
            "fixed": {"valid": fixed_valid, "excluded": fixed_excluded, "uncertain": fixed_uncertain},
            "challenge": {"valid": challenge_valid, "excluded": challenge_excluded},
            "july": {"valid": july_valid, "excluded": july_excluded},
            "historical_reported": historical_cases,
            "september15_training_replay": sept15_valid,
        },
        "geometry_checks": geometry_cases,
        "known_v10_regression_ids": required_ids,
        "pending": [{
            "id": "gulftoday_2026-09-11_real_estate_auction",
            "issue": "gulftoday_2026-09-11",
            "classification": "excluded",
            "category": "real_estate_auction",
            "reported_confidence": 0.87,
            "data_role": "summary_only_pending",
            "reason": "No reviewed page number or region exists; retain as a non-spatial pending case.",
        }],
    }
    expected = {
        "target_valid": 21, "target_excluded": 8, "fixed_valid": 271, "fixed_excluded": 17,
        "fixed_uncertain": 2, "challenge_valid": 31, "challenge_excluded": 13,
        "july_valid": 548, "july_excluded": 2, "september15_training_replay_valid": 19,
        "geometry_checks": 29,
    }
    if inventory["counts"] != expected:
        raise RuntimeError(f"Acceptance inventory counts changed: {inventory['counts']!r}")
    OUTPUT.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(OUTPUT), "sha256": file_sha256(OUTPUT), "counts": inventory["counts"]}, indent=2))


if __name__ == "__main__":
    main()
