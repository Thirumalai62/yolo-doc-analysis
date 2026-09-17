"""Verify that v6 supplies complete frozen-box candidates for required recoveries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixed080_acceptance import file_sha256
from main import render_pdf


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = box
    return [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]


def prepare_image(case: dict, generated: dict[Path, str]) -> Path:
    image = ROOT / case["image"]
    if not image.is_file():
        issue = case["issue"]
        pdf = ROOT / "input" / f"{issue}.pdf"
        if not pdf.is_file():
            raise RuntimeError(f"Missing source PDF for geometry check: {pdf}")
        destination = image.parent
        before = image.exists()
        render_pdf(pdf, destination, 200, {case["page"]})
        if not before:
            generated[image] = case["image_sha256"]
    if not image.is_file() or file_sha256(image) != case["image_sha256"]:
        raise RuntimeError(f"Geometry image is missing or changed: {image}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="regression/legal_notice_v15_acceptance_inventory.json")
    parser.add_argument("--weights", default="runs/legal_notice_v6_clslogit_cpu_2_r1/weights/best.pt")
    parser.add_argument("--output", default="output/regression_audit/legal_notice_v15_geometry_feasibility.json")
    parser.add_argument("--minimum-iou", type=float, default=0.5)
    args = parser.parse_args()

    inventory_path, weights_path, output_path = (ROOT / args.inventory, ROOT / args.weights, ROOT / args.output)
    inventory = load(inventory_path)
    model = YOLO(str(weights_path))
    generated: dict[Path, str] = {}
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for case in inventory["geometry_checks"]:
        grouped.setdefault((case["issue"], case["page"], case["image"]), []).append(case)

    records = []
    try:
        for (_issue, _page, _image), cases in grouped.items():
            image = prepare_image(cases[0], generated)
            result = model.predict(
                str(image), conf=0.0001, iou=0.99, max_det=3000, imgsz=1280,
                device="cpu", verbose=False, save=False,
            )[0]
            width, height = result.orig_shape[1], result.orig_shape[0]
            predicted_boxes = result.boxes.xyxy.detach().cpu().float()
            probabilities = result.boxes.conf.detach().cpu().float()
            for case in cases:
                reference = case.get("xyxy") or xywhn_to_xyxy(case["xywhn"], width, height)
                target = torch.tensor([reference], dtype=torch.float32)
                overlaps = box_iou(target, predicted_boxes)[0] if len(predicted_boxes) else torch.zeros(0)
                best = int(overlaps.argmax()) if len(overlaps) else None
                best_iou = float(overlaps[best]) if best is not None else 0.0
                eligible = torch.where(overlaps >= args.minimum_iou)[0]
                best_probability = float(probabilities[eligible].max()) if len(eligible) else 0.0
                records.append({
                    "id": case["id"], "issue": case["issue"], "page": case["page"],
                    "data_role": case["data_role"], "reference_xyxy": reference,
                    "best_candidate_iou": best_iou, "best_matching_probability": best_probability,
                    "matching_candidates": int(len(eligible)), "passed": bool(len(eligible)),
                })
            if image in generated:
                if file_sha256(image) != generated[image]:
                    raise RuntimeError(f"Generated geometry image changed before cleanup: {image}")
                image.unlink()
                generated.pop(image)
    finally:
        for image, expected_hash in list(generated.items()):
            if image.exists() and file_sha256(image) == expected_hash:
                image.unlink()

    failures = [record["id"] for record in records if not record["passed"]]
    report = {
        "status": "passed" if not failures else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": relative(inventory_path), "inventory_sha256": file_sha256(inventory_path),
        "weights": relative(weights_path), "weights_sha256": file_sha256(weights_path),
        "inference": {"confidence": 0.0001, "nms_iou": 0.99, "max_det": 3000, "image_size": 1280, "device": "cpu"},
        "minimum_candidate_iou": args.minimum_iou,
        "totals": {"required": len(records), "passed": len(records) - len(failures), "failed": len(failures)},
        "failures": failures,
        "cases": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(output_path), "status": report["status"], "totals": report["totals"]}, indent=2))
    if failures:
        raise RuntimeError(f"Frozen v6 geometry cannot support required cases: {', '.join(failures)}")


if __name__ == "__main__":
    main()
