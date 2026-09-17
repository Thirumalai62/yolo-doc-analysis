"""Build a hashed training overlay from the reviewed v10 CVAT imports."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output/cvat_import/legal_notice_v10_targeted_correction/import_report.json"
REVIEW = ROOT / "regression/legal_notice_v10_correction_review.json"
DESTINATION = ROOT / "output/correction_data/legal_notice_v10_targeted_correction"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def parse_labels(source: str, content: str) -> list[list[float]]:
    labels = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        require(len(values) == 5, f"{source}:{line_number} needs five values")
        class_id = int(values[0])
        box = [float(value) for value in values[1:]]
        x, y, width, height = box
        require(class_id == 0, f"{source}:{line_number} must use class 0")
        require(
            width > 0 and height > 0 and x - width / 2 >= 0 and x + width / 2 <= 1
            and y - height / 2 >= 0 and y + height / 2 <= 1,
            f"{source}:{line_number} has an invalid box",
        )
        labels.append(box)
    return labels


def corners(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = box
    return [
        (x - w / 2) * width,
        (y - h / 2) * height,
        (x + w / 2) * width,
        (y + h / 2) * height,
    ]


def overlaps(first: list[float], second: list[float]) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh this script's existing generated overlay")
    args = parser.parse_args()
    require(REPORT.is_file(), f"Missing import report: {REPORT}")
    require(REVIEW.is_file(), f"Missing correction review: {REVIEW}")
    require(not DESTINATION.exists() or args.refresh, f"Refusing to overwrite correction overlay: {DESTINATION}")

    report = load(REPORT)
    reviewed = load(REVIEW)
    pages = {(page["issue_id"], page["pdf_page"]): page for page in report["pages"]}
    decisions = {(page["issue"], page["page"]): page for page in reviewed["pages"]}
    require(pages.keys() == decisions.keys() and len(pages) == 7, "Review and CVAT pages do not match exactly")

    images = DESTINATION / "images"
    labels = DESTINATION / "labels"
    images.mkdir(parents=True, exist_ok=args.refresh)
    labels.mkdir(exist_ok=args.refresh)
    manifest_pages = []

    for key in sorted(pages):
        page = pages[key]
        decision = decisions[key]
        image_source = ROOT / page["image"]
        archive = ROOT / "input" / page["cvat_archive"]
        require(image_source.is_file() and archive.is_file(), f"Missing source for {key}")
        with Image.open(image_source) as source_image:
            width, height = source_image.size

        with ZipFile(archive) as cvat:
            label_entry = f"obj_train_data/{Path(page['image_name']).stem}.txt"
            content = cvat.read(label_entry).decode("utf-8")
        parsed = parse_labels(page["cvat_archive"], content)
        require(len(parsed) == page["notice_count"], f"Notice-count mismatch for {key}")
        require((page["role"] == "positive") == bool(parsed), f"Role/label mismatch for {key}")

        valid_boxes = [corners(box, width, height) for box in parsed]
        normalized_regions = []
        for region in decision["negative_regions_xyxy"]:
            x1, y1, x2, y2 = region
            require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"Invalid exclusion region for {key}")
            require(not any(overlaps(region, valid) for valid in valid_boxes), f"Exclusion overlaps a valid notice for {key}")
            normalized_regions.append([x1 / width, y1 / height, x2 / width, y2 / height])

        output_name = f"{page['issue_id']}_page_{page['pdf_page']:04d}.png"
        output_image = images / output_name
        output_label = labels / output_name.replace(".png", ".txt")
        shutil.copy2(image_source, output_image)
        output_label.write_text(content, encoding="utf-8")
        manifest_pages.append({
            "issue": page["issue_id"],
            "page": page["pdf_page"],
            "image_name": output_name,
            "image": str(output_image.relative_to(ROOT)),
            "label": str(output_label.relative_to(ROOT)),
            "role": page["role"],
            "category": decision["category"],
            "valid_notice_count": len(parsed),
            "negative_regions_xyxyn": normalized_regions,
            "width": width,
            "height": height,
            "image_sha256": file_sha256(output_image),
            "label_sha256": file_sha256(output_label),
            "cvat_archive": str(archive.relative_to(ROOT)),
            "cvat_archive_sha256": file_sha256(archive),
            "baseline_evidence": decision["baseline_evidence"],
        })

    manifest = {
        "name": "legal_notice_v10_targeted_correction",
        "status": "prepared_not_trained",
        "class_name": "legal_notice",
        "source_import_report": str(REPORT.relative_to(ROOT)),
        "source_import_report_sha256": file_sha256(REPORT),
        "review": str(REVIEW.relative_to(ROOT)),
        "review_sha256": file_sha256(REVIEW),
        "totals": {
            "pages": len(manifest_pages),
            "positive_pages": sum(page["role"] == "positive" for page in manifest_pages),
            "negative_pages": sum(page["role"] == "negative" for page in manifest_pages),
            "valid_notices": sum(page["valid_notice_count"] for page in manifest_pages),
            "negative_regions": sum(len(page["negative_regions_xyxyn"]) for page in manifest_pages),
        },
        "pages": manifest_pages,
    }
    manifest_path = DESTINATION / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), **manifest["totals"]}, indent=2))


if __name__ == "__main__":
    main()
