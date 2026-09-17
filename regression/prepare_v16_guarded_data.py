"""Restore and pin the unique-page V16 guarded-correction training inventory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image

from fixed080_acceptance import file_sha256
import prepare_corrected_dataset as dataset_tools


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v16_guarded_config.json"
DATASET_HASH_ROOTS = ("data.yaml", "train_sampling.txt", "images", "labels")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_file(value: str, expected: str, description: str) -> Path:
    path = project_path(value)
    require(path.is_file(), f"Missing {description}: {path}")
    require(file_sha256(path) == expected, f"Changed {description}: {path}")
    return path


def restore_page(issue: str, page_number: int, destination: Path, expected_hash: str) -> Path:
    if destination.is_file():
        require(file_sha256(destination) == expected_hash, f"Changed restored image: {destination}")
        return destination
    pdf = ROOT / "input" / f"{issue}.pdf"
    require(pdf.is_file(), f"Missing replay PDF: {pdf}")
    from main import render_pdf
    rendered = render_pdf(pdf, destination.parent, 200, {page_number})
    require(rendered == [destination], f"Unexpected render output for {issue} page {page_number}")
    require(file_sha256(destination) == expected_hash, f"Rendered image hash mismatch: {destination}")
    return destination


def page_record(
    image: Path,
    label: Path,
    role: str,
    negative_regions: list[list[float]] | None = None,
    image_name: str | None = None,
) -> dict:
    require(image.is_file(), f"Missing training image: {image}")
    require(label.is_file(), f"Missing training label: {label}")
    image_name = image_name or image.name
    require("_page_" in image_name, f"Unrecognized page identity: {image_name}")
    issue, page_text = Path(image_name).stem.rsplit("_page_", 1)
    lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    with Image.open(image) as opened:
        width, height = opened.size
    return {
        "id": f"{issue}:p{int(page_text):04d}",
        "issue": issue,
        "page": int(page_text),
        "image_name": image_name,
        "image": relative(image),
        "image_sha256": file_sha256(image),
        "label": relative(label),
        "label_sha256": file_sha256(label),
        "width": width,
        "height": height,
        "valid_notice_count": len(lines),
        "negative_regions_xyxyn": negative_regions or [],
        "review_coverage": "complete_page",
        "data_role": role,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load(config_path)
    require(config["training_authorized"] is True, "V16 preparation is not authorized")
    verify_file(config["starting_weights"], config["starting_weights_sha256"], "starting weights")
    verify_file(config["dataset_manifest"], config["dataset_manifest_sha256"], "dataset manifest")
    verify_file(config["target_manifest"], config["target_manifest_sha256"], "target manifest")
    verify_file(config["release_inventory"], config["release_inventory_sha256"], "release inventory")
    replay = config["replay_sources"]
    september15_path = verify_file(replay["september15"], replay["september15_sha256"], "September 15 manifest")
    september09_path = verify_file(replay["september09_guard"], replay["september09_guard_sha256"], "September 9 guard manifest")
    overlay_path = verify_file(replay["historical_overlay"], replay["historical_overlay_sha256"], "historical overlay")
    for issue, expected in config["source_pdfs"].items():
        verify_file(f"input/{issue}.pdf", expected, f"source PDF {issue}")

    dataset_root = project_path(config["dataset_root"])
    require(dataset_tools.tree_sha256(dataset_root, DATASET_HASH_ROOTS) == config["dataset_sha256"], "Consolidated dataset changed")
    target = load(project_path(config["target_manifest"]))
    target_regions = {page["image_name"]: page["negative_regions_xyxyn"] for page in target["pages"]}
    pages = []
    for image in sorted((dataset_root / "images/train").glob("*.png")):
        label = dataset_root / "labels/train" / f"{image.stem}.txt"
        pages.append(page_record(image, label, "consolidated_train", target_regions.get(image.name)))

    september15 = load(september15_path)
    for source in september15["pages"]:
        image = restore_page("gulftoday_2026-09-15", source["page"], project_path(source["image"]), source["image_sha256"])
        label = verify_file(source["labels"], source["labels_sha256"], "September 15 labels")
        pages.append(page_record(image, label, "september15_replay", image_name=Path(source["labels"]).with_suffix(".png").name))

    september09 = load(september09_path)
    guard_image = ROOT / "output/legal_notices/v10_gulftoday_2026-09-09/rendered/gulftoday_2026-09-09/page_0012.png"
    require(guard_image.is_file() and file_sha256(guard_image) == september09["image_sha256"], "Missing hash-equivalent September 9 guard image")
    guard_label = verify_file(september09["labels"], september09["labels_sha256"], "September 9 guard labels")
    pages.append(page_record(guard_image, guard_label, "september09_preservation_replay", image_name="gulftoday_2026-09-09_page_0012.png"))

    overlay = load(overlay_path)
    selected = {"gulftoday_2026-09-10_page_0007.png", "gulftoday_2026-07-03_page_0007.png"}
    output_root = project_path(config["audit_root"])
    empty_labels = output_root / "prepared_labels"
    empty_labels.mkdir(parents=True, exist_ok=True)
    overlay_pages = {page["image_name"]: page for page in overlay["pages"] if page["image_name"] in selected}
    require(set(overlay_pages) == selected, "Historical overlay is missing required negative replay pages")
    for image_name in sorted(selected):
        source = overlay_pages[image_name]
        image = restore_page(source["issue"], int(Path(image_name).stem.rsplit("_page_", 1)[1]), project_path(source["image"]), source["image_sha256"])
        label = empty_labels / f"{Path(image_name).stem}.txt"
        if not label.exists():
            label.write_text("", encoding="ascii")
        require(label.read_bytes() == b"", f"Negative replay label is not empty: {label}")
        pages.append(page_record(image, label, "reviewed_negative_replay", image_name=image_name))

    identities = [page["id"] for page in pages]
    require(len(identities) == len(set(identities)), "Training inventory contains duplicate issue/page identities")
    require(len(pages) == 96, f"Expected 96 unique pages, found {len(pages)}")
    require(sum(page["valid_notice_count"] for page in pages) == 352, "Expected 352 valid notice boxes")
    require(sum(len(page["negative_regions_xyxyn"]) for page in pages) == 8, "Expected eight explicit target-negative regions")
    by_role = {}
    for page in pages:
        role = page["data_role"]
        by_role.setdefault(role, {"pages": 0, "valid_notices": 0, "negative_regions": 0})
        by_role[role]["pages"] += 1
        by_role[role]["valid_notices"] += page["valid_notice_count"]
        by_role[role]["negative_regions"] += len(page["negative_regions_xyxyn"])
    manifest = {
        "name": "legal_notice_v16_guarded_training_inventory",
        "status": "prepared_not_fitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": file_sha256(config_path),
        "starting_weights_sha256": config["starting_weights_sha256"],
        "counts": {
            "unique_pages": len(pages),
            "valid_notices": sum(page["valid_notice_count"] for page in pages),
            "explicit_negative_regions": sum(len(page["negative_regions_xyxyn"]) for page in pages),
            "by_role": by_role,
        },
        "pages": pages,
    }
    output = project_path(config["prepared_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "output": relative(output), "sha256": file_sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
