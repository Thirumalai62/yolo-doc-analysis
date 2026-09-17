"""Build the V17 correction inventory directly from the complete release contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

import evaluate_v17_release as release
from fixed080_acceptance import file_sha256
from prepare_v16_guarded_data import restore_page


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v17_guarded_config.json"


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


def label_boxes(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, box_width, box_height = map(float, line.split())
        require(cls == 0, f"Unexpected class in {path}")
        boxes.append([(x - box_width / 2) * width, (y - box_height / 2) * height, (x + box_width / 2) * width, (y + box_height / 2) * height])
    return boxes


def normalized_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load(config_path)
    require(config["training_authorized"] is True, "V17 preparation is not authorized")
    require(file_sha256(project_path(config["starting_weights"])) == config["starting_weights_sha256"], "Starting weights changed")
    release_config_path = project_path(config["pinned_artifacts"]["release_config"]["path"])
    require(file_sha256(release_config_path) == config["pinned_artifacts"]["release_config"]["sha256"], "Release config changed")
    release_config = load(release_config_path)
    manifests = {name: load(release.verify_artifact(artifact, f"{name} manifest")) for name, artifact in release_config["manifests"].items()}
    for name in ("fixed", "challenge", "july", "historical"):
        release.restore_manifest_images(manifests[name])
    for page in manifests["september15"]["pages"]:
        restore_page("gulftoday_2026-09-15", page["page"], project_path(page["image"]), page["image_sha256"])
    september11_image = project_path(manifests["september11"]["image"])
    september11_issue = Path(manifests["september11"]["source_pdf"]).stem
    restore_page(september11_issue, manifests["september11"]["page"], september11_image, manifests["september11"]["image_sha256"])

    pages: dict[tuple[str, int], dict] = {}

    def add(issue: str, page_number: int, image_value: str | Path, image_sha256: str, valid: list[list[float]], negatives: list[list[float]], role: str) -> None:
        image = project_path(str(image_value))
        require(image.is_file() and file_sha256(image) == image_sha256, f"Changed source image: {image}")
        with Image.open(image) as opened:
            width, height = opened.size
        key = (issue, page_number)
        if key not in pages:
            pages[key] = {"issue": issue, "page": page_number, "image": image, "image_sha256": image_sha256, "width": width, "height": height, "valid": [], "negatives": [], "roles": set()}
        record = pages[key]
        require(record["image_sha256"] == image_sha256 and (record["width"], record["height"]) == (width, height), f"Conflicting page source: {issue} p{page_number}")
        for box in valid:
            rounded = [round(float(value), 4) for value in box]
            if rounded not in record["valid"]:
                record["valid"].append(rounded)
        for region in negatives:
            rounded = [round(float(value), 8) for value in region]
            if rounded not in record["negatives"]:
                record["negatives"].append(rounded)
        record["roles"].add(role)

    for name in ("fixed", "historical"):
        for page in manifests[name]["pages"]:
            valid = [reference["xyxy"] for reference in page["references"] if reference["classification"] == "valid"]
            excluded = [reference["xyxy"] for reference in page["references"] if reference["classification"] == "excluded"]
            with Image.open(project_path(page["image"])) as opened:
                width, height = opened.size
            add(page["issue"], page["page"], page["image"], page["image_sha256"], valid, [[box[0] / width, box[1] / height, box[2] / width, box[3] / height] for box in excluded], name)
    for name in ("challenge", "july"):
        for page in manifests[name]["pages"]:
            with Image.open(project_path(page["image"])) as opened:
                width, height = opened.size
            add(page["issue"], page["page"], page["image"], page["image_sha256"], [reference["xyxy"] for reference in page["valid"]], [[reference["xyxy"][0] / width, reference["xyxy"][1] / height, reference["xyxy"][2] / width, reference["xyxy"][3] / height] for reference in page["excluded"]], name)
    for page in manifests["target"]["pages"]:
        image = project_path(page["image"])
        with Image.open(image) as opened:
            width, height = opened.size
        add(page["issue"], page["page"], page["image"], page["image_sha256"], label_boxes(project_path(page["label"]), width, height), page["negative_regions_xyxyn"], "target")
    for page in manifests["september15"]["pages"]:
        image = project_path(page["image"])
        with Image.open(image) as opened:
            width, height = opened.size
        add("gulftoday_2026-09-15", page["page"], page["image"], page["image_sha256"], label_boxes(project_path(page["labels"]), width, height), [], "september15")

    for review in manifests["semantic_reviews"]["reviews"]:
        key = (review["issue"], review["page"])
        require(key in pages, f"Semantic review has no primary release page: {key}")
        record = pages[key]
        if review["classification"] == "valid":
            add(*key, record["image"], record["image_sha256"], [review["xyxy"]], [], "semantic_review")
        else:
            box = review["xyxy"]
            region = [box[0] / record["width"], box[1] / record["height"], box[2] / record["width"], box[3] / record["height"]]
            add(*key, record["image"], record["image_sha256"], [], [region], "semantic_review")
    add(september11_issue, manifests["september11"]["page"], september11_image, manifests["september11"]["image_sha256"], [], [region["xyxyn"] for region in manifests["september11"]["regions"]], "september11")

    output_root = project_path(config["audit_root"])
    label_root = output_root / "prepared_labels"
    label_root.mkdir(parents=True, exist_ok=True)
    output_pages = []
    for (issue, page_number), page in sorted(pages.items()):
        label = label_root / f"{issue}_page_{page_number:04d}.txt"
        lines = ["0 " + " ".join(f"{value:.10f}" for value in normalized_box(box, page["width"], page["height"])) for box in page["valid"]]
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
        output_pages.append({
            "id": f"{issue}:p{page_number:04d}", "issue": issue, "page": page_number,
            "image_name": f"{issue}_page_{page_number:04d}.png", "image": relative(page["image"]), "image_sha256": page["image_sha256"],
            "label": relative(label), "label_sha256": file_sha256(label), "width": page["width"], "height": page["height"],
            "valid_notice_count": len(page["valid"]), "negative_regions_xyxyn": page["negatives"],
            "review_coverage": "complete_page", "data_role": "+".join(sorted(page["roles"])),
        })
    counts = {
        "unique_pages": len(output_pages),
        "valid_notices": sum(page["valid_notice_count"] for page in output_pages),
        "explicit_negative_regions": sum(len(page["negative_regions_xyxyn"]) for page in output_pages),
    }
    require(counts == config["inventory_counts"], f"Complete release inventory count mismatch: {counts}")
    manifest = {
        "name": "legal_notice_v17_complete_release_training_inventory", "status": "prepared_not_fitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": relative(config_path),
        "config_sha256": file_sha256(config_path), "starting_weights_sha256": config["starting_weights_sha256"],
        "release_config": relative(release_config_path), "release_config_sha256": file_sha256(release_config_path),
        "counts": counts, "pages": output_pages,
    }
    output = project_path(config["prepared_manifest"])
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "counts": counts, "output": relative(output), "sha256": file_sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
