"""Create or verify the immutable v11 full-page epoch-training dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from fixed080_acceptance import file_sha256
import prepare_corrected_dataset as dataset_tools


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "regression/legal_notice_v11_dataset_config.json"
HASH_ROOTS = ("data.yaml", "train_sampling.txt", "images", "labels")


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


def issue_name(image: Path) -> str:
    return image.stem.rsplit("_page_", 1)[0]


def verify_counts(actual: dict, expected: dict) -> None:
    require(actual == expected, f"Dataset counts differ: expected {expected}, found {actual}")


def verify_issue_isolation(dataset: Path) -> dict[str, list[str]]:
    issues = {}
    for split in ("train", "val", "test"):
        issues[split] = sorted({issue_name(path) for path in (dataset / "images" / split).glob("*.png")})
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(issues[first]) & set(issues[second])
        require(not overlap, f"Issue leakage between {first} and {second}: {sorted(overlap)}")
    return issues


def write_sampling_schedule(dataset: Path, config: dict) -> dict:
    overrides = config["sampling_overrides"]
    train_images = sorted((dataset / "images/train").glob("*.png"), key=lambda path: path.name)
    names = {path.name for path in train_images}
    require(set(overrides) <= names, f"Sampling overrides are absent from train: {sorted(set(overrides) - names)}")
    lines = []
    categories = {}
    for image in train_images:
        override = overrides.get(image.name, {"category": "standard", "multiplier": 1})
        multiplier = int(override["multiplier"])
        require(multiplier >= 1, f"Invalid sampling multiplier for {image.name}")
        lines.extend([f"./images/train/{image.name}"] * multiplier)
        category = override["category"]
        categories.setdefault(category, {"unique_pages": 0, "epoch_samples": 0})
        categories[category]["unique_pages"] += 1
        categories[category]["epoch_samples"] += multiplier
    require(len(lines) == config["expected_epoch_samples"], "Unexpected effective epoch sample count")
    (dataset / "train_sampling.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"unique_pages": len(train_images), "epoch_samples": len(lines), "categories": categories}


def validate_sources(config_path: Path, config: dict) -> tuple[Path, dict, dict]:
    source = project_path(config["source_dataset"])
    overlay_path = project_path(config["correction_overlay_manifest"])
    extra_source_path = project_path(config["reviewed_extra_source"])
    require(source.is_dir(), f"Missing source dataset: {source}")
    require(dataset_tools.tree_sha256(source) == config["source_dataset_sha256"], "Source dataset changed")
    require(file_sha256(overlay_path) == config["correction_overlay_manifest_sha256"], "Correction overlay changed")
    require(file_sha256(extra_source_path) == config["reviewed_extra_source_sha256"], "Reviewed extra-page source changed")
    require(config.get("training_authorized") is False, "Dataset preparation must not authorize training")
    return source, load(overlay_path), load(extra_source_path)


def create(config_path: Path, config: dict) -> Path:
    source, overlay, extra_source = validate_sources(config_path, config)
    destination = project_path(config["destination_dataset"])
    manifest_path = project_path(config["manifest_output"])
    require(not destination.exists(), f"Refusing to overwrite existing dataset: {destination}")
    require(not manifest_path.exists(), f"Refusing to overwrite existing manifest: {manifest_path}")
    source_before = dataset_tools.tree_sha256(source)
    moved, added, pdf_sources = [], [], []

    try:
        shutil.copytree(source, destination)
        for cache in destination.rglob("*.cache"):
            cache.unlink()

        move_issues = set(config["move_issues_to_train"])
        for split in ("val", "test"):
            for image in sorted((destination / "images" / split).glob("*.png")):
                if issue_name(image) not in move_issues:
                    continue
                label = destination / "labels" / split / image.with_suffix(".txt").name
                require(label.is_file(), f"Missing label for {image}")
                target_image = destination / "images/train" / image.name
                target_label = destination / "labels/train" / label.name
                require(not target_image.exists() and not target_label.exists(), f"Move collision for {image.name}")
                image.rename(target_image)
                label.rename(target_label)
                moved.append({"image": image.name, "from": split, "to": "train"})

        for page in overlay["pages"]:
            image = project_path(page["image"])
            label = project_path(page["label"])
            require(file_sha256(image) == page["image_sha256"], f"Changed overlay image: {image}")
            require(file_sha256(label) == page["label_sha256"], f"Changed overlay label: {label}")
            target_image = destination / "images/train" / page["image_name"]
            target_label = destination / "labels/train" / target_image.with_suffix(".txt").name
            require(not target_image.exists() and not target_label.exists(), f"Overlay collision: {page['image_name']}")
            shutil.copy2(image, target_image)
            shutil.copy2(label, target_label)
            added.append({"image": page["image_name"], "source": "reviewed_v10_cvat_overlay", "boxes": page["valid_notice_count"]})

        extra_by_name = {page["image_name"]: page for page in extra_source["extra_train_pages"]}
        require(set(config["reviewed_extra_pages"]) <= set(extra_by_name), "Reviewed extra-page record is missing")
        from main import render_pdf

        render_root = destination / "_render_tmp"
        for image_name in config["reviewed_extra_pages"]:
            record = extra_by_name[image_name]
            require(record.get("reviewed_no_valid_notices") is True, f"Extra page was not fully reviewed: {image_name}")
            page_number = int(Path(image_name).stem.rsplit("_page_", 1)[1])
            pdf = ROOT / "input" / f"{record['issue']}.pdf"
            require(pdf.is_file(), f"Missing source PDF: {pdf}")
            rendered = render_pdf(pdf, render_root / record["issue"], config["render_dpi"], {page_number})
            require(len(rendered) == 1, f"Could not render exactly one page for {image_name}")
            target_image = destination / "images/train" / image_name
            target_label = destination / "labels/train" / Path(image_name).with_suffix(".txt").name
            require(not target_image.exists() and not target_label.exists(), f"Extra-page collision: {image_name}")
            rendered[0].replace(target_image)
            target_label.write_text("", encoding="utf-8")
            pdf_sources.append({"path": relative(pdf), "sha256": file_sha256(pdf)})
            added.append({"image": image_name, "source": relative(pdf), "boxes": 0, "review": record["review"]})
        shutil.rmtree(render_root)

        (destination / "data.yaml").write_text(
            "train: train_sampling.txt\nval: images/val\ntest: images/test\nnames: [legal_notice]\n",
            encoding="utf-8",
        )
        sampling = write_sampling_schedule(destination, config)
        counts = dataset_tools.dataset_counts(destination)
        verify_counts(counts, config["expected_counts"])
        issues = verify_issue_isolation(destination)
        require(dataset_tools.tree_sha256(source) == source_before, "Source dataset changed during consolidation")

        manifest = {
            "name": config["name"],
            "status": "prepared_training_not_authorized",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": relative(config_path),
            "config_sha256": file_sha256(config_path),
            "source_dataset": relative(source),
            "source_dataset_sha256_before": source_before,
            "source_dataset_sha256_after": dataset_tools.tree_sha256(source),
            "dataset": relative(destination),
            "dataset_sha256": dataset_tools.tree_sha256(destination, HASH_ROOTS),
            "counts": counts,
            "sampling": sampling,
            "split_issues": issues,
            "moved_pages": moved,
            "added_pages": added,
            "pdf_sources": pdf_sources,
            "overlay_manifest": config["correction_overlay_manifest"],
            "overlay_manifest_sha256": config["correction_overlay_manifest_sha256"],
            "training_authorized": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        if manifest_path.exists():
            manifest_path.unlink()
        raise

    print(json.dumps({"dataset": relative(destination), "manifest": relative(manifest_path), "counts": counts, "sampling": sampling}, indent=2))
    return destination


def verify(config_path: Path, config: dict) -> dict:
    source, overlay, _ = validate_sources(config_path, config)
    destination = project_path(config["destination_dataset"])
    manifest_path = project_path(config["manifest_output"])
    require(destination.is_dir() and manifest_path.is_file(), "Prepared dataset or manifest is missing")
    manifest = load(manifest_path)
    require(manifest["config_sha256"] == file_sha256(config_path), "Dataset config changed")
    require(manifest["overlay_manifest_sha256"] == file_sha256(project_path(config["correction_overlay_manifest"])), "Overlay provenance changed")
    require(manifest["dataset_sha256"] == dataset_tools.tree_sha256(destination, HASH_ROOTS), "Consolidated dataset changed")
    require(manifest["source_dataset_sha256_after"] == dataset_tools.tree_sha256(source), "Source dataset changed")
    counts = dataset_tools.dataset_counts(destination)
    verify_counts(counts, config["expected_counts"])
    issues = verify_issue_isolation(destination)
    schedule = (destination / "train_sampling.txt").read_text(encoding="utf-8").splitlines()
    require(len(schedule) == config["expected_epoch_samples"], "Training schedule length changed")
    require(all((destination / line.removeprefix("./")).is_file() for line in schedule), "Training schedule references a missing image")
    for page in overlay["pages"]:
        image = destination / "images/train" / page["image_name"]
        label = destination / "labels/train" / Path(page["image_name"]).with_suffix(".txt").name
        require(file_sha256(image) == page["image_sha256"] and file_sha256(label) == page["label_sha256"], f"Overlay copy changed: {page['image_name']}")
    result = {"status": "verified_training_not_authorized", "dataset": relative(destination), "dataset_sha256": manifest["dataset_sha256"], "counts": counts, "epoch_samples": len(schedule), "split_issue_counts": {key: len(value) for key, value in issues.items()}}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load(config_path)
    verify(config_path, config) if args.verify_existing else create(config_path, config)


if __name__ == "__main__":
    main()
