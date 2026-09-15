"""Create a corrected YOLO dataset version from an immutable source dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, relative_roots: tuple[str, ...] = ("data.yaml", "images", "labels")) -> str:
    digest = hashlib.sha256()
    files = []
    for relative_root in relative_roots:
        path = root / relative_root
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix != ".cache")
        else:
            raise SystemExit(f"Dataset path is missing: {path}")
    for path in sorted(files, key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_counts(dataset: Path) -> dict[str, dict[str, int]]:
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        labels = sorted(label_dir.rglob("*.txt"))
        counts[split] = {
            "images": len(images),
            "label_files": len(labels),
            "boxes": sum(len(path.read_text(encoding="utf-8").splitlines()) for path in labels),
        }
    return counts


def apply_removals(destination: Path, removals: list[dict]) -> list[dict]:
    applied = []
    for removal in removals:
        split = removal["split"]
        image_name = removal["image"]
        image_relative = Path(image_name)
        if split not in {"train", "val", "test"}:
            raise SystemExit(f"Unsupported correction split: {split}")
        if image_relative.name != image_name or image_relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Correction image must be a plain supported filename: {image_name}")
        image_path = destination / "images" / split / image_relative
        label_root = (destination / "labels" / split).resolve()
        label_path = (label_root / image_relative.with_suffix(".txt")).resolve()
        if not label_path.is_relative_to(label_root):
            raise SystemExit(f"Correction label escapes the destination: {label_path}")
        if not image_path.is_file():
            raise SystemExit(f"Correction image does not exist: {image_path}")
        if not label_path.is_file():
            raise SystemExit(f"Correction label does not exist: {label_path}")

        lines = label_path.read_text(encoding="utf-8").splitlines()
        line_index = removal["label_line"] - 1
        if line_index < 0 or line_index >= len(lines):
            raise SystemExit(f"Correction line is out of range: {label_path}:{removal['label_line']}")
        if lines[line_index] != removal["expected_label"]:
            raise SystemExit(
                f"Correction source mismatch at {label_path}:{removal['label_line']}\n"
                f"Expected: {removal['expected_label']}\n"
                f"Actual:   {lines[line_index]}"
            )

        removed_label = lines.pop(line_index)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        applied.append({**removal, "removed_label": removed_label})
    return applied


def verify_counts(actual: dict, expected: dict) -> None:
    errors = []
    for split, expected_split in expected.items():
        actual_split = actual.get(split, {})
        for key, value in expected_split.items():
            if actual_split.get(key) != value:
                errors.append(f"{split} {key}: expected {value}, found {actual_split.get(key)}")
        if actual_split.get("label_files") != actual_split.get("images"):
            errors.append(
                f"{split}: {actual_split.get('images')} images but "
                f"{actual_split.get('label_files')} label files"
            )
    if errors:
        raise SystemExit("Corrected dataset count verification failed:\n- " + "\n- ".join(errors))


def verify_existing_dataset(corrections_path: Path, output_path: Path) -> None:
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    source = project_path(corrections["source_dataset"])
    destination = project_path(corrections["destination_dataset"])
    if not source.is_dir() or not destination.is_dir():
        raise SystemExit("Both source and corrected datasets must exist for verification.")

    counts = dataset_counts(destination)
    verify_counts(counts, corrections["expected_counts"])
    removals_by_label = {}
    for removal in corrections["removals"]:
        relative = Path("labels") / removal["split"] / Path(removal["image"]).with_suffix(".txt")
        removals_by_label.setdefault(relative.as_posix(), []).append(removal)

    source_labels = {path.relative_to(source).as_posix(): path for path in source.glob("labels/*/*.txt")}
    destination_labels = {
        path.relative_to(destination).as_posix(): path for path in destination.glob("labels/*/*.txt")
    }
    if set(source_labels) != set(destination_labels):
        raise SystemExit("Source and corrected dataset label-file sets do not match.")
    for relative, source_label in source_labels.items():
        expected_lines = source_label.read_text(encoding="utf-8").splitlines()
        for removal in sorted(removals_by_label.get(relative, []), key=lambda item: item["label_line"], reverse=True):
            line_index = removal["label_line"] - 1
            if line_index < 0 or line_index >= len(expected_lines):
                raise SystemExit(f"Verification correction line is out of range: {relative}")
            if expected_lines[line_index] != removal["expected_label"]:
                raise SystemExit(f"Source correction label changed: {relative}:{removal['label_line']}")
            expected_lines.pop(line_index)
        actual_lines = destination_labels[relative].read_text(encoding="utf-8").splitlines()
        if actual_lines != expected_lines:
            raise SystemExit(f"Unexpected corrected label difference: {relative}")

    if (source / "data.yaml").read_bytes() != (destination / "data.yaml").read_bytes():
        raise SystemExit("Corrected data.yaml does not match the source.")
    source_images_sha256 = tree_sha256(source, ("images",))
    destination_images_sha256 = tree_sha256(destination, ("images",))
    source_test_sha256 = tree_sha256(source, ("images/test", "labels/test"))
    destination_test_sha256 = tree_sha256(destination, ("images/test", "labels/test"))
    if source_images_sha256 != destination_images_sha256:
        raise SystemExit("Corrected dataset images do not match the source images.")
    if source_test_sha256 != destination_test_sha256:
        raise SystemExit("Corrected dataset test split does not match the source test split.")

    report = {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source.relative_to(ROOT)),
        "corrected_dataset": str(destination.relative_to(ROOT)),
        "source_dataset_sha256": tree_sha256(source),
        "corrected_dataset_sha256": tree_sha256(destination),
        "images_sha256": source_images_sha256,
        "test_split_sha256": source_test_sha256,
        "only_label_differences": sorted(removals_by_label),
        "removed_boxes": len(corrections["removals"]),
        "counts": counts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Dataset verification report: {output_path.relative_to(ROOT)}")


def create_dataset(corrections_path: Path) -> Path:
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    source = project_path(corrections["source_dataset"])
    destination = project_path(corrections["destination_dataset"])
    if not source.is_dir():
        raise SystemExit(f"Source dataset does not exist: {source}")
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}. Refusing to overwrite it.")

    source_before_sha256 = tree_sha256(source)
    source_images_sha256 = tree_sha256(source, ("images",))
    source_test_sha256 = tree_sha256(source, ("images/test", "labels/test"))
    try:
        shutil.copytree(source, destination)
        applied = apply_removals(destination, corrections["removals"])
        for cache_path in destination.rglob("*.cache"):
            cache_path.unlink()
        counts = dataset_counts(destination)
        verify_counts(counts, corrections["expected_counts"])
        source_after_sha256 = tree_sha256(source)
        if source_after_sha256 != source_before_sha256:
            raise SystemExit("Source dataset changed while creating the corrected copy.")
        if tree_sha256(destination, ("images",)) != source_images_sha256:
            raise SystemExit("Corrected dataset images do not match the source images.")
        if tree_sha256(destination, ("images/test", "labels/test")) != source_test_sha256:
            raise SystemExit("Corrected dataset test split does not match the source test split.")
        record = {
            "status": "applied",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source.relative_to(ROOT)),
            "source_data_yaml_sha256": file_sha256(source / "data.yaml"),
            "source_dataset_sha256_before": source_before_sha256,
            "source_dataset_sha256_after": source_after_sha256,
            "images_sha256": source_images_sha256,
            "test_split_sha256": source_test_sha256,
            "corrections_file": str(corrections_path.relative_to(ROOT)),
            "corrections_file_sha256": file_sha256(corrections_path),
            "removals": applied,
            "counts": counts,
            "source_dataset_unchanged": source_before_sha256 == source_after_sha256,
        }
        (destination / "CORRECTIONS_APPLIED.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(destination)
        raise

    print(json.dumps(counts, indent=2))
    print(f"Corrected dataset created: {destination.relative_to(ROOT)}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corrections",
        default="regression/dataset_v5_corrections.json",
        help="Correction manifest relative to the project root",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify an existing corrected dataset instead of creating it",
    )
    parser.add_argument(
        "--verification-output",
        default="output/regression_audit/dataset_v5_corrected_verification.json",
    )
    args = parser.parse_args()
    corrections_path = project_path(args.corrections)
    if args.verify_existing:
        verify_existing_dataset(corrections_path, project_path(args.verification_output))
    else:
        create_dataset(corrections_path)


if __name__ == "__main__":
    main()
