"""Prepare and clean a hash-verified, PDF-backed evaluation image cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE_ROOT = (ROOT / "output/legal_notices").resolve()
MANIFEST_KEYS = ("fixed_manifest", "challenge_manifest", "july_manifest")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def expected_pages(
    config: dict,
    manifest_keys: tuple[str, ...] = MANIFEST_KEYS,
    issues: tuple[str, ...] | None = None,
) -> dict[Path, dict]:
    expected = {}
    for key in manifest_keys:
        if key not in MANIFEST_KEYS:
            raise RuntimeError(f"Unsupported evaluation manifest key: {key}")
        manifest = load(project_path(config[key]))
        for page in manifest["pages"]:
            if issues is not None and page["issue"] not in issues:
                continue
            path = project_path(page["image"]).resolve()
            record = {"path": path, "sha256": page["image_sha256"], "issue": page["issue"], "page": page["page"], "manifest": key}
            if path in expected and expected[path]["sha256"] != record["sha256"]:
                raise RuntimeError(f"Conflicting evaluation hashes for {path}")
            expected[path] = record
    return expected


def validate_inputs(config: dict, manifest_keys: tuple[str, ...] = MANIFEST_KEYS) -> dict:
    expected = expected_pages(config, manifest_keys)
    existing = 0
    sources = set()
    for record in expected.values():
        path = record["path"]
        if path.exists():
            if file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"Evaluation image differs from its pinned hash: {path}")
            existing += 1
            continue
        if not path.is_relative_to(CACHE_ROOT):
            raise RuntimeError(f"Non-cache evaluation image is missing: {path}")
        pdf = ROOT / "input" / f"{record['issue']}.pdf"
        if not pdf.is_file():
            raise RuntimeError(f"Missing source PDF for evaluation cache: {pdf}")
        sources.add(relative(pdf))
    return {
        "manifest_keys": list(manifest_keys),
        "total_images": len(expected),
        "existing_images": existing,
        "source_pdfs": sorted(sources),
    }


def prepare(
    config: dict,
    report_path: Path,
    manifest_keys: tuple[str, ...] = MANIFEST_KEYS,
    issues: tuple[str, ...] | None = None,
) -> dict:
    from main import render_pdf

    expected = expected_pages(config, manifest_keys, issues)
    missing_cache = []
    existing = []
    for record in expected.values():
        path = record["path"]
        if path.exists():
            if file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"Evaluation image differs from its pinned hash: {path}")
            existing.append(record)
        elif path.is_relative_to(CACHE_ROOT):
            missing_cache.append(record)
        else:
            raise RuntimeError(f"Non-cache evaluation image is missing: {path}")

    sources = {}
    grouped = {}
    for record in missing_cache:
        grouped.setdefault((record["issue"], record["path"].parent), []).append(record)
    created = []
    try:
        for (issue, destination), records in sorted(grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))):
            pdf = ROOT / "input" / f"{issue}.pdf"
            if not pdf.is_file():
                raise RuntimeError(f"Missing source PDF for evaluation cache: {pdf}")
            sources[relative(pdf)] = file_sha256(pdf)
            created.extend(records)
            rendered = render_pdf(pdf, destination, 200, {record["page"] for record in records})
            if len(rendered) != len(records):
                raise RuntimeError(f"Rendered the wrong evaluation-page count for {issue}")

        for record in expected.values():
            path = record["path"]
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"Prepared evaluation image failed its pinned hash: {path}")
    except BaseException:
        for record in created:
            path = record["path"]
            if path.is_relative_to(CACHE_ROOT) and path.exists():
                path.unlink()
        raise

    report = {
        "status": "prepared",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": relative(CACHE_ROOT),
        "render_dpi": 200,
        "manifest_keys": list(manifest_keys),
        "issues": list(issues) if issues is not None else None,
        "total_images": len(expected),
        "existing_images": len(existing),
        "created_images": [
            {"path": relative(record["path"]), "sha256": record["sha256"], "bytes": record["path"].stat().st_size}
            for record in created
        ],
        "created_bytes": sum(record["path"].stat().st_size for record in created),
        "source_pdfs": [{"path": path, "sha256": sha256} for path, sha256 in sorted(sources.items())],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class IssueScopedCache:
    """Keep only the current issue's generated evaluation pages on disk."""

    def __init__(self, config: dict, manifest_key: str, report_root: Path):
        self.config = config
        self.manifest_key = manifest_key
        self.report_root = report_root
        self.current_issue: str | None = None
        self.current_report: Path | None = None
        self.reports: list[str] = []

    def activate(self, page: dict) -> None:
        issue = page["issue"]
        if issue == self.current_issue:
            return
        self.close()
        report = self.report_root / f"{self.manifest_key}_{issue}.json"
        prepare(self.config, report, (self.manifest_key,), (issue,))
        self.current_issue = issue
        self.current_report = report
        self.reports.append(relative(report))

    def close(self) -> None:
        if self.current_report is not None:
            cleanup(self.current_report)
        self.current_issue = None
        self.current_report = None


def cleanup(report_path: Path) -> dict:
    report = load(report_path)
    if report["status"] == "cleaned":
        return report
    if report["status"] != "prepared":
        raise RuntimeError(f"Evaluation cache report is not prepared: {report_path}")
    removed_bytes = 0
    removed = []
    for record in report["created_images"]:
        path = project_path(record["path"]).resolve()
        if not path.is_relative_to(CACHE_ROOT):
            raise RuntimeError(f"Refusing to clean outside evaluation cache: {path}")
        if not path.exists():
            continue
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Refusing to clean changed evaluation image: {path}")
        removed_bytes += path.stat().st_size
        path.unlink()
        removed.append(relative(path))
        parent = path.parent
        while parent != CACHE_ROOT and parent.is_relative_to(CACHE_ROOT):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    report.update({
        "status": "cleaned",
        "cleaned_at_utc": datetime.now(timezone.utc).isoformat(),
        "removed_images": removed,
        "removed_bytes": removed_bytes,
    })
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def purge_verified_manifest_cache(config: dict, report_path: Path, manifest_keys: tuple[str, ...] = MANIFEST_KEYS) -> dict:
    removed, removed_bytes = [], 0
    for record in expected_pages(config, manifest_keys).values():
        path = record["path"]
        if not path.is_relative_to(CACHE_ROOT) or not path.exists():
            continue
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Refusing to purge changed evaluation image: {path}")
        removed_bytes += path.stat().st_size
        path.unlink()
        removed.append(relative(path))
    report = {
        "status": "purged_verified_manifest_cache",
        "purged_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": relative(CACHE_ROOT),
        "manifest_keys": list(manifest_keys),
        "removed_images": removed,
        "removed_bytes": removed_bytes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="regression/legal_notice_v11_feature_finetune_config.json")
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest-key", action="append", choices=MANIFEST_KEYS, dest="manifest_keys")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--cleanup", action="store_true")
    action.add_argument("--purge-verified", action="store_true")
    args = parser.parse_args()
    report_path = project_path(args.report)
    if args.cleanup:
        result = cleanup(report_path)
    else:
        config = load(project_path(args.config))
        manifest_keys = tuple(args.manifest_keys or MANIFEST_KEYS)
        result = prepare(config, report_path, manifest_keys) if args.prepare else purge_verified_manifest_cache(config, report_path, manifest_keys)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
