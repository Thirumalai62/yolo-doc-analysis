"""Restore pinned manifest images from source PDFs at 200 DPI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from fixed080_acceptance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "output/regression_audit/legal_notice_v8_protected_head_cpu_r7_paa_analogue/july_manifest.json"
DEFAULT_REPORT = ROOT / "output/regression_audit/legal_notice_v11_july_render_restore.json"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    args = parser.parse_args()
    manifest_path = project_path(args.manifest)
    report_path = project_path(args.report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: dict[str, list[dict]] = {}
    for page in manifest["pages"]:
        issues.setdefault(page["issue"], []).append(page)

    from main import render_pdf

    sources, restored = [], []
    for issue, pages in sorted(issues.items()):
        pdf = ROOT / "input" / f"{issue}.pdf"
        if not pdf.is_file():
            raise RuntimeError(f"Missing source PDF: {pdf}")
        sources.append({"path": pdf.relative_to(ROOT).as_posix(), "sha256": file_sha256(pdf)})
        missing = []
        for page in pages:
            image = project_path(page["image"])
            if image.exists():
                if file_sha256(image) != page["image_sha256"]:
                    raise RuntimeError(f"Existing image differs from its pinned hash: {image}")
            else:
                missing.append(page)
        if missing:
            destinations = {project_path(page["image"]).parent for page in missing}
            if len(destinations) != 1:
                raise RuntimeError(f"Issue has inconsistent render destinations: {issue}")
            rendered = render_pdf(pdf, destinations.pop(), 200, {page["page"] for page in missing})
            if len(rendered) != len(missing):
                raise RuntimeError(f"Rendered the wrong number of pages for {issue}")
            restored.extend(path.relative_to(ROOT).as_posix() for path in rendered)

    for page in manifest["pages"]:
        image = project_path(page["image"])
        if not image.is_file() or file_sha256(image) != page["image_sha256"]:
            raise RuntimeError(f"Restored image failed its pinned hash: {image}")

    report = {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
        "render_dpi": 200,
        "issues": len(issues),
        "pages": len(manifest["pages"]),
        "restored_pages": len(restored),
        "source_pdfs": sources,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
