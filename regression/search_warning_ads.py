"""Search eligible development PDFs for private warning advertisements."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset_v5_corrected"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "regression_audit"
    / "legal_notice_v5_corrective_3_monitor"
    / "warning_ad_search.json"
)
ISSUE_FROM_IMAGE = re.compile(r"(.+_\d{4}-\d{2}-\d{2})_page_\d{4}$")
PRESERVATION_DATES = {"2026-09-09", "2026-09-10"}
SEARCH_TERMS = (
    "قطع عال",
    "عالقتها بال",
    "كما نحذر",
    "نحذر",
    "حتذير",
    "تحذير",
    "إخالء م",
    "اخالء م",
    "تبرؤ",
    "براءة ذمة",
    "no longer associated",
    "not associated with",
    "shall not be responsible",
    "will not be responsible",
    "public warning",
    "warning notice",
    "disclaimer",
    "disassociation",
    "termination of relationship",
    "ceased to be",
)


def issue_from_page_name(name: str) -> str | None:
    match = ISSUE_FROM_IMAGE.match(Path(name).stem)
    return match.group(1) if match else None


def development_issues(dataset: Path) -> set[str]:
    issues: set[str] = set()
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        if not image_dir.is_dir():
            raise SystemExit(f"Dataset split does not exist: {image_dir}")
        for image in image_dir.glob("*.png"):
            issue = issue_from_page_name(image.name)
            if issue:
                issues.add(issue)
    return issues


def held_out_issues() -> set[str]:
    issues: set[str] = set()
    for manifest_path in (PROJECT_ROOT / "annotation_batches").glob("*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for page in manifest.get("pages", []):
            if page.get("split") != "test":
                continue
            issue = page.get("issue_id")
            if not issue:
                source_pdf = page.get("source_pdf")
                issue = Path(source_pdf).stem if isinstance(source_pdf, str) else None
            if issue:
                issues.add(issue)
    return issues


def issue_date(issue: str) -> str:
    return issue.rsplit("_", maxsplit=1)[-1]


def matching_lines(text: str, matches: list[str]) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and any(term.casefold() in line.casefold() for term in matches)
    ]


def scan(dataset: Path) -> dict[str, object]:
    import pypdfium2 as pdfium

    held_out = held_out_issues()
    eligible = sorted(
        issue
        for issue in development_issues(dataset)
        if issue not in held_out and issue_date(issue) not in PRESERVATION_DATES
    )
    candidates: list[dict[str, object]] = []
    missing_pdfs: list[str] = []

    for issue in eligible:
        pdf_path = PROJECT_ROOT / "input" / f"{issue}.pdf"
        if not pdf_path.is_file():
            missing_pdfs.append(str(pdf_path.relative_to(PROJECT_ROOT)))
            continue

        document = pdfium.PdfDocument(str(pdf_path))
        try:
            for page_index in range(len(document)):
                page = document[page_index]
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_range()
                finally:
                    text_page.close()
                    page.close()

                matches = [term for term in SEARCH_TERMS if term.casefold() in text.casefold()]
                if matches:
                    candidates.append(
                        {
                            "issue_id": issue,
                            "source_pdf": str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                            "pdf_page": page_index + 1,
                            "matches": matches,
                            "matching_lines": matching_lines(text, matches),
                        }
                    )
        finally:
            document.close()

    return {
        "searched_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "dataset": str(dataset.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "included_splits": ["train", "val"],
            "excluded_manifest_splits": ["test"],
            "excluded_preservation_dates": sorted(PRESERVATION_DATES),
            "held_out_issue_ids_excluded": sorted(held_out),
        },
        "search_terms": list(SEARCH_TERMS),
        "eligible_issue_ids": eligible,
        "missing_pdfs": missing_pdfs,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if not output.parent.is_dir():
        raise SystemExit(f"Output parent does not exist: {output.parent}")
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")

    report = scan(dataset)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Eligible issues: {len(report['eligible_issue_ids'])}")
    print(f"Candidate pages: {len(report['candidates'])}")
    print(f"Missing PDFs: {len(report['missing_pdfs'])}")
    print(f"Saved: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
