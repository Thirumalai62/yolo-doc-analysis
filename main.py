"""CPU-only workflow for detecting legal notices in UAE e-paper PDFs.

Run `py main.py --help` to see the available commands. Start with
`py main.py setup`, then put PDF files in `input/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import shutil
import subprocess
import sys
from zipfile import ZipFile
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
RUNS_DIR = PROJECT_ROOT / "runs"
DATASET_DIR = PROJECT_ROOT / "dataset"
LAYOUT_REPOSITORY = "Armaggheddon/yolo26-document-layout"
LAYOUT_MODEL_FILE = "yolo26m_doc_layout.pt"
LEGAL_NOTICE_CLASS = "legal_notice"
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def project_path(value: str) -> Path:
    """Interpret relative command arguments from the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_directories() -> None:
    """Create directories that hold local input and generated artifacts."""
    for directory in (INPUT_DIR, MODELS_DIR, OUTPUT_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def virtualenv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run_setup(_: argparse.Namespace) -> None:
    """Create a virtual environment and install CPU-only runtime dependencies."""
    ensure_directories()
    venv_dir = PROJECT_ROOT / ".venv"
    python = virtualenv_python(venv_dir)

    if not python.exists():
        print(f"Creating virtual environment: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    else:
        print(f"Using existing virtual environment: {venv_dir}")

    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        [str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
    ]
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)

    print("\nScaffold complete.")
    print("1. Put newspaper PDFs in input/.")
    print("2. Download the local layout model:")
    print(r"   .\.venv\Scripts\python.exe main.py download-model")
    print("3. Render pages or inspect layout regions:")
    print(r"   .\.venv\Scripts\python.exe main.py render --input input")
    print(r"   .\.venv\Scripts\python.exe main.py preview-layout --input input")
    print("Read README.md before labeling and training a legal-notice model.")


def require_packages(*package_names: str) -> None:
    """Give a useful error when an ML command runs before setup."""
    try:
        for package_name in package_names:
            __import__(package_name)
    except ImportError as error:
        missing = error.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Run `py main.py setup` first."
        ) from error


def input_files(input_path: Path, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in extensions else []
    elif input_path.is_dir():
        files = [path for path in input_path.rglob("*") if path.suffix.lower() in extensions]
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")
    if not files:
        extension_list = ", ".join(sorted(extensions))
        raise SystemExit(f"No supported files found in {input_path} ({extension_list}).")
    return sorted(files)


def render_pdf(pdf_path: Path, destination: Path, dpi: int, page_numbers: set[int] | None = None) -> list[Path]:
    """Render each PDF page with pypdfium2; no external PDF utility is needed."""
    from PIL import Image
    import pypdfium2 as pdfium

    scale = dpi / 72
    document = pdfium.PdfDocument(str(pdf_path))
    rendered_paths: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for page_number in range(len(document)):
            if page_numbers is not None and page_number + 1 not in page_numbers:
                continue
            page = document[page_number]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil().convert("RGB")
            output_path = destination / f"page_{page_number + 1:04d}.png"
            pil_image.save(output_path)
            rendered_paths.append(output_path)
    finally:
        document.close()
    return rendered_paths


def render_input_pdfs(
    input_path: Path, dpi: int, destination_root: Path, selections: dict[str, set[int]] | None = None
) -> list[Path]:
    pages: list[Path] = []
    for pdf_path in input_files(input_path, {".pdf"}):
        page_numbers = selections.get(pdf_path.name) if selections else None
        if selections and page_numbers is None:
            continue
        relative_name = pdf_path.relative_to(input_path) if input_path.is_dir() else Path(pdf_path.name)
        destination = destination_root / relative_name.with_suffix("")
        rendered = render_pdf(pdf_path, destination, dpi, page_numbers)
        print(f"Rendered {len(rendered)} page(s): {pdf_path.name}")
        pages.extend(rendered)
    return pages


def create_run_directory(kind: str, name: str | None = None) -> Path:
    """Keep generated artifacts from separate commands from overwriting each other."""
    run_name = name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = OUTPUT_DIR / kind / run_name
    if destination.exists():
        raise SystemExit(f"Output run already exists: {destination}. Choose a different --name.")
    destination.mkdir(parents=True)
    return destination


def run_render(args: argparse.Namespace) -> None:
    ensure_directories()
    output_root = create_run_directory("rendered", args.name)
    pages = render_input_pdfs(project_path(args.input), args.dpi, output_root)
    print(f"Saved {len(pages)} rendered page(s): {output_root.relative_to(PROJECT_ROOT)}")


def download_layout_model(_: argparse.Namespace) -> None:
    """Download the supplied medium layout model once and retain it locally."""
    require_packages("huggingface_hub")
    from huggingface_hub import hf_hub_download

    ensure_directories()
    model_path = hf_hub_download(
        repo_id=LAYOUT_REPOSITORY,
        filename=LAYOUT_MODEL_FILE,
        local_dir=MODELS_DIR,
    )
    print(f"Layout model ready: {model_path}")
    print("This model finds document-layout regions only; it is not a legal-notice detector.")


def load_yolo(weights: Path):
    require_packages("ultralytics")
    if not weights.is_file():
        raise SystemExit(f"Weights file does not exist: {weights}")
    from ultralytics import YOLO

    return YOLO(str(weights))


def layout_model_path() -> Path:
    path = MODELS_DIR / LAYOUT_MODEL_FILE
    if not path.is_file():
        raise SystemExit(
            "The local layout model is missing. Run `py main.py download-model` first."
        )
    return path


def run_preview_layout(args: argparse.Namespace) -> None:
    """Annotate general page-layout regions using the downloaded medium model."""
    ensure_directories()
    output_root = create_run_directory("layout_preview", args.name)
    rendered_root = output_root / "rendered"
    pages = render_input_pdfs(project_path(args.input), args.dpi, rendered_root)
    model = load_yolo(layout_model_path())
    preview_dir = output_root / "annotated"
    for page in pages:
        relative_page = page.relative_to(rendered_root)
        destination = preview_dir / relative_page
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = model.predict(str(page), imgsz=args.image_size, conf=args.confidence, device="cpu", verbose=False)[0]
        result.save(filename=str(destination))
        print(f"Layout preview saved: {destination.relative_to(PROJECT_ROOT)}")
    print(f"Preview complete: {output_root.relative_to(PROJECT_ROOT)}")
    print("These boxes are layout classes, not legal-notice predictions.")


def class_names(model) -> list[str]:
    names = model.names
    return list(names.values()) if isinstance(names, dict) else list(names)


def require_legal_notice_model(model) -> None:
    names = class_names(model)
    if names != [LEGAL_NOTICE_CLASS]:
        raise SystemExit(
            "The supplied checkpoint must contain exactly one class named `legal_notice`. "
            f"Its classes are: {names}. Train a custom model before running detect."
        )


def run_detect(args: argparse.Namespace) -> None:
    """Detect, crop, annotate, and report legal notices from rendered PDF pages."""
    from PIL import Image

    ensure_directories()
    output_root = create_run_directory("legal_notices", args.name)
    rendered_root = output_root / "rendered"
    pages = render_input_pdfs(project_path(args.input), args.dpi, rendered_root)
    model = load_yolo(project_path(args.weights))
    require_legal_notice_model(model)
    report: list[dict] = []

    for page in pages:
        relative_page = page.relative_to(rendered_root)
        page_root = output_root / "pages" / relative_page.parent / relative_page.stem
        crop_dir = page_root / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        result = model.predict(str(page), imgsz=args.image_size, conf=args.confidence, device="cpu", verbose=False)[0]
        annotated_path = page_root / "annotated.png"
        result.save(filename=str(annotated_path))
        image = Image.open(page).convert("RGB")
        detections: list[dict] = []
        for number, box in enumerate(result.boxes, start=1):
            x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
            x1, x2 = max(0, x1), min(image.width, x2)
            y1, y2 = max(0, y1), min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_path = crop_dir / f"notice_{number:03d}.png"
            image.crop((x1, y1, x2, y2)).save(crop_path)
            detections.append({
                "id": number,
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "crop": str(crop_path.relative_to(PROJECT_ROOT)),
            })
        report.append({
            "page": str(page.relative_to(PROJECT_ROOT)), "detections": detections
        })
        print(f"{page.name}: {len(detections)} legal notice(s)")

    report_path = output_root / "detections.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path.relative_to(PROJECT_ROOT)}")


def parse_page_selection(values: list[str] | None) -> dict[str, set[int]]:
    """Parse repeated FILE.pdf:PAGE,PAGE options for annotation review."""
    selections: dict[str, set[int]] = {}
    for value in values or []:
        try:
            filename, page_numbers = value.rsplit(":", 1)
            numbers = {int(number) for number in page_numbers.split(",")}
        except ValueError as error:
            raise SystemExit(f"Invalid --include value: {value}. Use FILE.pdf:PAGE,PAGE.") from error
        if not filename or not numbers or any(number <= 0 for number in numbers):
            raise SystemExit(f"Invalid --include value: {value}. Page numbers must be positive.")
        selections.setdefault(Path(filename).name, set()).update(numbers)
    return selections


def read_annotation_plan(plan_path: Path) -> list[dict[str, object]]:
    """Load and validate a batch plan before rendering any annotation images."""
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid annotation plan: {plan_path}") from error
    pages = plan.get("pages") if isinstance(plan, dict) else None
    if not isinstance(pages, list) or not pages:
        raise SystemExit("Annotation plan must contain a non-empty 'pages' list.")
    seen_images: set[str] = set()
    seen_sources: set[tuple[Path, int]] = set()
    issue_splits: dict[str, str] = {}
    issue_sources: dict[str, Path] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise SystemExit("Every annotation-plan page must be an object.")
        source_pdf = page.get("source_pdf")
        pdf_page = page.get("pdf_page")
        image_name = page.get("image_name")
        split = page.get("split")
        role = page.get("role")
        if not isinstance(source_pdf, str) or not isinstance(pdf_page, int) or not isinstance(image_name, str):
            raise SystemExit("Every annotation-plan page needs source_pdf, pdf_page, and image_name.")
        source_path = project_path(source_pdf).resolve()
        newspaper = page.get("newspaper", source_path.stem.split("_")[0])
        issue_id = page.get("issue_id", source_path.stem)
        if not isinstance(newspaper, str) or not newspaper or not isinstance(issue_id, str) or not issue_id:
            raise SystemExit("Annotation-plan newspaper and issue_id values must be non-empty strings.")
        page["newspaper"] = newspaper
        page["issue_id"] = issue_id
        if split not in {"train", "val", "test"} or role not in {"positive", "negative"}:
            raise SystemExit("Every annotation-plan page needs split train/val/test and role positive/negative.")
        if pdf_page <= 0 or Path(image_name).name != image_name or Path(image_name).suffix.lower() != ".png":
            raise SystemExit(f"Invalid page or image name in annotation plan: {source_pdf}:{pdf_page}")
        if image_name in seen_images or (source_path, pdf_page) in seen_sources:
            raise SystemExit(f"Duplicate annotation-plan page: {source_pdf}:{pdf_page}")
        if issue_id in issue_splits and issue_splits[issue_id] != split:
            raise SystemExit(f"Issue {issue_id} is assigned to multiple splits.")
        if issue_id in issue_sources and issue_sources[issue_id] != source_path:
            raise SystemExit(f"Issue {issue_id} maps to multiple source PDFs.")
        seen_images.add(image_name)
        seen_sources.add((source_path, pdf_page))
        issue_splits[issue_id] = split
        issue_sources[issue_id] = source_path
    return pages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_prepare_annotation(args: argparse.Namespace) -> None:
    """Render a manifest-driven batch with unique CVAT image names."""
    ensure_directories()
    output_root = create_run_directory("annotation_review", args.name)
    plan_path = project_path(args.plan)
    pages = read_annotation_plan(plan_path)
    plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
    scope = plan_document.get("scope", {}) if isinstance(plan_document, dict) else {}
    scope_exclusions = scope.get("exclude") if isinstance(scope, dict) else None
    split_notes = plan_document.get("split_notes", []) if isinstance(plan_document, dict) else []
    if not isinstance(split_notes, list) or not all(isinstance(note, str) for note in split_notes):
        raise SystemExit("Annotation-plan split_notes must be a list of strings.")
    upload_dir = output_root / "cvat_upload"
    selected: list[dict[str, object]] = []
    for planned_page in pages:
        source_path = project_path(str(planned_page["source_pdf"]))
        if not source_path.is_file() or source_path.suffix.lower() != ".pdf":
            raise SystemExit(f"Annotation-plan source PDF does not exist: {source_path}")
        temporary_dir = output_root / "rendered" / source_path.stem
        rendered = render_pdf(source_path, temporary_dir, args.dpi, {int(planned_page["pdf_page"])})
        if len(rendered) != 1:
            raise SystemExit(f"{source_path.name} does not contain PDF page {planned_page['pdf_page']}.")
        # Issue folders make the intended one-task-per-issue CVAT upload unambiguous.
        image_path = upload_dir / source_path.stem / str(planned_page["image_name"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rendered[0]), image_path)
        from PIL import Image
        with Image.open(image_path) as image:
            width, height = image.size
        selected.append({
            **planned_page,
            "image": str(image_path.relative_to(PROJECT_ROOT)),
            "width": width,
            "height": height,
            "sha256": sha256_file(image_path),
            "status": "needs_manual_annotation",
        })
    instructions = "Draw one tight rectangle around each complete published court or authority-issued legal notice. Include attached authority/header, logo, body, border, internal table, and notice-specific footer. Exclude advertisements, editorial content, private name-change announcements, lost-passport notices, lost-share-certificate classifieds, and page-level newspaper furniture. Keep adjacent notices separate."
    if isinstance(scope_exclusions, str) and scope_exclusions:
        instructions += f" Batch-specific exclusions: {scope_exclusions}"
    manifest = {
        "batch_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "class_name": LEGAL_NOTICE_CLASS,
        "instructions": instructions,
        "split_notes": split_notes,
        "pages": selected,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checklist = ["# CVAT Annotation Checklist", "", "Create one CVAT task per source issue, using only that issue's images.", "", "| Image | Split | Role | Review note |", "|---|---|---|---|"]
    checklist.extend(
        f"| `{Path(page['image']).parent.name}/{page['image_name']}` | {page['split']} | {page['role']} | {page.get('reason', '')} |"
        for page in selected
    )
    checklist.extend(["", "Positive pages: draw one box per complete court or authority-issued notice.", "Negative pages: draw no boxes; their presence in the CVAT export is required.", f"Exclusions: {scope_exclusions}" if scope_exclusions else "Exclude private name changes, lost passports, lost-share certificates, advertisements, and editorial content."])
    if split_notes:
        checklist.extend(["", "## Split Notes", "", *(f"- {note}" for note in split_notes)])
    (output_root / "CHECKLIST.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} page(s): {output_root.relative_to(PROJECT_ROOT)}")
    print(f"Upload these uniquely named images: {upload_dir.relative_to(PROJECT_ROOT)}")
    print(f"Review manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


def parse_yolo_labels(label_path: str, content: str) -> list[tuple[float, float, float, float]]:
    """Read one-class YOLO labels and reject invalid rectangles before rendering crops."""
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise SystemExit(f"{label_path}:{line_number} needs 5 values")
        try:
            class_id = int(values[0])
            x_center, y_center, width, height = (float(value) for value in values[1:])
        except ValueError as error:
            raise SystemExit(f"{label_path}:{line_number} contains non-numeric values") from error
        coordinates = (x_center, y_center, width, height)
        if class_id != 0 or not all(math.isfinite(value) for value in coordinates):
            raise SystemExit(f"{label_path}:{line_number} must use class 0 and finite coordinates")
        if width <= 0 or height <= 0 or x_center - width / 2 < 0 or x_center + width / 2 > 1 or y_center - height / 2 < 0 or y_center + height / 2 > 1:
            raise SystemExit(f"{label_path}:{line_number} has an empty or out-of-bounds box")
        boxes.append(coordinates)
    return boxes


def save_annotation_review(image_path: Path, boxes: list[tuple[float, float, float, float]], destination: Path) -> list[dict[str, object]]:
    """Save a numbered overlay and complete-notice crops for manual annotation QA."""
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    crop_dir = destination / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    notices: list[dict[str, object]] = []
    for number, (x_center, y_center, width, height) in enumerate(boxes, start=1):
        x1 = round((x_center - width / 2) * image.width)
        y1 = round((y_center - height / 2) * image.height)
        x2 = round((x_center + width / 2) * image.width)
        y2 = round((y_center + height / 2) * image.height)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=max(3, image.width // 500))
        draw.text((x1 + 4, y1 + 4), str(number), fill="red", stroke_width=1, stroke_fill="white")
        crop_path = crop_dir / f"notice_{number:03d}.png"
        image.crop((x1, y1, x2, y2)).save(crop_path)
        notices.append({
            "id": number,
            "bbox_xyxy": [x1, y1, x2, y2],
            "crop": str(crop_path.relative_to(PROJECT_ROOT)),
        })
    destination.mkdir(parents=True, exist_ok=True)
    overlay.save(destination / "annotated.png")
    return notices


def run_import_cvat(args: argparse.Namespace) -> None:
    """Import a manifest-backed CVAT export and create visual QA artifacts."""
    ensure_directories()
    output_root = create_run_directory("cvat_import", args.name)
    manifest_path = project_path(args.manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid annotation manifest: {manifest_path}") from error
    pages = manifest.get("pages") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or manifest.get("class_name") != LEGAL_NOTICE_CLASS or not isinstance(pages, list) or not pages:
        raise SystemExit("Annotation manifest must contain the legal_notice class and at least one page.")
    planned_by_image: dict[str, dict] = {}
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("image_name"), str):
            raise SystemExit("Annotation manifest has an invalid page entry.")
        image_name = page["image_name"]
        if image_name in planned_by_image:
            raise SystemExit(f"Duplicate image name in annotation manifest: {image_name}")
        if page.get("split") not in {"train", "val", "test"} or page.get("role") not in {"positive", "negative"}:
            raise SystemExit(f"Annotation manifest lacks split or role for: {image_name}")
        planned_by_image[image_name] = page
    report: list[dict[str, object]] = []
    exported: dict[str, tuple[Path, str | None]] = {}
    for archive_value in args.archive:
        archive_path = project_path(archive_value)
        if not archive_path.is_file():
            raise SystemExit(f"CVAT archive does not exist: {archive_path}")
        with ZipFile(archive_path) as archive:
            try:
                names = archive.read("obj.names").decode("utf-8").splitlines()
            except KeyError as error:
                raise SystemExit(f"{archive_path} is not a supported CVAT YOLO export (missing obj.names).") from error
            if names != [LEGAL_NOTICE_CLASS]:
                raise SystemExit(f"{archive_path} must contain only the '{LEGAL_NOTICE_CLASS}' class.")
            try:
                image_entries = [Path(line.strip()).name for line in archive.read("train.txt").decode("utf-8").splitlines() if line.strip()]
            except KeyError as error:
                raise SystemExit(f"{archive_path} is missing train.txt, so reviewed negative pages cannot be verified.") from error
            label_entries = {Path(entry).stem: entry for entry in archive.namelist() if entry.startswith("obj_train_data/") and entry.endswith(".txt")}
            for image_name in image_entries:
                if image_name in exported:
                    raise SystemExit(f"CVAT image appears in more than one archive: {image_name}")
                exported[image_name] = (archive_path, label_entries.get(Path(image_name).stem))
            unexpected_labels = set(label_entries) - {Path(name).stem for name in image_entries}
            if unexpected_labels:
                raise SystemExit(f"CVAT archive has labels without matching images: {sorted(unexpected_labels)[0]}.txt")

    missing = set(planned_by_image) - set(exported)
    unexpected = set(exported) - set(planned_by_image)
    if missing or unexpected:
        detail = f"missing {sorted(missing)[0]}" if missing else f"unexpected {sorted(unexpected)[0]}"
        raise SystemExit(f"CVAT export does not exactly match the annotation manifest: {detail}.")
    for image_name, planned_page in planned_by_image.items():
        archive_path, label_entry = exported[image_name]
        with ZipFile(archive_path) as archive:
            content = archive.read(label_entry).decode("utf-8") if label_entry else ""
        boxes = parse_yolo_labels(f"{archive_path.name}:{label_entry or image_name}", content)
        if planned_page["role"] == "positive" and not boxes:
            raise SystemExit(f"Expected at least one notice on positive page: {image_name}")
        if planned_page["role"] == "negative" and boxes:
            raise SystemExit(f"Negative page contains legal_notice boxes: {image_name}")
        image_path = project_path(str(planned_page.get("image")))
        if not image_path.is_file() or sha256_file(image_path) != planned_page.get("sha256"):
            raise SystemExit(f"Prepared image is missing or changed since manifest creation: {image_name}")
        source_pdf = str(planned_page["source_pdf"])
        page_number = int(planned_page["pdf_page"])
        page_root = output_root / "review" / Path(image_name).stem
        notices = save_annotation_review(image_path, boxes, page_root)
        report.append({
            "cvat_archive": archive_path.name,
            "source_pdf": source_pdf,
            "pdf_page": page_number,
            "image_name": image_name,
            "image": str(image_path.relative_to(PROJECT_ROOT)),
            "split": planned_page["split"],
            "role": planned_page["role"],
            "newspaper": planned_page.get("newspaper"),
            "issue_id": planned_page.get("issue_id"),
            "annotated_image": str((page_root / "annotated.png").relative_to(PROJECT_ROOT)),
            "notice_count": len(notices),
            "notices": notices,
        })

    report_path = output_root / "import_report.json"
    report_path.write_text(json.dumps({"manifest": str(manifest_path.relative_to(PROJECT_ROOT)), "pages": report}, indent=2), encoding="utf-8")
    print(f"Imported {len(report)} annotated page(s) and {sum(page['notice_count'] for page in report)} notice box(es).")
    print(f"Review report: {report_path.relative_to(PROJECT_ROOT)}")


def run_prepare_dataset(args: argparse.Namespace) -> None:
    """Promote approved manifest-backed pages into a separate YOLO dataset."""
    report_path = project_path(args.report)
    if not report_path.is_file():
        raise SystemExit(f"Import report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid import report: {report_path}") from error
    pages = report.get("pages") if isinstance(report, dict) else None
    if not isinstance(pages, list) or not pages:
        raise SystemExit(f"Import report contains no annotated pages: {report_path}")
    dataset_dir = project_path(args.dataset)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise SystemExit(f"Dataset destination already exists and is not empty: {dataset_dir}. Do not overwrite labeled data.")

    staged: list[tuple[dict, str, Path, Path]] = []
    for page in pages:
        source_pdf = page.get("source_pdf")
        image_value = page.get("image")
        notices = page.get("notices")
        page_number = page.get("pdf_page")
        split = page.get("split")
        if not isinstance(source_pdf, str) or not isinstance(image_value, str) or not isinstance(notices, list) or not isinstance(page_number, int) or split not in {"train", "val", "test"}:
            raise SystemExit("Import report has an invalid page entry.")
        image_path = project_path(image_value)
        if not image_path.is_file():
            raise SystemExit(f"Imported source image is missing: {image_path}")
        image_name = f"{Path(source_pdf).stem}_page_{page_number:04d}.png"
        destination_image = dataset_dir / "images" / split / image_name
        destination_label = dataset_dir / "labels" / split / image_name.replace(".png", ".txt")
        if destination_image.exists() or destination_label.exists():
            raise SystemExit(f"Dataset destination already exists: {destination_image}. Do not overwrite labeled data.")
        staged.append((page, split, destination_image, destination_label))

    for page, split, destination_image, destination_label in staged:
        from PIL import Image

        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_path(page["image"]), destination_image)
        with Image.open(destination_image) as image:
            lines = []
            for notice in page["notices"]:
                x1, y1, x2, y2 = notice["bbox_xyxy"]
                x_center = (x1 + x2) / 2 / image.width
                y_center = (y1 + y2) / 2 / image.height
                width = (x2 - x1) / image.width
                height = (y2 - y1) / image.height
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Added {split}: {destination_image.relative_to(PROJECT_ROOT)} ({len(page['notices'])} notices)")

    data_path = dataset_dir / "data.yaml"
    data_path.write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [legal_notice]\n", encoding="utf-8")
    validate_dataset(data_path, ("train", "val", "test"))
    print(f"Dataset preparation complete: {dataset_dir.relative_to(PROJECT_ROOT)}")


def read_dataset_config(data_path: Path) -> dict:
    require_packages("yaml")
    import yaml

    if not data_path.is_file():
        raise SystemExit(f"Dataset configuration does not exist: {data_path}")
    with data_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if config.get("names") != [LEGAL_NOTICE_CLASS]:
        raise SystemExit("dataset/data.yaml must define exactly: names: [legal_notice]")
    return config


def dataset_root(data_path: Path, config: dict) -> Path:
    root = Path(config.get("path", "."))
    return root if root.is_absolute() else (data_path.parent / root).resolve()


def labels_for_split(data_path: Path, split: str, config: dict) -> Iterator[tuple[Path, Path]]:
    split_path = Path(config.get(split, f"images/{split}"))
    root = dataset_root(data_path, config)
    image_dir = split_path if split_path.is_absolute() else root / split_path
    if not image_dir.exists():
        return
    for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in SUPPORTED_IMAGES):
        relative = image_path.relative_to(image_dir)
        label_dir = root / "labels" / split
        yield image_path, label_dir / relative.with_suffix(".txt")


def validate_dataset(data_path: Path, required_splits: tuple[str, ...] = ("train", "val")) -> None:
    """Check YOLO label files before spending CPU time on a training run."""
    config = read_dataset_config(data_path)
    image_count = 0
    label_count = 0
    errors: list[str] = []
    split_counts: dict[str, int] = {}
    for split in required_splits:
        split_count = 0
        for image_path, label_path in labels_for_split(data_path, split, config):
            image_count += 1
            split_count += 1
            try:
                from PIL import Image
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as error:
                errors.append(f"Unreadable image: {image_path} ({error})")
                continue
            if not label_path.is_file():
                errors.append(f"Missing label file: {label_path} (use an empty file for a negative page)")
                continue
            label_count += 1
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                values = line.split()
                if len(values) != 5:
                    errors.append(f"{label_path}:{line_number} needs 5 values")
                    continue
                try:
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number} contains non-numeric values")
                    continue
                x_center, y_center, width, height = coordinates
                if class_id != 0 or not all(math.isfinite(value) for value in coordinates):
                    errors.append(f"{label_path}:{line_number} must use class 0 and finite coordinates")
                elif width <= 0 or height <= 0 or x_center - width / 2 < 0 or x_center + width / 2 > 1 or y_center - height / 2 < 0 or y_center + height / 2 > 1:
                    errors.append(f"{label_path}:{line_number} has an empty or out-of-bounds box")
        split_counts[split] = split_count
        if split_count == 0:
            errors.append(f"No images found for required split '{split}'.")
    if errors:
        message = "\n".join(f"- {error}" for error in errors[:20])
        extra = "\n- More errors omitted." if len(errors) > 20 else ""
        raise SystemExit(f"Dataset validation failed:\n{message}{extra}")
    summary = ", ".join(f"{split}: {count}" for split, count in split_counts.items())
    print(f"Dataset validation passed: {image_count} images, {label_count} label files ({summary}).")


def run_validate_dataset(args: argparse.Namespace) -> None:
    validate_dataset(project_path(args.data))


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Return IoU for normalized center-width-height boxes."""
    first_x1, first_y1 = first[0] - first[2] / 2, first[1] - first[3] / 2
    first_x2, first_y2 = first[0] + first[2] / 2, first[1] + first[3] / 2
    second_x1, second_y1 = second[0] - second[2] / 2, second[1] - second[3] / 2
    second_x2, second_y2 = second[0] + second[2] / 2, second[1] + second[3] / 2
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union else 0.0


def match_boxes(predictions: list[tuple[float, float, float, float]], targets: list[tuple[float, float, float, float]]) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match highest-IoU predictions and targets once each."""
    candidates = sorted(
        ((box_iou(prediction, target), prediction_index, target_index)
         for prediction_index, prediction in enumerate(predictions)
         for target_index, target in enumerate(targets)),
        reverse=True,
    )
    matches: list[tuple[int, int, float]] = []
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    for iou, prediction_index, target_index in candidates:
        if iou < 0.5:
            break
        if prediction_index not in matched_predictions and target_index not in matched_targets:
            matches.append((prediction_index, target_index, iou))
            matched_predictions.add(prediction_index)
            matched_targets.add(target_index)
    return matches, [index for index in range(len(predictions)) if index not in matched_predictions], [index for index in range(len(targets)) if index not in matched_targets]


def threshold_values(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise SystemExit("--thresholds must be comma-separated numbers between 0 and 1.") from error
    if not values or any(value <= 0 or value >= 1 for value in values):
        raise SystemExit("--thresholds must be comma-separated numbers between 0 and 1.")
    return sorted(set(values))


def run_review_validation(args: argparse.Namespace) -> None:
    """Score and render only validation pages across candidate confidence thresholds."""
    from PIL import Image, ImageDraw

    data_path = project_path(args.data)
    validate_dataset(data_path)
    config = read_dataset_config(data_path)
    model = load_yolo(project_path(args.weights))
    require_legal_notice_model(model)
    pages = list(labels_for_split(data_path, "val", config))
    if not pages:
        raise SystemExit("No validation pages found.")
    thresholds = threshold_values(args.thresholds)
    targets_by_image = {
        image_path: parse_yolo_labels(str(label_path), label_path.read_text(encoding="utf-8"))
        for image_path, label_path in pages
    }
    results = model.predict([str(image_path) for image_path, _ in pages], conf=min(thresholds), imgsz=args.image_size, device="cpu", verbose=False)
    minimum_threshold_predictions = {
        image_path: [(*box.xywhn[0].tolist(), float(box.conf[0])) for box in result.boxes]
        for (image_path, _), result in zip(pages, results)
    }
    predictions_by_threshold = {
        threshold: {
            image_path: [box for box in predictions if box[4] >= threshold]
            for image_path, predictions in minimum_threshold_predictions.items()
        }
        for threshold in thresholds
    }

    summaries: dict[str, dict[str, float | int]] = {}
    for threshold, by_image in predictions_by_threshold.items():
        true_positive = false_positive = false_negative = boundary_issues = 0
        threshold_newspapers: dict[str, dict[str, int]] = {}
        for image_path, _ in pages:
            predictions = [box[:4] for box in by_image[image_path]]
            matches, unmatched_predictions, unmatched_targets = match_boxes(predictions, targets_by_image[image_path])
            true_positive += len(matches)
            false_positive += len(unmatched_predictions)
            false_negative += len(unmatched_targets)
            boundary_issues += sum(iou < 0.75 for _, _, iou in matches)
            newspaper = image_path.name.split("_")[0]
            newspaper_totals = threshold_newspapers.setdefault(newspaper, {"pages": 0, "targets": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0, "boundary_issues": 0})
            newspaper_totals["pages"] += 1
            newspaper_totals["targets"] += len(targets_by_image[image_path])
            newspaper_totals["true_positives"] += len(matches)
            newspaper_totals["false_positives"] += len(unmatched_predictions)
            newspaper_totals["false_negatives"] += len(unmatched_targets)
            newspaper_totals["boundary_issues"] += sum(iou < 0.75 for _, _, iou in matches)
        newspaper_metrics: dict[str, dict[str, float | int]] = {}
        for newspaper, totals in threshold_newspapers.items():
            newspaper_precision = totals["true_positives"] / (totals["true_positives"] + totals["false_positives"]) if totals["true_positives"] + totals["false_positives"] else 0.0
            newspaper_recall = totals["true_positives"] / (totals["true_positives"] + totals["false_negatives"]) if totals["true_positives"] + totals["false_negatives"] else 0.0
            newspaper_f1 = 2 * newspaper_precision * newspaper_recall / (newspaper_precision + newspaper_recall) if newspaper_precision + newspaper_recall else 0.0
            newspaper_metrics[newspaper] = {**totals, "precision": round(newspaper_precision, 4), "recall": round(newspaper_recall, 4), "f1": round(newspaper_f1, 4)}
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_f1 = sum(metrics["f1"] for metrics in newspaper_metrics.values()) / len(newspaper_metrics)
        summaries[f"{threshold:.3f}"] = {"threshold": threshold, "true_positives": true_positive, "false_positives": false_positive, "false_negatives": false_negative, "boundary_issues": boundary_issues, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "macro_f1": round(macro_f1, 4), "newspapers": newspaper_metrics}
    recommended_key = max(summaries, key=lambda key: (summaries[key]["f1"], summaries[key]["precision"]))
    recommended_threshold = float(summaries[recommended_key]["threshold"])
    recommended_macro_key = max(summaries, key=lambda key: (summaries[key]["macro_f1"], summaries[key]["precision"]))
    recommended_macro_threshold = float(summaries[recommended_macro_key]["threshold"])

    output_root = create_run_directory("validation_review", args.name)
    pages_report: list[dict[str, object]] = []
    newspapers: dict[str, dict[str, int]] = {}
    for image_path, _ in pages:
        targets = targets_by_image[image_path]
        predictions_with_confidence = predictions_by_threshold[recommended_threshold][image_path]
        predictions = [box[:4] for box in predictions_with_confidence]
        matches, unmatched_predictions, unmatched_targets = match_boxes(predictions, targets)
        newspaper = image_path.name.split("_")[0]
        totals = newspapers.setdefault(newspaper, {"pages": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0, "boundary_issues": 0})
        totals["pages"] += 1
        totals["true_positives"] += len(matches)
        totals["false_positives"] += len(unmatched_predictions)
        totals["false_negatives"] += len(unmatched_targets)
        totals["boundary_issues"] += sum(iou < 0.75 for _, _, iou in matches)
        image = Image.open(image_path).convert("RGB")
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)

        def draw_box(box: tuple[float, float, float, float], color: str) -> None:
            x_center, y_center, width, height = box
            draw.rectangle(((x_center - width / 2) * image.width, (y_center - height / 2) * image.height, (x_center + width / 2) * image.width, (y_center + height / 2) * image.height), outline=color, width=max(3, image.width // 500))

        for prediction_index, target_index, iou in matches:
            draw_box(predictions[prediction_index], "blue" if iou >= 0.75 else "orange")
        for prediction_index in unmatched_predictions:
            draw_box(predictions[prediction_index], "red")
        for target_index in unmatched_targets:
            draw_box(targets[target_index], "magenta")
        destination = output_root / "pages" / newspaper / image_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(destination)
        pages_report.append({"image": str(image_path.relative_to(PROJECT_ROOT)), "newspaper": newspaper, "overlay": str(destination.relative_to(PROJECT_ROOT)), "targets": len(targets), "predictions": len(predictions), "matches": len(matches), "false_positive_boxes": [predictions_with_confidence[index] for index in unmatched_predictions], "missed_target_boxes": [targets[index] for index in unmatched_targets], "boundary_issue_matches": [{"prediction": prediction_index, "target": target_index, "iou": round(iou, 4)} for prediction_index, target_index, iou in matches if iou < 0.75]})
    report = {"weights": str(project_path(args.weights).relative_to(PROJECT_ROOT)), "split": "val", "matching_iou": 0.5, "boundary_issue_iou_below": 0.75, "recommended_threshold": recommended_threshold, "recommended_macro_threshold": recommended_macro_threshold, "thresholds": summaries, "newspapers": newspapers, "pages": pages_report, "legend": {"blue": "matched box, IoU >= 0.75", "orange": "matched box with boundary issue", "red": "false-positive prediction", "magenta": "missed ground-truth notice"}}
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation review complete: {output_root.relative_to(PROJECT_ROOT)}")
    print(f"Recommended confidence threshold: {recommended_threshold:.3f}")
    print(f"Recommended macro confidence threshold: {recommended_macro_threshold:.3f}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


def run_train(args: argparse.Namespace) -> None:
    data_path = project_path(args.data)
    validate_dataset(data_path)
    weights = project_path(args.weights) if args.weights else layout_model_path()
    run_dir = RUNS_DIR / args.name
    if run_dir.exists():
        raise SystemExit(f"Training run already exists: {run_dir}. Use a new --name to preserve prior checkpoints.")
    if args.learning_rate <= 0 or not 0 <= args.momentum <= 1 or args.warmup_bias_lr < 0:
        raise SystemExit("Training rates must use --learning-rate > 0, --momentum between 0 and 1, and --warmup-bias-lr >= 0.")
    if any(value < 0 for value in (args.mosaic, args.scale, args.translate, args.fliplr)) or args.mosaic > 1 or args.translate > 1 or args.fliplr > 1:
        raise SystemExit("Augmentation probabilities must be non-negative; mosaic, translate, and fliplr cannot exceed 1.")
    if args.close_mosaic < 0 or args.save_period == 0 or args.save_period < -1:
        raise SystemExit("Use --close-mosaic >= 0 and --save-period -1 or a positive epoch interval.")
    model = load_yolo(weights)
    results = model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.image_size, batch=args.batch,
        device="cpu", workers=0, project=str(RUNS_DIR), name=args.name, exist_ok=False,
        optimizer=args.optimizer, lr0=args.learning_rate, momentum=args.momentum,
        warmup_bias_lr=args.warmup_bias_lr, mosaic=args.mosaic, scale=args.scale,
        translate=args.translate, fliplr=args.fliplr, close_mosaic=args.close_mosaic,
        save_period=args.save_period,
    )
    print(f"Training complete. Results: {results.save_dir}")
    print("Use the generated weights/best.pt with the detect command.")


def run_evaluate(args: argparse.Namespace) -> None:
    data_path = project_path(args.data)
    validate_dataset(data_path, (args.split,))
    model = load_yolo(project_path(args.weights))
    require_legal_notice_model(model)
    metrics = model.val(data=str(data_path), split=args.split, imgsz=args.image_size, batch=args.batch, device="cpu", workers=0)
    print(f"Evaluation complete. mAP50-95: {metrics.box.map:.4f}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Create .venv and install CPU-only dependencies").set_defaults(func=run_setup)
    commands.add_parser("download-model", help="Download the supplied medium layout model locally").set_defaults(func=download_layout_model)

    render = commands.add_parser("render", help="Render input PDF pages to output/rendered")
    render.add_argument("--input", default="input", help="PDF file or directory (default: input)")
    render.add_argument("--dpi", type=int, default=200, help="PDF render DPI (default: 200)")
    render.add_argument("--name", help="Optional unique output run name")
    render.set_defaults(func=run_render)

    preview = commands.add_parser("preview-layout", help="Preview generic document-layout detections")
    preview.add_argument("--input", default="input")
    preview.add_argument("--dpi", type=int, default=200)
    preview.add_argument("--image-size", type=int, default=1280)
    preview.add_argument("--confidence", type=float, default=0.25)
    preview.add_argument("--name", help="Optional unique output run name")
    preview.set_defaults(func=run_preview_layout)

    detect = commands.add_parser("detect", help="Detect legal notices using a custom trained model")
    detect.add_argument("--input", default="input")
    detect.add_argument("--weights", required=True, help="Custom legal_notice checkpoint")
    detect.add_argument("--dpi", type=int, default=200)
    detect.add_argument("--image-size", type=int, default=1280)
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--name", help="Optional unique output run name")
    detect.set_defaults(func=run_detect)

    prepare = commands.add_parser("prepare-annotation", help="Render a manifest-driven CVAT annotation batch")
    prepare.add_argument("--plan", required=True, help="JSON batch plan with source pages, roles, and splits")
    prepare.add_argument("--dpi", type=int, default=200)
    prepare.add_argument("--name", help="Optional unique output run name")
    prepare.set_defaults(func=run_prepare_annotation)

    cvat_import = commands.add_parser("import-cvat", help="Import manifest-backed CVAT YOLO exports and generate review overlays")
    cvat_import.add_argument("--manifest", required=True, help="manifest.json created by prepare-annotation")
    cvat_import.add_argument("--archive", action="append", required=True, help="CVAT YOLO annotation ZIP; repeat for separate tasks")
    cvat_import.add_argument("--name", help="Optional unique output run name")
    cvat_import.set_defaults(func=run_import_cvat)

    prepare_dataset = commands.add_parser("prepare-dataset", help="Copy approved CVAT review pages into a separate YOLO dataset")
    prepare_dataset.add_argument("--report", required=True, help="import_report.json from import-cvat")
    prepare_dataset.add_argument("--dataset", default="dataset_v2", help="Empty destination dataset directory (default: dataset_v2)")
    prepare_dataset.set_defaults(func=run_prepare_dataset)

    validate = commands.add_parser("validate-dataset", help="Validate YOLO labels before training")
    validate.add_argument("--data", default="dataset/data.yaml")
    validate.set_defaults(func=run_validate_dataset)

    review = commands.add_parser("review-validation", help="Render validation-only predictions and threshold error reports")
    review.add_argument("--data", default="dataset_v2/data.yaml")
    review.add_argument("--weights", required=True)
    review.add_argument("--image-size", type=int, default=1280)
    review.add_argument("--thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3")
    review.add_argument("--name", help="Optional unique output run name")
    review.set_defaults(func=run_review_validation)

    train = commands.add_parser("train", help="Fine-tune a legal-notice model using CPU")
    train.add_argument("--data", default="dataset/data.yaml")
    train.add_argument("--weights", help="Optional starting checkpoint; defaults to medium layout model")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--image-size", type=int, default=1280)
    train.add_argument("--batch", type=int, default=1)
    train.add_argument("--name", default="legal_notice")
    train.add_argument("--optimizer", default="auto", choices=("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"))
    train.add_argument("--learning-rate", type=float, default=0.01, help="Initial optimizer learning rate (default: 0.01)")
    train.add_argument("--momentum", type=float, default=0.937, help="SGD momentum or Adam beta1 (default: 0.937)")
    train.add_argument("--warmup-bias-lr", type=float, default=0.1, help="Initial bias learning rate during warmup (default: 0.1)")
    train.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability (default: 1.0)")
    train.add_argument("--scale", type=float, default=0.5, help="Random scale gain (default: 0.5)")
    train.add_argument("--translate", type=float, default=0.1, help="Random translation fraction (default: 0.1)")
    train.add_argument("--fliplr", type=float, default=0.5, help="Horizontal-flip probability (default: 0.5)")
    train.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for the final N epochs (default: 10)")
    train.add_argument("--save-period", type=int, default=-1, help="Save a checkpoint every N epochs; -1 disables periodic saves")
    train.set_defaults(func=run_train)

    evaluate = commands.add_parser("evaluate", help="Evaluate a custom legal-notice checkpoint using CPU")
    evaluate.add_argument("--data", default="dataset/data.yaml")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--image-size", type=int, default=1280)
    evaluate.add_argument("--batch", type=int, default=1)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.set_defaults(func=run_evaluate)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
