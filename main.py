"""CPU-only workflow for detecting legal notices in UAE e-paper PDFs.

Run `py main.py --help` to see the available commands. Start with
`py main.py setup`, then put PDF files in `input/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import subprocess
import sys
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


def run_prepare_annotation(args: argparse.Namespace) -> None:
    """Render selected pages and record them for manual whole-notice annotation."""
    ensure_directories()
    output_root = create_run_directory("annotation_review", args.name)
    rendered_root = output_root / "pages"
    selections = parse_page_selection(args.include)
    pages = render_input_pdfs(project_path(args.input), args.dpi, rendered_root, selections or None)
    selected: list[dict[str, object]] = []
    for page in pages:
        source_pdf = page.parent.name + ".pdf"
        page_number = int(page.stem.removeprefix("page_"))
        if selections and page_number not in selections.get(source_pdf, set()):
            continue
        selected.append({
            "source_pdf": source_pdf,
            "pdf_page": page_number,
            "image": str(page.relative_to(PROJECT_ROOT)),
            "status": "needs_manual_annotation",
        })
    if selections and not selected:
        raise SystemExit("No rendered pages matched --include. Check the PDF filenames and page numbers.")
    manifest = {
        "class_name": LEGAL_NOTICE_CLASS,
        "instructions": "Draw one tight rectangle around each complete published legal notice. Include attached authority/header, logo, body, border, and notice-specific footer. Exclude advertisements, editorial content, and page-level newspaper furniture.",
        "pages": selected,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(selected)} page(s): {output_root.relative_to(PROJECT_ROOT)}")
    print(f"Review manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


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


def run_train(args: argparse.Namespace) -> None:
    data_path = project_path(args.data)
    validate_dataset(data_path)
    weights = project_path(args.weights) if args.weights else layout_model_path()
    model = load_yolo(weights)
    results = model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.image_size, batch=args.batch,
        device="cpu", workers=0, project=str(RUNS_DIR), name=args.name, exist_ok=True,
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

    prepare = commands.add_parser("prepare-annotation", help="Render pages and create a manual-annotation review manifest")
    prepare.add_argument("--input", default="input")
    prepare.add_argument("--dpi", type=int, default=200)
    prepare.add_argument("--include", action="append", help="Repeat FILE.pdf:PAGE,PAGE to select pages; default: all pages")
    prepare.add_argument("--name", help="Optional unique output run name")
    prepare.set_defaults(func=run_prepare_annotation)

    validate = commands.add_parser("validate-dataset", help="Validate YOLO labels before training")
    validate.add_argument("--data", default="dataset/data.yaml")
    validate.set_defaults(func=run_validate_dataset)

    train = commands.add_parser("train", help="Fine-tune a legal-notice model using CPU")
    train.add_argument("--data", default="dataset/data.yaml")
    train.add_argument("--weights", help="Optional starting checkpoint; defaults to medium layout model")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--image-size", type=int, default=1280)
    train.add_argument("--batch", type=int, default=1)
    train.add_argument("--name", default="legal_notice")
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
