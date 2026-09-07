# UAE E-Paper Legal Notice Detection

This project finds complete legal-notice blocks in UAE e-paper PDFs. Arabic,
English, and bilingual notices all use one class: `legal_notice`.

The supplied medium YOLO document-layout model is useful for viewing page
regions, but it does **not** know what a legal notice is. Train the custom
model with your labeled newspaper pages before using `detect`.

## Setup

Python 3.10 or later is required. From the project root in PowerShell:

```powershell
py main.py setup
```

The command creates `.venv` and installs CPU-only PyTorch and the application
dependencies. It may take several minutes on the first run.

## First Look At A PDF

1. Put one or more PDFs in `input/`.
2. Download the medium document-layout model into `models/`:

```powershell
.\.venv\Scripts\python.exe main.py download-model
```

3. Render pages. A 200 DPI render is a sensible starting point:

```powershell
.\.venv\Scripts\python.exe main.py render --input input
```

4. Create a generic layout preview:

```powershell
.\.venv\Scripts\python.exe main.py preview-layout --input input
```

Each command saves an isolated timestamped run under `output/`, so prior
artifacts are not overwritten. Layout boxes represent text, titles, pictures
and similar areas. They are not legal-notice results.

## Prepare Annotation Review

Create review assets for selected PDF pages before labeling. This creates
rendered originals plus a manifest with the complete-notice boundary rules; it
does not generate speculative boxes.

```powershell
.\.venv\Scripts\python.exe main.py prepare-annotation --input input --include gulftoday_2026-09-01.pdf:12,13 --include alfajr_2026-09-01.pdf:4,5,6
```

Omit `--include` to prepare every page. The generated `manifest.json` lists
the source PDF and page number for each page sent to annotation.

## Import CVAT Labels

CVAT Online may limit exports that include images. Export annotations in YOLO
format without images, retain the original PDF, then create local review
overlays and crops by pairing each ZIP with its source PDF:

```powershell
.\.venv\Scripts\python.exe main.py import-cvat --archive input/cvat_gulf_page_12.zip --source input/gulftoday_2026-09-01.pdf --archive input/cvat_alfajr_pages.zip --source input/alfajr_2026-09-01.pdf
```

The command expects one `--source` for every `--archive`, in the same order.
It verifies the sole class is `legal_notice`, maps `page_0012.txt` to PDF page
12, renders the source pages, and saves numbered overlays, notice crops, and
an `import_report.json` under `output/cvat_import/`.

After approving the overlays, create splits by assigning whole source issues to
training and leaving other issues for validation:

```powershell
.\.venv\Scripts\python.exe main.py prepare-dataset --report output/cvat_import/initial_cvat_review/import_report.json --train-source input/alfajr_2026-09-01.pdf
```

The command refuses to overwrite existing dataset files and validates the
result. It reconstructs labels from the reviewed pixel boxes, so use it only
after annotation approval.

## Label Training Data

Use an annotation tool that can export **YOLO detection** labels, such as
CVAT, Label Studio, or Roboflow. Draw one rectangle around each complete legal
notice, not around individual text lines.

For the example screenshot, annotate the eight legal notices and do not
annotate the Gulf Today advertisement. Include Arabic, English, bilingual,
small, large, bordered and unbordered notices. Also include pages with normal
advertisements but no legal notices.

Place exported files in this exact structure:

```text
dataset/
  images/train/issue_001_page_0001.png
  images/val/issue_010_page_0001.png
  images/test/issue_020_page_0001.png
  labels/train/issue_001_page_0001.txt
  labels/val/issue_010_page_0001.txt
  labels/test/issue_020_page_0001.txt
```

Each image has a matching `.txt` label file. A page with no legal notices must
still have an empty matching `.txt` file. Each notice uses class `0`:

```text
0 center_x center_y width height
```

Coordinates are normalized between `0` and `1`. Keep all pages from one issue
in only one split. This prevents nearly identical notices appearing in both
training and evaluation data.

## Train And Evaluate

Check the dataset first:

```powershell
.\.venv\Scripts\python.exe main.py validate-dataset
```

Fine-tune the local medium model on CPU:

```powershell
.\.venv\Scripts\python.exe main.py train --epochs 50 --batch 1
```

CPU training with 1280 pixel pages is slow. Start with a small labeled set to
confirm the process, then increase data and epochs. The best checkpoint is
normally written to `runs/legal_notice/weights/best.pt`.

```powershell
.\.venv\Scripts\python.exe main.py evaluate --weights runs/legal_notice/weights/best.pt --split test
```

Review precision, recall, and missed notices on the held-out test issues. A
zero prediction means the model found no notice; it does not prove the page
contains none.

## Detect Legal Notices

Only a checkpoint trained with exactly the `legal_notice` class can run this
command:

```powershell
.\.venv\Scripts\python.exe main.py detect --input input --weights runs/legal_notice/weights/best.pt
```

The command saves annotated pages, an image crop per notice, and a
`detections.json` report in its isolated run directory. All inference uses CPU.

## License Note

Ultralytics publishes YOLO under AGPL-3.0 with a separate enterprise license.
Review its terms before deploying this in a business product or workflow.
