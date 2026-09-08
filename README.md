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

Create review assets from a versioned JSON batch plan before labeling. The plan
assigns each page a split and a positive or negative review role. The command
uses unique issue-prefixed filenames, so one CVAT export can safely contain
multiple issues.

```powershell
 .\.venv\Scripts\python.exe main.py prepare-annotation --plan annotation_batches/legal_notice_v2.json --name legal_notice_v2
```

Upload the images from the generated `cvat_upload/` folder. The generated
`CHECKLIST.md` lists every frame, assigned split, role, and review note.
Create one CVAT task per source issue. A negative page must remain in the task
with no shapes; do not omit it from the export.

## Import CVAT Labels

CVAT Online may limit exports that include images. Export annotations in YOLO
format without images, then create local review overlays and crops from the
prepared batch manifest:

```powershell
 .\.venv\Scripts\python.exe main.py import-cvat --manifest output/annotation_review/legal_notice_v2/manifest.json --archive input/cvat_alfajr_2026-09-02.zip --archive input/cvat_gulftoday_2026-09-02.zip --name legal_notice_v2
```

It verifies that every manifest image appears exactly once across the exports,
the sole class is `legal_notice`, positive pages contain boxes, and negative
pages contain none. It also verifies the prepared image hashes before saving
numbered overlays, notice crops, and an `import_report.json`.

After approving the overlays, create the manifest-defined train, validation,
and test splits in a new destination. The default `dataset_v2/` protects the
existing smoke-test dataset from being overwritten:

```powershell
 .\.venv\Scripts\python.exe main.py prepare-dataset --report output/cvat_import/legal_notice_v2/import_report.json --dataset dataset_v2
```

The command refuses non-empty destinations, creates explicit empty `.txt`
files for approved negative pages, and validates train, validation, and test.
It reconstructs labels from the reviewed pixel boxes, so use it only after
annotation approval.

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

## Review Validation Predictions

Before evaluating the held-out test split, render every validation page with a
candidate checkpoint and compare confidence thresholds. The command writes
full-resolution overlays and a JSON report. Blue boxes are well-matched,
orange boxes have a boundary issue, red boxes are false positives, and magenta
boxes are missed approved notices.

```powershell
.\.venv\Scripts\python.exe main.py review-validation --data dataset_v2/data.yaml --weights runs/legal_notice_v2_experiment_50/weights/best.pt --name legal_notice_v2_experiment_50
```

Use the report's recommended threshold only after reviewing its false positives
and misses. This command uses `val` only; it does not load or evaluate `test`.

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
