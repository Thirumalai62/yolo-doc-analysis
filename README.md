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

Rendered pages are saved in `output/rendered/`; previews are saved in
`output/layout_preview/`. Layout boxes represent text, titles, pictures and
similar areas. They are not legal-notice results.

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

Each image has a matching `.txt` label file. Each notice uses class `0`:

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
.\.venv\Scripts\python.exe main.py evaluate --weights runs/legal_notice/weights/best.pt
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

The command saves annotated pages, an image crop per notice, and
`output/legal_notices/detections.json`. All inference uses CPU.

## License Note

Ultralytics publishes YOLO under AGPL-3.0 with a separate enterprise license.
Review its terms before deploying this in a business product or workflow.
