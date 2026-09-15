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

For full-page document fine-tuning, explicitly reduce augmentation that can
mirror or clip large notices and save intermediate checkpoints:

```powershell
.\.venv\Scripts\python.exe main.py train --data dataset_v3/data.yaml --weights runs/legal_notice_v2_experiment_50/weights/best.pt --epochs 50 --batch 1 --optimizer AdamW --learning-rate 0.0005 --momentum 0.9 --warmup-bias-lr 0 --mosaic 0 --scale 0.1 --translate 0.02 --fliplr 0 --close-mosaic 0 --save-period 5 --name legal_notice_v3_controlled_50
```

Training refuses to reuse an existing run name so prior checkpoints are not
overwritten.

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
.\.venv\Scripts\python.exe main.py review-validation --data "dataset_v3/data.yaml" --weights "runs/legal_notice_v3_controlled_50/weights/best.pt" --image-size 1280 --thresholds "0.05,0.10,0.15,0.20,0.25,0.30" --name "v3_validation_review_repeat"
```

Use the report's recommended threshold only after reviewing its false positives
and misses. The JSON includes per-newspaper metrics at every threshold plus a
macro recommendation that gives each newspaper equal weight. Each page is
inferred separately at the minimum requested threshold to match `detect`;
higher thresholds filter those predictions. This command uses `val` only; it
does not load or evaluate `test`.
Every named output run must be unique, so change `--name` when repeating it.

## Fixed-0.80 Corrective-Candidate Gate

The corrective-candidate workflow uses a reviewed preservation suite from the
complete September 9-10 issues. It contains 271 valid v3 references, 17
definite exclusions, two non-gating uncertain references, and four strict Al
Bayan table-boundary checks. This is a preservation suite, not exhaustive page
ground truth.

Create the corrected dataset once. The command copies `dataset_v4`, verifies
the exact four reviewed labels before removing them, deletes stale YOLO caches,
and refuses to overwrite an existing destination:

```powershell
.\.venv\Scripts\python.exe regression/prepare_corrected_dataset.py
.\.venv\Scripts\python.exe main.py validate-dataset --data dataset_v5_corrected/data.yaml
.\.venv\Scripts\python.exe regression/prepare_corrected_dataset.py --verify-existing
```

`dataset_v5_corrected` contains 276 train boxes, 175 validation boxes, and the
unchanged 281-box test split. Do not evaluate the test split while developing
the next candidate.

The reviewed manifest is `regression/fixed080_manifest.json`. Check a future
checkpoint with one-page inference at the mandatory confidence `0.80`:

```powershell
.\.venv\Scripts\python.exe regression/fixed080_acceptance.py check --weights "runs/<candidate>/weights/best.pt" --output "output/regression_audit/<candidate>_fixed080.json" --enforce
```

Strict acceptance requires zero valid-reference losses, zero accepted known
exclusions, zero critical boundary failures, and zero unreviewed new accepted
boxes. Uncertain references are reported but do not affect the result. Review
any unmatched new box before adding it to the manifest.

### Monitored Corrective Pilot

The pinned pilot configuration is `regression/corrective_pilot_config.json`.
Running the monitor without an authorization flag performs preflight only and
does not create a training run:

```powershell
.\.venv\Scripts\python.exe regression/monitored_corrective_pilot.py
```

Preflight verifies the corrected dataset and untouched test split hashes, the
v3 starting checkpoint, the fresh v3 acceptance report, the regression
manifest, Ultralytics `8.4.146`, and zero image overlap between the development
dataset and the 172-page preservation suite.

After explicit approval, the same tool launches the pinned three-epoch pilot:

```powershell
.\.venv\Scripts\python.exe regression/monitored_corrective_pilot.py --start-training
```

The trainer saves `epoch0.pt`, `epoch1.pt`, and `epoch2.pt`. Its synchronous
`on_model_save` callback pauses training after each epoch and runs the full
fixed-`0.80` suite. It stops before the next epoch if a valid reference is
missed, a critical boundary fails, an unreviewed accepted box appears, or
prediction provenance fails. Known exclusions remain a final promotion gate
but do not stop an intermediate epoch while the pilot is still learning.

Per-epoch reports, prediction caches, and `monitor_summary.json` are written to
`output/regression_audit/legal_notice_v5_corrective_3_monitor/`. The process
returns a failure status if preservation fails or if no epoch passes every
promotion gate.

The prepared pilot was run on September 12, 2026. Epoch 1 preserved all 271
valid references and four critical boundaries but retained 15 of 17 known
exclusions. Epoch 2 lost one valid reference and failed one boundary check, so
the monitor stopped before epoch 3. No checkpoint passed promotion. See
`output/regression_audit/legal_notice_v5_corrective_3_monitor/PILOT_FINDINGS.md`;
do not rerun the same named configuration or use its `best.pt` in production.

## V3 New-Date Testing Reference

The validation-selected v3 candidate and its frozen evaluation settings are:

```text
Weights:    runs/legal_notice_v3_controlled_50/weights/best.pt
Confidence: 0.80
Image size: 1280
Device:     CPU
Render DPI: 200
```

Keep the v2 checkpoint operational until this v3 candidate passes the held-out
test and new-date review. Use PDFs from dates absent from the training,
validation, and held-out test splits. A different publication date is not
necessarily independent if it repeats the same notice from an earlier issue.

### Arrange New PDFs

The input can be one PDF or a directory. Directories are searched recursively,
so a convenient structure for all four sources is:

```text
input/new_dates/
  albayan/
  alfajr/
  gulftoday/
  khaleejtimes/
```

Put complete issues in these folders rather than selecting only expected notice
pages. Complete issues also test rejection of editorial pages, ordinary ads,
name changes, and lost-document notices.

### Required Acceptance Confidence: 0.80

Run this first for normal new-date testing:

```powershell
.\.venv\Scripts\python.exe main.py detect --input "input/new_dates" --weights "runs/legal_notice_v3_controlled_50/weights/best.pt" --dpi 200 --image-size 1280 --confidence 0.80 --name "v3_new_dates_conf080"
```

To process only one PDF, replace the input directory with its path:

```powershell
.\.venv\Scripts\python.exe main.py detect --input "input/new_dates/alfajr/alfajr_2026-09-10.pdf" --weights "runs/legal_notice_v3_controlled_50/weights/best.pt" --dpi 200 --image-size 1280 --confidence 0.80 --name "v3_alfajr_20260910_conf080"
```

Lower-confidence predictions may be generated for diagnosis, but they do not
satisfy the fixed-`0.80` acceptance contract. Do not tune the threshold on the
held-out test split.

### Detection Outputs

Each run is saved under `output/legal_notices/<name>/` and contains:

- `rendered/`: PDF pages rendered at the requested DPI.
- `pages/`: an annotated image and notice crops for each rendered page.
- `detections.json`: confidence scores, pixel bounding boxes, and crop paths.

The command refuses to overwrite an existing named run. Change `--name` for
each date, source, confidence, or repeat. Omitting `--name` creates a unique
timestamped run automatically.

### Manual Review Checklist

For each newspaper and date, review the annotated pages and crops for:

- Missed legal notices, especially small or low-contrast notices.
- False positives on editorial content, ordinary ads, private name changes,
  lost documents, passports, share certificates, recalls, and tenders.
- Boxes that omit headings, tables, signatures, or footer text.
- Adjacent notices incorrectly merged into one box.
- One complete notice incorrectly split into several boxes.
- Large Al Bayan panels that are fragmented or overlap another panel.
- Repeated notices that also appear in training or evaluation issues.

Normal detection on unannotated PDFs is useful for visual acceptance testing,
but it cannot calculate trustworthy precision and recall. Fully annotate a
separate representative batch when numerical new-date metrics are required.

### Held-Out Test

After the model, confidence, and image size are frozen, evaluate the existing
untouched `dataset_v3` test split once:

```powershell
.\.venv\Scripts\python.exe main.py evaluate --data "dataset_v3/data.yaml" --weights "runs/legal_notice_v3_controlled_50/weights/best.pt" --image-size 1280 --batch 1 --split test
```

Record this result as the final held-out measurement. If the model or threshold
is changed in response to the result, the same split is no longer an independent
test for the changed system; use a new held-out set. `evaluate` reports standard
YOLO metrics across its confidence curve; use the frozen `0.80` threshold for
the separate operational PDF review described above.

## Detect Legal Notices

Only a checkpoint trained with exactly the `legal_notice` class can run this
command. The general form is:

```powershell
.\.venv\Scripts\python.exe main.py detect --input input --weights runs/legal_notice/weights/best.pt --confidence 0.25
```

The command saves annotated pages, an image crop per notice, and a
`detections.json` report in its isolated run directory. All inference uses CPU.

## License Note

Ultralytics publishes YOLO under AGPL-3.0 with a separate enterprise license.
Review its terms before deploying this in a business product or workflow.
