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

`--input` also accepts a direct HTTP or HTTPS PDF URL, including a signed URL.
Quote the URL so shell characters in its query string are preserved:

```powershell
.\.venv\Scripts\python.exe main.py detect --input "https://example.com/newspaper.pdf?token=..." --weights runs/legal_notice/weights/best.pt --confidence 0.25
```

For URL input, the source PDF is downloaded into memory and passed directly to
the PDF renderer. The source PDF is not saved locally. Rendered page images and
the normal detection artifacts are still written to the isolated output run.
Local PDF and directory inputs continue to use the existing behavior.

The command saves annotated pages, an image crop per notice, and a
`detections.json` report in its isolated run directory. All inference uses CPU.

## Run R7 Locally With Docker

This workflow runs the inference-only `doc_detector` package in the same custom
Linux image intended for OpenSandbox. It is separate from the training-oriented
`main.py` workflow above.

The image contains the approved R7 model and its Python dependencies. A direct
HTTP or HTTPS PDF is downloaded into bounded container memory and is not saved
as a source PDF. Requested rendered pages, annotations, crops, and JSON reports
are written to a host-mounted output directory.

The commands in this section use Windows PowerShell from the repository root:

```powershell
Set-Location "E:\yolo-doc-analysis"
```

The Windows `.venv` does not need to be activated when running the Docker image.

### 1. Start Docker Desktop

If Docker Desktop is closed:

1. Open **Docker Desktop** from the Windows Start menu.
2. Wait until Docker Desktop reports that its engine is running.
3. Ensure Docker Desktop is using Linux containers.
4. Open PowerShell and move to the repository root as shown above.

Docker Desktop can also be started from PowerShell when installed in its default
location:

```powershell
Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
```

Starting the application does not mean the engine is immediately ready. Wait
for Docker Desktop to finish initialization before continuing.

Verify the client and Linux engine:

```powershell
docker version
```

A ready installation prints both `Client` and `Server` sections. If it prints
only the client followed by an error for
`dockerDesktopLinuxEngine`, Docker Desktop is not ready yet. Wait for startup
to finish and run `docker version` again.

### 2. Check Whether The Image Exists

```powershell
docker image inspect "legal-notice-detector:r7" --format "{{.Id}}"
```

If this prints an image SHA, continue to the smoke test. Build the image if the
command reports that the image does not exist, or whenever the Dockerfile,
dependency lock, packaged detector code, model manifest, or model changes.

### 3. Build The Image

The default build reads the approved checkpoint from:

```text
runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt
```

It verifies SHA256
`2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76`
before building:

```powershell
.\scripts\build-opensandbox-image.ps1 `
    -Image "legal-notice-detector:r7"
```

The first build downloads the OpenSandbox base image and CPU inference
dependencies and can take several minutes. Later builds normally reuse Docker's
cache. A successful build ends with:

```text
Built legal-notice-detector:r7
```

Do not close Docker Desktop while the build is running.

### 4. Run The Model Smoke Test

```powershell
.\scripts\smoke-test-opensandbox-image.ps1 `
    -Image "legal-notice-detector:r7"
```

The smoke test selects Python 3.13 inside the image, validates the packaged R7
checkpoint, loads it, and performs one synthetic warmup prediction. Successful
output includes:

```text
'status': 'ready'
'model_version': 'r7'
'warmed': True
OpenSandbox detector image smoke test passed: legal-notice-detector:r7
```

This confirms that the model starts correctly. It does not process a real PDF.

### 5. Run A Direct PDF URL

Use the following procedure in one PowerShell window. No local input-directory
mount is needed because the detector downloads the URL into memory.

Create or confirm the host output directory:

```powershell
New-Item -ItemType Directory -Force ".\output" | Out-Null
$OutputDir = (Resolve-Path ".\output").Path
```

Set a direct PDF URL. For example:

```powershell
$env:DETECTOR_PDF_URL = "https://alfajr-news.net/uploads/posts/56b19cbeed56fe0bd788e6dc74eb7e0d.pdf"
```

For a signed URL or a URL you do not want stored in PowerShell history, prompt
for it instead:

```powershell
$env:DETECTOR_PDF_URL = Read-Host "Enter the direct HTTP or HTTPS PDF URL"
```

Define the Python task that will run inside the container:

```powershell
$PythonCode = @'
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from doc_detector import get_detector

job_id = f"url-{uuid4().hex[:12]}"
output_dir = Path("/host-output/docker-url-tests") / job_id

detector = get_detector(model_version="r7")

started = time.perf_counter()
readiness = detector.warmup()
warmup_seconds = time.perf_counter() - started
model_identity = detector.model_identity

started = time.perf_counter()
result = detector.detect(
    job_id=job_id,
    pdf_url=os.environ["DETECTOR_PDF_URL"],
    output_dir=output_dir,
    outputs=["json", "crops", "annotated_pages", "rendered_pages"],
)
detection_seconds = time.perf_counter() - started

print(json.dumps({
    "readiness": readiness,
    "result": result.as_dict(),
    "warmup_seconds": round(warmup_seconds, 2),
    "detection_seconds": round(detection_seconds, 2),
    "model_reused": detector.model_identity == model_identity,
}, indent=2))
'@
```

Run that task in the image:

```powershell
$PythonCode | docker run --rm -i `
    --platform linux/amd64 `
    --entrypoint /bin/bash `
    --env DETECTOR_PDF_URL `
    --env "DOC_DETECTOR_OUTPUT_ROOT=/host-output/docker-url-tests" `
    --mount "type=bind,source=$OutputDir,target=/host-output" `
    "legal-notice-detector:r7" `
    -lc 'source /opt/code-interpreter/code-interpreter-env.sh python 3.13 && python -'

if ($LASTEXITCODE -ne 0) {
    throw "URL detection failed with exit code $LASTEXITCODE."
}
```

PowerShell sends the task through standard input to avoid nested quoting issues
between PowerShell, Docker, Bash, and Python. `--rm` removes the stopped
container after the task; it does not remove the image or host-mounted results.

Each execution generates a unique job ID, so rerunning the command does not
overwrite an earlier job.

### 6. Inspect The Results

Results are saved under:

```text
E:\yolo-doc-analysis\output\docker-url-tests\url-<unique-id>\
```

Find the newest job from PowerShell:

```powershell
$LatestJob = Get-ChildItem ".\output\docker-url-tests" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$LatestJob.FullName
Get-Content (Join-Path $LatestJob.FullName "result.json")
```

A completed job contains:

```text
url-<unique-id>/
  rendered/
    page_0001.png
    ...
  pages/
    page_0001/
      annotated.png
      crops/
    ...
  detections.json
  result.json
```

`result.json` records the job status, source type, approved model identity,
inference settings, page count, detection count, and relative artifact paths.
`detections.json` contains the confidence and bounding box for each accepted
notice. Review annotated pages and crops to assess detection quality.

The source PDF itself should not appear in the job directory. Only generated
artifacts are persisted. With `outputs=["json"]`, rendered pages remain temporary
and are deleted after inference.

The verified Alfajr URL example completed with 20 pages and 87 detections. On
the development machine, model warmup took approximately 4 seconds and the PDF
job approximately 68 seconds. Timings vary by CPU, available memory, network,
PDF size, and Docker configuration.

### 7. Choose Which Artifacts To Keep

JSON reports are always written. Edit the `outputs` value in `$PythonCode` to
control retained visual files:

```python
outputs=["json"]
```

or:

```python
outputs=["json", "crops", "annotated_pages", "rendered_pages"]
```

Supported values are:

- `json`
- `crops`
- `annotated_pages`
- `rendered_pages`

### 8. Test Local Detector Code Without Rebuilding

For development, mount the local `doc_detector` directory over the packaged
copy. This lets a new container run the latest local Python source without
rebuilding the image.

Resolve the source directory:

```powershell
$SourceDir = (Resolve-Path ".\doc_detector").Path
```

Then add this mount to the `docker run` command in step 5, before the image name:

```powershell
--mount "type=bind,source=$SourceDir,target=/opt/doc-detector/doc_detector,readonly" `
```

The development cycle is:

1. Edit files under `doc_detector/`.
2. Run the URL task again with the source mount.
3. Inspect the new job's JSON, annotations, and crops.
4. Run the unit and regression tests.
5. Repeat until the change is ready.

The source mount is only for development. Before sharing an image, rebuild it
and test it again without this mount so the test exercises packaged code.

Changes that require rebuilding include:

- `Dockerfile.opensandbox`
- `requirements-sandbox.in` or `requirements-sandbox.lock`
- `model-manifest.json`
- The packaged checkpoint
- Any final `doc_detector/` change that must be included in the shared image

### 9. Build And Verify A Development Version

Use a new tag instead of overwriting a previously tested image:

```powershell
.\scripts\build-opensandbox-image.ps1 `
    -Image "legal-notice-detector:r7-dev1"

.\scripts\smoke-test-opensandbox-image.ps1 `
    -Image "legal-notice-detector:r7-dev1"
```

Repeat the real URL test with `legal-notice-detector:r7-dev1` and no source-code
mount. Review all generated results before selecting that image for handoff.

### 10. Stop Docker And Resume Later

The commands above use temporary containers. After a run, this should normally
show no detector container:

```powershell
docker ps
```

The built images remain available:

```powershell
docker image ls "legal-notice-detector"
```

It is safe to close Docker Desktop after no builds or runs are active. Output
files remain under `output/docker-url-tests/` because they are stored on the
Windows host.

After restarting Windows or Docker Desktop:

1. Wait for the Docker Linux engine to become ready.
2. Run `docker version`.
3. Check the image with `docker image inspect`.
4. Run the smoke test if the Docker or image environment changed.
5. Recreate `$OutputDir`, `$env:DETECTOR_PDF_URL`, and `$PythonCode`.
6. Run the URL task again.

PowerShell variables do not persist after closing the PowerShell window.

### Docker Troubleshooting

#### Docker command is not found

Install or repair Docker Desktop, then open a new PowerShell window so its PATH
changes are available.

#### Docker cannot connect to `dockerDesktopLinuxEngine`

Docker Desktop is closed, still starting, or not using Linux containers. Start
Docker Desktop, wait for the engine, and rerun `docker version`.

#### The image does not exist

Build it with `scripts/build-opensandbox-image.ps1` as shown in step 3. Docker
images are local to the active Docker context.

#### PowerShell blocks a script

Use a process-scoped execution policy rather than changing the machine policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run the build or smoke-test script again from the same PowerShell window.

#### The URL response is not a PDF

Use a direct PDF URL. Viewer pages, login pages, expired signed links, and HTML
error responses are rejected.

#### The PDF host is rejected as non-public

The detector rejects loopback, private, link-local, reserved, and other
non-public destinations by default. Do not disable this check for untrusted URL
input. Internal sources require a separately reviewed network policy.

#### The download exceeds a timeout or size limit

The runtime enforces connection, total download, source-size, page-count,
rendered-page, and aggregate artifact limits from `model-manifest.json`. Confirm
that the source responds promptly and that the PDF is within the approved
limits.

#### The output destination already exists

Every job directory is immutable. Use a new job ID or let the example generate
one. Do not delete or overwrite prior results until they have been reviewed.

#### Python reports a syntax error after `python -c`

Nested quoting can be stripped when a command passes through PowerShell, Docker,
Bash, and Python. Use the stdin-based commands in this README and the current
`smoke-test-opensandbox-image.ps1` script.

## OpenSandbox Deployment

Complete local Docker testing before platform integration. The inference-only
`doc_detector` package and custom Code Interpreter image keep the deployment
runtime separate from the training-oriented `main.py`. See
[`docs/opensandbox-deployment.md`](docs/opensandbox-deployment.md) for private
registry publishing, persistent Code Interpreter context execution, network
policy, artifact retrieval, and platform release gates.

The approved implementation plan is recorded in
[`DOCKER_DEPLOYMENT_PLAN.md`](DOCKER_DEPLOYMENT_PLAN.md).

## License Note

Ultralytics publishes YOLO under AGPL-3.0 with a separate enterprise license.
Review its terms before deploying this in a business product or workflow.
