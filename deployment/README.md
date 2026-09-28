# Legal PDF Detector

Deployment-only repository for the R7 legal-notice detector. It builds a custom
OpenSandbox Code Interpreter image for CPU inference and intentionally excludes
training code, datasets, source PDFs, experiment history, and development
checkpoints.

## Runtime Contract

- Architecture: `linux/amd64`
- Base image: `opensandbox/code-interpreter:v1.1.0`
- Python: `3.13`
- Model: R7, class `legal_notice`
- Model SHA256: `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76`
- Device: CPU
- Confidence: `0.80`
- Image size: `1280`
- PDF render DPI: `200`
- Maximum detections per page: `300`

For URL input, the source PDF is downloaded into bounded memory and is not
saved. Requested JSON reports, crops, annotations, and rendered pages are saved
under the selected job output directory.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `doc_detector/` | Inference-only Python package |
| `models/legal_notice_r7.pt` | Approved checkpoint tracked by Git LFS |
| `model-manifest.json` | Model identity, inference settings, and safety limits |
| `requirements-sandbox.lock` | Complete hash-verified Linux dependency lock |
| `Dockerfile.opensandbox` | Custom OpenSandbox image |
| `scripts/` | Build, smoke-test, and local URL-runner commands |
| `examples/sandbox_detection_task.py` | Persistent-context OpenSandbox example |
| `tests/` | Inference runtime tests |
| `release-info.json` | Source and model provenance for this export |
| `.deployment-sync-state.json` | Managed-file hashes used to detect unsafe sync overwrites |

## Prerequisites

Install:

- Git and Git LFS
- Docker Desktop with Linux containers, or Docker Engine on Linux
- PowerShell 5.1+ for `.ps1` commands, or Bash and Python 3 for `.sh` commands

After cloning:

```powershell
git lfs install
git lfs pull
```

Confirm that `models/legal_notice_r7.pt` is a real model file rather than a Git
LFS pointer:

```powershell
Get-Item ".\models\legal_notice_r7.pt"
Get-FileHash ".\models\legal_notice_r7.pt" -Algorithm SHA256
```

The build script also rejects unresolved LFS pointers and checksum mismatches.

## Build

Start Docker Desktop and wait until both client and server are available:

```powershell
docker version
```

Build the local image:

```powershell
.\scripts\build-opensandbox-image.ps1 `
    -Image "legal-notice-detector:r7"
```

Linux or WSL:

```bash
IMAGE=legal-notice-detector:r7 bash scripts/build-opensandbox-image.sh
```

## Smoke Test

```powershell
.\scripts\smoke-test-opensandbox-image.ps1 `
    -Image "legal-notice-detector:r7"
```

Successful output reports model version `r7`, the approved SHA256, and
`warmed: True`.

## Run A Direct PDF URL Locally

The PowerShell runner starts a temporary container, warms R7, downloads the
source PDF into memory, runs inference, and writes generated artifacts under
`output/<job-id>/`:

```powershell
.\scripts\run-local-detector.ps1 `
    -Image "legal-notice-detector:r7" `
    -PdfUrl "https://example.com/newspaper.pdf"
```

Use a specific job ID when required:

```powershell
.\scripts\run-local-detector.ps1 `
    -Image "legal-notice-detector:r7" `
    -PdfUrl "https://example.com/newspaper.pdf" `
    -JobId "staging-check-001" `
    -Outputs "json,crops,annotated_pages,rendered_pages"
```

Linux or WSL:

```bash
IMAGE=legal-notice-detector:r7 \
  bash scripts/run-local-detector.sh "https://example.com/newspaper.pdf"
```

For local source-code iteration only, add `-Dev` to the PowerShell runner or set
`DEV=true` for Bash. Rebuild and test without development mounts before sharing
an image.

## Unit Tests

After building the image, run tests against the exported source using the
image's Python environment and installed dependencies:

```powershell
$RepositoryRoot = (Resolve-Path ".").Path

docker run --rm `
    --platform linux/amd64 `
    --entrypoint /bin/bash `
    --env "DOC_DETECTOR_OUTPUT_ROOT=" `
    --mount "type=bind,source=$RepositoryRoot,target=/repo,readonly" `
    "legal-notice-detector:r7" `
    -lc 'source /opt/code-interpreter/code-interpreter-env.sh python 3.13 && cd /repo && python -m unittest discover -v'
```

Image build, smoke test, and a real URL job are the required self-contained
handoff checks.

## First Repository Publication

The initial export is deliberately left uncommitted so it can be reviewed before
publication. Confirm that the private remote supports Git LFS, then run:

```powershell
git status
git add .
git lfs status
git commit -m "Add R7 legal PDF detector runtime"
git remote add origin "<private-repository-url>"
git push -u origin main
```

Do not commit if `git lfs status` does not list
`models/legal_notice_r7.pt` as an LFS object. Do not commit generated `output/`
files, credentials, source PDFs, or `.env` files.

Validate the publication from a separate clean directory before handoff:

```powershell
git clone "<private-repository-url>" legal-pdf-detector-clean
Set-Location ".\legal-pdf-detector-clean"
git lfs pull
Get-FileHash ".\models\legal_notice_r7.pt" -Algorithm SHA256
```

The expected model SHA256 is recorded in `model-manifest.json`. Build, smoke-test,
and run one real URL from the clean clone before publishing the final image.

## Release Build

Release builds require an immutable base-image digest approved by the platform
team:

```powershell
.\scripts\build-opensandbox-image.ps1 `
    -Release `
    -BaseImage "opensandbox/code-interpreter:v1.1.0@sha256:<approved-digest>" `
    -Image "<private-registry>/legal-notice-detector:r7-v1"
```

Push the tested image and give DevOps its immutable registry digest, source
commit, `release-info.json`, and `docs/validation.md`.

## OpenSandbox Integration

The image is a Code Interpreter runtime, not a standalone HTTP API. Create one
Python context, call `get_detector().warmup()` once, and run subsequent jobs in
that same context to reuse the loaded model. See:

- `examples/sandbox_detection_task.py`
- `docs/opensandbox-deployment.md`

The platform must provide a writable output path, retrieve artifacts before the
sandbox expires, and enforce default-deny egress with explicit PDF/redirect host
allowlists and non-public destination-IP rejection.

## Updating This Repository

Model development remains in the internal training workspace. Approved runtime
updates are exported by running its guarded sync command from that source
workspace:

```powershell
.\scripts\sync-deployment-repo.ps1 `
    -Destination "E:\legal-pdf-detector" `
    -DryRun

.\scripts\sync-deployment-repo.ps1 `
    -Destination "E:\legal-pdf-detector"
```

Review, rebuild, smoke-test, run a real PDF, and commit the deployment repository
after every sync. Do not develop training features in this repository.
