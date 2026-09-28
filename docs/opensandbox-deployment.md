# OpenSandbox Deployment Guide

This guide covers the implemented inference-only base flow. The detector runs in
a custom OpenSandbox Code Interpreter image and does not import or modify the
project's training-oriented `main.py`.

## Runtime Contract

- Model: R7 `models/legal_notice_r7.pt`
- Model SHA256: `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76`
- Confidence: `0.80`
- Image size: `1280`
- PDF render DPI: `200`
- Device: CPU
- Maximum detections per page: `300`
- Aggregate retained-artifact limit: `3221225472` bytes
- Target platform: `linux/amd64`
- Python target: `3.13`

The package supports one of `pdf_url`, `pdf_path`, or `pdf_bytes` for each job.
URL source PDFs stay in memory. Rendered pages and requested detection artifacts
are written to the caller's output directory.

## Files

| File | Purpose |
| --- | --- |
| `doc_detector/` | Independent inference runtime |
| `model-manifest.json` | Approved model identity, settings, and safety limits |
| `requirements-sandbox.in` | Pinned direct runtime constraints used to resolve the lock |
| `requirements-sandbox.lock` | Fully resolved, hashed Linux/Python 3.13 CPU dependency lock |
| `Dockerfile.opensandbox` | Custom Code Interpreter image |
| `Dockerfile.opensandbox.dockerignore` | Minimal runtime build context |
| `scripts/build-opensandbox-image.sh` | Verify R7 and build `linux/amd64` image |
| `scripts/smoke-test-opensandbox-image.sh` | Load and warm R7 inside the image |
| `scripts/run-local-detector.sh` | Run a direct PDF URL through the local image |
| `examples/local_detection_task.py` | Local URL task executed inside the image |
| `examples/sandbox_detection_task.py` | Persistent-context SDK example |

## Prerequisites

Confirm these values with the platform team before publishing:

1. OpenSandbox server version.
2. `opensandbox` and `opensandbox-code-interpreter` SDK versions.
3. Approved Code Interpreter base-image tag and immutable digest.
4. Private registry location and image pull permissions.
5. CPU, memory, writable storage, egress, and sandbox TTL limits.

Docker must be running to build or smoke-test the image.

The image installs `requirements-sandbox.lock` in one `--require-hashes` step.
Regenerate it after intentional dependency changes with the approved `uv`
version and review the resulting diff:

```powershell
uv pip compile requirements-sandbox.in `
  --output-file requirements-sandbox.lock `
  --python-version 3.13 `
  --python-platform x86_64-manylinux_2_28 `
  --torch-backend cpu `
  --extra-index-url https://download.pytorch.org/whl/cpu `
  --index-strategy unsafe-best-match `
  --generate-hashes `
  --only-binary :all: `
  --emit-index-url
```

## Build The Image

Run from Git Bash, WSL, or a Linux build worker:

```bash
IMAGE=registry.example.com/legal-notice-detector:r7-v1 \
BASE_IMAGE=opensandbox/code-interpreter:v1.1.0 \
bash scripts/build-opensandbox-image.sh
```

On Windows PowerShell with Docker Desktop running:

```powershell
.\scripts\build-opensandbox-image.ps1 `
  -Image "registry.example.com/legal-notice-detector:r7-v1" `
  -BaseImage "opensandbox/code-interpreter:v1.1.0"
```

The build script:

1. Resolves R7 and its checksum from `model-manifest.json`.
2. Reads `models/legal_notice_r7.pt` by default and rejects unresolved Git LFS pointers.
3. Verifies the approved checksum.
4. Copies only that checkpoint into a temporary ignored build directory.
5. Builds the custom `linux/amd64` image.
6. Removes the temporary model staging directory.

Override the local checkpoint location when required:

```bash
MODEL_SOURCE=/secure/artifacts/legal_notice_r7.pt \
IMAGE=registry.example.com/legal-notice-detector:r7-v1 \
bash scripts/build-opensandbox-image.sh
```

For production, replace the base tag with the immutable digest approved for the
installed OpenSandbox version:

```bash
RELEASE=true \
BASE_IMAGE='opensandbox/code-interpreter:v1.1.0@sha256:<approved-digest>' \
IMAGE=registry.example.com/legal-notice-detector:r7-v1 \
bash scripts/build-opensandbox-image.sh
```

PowerShell release build:

```powershell
.\scripts\build-opensandbox-image.ps1 `
  -Release `
  -BaseImage "opensandbox/code-interpreter:v1.1.0@sha256:<approved-digest>" `
  -Image "registry.example.com/legal-notice-detector:r7-v1"
```

## Smoke Test

```bash
IMAGE=registry.example.com/legal-notice-detector:r7-v1 \
bash scripts/smoke-test-opensandbox-image.sh
```

Windows PowerShell:

```powershell
.\scripts\smoke-test-opensandbox-image.ps1 `
  -Image "registry.example.com/legal-notice-detector:r7-v1"
```

The image smoke test overrides the entrypoint, selects the configured Python,
validates R7, and runs a synthetic warm-up prediction. It is not a substitute
for the release-gate OpenSandbox test, which must start the official entrypoint,
create a Code Interpreter context, run two jobs, and retrieve JSON and PNG files.

## Publish

After all tests pass:

```bash
docker push registry.example.com/legal-notice-detector:r7-v1
docker inspect --format='{{index .RepoDigests 0}}' \
  registry.example.com/legal-notice-detector:r7-v1
```

Configure the platform with the resulting immutable digest rather than only the
mutable tag.

## Platform Preparation

The platform must create the sandbox with:

- The private image digest.
- Entrypoint `/opt/code-interpreter/code-interpreter.sh`.
- `PYTHON_VERSION=3.13`.
- Measured CPU, memory, storage, timeout, and TTL values.
- A default-deny network policy that allows the PDF host and every legitimate
  redirect host. The example reads additional redirect hosts from
  `DETECTOR_EGRESS_HOSTS` as a comma-separated list.
- Destination-IP enforcement that rejects loopback, private, link-local,
  reserved, metadata-service, and other non-public addresses after DNS
  resolution and when the connection is established.

Create one Python context and initialize it once:

```python
from doc_detector import get_detector

detector = get_detector(model_version="r7")
readiness = detector.warmup()
```

Run later jobs through the same context to reuse the loaded model. A separate
`python main.py ...` command starts another process and does not reuse it.

## Run A Detection

```python
result = detector.detect(
    job_id="platform-job-123",
    output_dir="/tmp/doc-detector/jobs/platform-job-123",
    pdf_url="https://example.com/newspaper.pdf",
    outputs=["json", "crops", "annotated_pages", "rendered_pages"],
)

result.as_dict()
```

`DetectionResult.as_dict()` intentionally contains sandbox-local absolute paths
for immediate follow-up script steps. Persisted `result.json` and
`detections.json` use relative artifact paths and remain portable if the output
directory is moved.

Use `pdf_path` for a PDF already attached inside the sandbox or `pdf_bytes` for
bytes supplied by the caller. Supply exactly one input.

JSON reports are always generated. The `outputs` list controls the retained
visual artifacts. Supported values are:

- `json`
- `crops`
- `annotated_pages`
- `rendered_pages`

The destination must not already exist. This prevents retries from silently
overwriting prior output.

Rendered pages have their own intermediate-output limit. All retained rendered
pages, annotations, crops, and JSON files also share the aggregate artifact
limit from `model-manifest.json`. If a job fails, partial artifacts are removed
before the failed `result.json` is written.

## Retrieve Results

The job directory contains:

```text
<output-directory>/
  rendered/
  pages/
    page_0001/
      annotated.png
      crops/
  detections.json
  result.json
```

Paths stored inside JSON are relative to the output directory. Use OpenSandbox
text operations for JSON and binary or streaming operations for PNG files.

The sandbox filesystem is temporary unless the platform mounts persistent
storage. Retrieve or upload required artifacts before destroying or allowing the
sandbox to expire.

## Continued Operation

- Use one active detection at a time per detector context.
- Separate sandboxes can process jobs in parallel.
- Keep sandbox and context IDs when reusing a prepared detector.
- Reinitialize after a Python kernel restart.
- A failed task writes a failed `result.json` when its output directory was
  already created, then raises an exception for the platform to handle.
- Upload integrations and automatic artifact deletion are intentionally outside
  this base implementation.

By default, URL validation rejects hosts that resolve to loopback, private,
link-local, reserved, or other non-public addresses, and validates redirect
targets. Keep `allow_private_hosts` disabled unless the platform explicitly
requires a reviewed internal source and enforces access with its network policy.

Application DNS validation is defense in depth, not the SSRF security boundary:
DNS can change between validation and connection. Production release therefore
requires the platform network layer to apply the default-deny host allowlist and
reject non-public destination IPs at connection time. Do not rely on the Python
preflight check alone.

## Local Verification

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Use the current CLI independently for local training and regression workflows.
The OpenSandbox runtime package has no import dependency on `main.py`.
