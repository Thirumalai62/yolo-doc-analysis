# AGENTS.md

## Repository Purpose

This repository contains only the deployment runtime for legal-notice detection
in PDF newspapers. It builds a custom OpenSandbox Code Interpreter image and
supports local Docker validation.

Training pipelines, datasets, experiment history, and model-development code
belong in the internal source workspace and must not be added here.

## Runtime Contract

Preserve these release settings unless an approved model release explicitly
changes them:

- Model version: `r7`
- Model file: `models/legal_notice_r7.pt`
- Model SHA256: `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76`
- Class list: `legal_notice` only
- Confidence: `0.80`
- Image size: `1280`
- Render DPI: `200`
- Device: CPU
- Maximum detections per page: `300`
- Target architecture: `linux/amd64`
- Python: `3.13.13`
- Node.js: `22.2.0`
- Kernels: Python, Bash, JavaScript, and TypeScript

`model-manifest.json` is the authoritative runtime configuration. The build and
runtime must reject a missing model, unresolved Git LFS pointer, checksum
mismatch, or unexpected model class.

## Input And Output Rules

- Each job must provide exactly one of `pdf_url`, `pdf_path`, or `pdf_bytes`.
- URL source PDFs must remain in bounded memory and must not be persisted.
- Keep download, page, render, pixel, and aggregate artifact limits enabled.
- Keep private and non-public URL destinations blocked by default.
- Treat application DNS validation as defense in depth. Production still
  requires connection-time network enforcement by the platform.
- Keep all generated paths inside the selected job output directory.
- Never overwrite an existing job directory.
- Persisted JSON must use relative artifact paths.
- One detector context processes one operation at a time.
- Reuse the same Python context when model reuse is required.

## Repository Boundaries

Do not add or depend on:

- `main.py`
- Training or regression pipelines
- Datasets or annotation exports
- Source PDFs or generated output folders
- Historical checkpoints
- Local virtual environments
- Credentials, signed URLs, API keys, or `.env` files

The Docker build context intentionally contains only the runtime package,
manifest, dependency lock, and staged approved checkpoint.

## Model And Git LFS

Model checkpoints are tracked through Git LFS:

```powershell
git lfs install
git lfs pull
git lfs status
```

Before building, verify that `models/legal_notice_r7.pt` is not a small text LFS
pointer and that its SHA256 matches `model-manifest.json`.

Do not replace or rename the checkpoint without updating the manifest, build
scripts, documentation, tests, and release evidence together.

## Required Validation

After runtime changes, run the inference tests:

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

Build and smoke-test a versioned image:

```powershell
.\scripts\build-opensandbox-image.ps1 `
    -Image "legal-notice-detector:<version>"

.\scripts\smoke-test-opensandbox-image.ps1 `
    -Image "legal-notice-detector:<version>"
```

Run at least one direct HTTPS PDF through the packaged image without a
development source mount:

```powershell
.\scripts\run-local-detector.ps1 `
    -Image "legal-notice-detector:<version>" `
    -PdfUrl "<direct-pdf-url>"
```

Verify `result.json`, `detections.json`, every referenced artifact, model reuse,
and the absence of a persisted source PDF. Update `docs/validation.md` with the
tested image identity and observed results.

## Change Guidelines

- Prefer the smallest correct runtime change.
- Keep `doc_detector` independent from the original training workspace.
- Preserve manifest-driven settings rather than duplicating constants.
- Keep dependency versions fully pinned and hash verified.
- Regenerate `requirements-sandbox.lock` or `requirements-code-interpreter.lock`
  only after intentional dependency changes and review the complete diff.
- Do not weaken SSRF controls, resource limits, output containment, checksum
  validation, or failure cleanup.
- Add or update focused tests for behavioral changes.
- Do not commit files under `output/`, `.build/`, caches, or local environments.
- Do not commit, push, or create releases unless explicitly requested.

## Internal Synchronization

This deployment repository is exported from the internal development workspace
using `scripts/sync-deployment-repo.ps1` in that source workspace. The sync uses
an explicit allowlist and `.deployment-sync-state.json` to detect conflicting
destination edits.

When changing a managed file directly in this repository, coordinate the same
change back into the source workspace before the next sync. Otherwise, the sync
will stop with a conflict or the repositories will diverge.

After every approved sync:

1. Review the complete deployment-repository diff.
2. Confirm the model remains a Git LFS object.
3. Run tests, image build, smoke test, and a real URL job.
4. Update release provenance and validation evidence.
5. Commit and publish only after all required checks pass.

## Platform Release Gates

Local Docker success is necessary but not sufficient. DevOps must also verify:

- Private-registry push and immutable image digest
- OpenSandbox entrypoint startup
- Two jobs using one persistent Code Interpreter Python context
- Artifact retrieval before sandbox expiration
- CPU, memory, writable storage, timeout, and TTL sizing
- Default-deny egress and explicit source/redirect host allowlists
- Connection-time rejection of private and non-public destination IPs

See `README.md`, `docs/opensandbox-deployment.md`, and `docs/validation.md` for
the operator workflow and current release evidence.
