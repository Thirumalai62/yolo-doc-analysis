# Final OpenSandbox Detector Implementation Plan

**Status:** Draft for team review. Implementation has not started.

## 1. Objective

Build a small, inference-only Python package that runs inside a custom
OpenSandbox Code Interpreter image.

Our platform will prepare and execute a script that:

1. Selects the available finalized model.
2. Supplies a PDF URL, attached PDF, or PDF bytes.
3. Specifies the output directory and required artifacts.
4. Runs detection using the prewarmed model.
5. Receives structured results and locally generated file paths.
6. Continues its workflow using those files.

Upload integrations will be added to the platform script in a later phase.

The existing `main.py` will remain unchanged for local training, evaluation,
experiments, and detection testing.

## 2. Agreed Initial Configuration

| Item | Initial value |
| --- | --- |
| Runtime | OpenSandbox Code Interpreter |
| Execution | Persistent Python context |
| Architecture | Linux Intel/AMD: `linux/amd64` |
| Device | CPU |
| Selected model | R7 `candidate.pt` |
| Current checkpoint | `runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt` |
| Confidence | `0.80` |
| Render DPI | `200` |
| Inference image size | `1280` |
| Image distribution | Private container registry |
| Model distribution | Included in the private image |
| Default outputs | Detection JSON, crops, annotated pages, and rendered pages |

The selected checkpoint, checksum, runtime settings, and dependency versions
will be recorded in a model manifest.

## 3. Final Architecture

```text
BUILD ONCE PER RELEASE
--------------------------------------------------
Official OpenSandbox Code Interpreter image
    + tested CPU inference dependencies
    + independent detector package
    + selected R7 checkpoint
    + model/version manifest
                    |
                    v
          Private container registry


PREPARE THE RUNNING ENVIRONMENT
--------------------------------------------------
Platform creates or acquires a sandbox
                    |
                    v
Creates a Python execution context
                    |
                    v
Loads and validates R7
                    |
                    v
Runs warm-up prediction
                    |
                    v
Records sandbox/context IDs and marks detector ready


EXECUTE EACH TASK
--------------------------------------------------
Platform script supplies PDF and output options
                    |
                    v
Detector runs in the prepared Python context
                    |
                    v
Results saved in a job-specific local directory
                    |
                    v
Script receives structured results and file paths
                    |
                    v
Platform continues processing
    `-- Upload integration added later
```

The platform may keep the sandbox and Python context running for multiple
sequential tasks.

Keeping the sandbox alive preserves the prewarmed model only while the same
Python process and context remain alive. A new subprocess or restarted kernel
requires loading and warming the model again.

## 4. Scope And Responsibilities

### Detector Package

- Validate and load the selected model.
- Warm the model.
- Accept supported PDF input forms.
- Render pages and perform inference.
- Generate requested artifacts.
- Return structured results and file locations.
- Release per-task resources while retaining the model.

### Platform Script And OpenSandbox Integration

- Create or connect to the sandbox.
- Maintain the Python context.
- Supply task IDs, inputs, and output options.
- Track task status and decide retries.
- Handle execution limits and sandbox lifetime.
- Read or retrieve generated files.
- Perform downstream processing.
- Control eventual cleanup.

### Existing Local Project

`main.py`, training workflows, datasets, and regression scripts remain available
for local development.

The new runtime package will neither modify nor import `main.py`. Only the
necessary inference behavior will be adapted into the new package and checked
against the current detector.

## 5. Proposed Project Structure

```text
main.py                              Existing local workflow; unchanged
requirements.txt                     Existing local setup

doc_detector/
    __init__.py                      Public interface
    detector.py                      Model lifecycle and inference
    pdf_input.py                     PDF input and rendering
    outputs.py                       Crops, annotations, reports, manifest

tests/
    test_detector_inputs.py
    test_detector_outputs.py
    test_detector_lifecycle.py

examples/
    sandbox_detection_task.py        Platform integration example

Dockerfile.opensandbox
Dockerfile.opensandbox.dockerignore
requirements-sandbox.lock
model-manifest.json

docs/
    opensandbox-deployment.md
```

The runtime image will contain the new package, runtime dependencies, and
selected checkpoint. Training data and development scripts are not required by
that package.

## 6. Input And Output Contract

### Inputs

Each task supplies exactly one PDF source:

| Input | Handling |
| --- | --- |
| `pdf_url` | Fetch into memory; do not save the source PDF |
| `pdf_path` | Read an attached PDF already available inside the sandbox |
| `pdf_bytes` | Process supplied bytes directly |

An attached file path must exist inside the sandbox. A path on the caller's
computer is not automatically accessible there.

Each task also supplies:

- A platform-generated job ID.
- A writable output directory.
- Requested artifact types.

### Model Selection

Initially, `model_version="r7"` resolves to the checkpoint packaged in the image.

An unavailable model version produces a clear error. Future finalized models
will be distributed through new versioned images with matching manifests.

### Outputs

The initial supported formats are:

- **JSON:** Detections and execution manifest.
- **PNG:** Crops, annotated pages, and rendered pages.

Artifact selection is configurable. Any additional formats will require an
agreed contract.

Example directory:

```text
<output-directory>/
  rendered/
  pages/
    .../
      annotated.png
      crops/
  detections.json
  result.json
```

`result.json` will identify the job, model, settings, counts, and generated
artifacts. Artifact paths in the manifest will be relative to the output
directory for portability.

The detector will refuse to overwrite an existing job destination. The platform
controls retry naming and recovery.

### Proposed Script Interface

This is the target API to be implemented.

Initialize once during context preparation:

```python
from doc_detector import get_detector

detector = get_detector(model_version="r7")
detector.warmup()
```

Run each task in the same Python context:

```python
result = detector.detect(
    pdf_url=pdf_url,
    job_id=job_id,
    output_dir=output_directory,
    outputs=["json", "crops", "annotated_pages", "rendered_pages"],
)

print(result.report_path)
print(result.output_dir)

# Future upload and downstream operations go here.
```

Failures will use clear Python exceptions that the platform adapter handles.
Library code will not terminate the interpreter with `SystemExit`.

## 7. Image And Dependency Preparation

### Base Image

Use a pinned, platform-compatible release and digest of:

```text
opensandbox/code-interpreter
```

Preserve its Code Interpreter startup integration and explicitly select the
Python version.

Python 3.13 is the initial compatibility target because the current project uses
it. The final image and dependency combination must pass Linux testing.

### Dependencies

Prepare a Linux-tested lock file covering:

- CPU-only PyTorch and torchvision.
- `ultralytics==8.4.146`.
- `pypdfium2`.
- Pillow.
- Required transitive packages and Linux libraries.

Install dependencies into the Python environment actually used by the execution
kernel.

Dependencies and model files will be prepared at image-build time rather than
installed for each detection task. Model loading into memory happens during
runtime preparation.

Do not copy the local Windows `.venv` into the image.

### Model Packaging

Supply only the selected R7 checkpoint as a controlled build input.

- Keep the binary outside Git tracking.
- Verify its checksum.
- Include it in the private image.
- Record model identity and settings.
- Verify identity again when initializing the detector.

Configure library caches and generated outputs to use sandbox-writable
locations.

## 8. Prewarming And Continued Operation

The platform preparation step will:

1. Start or connect to a compatible sandbox.
2. Create or locate the intended Python context.
3. Initialize the detector.
4. Validate model identity and the `legal_notice` class.
5. Run a warm-up inference.
6. Verify readiness in that context.
7. Retain the sandbox ID and context ID.

Run one active detection per context initially. Parallel processing should use
separate prepared sandboxes.

If the platform already uses a warm pool, its preparation hook can perform these
steps before a sandbox becomes available. Cold-created sandboxes must undergo
the same preparation.

A missing or restarted kernel invalidates model readiness and triggers
initialization again.

## 9. Local Results And Later Uploads

"Local output" means local to the sandbox where the task executes.

The platform script can immediately open the generated files and perform
further processing. Alternatively, the platform can retrieve them using
OpenSandbox file operations.

For the first implementation:

- Preserve generated artifacts after detection.
- Release PDF bytes and image resources.
- Keep the model loaded.
- Return a manifest and usable file paths.
- Leave uploads and automatic retention policies for the next phase.

Required results must be transferred to persistent storage before the sandbox
expires or is destroyed unless the platform provides a persistent volume.

The integration example will distinguish between:

- Using an existing retained sandbox.
- Creating a temporary sandbox whose lifecycle the example owns.

## 10. GitHub And Registry Contents

| Destination | Contents |
| --- | --- |
| GitHub | Detector source, tests, Dockerfile, dependency lock, model manifest, examples, documentation |
| Existing GitHub development files | `main.py`, regression source, configuration, and relevant development documentation |
| Private image registry | Built runtime image, including R7 |
| Ignored local/build storage | Checkpoints, datasets, PDFs, generated outputs, environments, credentials |

The Docker build context will explicitly include only required runtime files and
the selected model build input. `.gitignore` alone does not control Docker's
build contents.

## 11. Implementation Stages And Acceptance Criteria

| Stage | Work | Completion criteria |
| --- | --- | --- |
| **1. Confirm platform contract** | Record SDK/server versions, base image, registry, writable paths, resources, and lifetime settings | Team agrees on the execution environment |
| **2. Implement inference-only package** | Add model lifecycle, PDF handling, detection, and configurable outputs | Runs independently of `main.py`; input/output tests pass |
| **3. Verify detection equivalence** | Compare against the existing CLI using the same PDFs, model, and settings | Counts, boxes, and scores agree within documented numerical tolerances |
| **4. Build custom image** | Install pinned Linux dependencies and package R7 | Selected Python kernel imports dependencies, validates R7, and completes inference |
| **5. Integrate with OpenSandbox** | Add preparation and task-script examples | Multiple tasks reuse the model; files are accessible locally and retrievable |
| **6. Validate operational behavior** | Test failures, context restart, limits, and resource use | Clear failure recovery; measured settings support representative complete PDFs |
| **7. Publish release** | Publish private image and record immutable digest | Platform can execute the verified image using the deployment guide |

### Essential Verification

- `main.py` remains unchanged.
- The new runtime has no dependency on importing `main.py`.
- URL and byte inputs create no source PDF files.
- Attached PDFs work without creating another source copy.
- Requested artifacts and manifests are correct.
- Successive tasks reuse the same model instance.
- Failed tasks release resources and do not corrupt later tasks.
- Separate job directories prevent accidental overwrites.
- Results remain available for subsequent script steps.
- Actual OpenSandbox inference and artifact retrieval succeed.

Measure model warmup, full-PDF processing time, peak RAM, CPU usage, and output
storage before choosing final limits.

## 12. Detailed Implementation Steps

### Step 1: Confirm Platform Contract

Collect and record:

- OpenSandbox server version.
- `opensandbox` SDK version.
- `opensandbox-code-interpreter` SDK version.
- Approved Code Interpreter image version and digest.
- Private registry location and pull permissions.
- Existing platform sandbox creation and execution example.
- CPU and memory limits.
- Writable paths and available temporary storage.
- Outbound network policy for PDF URLs and redirects.
- SDK request timeout, command timeout, and sandbox lifetime.
- Whether the platform already has a warm pool or preparation hook.
- How sandbox and Python context IDs are retained between tasks.

Deliverable: a short compatibility and resource record approved by the team.

### Step 2: Implement The Inference-Only Package

Add the independent `doc_detector` package without modifying or importing
`main.py`.

Implement:

- Model manifest loading.
- Model checksum and class validation.
- Model caching by approved version.
- Explicit model warmup.
- URL, attached-file, and byte input validation.
- In-memory URL download.
- PDF rendering.
- Page-by-page inference.
- Configurable artifact generation.
- Relative-path JSON reports.
- Structured Python results and exceptions.
- Per-task resource cleanup.

Deliverable: unit tests and a small local inference example.

### Step 3: Verify Existing Detector Equivalence

Run the current `main.py detect` and the new package using:

- The same R7 checkpoint.
- Confidence `0.80`.
- Image size `1280`.
- Render DPI `200`.
- The same representative PDFs.

Compare rendered dimensions, page count, detection count, confidence, bounding
boxes, crops, and annotated output. Document any expected platform-level
numerical tolerance.

Deliverable: a parity report proving the inference-only implementation preserves
the approved behavior.

### Step 4: Build The OpenSandbox Image

Create:

```text
Dockerfile.opensandbox
Dockerfile.opensandbox.dockerignore
requirements-sandbox.lock
model-manifest.json
```

The build will:

1. Use an approved immutable OpenSandbox base-image digest.
2. Select the approved Python environment.
3. Install CPU-only dependencies into that environment.
4. Copy only runtime package files and examples.
5. Verify and copy only the approved R7 checkpoint.
6. Configure writable cache and output paths.
7. Preserve the official Code Interpreter entrypoint.
8. Run import, model-validation, and inference smoke tests.

Deliverable: a tested local `linux/amd64` image.

### Step 5: Add The Platform Integration Example

The example will demonstrate:

```text
Connect to OpenSandbox
  -> create or attach to a sandbox
  -> create or locate the persistent Python context
  -> load and warm the detector once
  -> pass task inputs safely
  -> run detection in the same context
  -> read JSON results
  -> access or retrieve binary artifacts
  -> retain or destroy the sandbox according to caller ownership
```

Task inputs will be serialized safely instead of directly concatenated into
executable Python source.

Deliverable: an example that the platform team can adapt to its own task script.

### Step 6: Validate End To End

Test:

- All dependencies import in the selected Python kernel.
- R7 checksum and class validation pass.
- Warmup completes before readiness.
- Two tasks in the same context reuse one model instance.
- Source PDF URLs remain in memory.
- Attached PDFs and PDF bytes work.
- Configurable output combinations work.
- All artifacts are available in the chosen output directory.
- Failed tasks do not break later tasks in the same healthy context.
- Kernel loss is detected and followed by reinitialization.
- Job paths cannot overwrite or escape another job directory.
- Signed URL query values are not written to logs or reports.
- Resource use fits approved sandbox limits.

Deliverable: an end-to-end report with timing, memory, output size, and behavior
results.

### Step 7: Publish And Operate

1. Build a versioned image.
2. Run unit, parity, image, and OpenSandbox tests.
3. Push the image to the private registry.
4. Record its immutable digest.
5. Configure the platform to use that digest.
6. Warm one sandbox and run a production-like PDF.
7. Review all generated artifacts.
8. Roll out to the intended workflow.

Deliverable: an operational guide covering build, publish, rollback, execution,
troubleshooting, output retrieval, and cleanup.

## 13. Security And Operational Rules

- Accept only HTTP and HTTPS URL inputs.
- Apply PDF download timeout and maximum size limits.
- Apply a maximum page count.
- Do not log signed URL query strings.
- Validate job IDs before using them as path components.
- Refuse output-directory overwrites.
- Keep artifact paths inside the selected job directory.
- Keep credentials outside source code, image layers, logs, and results.
- Validate model checksum during build and runtime initialization.
- Require exactly one model class named `legal_notice`.
- Start with one detection at a time per Python context.
- Persist required results before sandbox expiration or destruction.

## 14. Items For Team Confirmation

Before implementation, confirm:

1. **Platform versions:** OpenSandbox server, SDK, Code Interpreter image, and
   controller language.
2. **Registry:** Private image location and image-pull permissions.
3. **Runtime:** Permitted CPU/RAM, writable paths, outbound PDF access,
   execution timeout, and sandbox lifetime.
4. **Context ownership:** How the platform retains and reuses sandbox/context
   IDs.
5. **Output contract:** JSON and PNG with configurable artifact selection are
   sufficient for the base release.
6. **Model release:** R7 at confidence `0.80` is the selected initial checkpoint.

## 15. Implementation Order

```text
Confirm platform versions and limits
  -> Build independent inference-only package
  -> Verify parity with current local detection
  -> Build custom OpenSandbox image with R7
  -> Add persistent-context warmup
  -> Add platform task example
  -> Verify resource use and artifact access
  -> Publish immutable private image
  -> Integrate into the platform workflow
  -> Add upload operations in a later phase
```

## 16. Final Recommendation

Deliver an independent inference package and a self-contained OpenSandbox image.
The platform script will use the prepared Python context, run detection, and
receive all requested outputs in its chosen local sandbox directory.

Do not change or import the current `main.py` for the OpenSandbox runtime. This
preserves the existing local training and testing workflow while giving the
platform a small, stable, inference-only integration surface.

## References

- [OpenSandbox Code Interpreter example](https://open-sandbox.ai/examples/code-interpreter)
- [Official OpenSandbox sandbox images](https://github.com/opensandbox-group/sandbox-images)
- [OpenSandbox Code Interpreter Python SDK](https://github.com/opensandbox-group/OpenSandbox/blob/main/sdks/code-interpreter/python/README.md)
- [OpenSandbox Python SDK](https://open-sandbox.ai/sdks/python)
- [OpenSandbox client pool](https://open-sandbox.ai/guides/client-pool)
