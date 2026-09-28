# Local Release Validation

## Validated Build

| Item | Value |
| --- | --- |
| Validation date | 2026-09-28 |
| Image tag | `legal-notice-detector:r7-handoff` |
| Local image ID | `sha256:cf9cec9ba9185f5c6762b87c51ddc92900fc08a549f0721c44f4943d53ae6335` |
| Platform | `linux/amd64` |
| Image size | 3,128,075,932 bytes |
| Base image | `opensandbox/code-interpreter:v1.1.0` |
| Resolved base digest | `sha256:133a3c1720dd52291a019740c2987e7164ea6de79e23d8198798e58950ae2e6e` |
| Model version | R7 |
| Model SHA256 | `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76` |

The image was built from the self-contained `E:\legal-pdf-detector` export. No
runtime or build file depended on the training workspace.

## Checks

| Check | Result |
| --- | --- |
| Inference unit tests | 25 passed |
| Python compilation | Passed |
| PowerShell parser checks | Passed |
| Bash syntax checks in Linux container | Passed |
| Checkpoint checksum verification | Passed |
| Docker image build | Passed |
| Container R7 warmup | Passed |
| Real HTTPS PDF job | Passed |

## Real URL Test

URL:

```text
https://alfajr-news.net/uploads/posts/bdf606a7e4ee04742b92cbe9c074d12e.pdf
```

Command:

```powershell
.\scripts\run-local-detector.ps1 `
    -Image "legal-notice-detector:r7-handoff" `
    -PdfUrl "https://alfajr-news.net/uploads/posts/bdf606a7e4ee04742b92cbe9c074d12e.pdf" `
    -JobId "handoff-alfajr-bdf606a7"
```

Observed result:

| Metric | Value |
| --- | --- |
| Status | `completed` |
| Source type | `url` |
| Pages | 20 |
| Detections/crops | 58 |
| Rendered pages | 20 |
| Annotated pages | 20 |
| Warmup time | 3.32 seconds |
| Detection time | 92.04 seconds |
| Warmed model reused | Yes |
| Source PDF persisted | No |

Local artifacts are under
`output/handoff-alfajr-bdf606a7/`. The `output/` directory is intentionally
ignored and is not part of the deployment repository.

## Remaining Release Gates

Local Docker validation does not replace staging validation. DevOps must still:

1. Build or pull the image through the approved private registry.
2. Record and deploy the immutable registry digest.
3. Start the image through the OpenSandbox Code Interpreter entrypoint.
4. Warm one Python context and run two jobs in that same context.
5. Verify artifact retrieval before sandbox expiration.
6. Measure CPU, peak memory, writable storage, timeout, and TTL requirements.
7. Verify default-deny egress, explicit PDF/redirect host allowlists, and
   connection-time rejection of non-public destination IPs.

The source export was created while the source workspace contained uncommitted
deployment changes. Commit the reviewed source and deployment repositories, run
the sync again, and repeat the build/smoke/URL checks for the final registry
release tag.
