# Local Release Validation

## Optimized Build

| Item | Value |
| --- | --- |
| Validation date | 2026-09-29 |
| Image tag | `legal-notice-detector:r7-python-node-slim-deployment` |
| Local image ID | `sha256:6ca646de1457bbbf073ba2d1d8b8fc30b95e266e91cdcd5ebe7a3cd2845681d9` |
| Platform | `linux/amd64` |
| Docker CLI size | 3.3 GB, reduced from 12.6 GB |
| Docker inspect content size | 755,604,891 bytes, reduced from 3,128,075,932 bytes |
| Base image | `python:3.13.13-slim-bookworm` |
| Pinned base digest | `sha256:355bfa66770995d7e9a0da4b3473b44d0cb451f6b56f5615ad9c39e3c4eca03f` |
| Python / Node.js | `3.13.13` / `22.2.0` |
| Model version | R7 |
| Model SHA256 | `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76` |

The validated image was built from the self-contained deployment export. The
rollback image remains available as `legal-notice-detector:r7-handoff`
(`sha256:edd1f77d5c4275c06006df5f77269e506c33a9d99caee3ee7ae6a6155b1367fd`).
It was rebuilt from the preserved full Dockerfile and the original immutable
OpenSandbox base digest, then passed model warmup.

## Checks

| Check | Result |
| --- | --- |
| Source inference unit tests | 31 passed |
| Deployment inference unit tests | 25 passed |
| Python compilation | Passed |
| PowerShell parser checks | Passed |
| Bash syntax checks in Linux container | Passed |
| Hash-verified focused dependency lock | 122 packages resolved and installed |
| Checkpoint checksum verification | Passed |
| Docker image build | Passed |
| Container R7 warmup | Passed |
| Jupyter entrypoint startup | Passed |
| Persistent Python state | Passed |
| Bash, JavaScript, and TypeScript kernels | Passed |
| `EXECD_CLONE3_COMPAT=true` startup | Passed |
| Required real HTTPS PDF parity job | Passed |

## Real URL Test

URL:

```text
https://alfajr-news.net/uploads/posts/56b19cbeed56fe0bd788e6dc74eb7e0d.pdf
```

Command:

```powershell
.\scripts\run-local-detector.ps1 `
    -Image "legal-notice-detector:r7-python-node-slim-deployment" `
    -PdfUrl "https://alfajr-news.net/uploads/posts/56b19cbeed56fe0bd788e6dc74eb7e0d.pdf" `
    -JobId "deployment-slim-alfajr-56b19cbe"
```

Observed result:

| Metric | Value |
| --- | --- |
| Status | `completed` |
| Source type | `url` |
| Pages | 20 |
| Detections/crops | 87 |
| Rendered pages | 20 |
| Annotated pages | 20 |
| Referenced artifacts | 127, all present |
| Warmup time | 40.78 seconds |
| Detection time | 90.11 seconds |
| Warmed model reused | Yes |
| Source PDF persisted | No |

Local artifacts are under
`output/deployment-slim-alfajr-56b19cbe/`. The result exactly matches the previously
established 20-page/87-detection baseline. Timings are local observations, not
release performance targets. The `output/` directory is intentionally ignored
and is not part of the deployment repository.

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

Commit the reviewed source and deployment repositories, then repeat the
build/smoke/URL checks for the final immutable registry release tag.
