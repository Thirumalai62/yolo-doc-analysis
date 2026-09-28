# R7 Inference Parity Record

## Purpose

Confirm that the independent `doc_detector` package preserves the current R7
detection behavior without importing or modifying `main.py`.

## Configuration

| Setting | Value |
| --- | --- |
| Date verified | 2026-09-28 |
| PDF | `input/gulftoday_2026-09-07.pdf` |
| Pages | 16 |
| Model | R7 `candidate.pt` |
| Model SHA256 | `2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76` |
| Confidence | `0.80` |
| Image size | `1280` |
| Render DPI | `200` |
| Device | CPU |
| Maximum detections per page | `300` |

## Result

The existing detection output and the new inference-only package each produced:

- 16 page records.
- 9 accepted detections.
- Identical per-page detection counts.
- Identical four-decimal confidence values.
- Identical rounded `bbox_xyxy` pixel coordinates.

The comparison reported `exact_detection_parity: true` with no differing pages.

The package used the same loaded model instance for warmup and detection. Its
source-PDF path input was read directly, and its JSON-only test removed temporary
rendered pages after inference.

## Scope

This is a local Windows CPU parity check against the existing ignored output
`output/legal_notices/gulftoday_2026-09-07/detections.json`. Linux CPU parity
inside the custom OpenSandbox image remains a release gate because the Docker
daemon was not available during this local implementation session.
