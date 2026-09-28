"""Run one URL job inside the detector image using environment configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from uuid import uuid4

from doc_detector import get_detector


def main() -> None:
    pdf_url = os.environ.get("DETECTOR_PDF_URL", "").strip()
    if not pdf_url:
        raise RuntimeError("DETECTOR_PDF_URL is required.")

    job_id = os.environ.get("DETECTOR_JOB_ID", "").strip() or f"url-{uuid4().hex[:12]}"
    output_root = Path(
        os.environ.get("DOC_DETECTOR_OUTPUT_ROOT", "/tmp/doc-detector/jobs")
    )
    outputs = [
        value.strip()
        for value in os.environ.get(
            "DETECTOR_OUTPUTS", "json,crops,annotated_pages,rendered_pages"
        ).split(",")
        if value.strip()
    ]

    detector = get_detector(model_version=os.environ.get("DETECTOR_MODEL_VERSION", "r7"))
    started = time.perf_counter()
    readiness = detector.warmup()
    warmup_seconds = time.perf_counter() - started
    model_identity = detector.model_identity

    started = time.perf_counter()
    result = detector.detect(
        job_id=job_id,
        pdf_url=pdf_url,
        output_dir=output_root / job_id,
        outputs=outputs,
    )
    detection_seconds = time.perf_counter() - started

    print(
        json.dumps(
            {
                "readiness": readiness,
                "result": result.as_dict(),
                "warmup_seconds": round(warmup_seconds, 2),
                "detection_seconds": round(detection_seconds, 2),
                "model_reused": detector.model_identity == model_identity,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
