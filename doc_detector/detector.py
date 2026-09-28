"""Reusable, prewarmable legal-notice inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import Lock
from typing import Any, Iterable, cast

from .errors import DetectorBusyError, DetectorError, ModelError, OutputError
from .outputs import ArtifactBudget, OutputSelection, parse_outputs, relative_path, write_json
from .pdf_input import render_pdf, resolve_pdf_source


DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "model-manifest.json"
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DETECTORS: dict[tuple[str, str, str], "LegalNoticeDetector"] = {}
_DETECTORS_LOCK = Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    version: str
    path: Path
    sha256: str
    class_names: tuple[str, ...]
    confidence: float
    image_size: int
    render_dpi: int
    device: str
    max_detections: int
    download_timeout_seconds: int
    total_download_timeout_seconds: int
    max_pdf_bytes: int
    max_pages: int
    max_page_pixels: int
    max_rendered_bytes: int
    max_artifact_bytes: int
    allow_private_hosts: bool


@dataclass(frozen=True)
class DetectionResult:
    job_id: str
    status: str
    output_dir: Path
    report_path: Path
    result_path: Path
    page_count: int
    detection_count: int
    model_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "report_path": str(self.report_path),
            "result_path": str(self.result_path),
            "page_count": self.page_count,
            "detection_count": self.detection_count,
            "model_version": self.model_version,
        }


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ModelError(f"Model manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelError(f"Model manifest is invalid: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ModelError("Model manifest must use schema_version 1.")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelError(f"Model manifest field {field} must be a positive integer.")
    return value


def _model_spec(
    manifest_path: Path,
    model_version: str,
    model_path_override: Path | None,
) -> ModelSpec:
    manifest = _load_manifest(manifest_path)
    models = manifest.get("models")
    entry = models.get(model_version) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        available = ", ".join(sorted(models)) if isinstance(models, dict) else "none"
        raise ModelError(f"Unknown model version {model_version!r}. Available: {available}.")

    file_value = entry.get("file")
    checksum = entry.get("sha256")
    class_names = entry.get("class_names")
    settings = entry.get("settings")
    limits = entry.get("limits")
    if not isinstance(file_value, str) or not file_value:
        raise ModelError("Model manifest field file must be a non-empty string.")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ModelError("Model manifest field sha256 must be a lowercase SHA256 value.")
    if class_names != ["legal_notice"]:
        raise ModelError("The deployment model must define only the legal_notice class.")
    if not isinstance(settings, dict) or not isinstance(limits, dict):
        raise ModelError("Model manifest must define settings and limits objects.")

    confidence = settings.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ModelError("Model confidence must be between 0 and 1.")
    device = settings.get("device")
    if device != "cpu":
        raise ModelError("The initial deployment model must use the cpu device.")

    configured_path = Path(file_value)
    model_path = model_path_override or (
        configured_path
        if configured_path.is_absolute()
        else manifest_path.parent / configured_path
    )
    return ModelSpec(
        version=model_version,
        path=model_path.expanduser().resolve(),
        sha256=checksum,
        class_names=tuple(cast(list[str], class_names)),
        confidence=float(confidence),
        image_size=_positive_int(settings.get("image_size"), "settings.image_size"),
        render_dpi=_positive_int(settings.get("render_dpi"), "settings.render_dpi"),
        device=device,
        max_detections=_positive_int(
            settings.get("max_detections"), "settings.max_detections"
        ),
        download_timeout_seconds=_positive_int(
            limits.get("download_timeout_seconds"), "limits.download_timeout_seconds"
        ),
        total_download_timeout_seconds=_positive_int(
            limits.get("total_download_timeout_seconds"),
            "limits.total_download_timeout_seconds",
        ),
        max_pdf_bytes=_positive_int(limits.get("max_pdf_bytes"), "limits.max_pdf_bytes"),
        max_pages=_positive_int(limits.get("max_pages"), "limits.max_pages"),
        max_page_pixels=_positive_int(
            limits.get("max_page_pixels"), "limits.max_page_pixels"
        ),
        max_rendered_bytes=_positive_int(
            limits.get("max_rendered_bytes"), "limits.max_rendered_bytes"
        ),
        max_artifact_bytes=_positive_int(
            limits.get("max_artifact_bytes"), "limits.max_artifact_bytes"
        ),
        allow_private_hosts=limits.get("allow_private_hosts") is True,
    )


class LegalNoticeDetector:
    """One validated YOLO model that can process sequential PDF jobs."""

    def __init__(
        self,
        *,
        model_version: str = "r7",
        manifest_path: str | Path | None = None,
        model_path: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        configured_manifest = manifest_path or os.getenv("DOC_DETECTOR_MANIFEST") or DEFAULT_MANIFEST
        configured_model = model_path or os.getenv("DOC_DETECTOR_MODEL_PATH")
        self.manifest_path = Path(configured_manifest).expanduser().resolve()
        self.spec = _model_spec(
            self.manifest_path,
            model_version,
            Path(configured_model) if configured_model else None,
        )
        configured_output_root = output_root or os.getenv("DOC_DETECTOR_OUTPUT_ROOT")
        self.output_root = (
            Path(configured_output_root).expanduser().resolve()
            if configured_output_root
            else None
        )
        self._model: Any | None = None
        self._load_lock = Lock()
        self._operation_lock = Lock()
        self._warmed = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def warmed(self) -> bool:
        return self._warmed

    @property
    def model_identity(self) -> int | None:
        return id(self._model) if self._model is not None else None

    def _create_model(self):
        yolo = importlib.import_module("ultralytics").YOLO
        return yolo(str(self.spec.path))

    def load(self) -> "LegalNoticeDetector":
        """Load and validate the model once for this detector instance."""
        if self._model is not None:
            return self
        with self._load_lock:
            if self._model is not None:
                return self
            if not self.spec.path.is_file():
                raise ModelError(f"Model file does not exist: {self.spec.path}")
            actual_checksum = sha256_file(self.spec.path)
            if actual_checksum != self.spec.sha256:
                raise ModelError(
                    f"Model checksum mismatch for {self.spec.version}: "
                    f"expected {self.spec.sha256}, got {actual_checksum}."
                )
            try:
                model = self._create_model()
            except Exception as error:
                raise ModelError(f"Could not load model {self.spec.version}: {error}") from error
            names = model.names
            actual_names = list(names.values()) if isinstance(names, dict) else list(names)
            if actual_names != list(self.spec.class_names):
                raise ModelError(
                    f"Model {self.spec.version} has classes {actual_names}; "
                    f"expected {list(self.spec.class_names)}."
                )
            self._model = model
        return self

    def warmup(self) -> dict[str, object]:
        """Load the model and run one synthetic CPU prediction."""
        if not self._operation_lock.acquire(blocking=False):
            raise DetectorBusyError("The detector is already processing another operation.")
        try:
            self.load()
            if not self._warmed:
                import numpy as np

                image = np.full((self.spec.image_size, self.spec.image_size, 3), 255, dtype=np.uint8)
                model = self._model
                if model is None:
                    raise ModelError("Model was not available after loading.")
                model.predict(
                    image,
                    imgsz=self.spec.image_size,
                    conf=self.spec.confidence,
                    device=self.spec.device,
                    max_det=self.spec.max_detections,
                    verbose=False,
                )
                self._warmed = True
            return {
                "status": "ready",
                "model_version": self.spec.version,
                "model_sha256": self.spec.sha256,
                "model_identity": self.model_identity,
                "warmed": self._warmed,
            }
        except DetectorError:
            raise
        except Exception as error:
            raise ModelError(f"Model warmup failed: {error}") from error
        finally:
            self._operation_lock.release()

    def detect(
        self,
        *,
        job_id: str,
        output_dir: str | Path,
        pdf_url: str | None = None,
        pdf_path: str | Path | None = None,
        pdf_bytes: bytes | bytearray | memoryview | None = None,
        outputs: Iterable[str] | None = None,
    ) -> DetectionResult:
        """Process one PDF using fixed, manifest-approved inference settings."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise OutputError(
                "job_id must contain 1-128 letters, numbers, dots, underscores, or hyphens "
                "and must start with a letter or number."
            )
        selection = parse_outputs(outputs)
        destination = Path(output_dir).expanduser().resolve()
        if self.output_root is not None:
            expected_destination = (self.output_root / job_id).resolve()
            if destination != expected_destination:
                raise OutputError(
                    f"Output destination must be {expected_destination} for job {job_id}."
                )
        if destination.exists():
            raise OutputError(f"Output destination already exists: {destination}")
        if not self._operation_lock.acquire(blocking=False):
            raise DetectorBusyError("The detector is already processing another job.")

        started_at = utc_now()
        try:
            source = resolve_pdf_source(
                pdf_url=pdf_url,
                pdf_path=pdf_path,
                pdf_bytes=pdf_bytes,
                timeout_seconds=self.spec.download_timeout_seconds,
                total_timeout_seconds=self.spec.total_download_timeout_seconds,
                max_bytes=self.spec.max_pdf_bytes,
                allow_private_hosts=self.spec.allow_private_hosts,
            )
            self.load()
            destination.mkdir(parents=True)
            try:
                return self._detect_locked(
                    job_id=job_id,
                    destination=destination,
                    source_value=source.value,
                    source_kind=source.kind,
                    selection=selection,
                    started_at=started_at,
                )
            except Exception as error:
                if destination.is_dir():
                    failure = {
                        "job_id": job_id,
                        "status": "failed",
                        "model_version": self.spec.version,
                        "started_at": started_at,
                        "completed_at": utc_now(),
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error)[:1024],
                        },
                    }
                    shutil.rmtree(destination, ignore_errors=True)
                    destination.mkdir(parents=True, exist_ok=True)
                    try:
                        write_json(
                            destination / "result.json",
                            failure,
                            budget=ArtifactBudget(self.spec.max_artifact_bytes),
                        )
                    except (OSError, OutputError):
                        shutil.rmtree(destination, ignore_errors=True)
                raise
        finally:
            self._operation_lock.release()

    def _detect_locked(
        self,
        *,
        job_id: str,
        destination: Path,
        source_value: Path | bytes,
        source_kind: str,
        selection: OutputSelection,
        started_at: str,
    ) -> DetectionResult:
        from PIL import Image

        model = self._model
        if model is None:
            raise ModelError("Model was not available after loading.")
        artifact_budget = ArtifactBudget(self.spec.max_artifact_bytes)

        rendered_destination = destination / "rendered"
        temporary_render = not selection.rendered_pages
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".rendered-", dir=str(destination))
        ) if temporary_render else rendered_destination

        try:
            pages = render_pdf(
                source_value,
                temporary_root,
                dpi=self.spec.render_dpi,
                max_pages=self.spec.max_pages,
                max_page_pixels=self.spec.max_page_pixels,
                max_rendered_bytes=self.spec.max_rendered_bytes,
            )
            if selection.rendered_pages:
                for page in pages:
                    artifact_budget.add_file(page)
            report: list[dict[str, object]] = []
            crop_paths: list[str] = []
            annotated_paths: list[str] = []
            rendered_paths: list[str] = []
            detection_count = 0

            for page_index, page in enumerate(pages, start=1):
                page_name = f"page_{page_index:04d}"
                page_root = destination / "pages" / page_name
                result = model.predict(
                    str(page),
                    imgsz=self.spec.image_size,
                    conf=self.spec.confidence,
                    device=self.spec.device,
                    max_det=self.spec.max_detections,
                    verbose=False,
                )[0]
                self._warmed = True

                if selection.annotated_pages:
                    page_root.mkdir(parents=True, exist_ok=True)
                    annotated_path = page_root / "annotated.png"
                    result.save(filename=str(annotated_path))
                    artifact_budget.add_file(annotated_path)
                    annotated_paths.append(relative_path(annotated_path, destination))

                detections: list[dict[str, object]] = []
                with Image.open(page) as opened:
                    image = opened.convert("RGB")
                try:
                    for number, box in enumerate(result.boxes, start=1):
                        x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
                        x1, x2 = max(0, x1), min(image.width, x2)
                        y1, y2 = max(0, y1), min(image.height, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        record: dict[str, object] = {
                            "id": number,
                            "confidence": round(float(box.conf[0]), 4),
                            "bbox_xyxy": [x1, y1, x2, y2],
                        }
                        if selection.crops:
                            crop_dir = page_root / "crops"
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            crop_path = crop_dir / f"notice_{number:03d}.png"
                            crop = image.crop((x1, y1, x2, y2))
                            try:
                                crop.save(crop_path)
                            finally:
                                crop.close()
                            artifact_budget.add_file(crop_path)
                            relative_crop = relative_path(crop_path, destination)
                            record["crop"] = relative_crop
                            crop_paths.append(relative_crop)
                        detections.append(record)
                finally:
                    image.close()

                detection_count += len(detections)
                page_value = (
                    relative_path(page, destination)
                    if selection.rendered_pages
                    else page_name
                )
                if selection.rendered_pages:
                    rendered_paths.append(page_value)
                report.append({"page": page_value, "detections": detections})

            report_path = destination / "detections.json"
            write_json(report_path, report, budget=artifact_budget)
            result_path = destination / "result.json"
            manifest = {
                "job_id": job_id,
                "status": "completed",
                "source_type": source_kind,
                "model": {
                    "version": self.spec.version,
                    "sha256": self.spec.sha256,
                    "class_names": list(self.spec.class_names),
                },
                "settings": {
                    "confidence": self.spec.confidence,
                    "image_size": self.spec.image_size,
                    "render_dpi": self.spec.render_dpi,
                    "device": self.spec.device,
                    "max_detections": self.spec.max_detections,
                },
                "started_at": started_at,
                "completed_at": utc_now(),
                "page_count": len(pages),
                "detection_count": detection_count,
                "reports": {"detections": relative_path(report_path, destination)},
                "artifacts": {
                    "rendered_pages": rendered_paths,
                    "annotated_pages": annotated_paths,
                    "crops": crop_paths,
                },
            }
            write_json(result_path, manifest, budget=artifact_budget)
            return DetectionResult(
                job_id=job_id,
                status="completed",
                output_dir=destination,
                report_path=report_path,
                result_path=result_path,
                page_count=len(pages),
                detection_count=detection_count,
                model_version=self.spec.version,
            )
        finally:
            if temporary_render:
                shutil.rmtree(temporary_root, ignore_errors=True)


def get_detector(
    *,
    model_version: str = "r7",
    manifest_path: str | Path | None = None,
    model_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> LegalNoticeDetector:
    """Return one reusable detector for an approved model configuration."""
    configured_manifest = Path(
        manifest_path or os.getenv("DOC_DETECTOR_MANIFEST") or DEFAULT_MANIFEST
    ).expanduser().resolve()
    configured_model_value = model_path or os.getenv("DOC_DETECTOR_MODEL_PATH")
    configured_model = (
        Path(configured_model_value).expanduser().resolve()
        if configured_model_value
        else None
    )
    configured_output_root_value = output_root or os.getenv("DOC_DETECTOR_OUTPUT_ROOT")
    configured_output_root = (
        Path(configured_output_root_value).expanduser().resolve()
        if configured_output_root_value
        else None
    )
    key = (
        model_version,
        str(configured_manifest),
        "|".join(
            (
                str(configured_model) if configured_model else "",
                str(configured_output_root) if configured_output_root else "",
            )
        ),
    )
    with _DETECTORS_LOCK:
        detector = _DETECTORS.get(key)
        if detector is None:
            detector = LegalNoticeDetector(
                model_version=model_version,
                manifest_path=configured_manifest,
                model_path=configured_model,
                output_root=configured_output_root,
            )
            _DETECTORS[key] = detector
        return detector
