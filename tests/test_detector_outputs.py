import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from doc_detector import LegalNoticeDetector, OutputError


class FakeValues:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self):
        self.xyxy = [FakeValues([5.2, 6.4, 40.1, 45.8])]
        self.conf = [0.91234]


class FakeResult:
    boxes = [FakeBox()]

    def save(self, filename):
        Image.new("RGB", (50, 60), "white").save(filename)


class FakeModel:
    names = {0: "legal_notice"}

    def __init__(self, fail=False):
        self.fail = fail
        self.predict_calls = 0
        self.predict_kwargs = []

    def predict(self, *args, **kwargs):
        self.predict_calls += 1
        self.predict_kwargs.append(kwargs)
        if self.fail:
            raise RuntimeError("prediction failed")
        return [FakeResult()]


def create_detector(
    root: Path, model: FakeModel, *, max_artifact_bytes: int = 200000
) -> LegalNoticeDetector:
    model_bytes = b"model"
    model_path = root / "model.pt"
    model_path.write_bytes(model_bytes)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "test": {
                        "file": "model.pt",
                        "sha256": hashlib.sha256(model_bytes).hexdigest(),
                        "class_names": ["legal_notice"],
                        "settings": {
                            "confidence": 0.8,
                            "image_size": 1280,
                            "render_dpi": 200,
                            "device": "cpu",
                            "max_detections": 300,
                        },
                        "limits": {
                            "download_timeout_seconds": 60,
                            "total_download_timeout_seconds": 120,
                            "max_pdf_bytes": 1000,
                            "max_pages": 10,
                            "max_page_pixels": 100000,
                            "max_rendered_bytes": 100000,
                            "max_artifact_bytes": max_artifact_bytes,
                            "allow_private_hosts": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    detector = LegalNoticeDetector(model_version="test", manifest_path=manifest)
    detector._create_model = lambda: model
    return detector


def fake_render(pdf_source, destination, **kwargs):
    destination.mkdir(parents=True, exist_ok=True)
    page = destination / "page_0001.png"
    Image.new("RGB", (50, 60), "white").save(page)
    return [page]


class DetectorOutputTests(unittest.TestCase):
    def test_detection_writes_portable_complete_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = FakeModel()
            detector = create_detector(root, model)
            output = root / "job-output"

            with patch("doc_detector.detector.render_pdf", side_effect=fake_render):
                result = detector.detect(
                    job_id="job-123",
                    output_dir=output,
                    pdf_bytes=b"%PDF-1.7\ncontent",
                )

            self.assertEqual(result.page_count, 1)
            self.assertEqual(result.detection_count, 1)
            self.assertEqual(model.predict_calls, 1)
            self.assertEqual(model.predict_kwargs[0]["max_det"], 300)
            report = json.loads((output / "detections.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report[0]["page"], "rendered/page_0001.png")
            self.assertEqual(report[0]["detections"][0]["bbox_xyxy"], [5, 6, 40, 46])
            self.assertEqual(report[0]["detections"][0]["confidence"], 0.9123)
            self.assertEqual(
                report[0]["detections"][0]["crop"],
                "pages/page_0001/crops/notice_001.png",
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["source_type"], "bytes")
            self.assertEqual(manifest["settings"]["max_detections"], 300)
            self.assertNotIn(str(root), json.dumps(manifest))
            self.assertTrue((output / "pages/page_0001/annotated.png").is_file())

    def test_json_only_removes_temporary_rendered_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = create_detector(root, FakeModel())
            output = root / "json-only"

            with patch("doc_detector.detector.render_pdf", side_effect=fake_render):
                detector.detect(
                    job_id="json-only",
                    output_dir=output,
                    pdf_bytes=b"%PDF-1.7\ncontent",
                    outputs=["json"],
                )

            report = json.loads((output / "detections.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report[0]["page"], "page_0001")
            self.assertNotIn("crop", report[0]["detections"][0])
            self.assertFalse((output / "rendered").exists())
            self.assertFalse((output / "pages").exists())
            self.assertEqual(manifest["artifacts"]["rendered_pages"], [])
            self.assertFalse(any(path.name.startswith(".rendered-") for path in output.iterdir()))

    def test_existing_output_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = create_detector(root, FakeModel())
            output = root / "existing"
            output.mkdir()

            with self.assertRaisesRegex(OutputError, "already exists"):
                detector.detect(
                    job_id="job",
                    output_dir=output,
                    pdf_bytes=b"%PDF-1.7\ncontent",
                )

    def test_configured_output_root_contains_job_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = create_detector(root, FakeModel())
            detector.output_root = root / "jobs"

            with self.assertRaisesRegex(OutputError, "Output destination must be"):
                detector.detect(
                    job_id="job-123",
                    output_dir=root / "outside" / "job-123",
                    pdf_bytes=b"%PDF-1.7\ncontent",
                )

    def test_failed_job_writes_failure_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = FakeModel(fail=True)
            detector = create_detector(root, model)
            output = root / "failed"

            with patch("doc_detector.detector.render_pdf", side_effect=fake_render):
                with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                    detector.detect(
                        job_id="failed-job",
                        output_dir=output,
                        pdf_bytes=b"%PDF-1.7\ncontent",
                    )

            manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["type"], "RuntimeError")

            model.fail = False
            with patch("doc_detector.detector.render_pdf", side_effect=fake_render):
                recovered = detector.detect(
                    job_id="recovered-job",
                    output_dir=root / "recovered",
                    pdf_bytes=b"%PDF-1.7\ncontent",
                    outputs=["json"],
                )
            self.assertEqual(recovered.status, "completed")

    def test_aggregate_artifact_limit_removes_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = create_detector(root, FakeModel(), max_artifact_bytes=1000)
            output = root / "limited"

            with patch("doc_detector.detector.render_pdf", side_effect=fake_render):
                with self.assertRaisesRegex(OutputError, "aggregate limit"):
                    detector.detect(
                        job_id="limited-job",
                        output_dir=output,
                        pdf_bytes=b"%PDF-1.7\ncontent",
                    )

            self.assertEqual([path.name for path in output.iterdir()], ["result.json"])
            manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["type"], "OutputError")

    def test_warmup_and_multiple_jobs_reuse_one_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = FakeModel()
            detector = create_detector(root, model)
            detector.warmup()
            identity = detector.model_identity

            with patch("doc_detector.detector.render_pdf", side_effect=fake_render):
                detector.detect(
                    job_id="first",
                    output_dir=root / "first",
                    pdf_bytes=b"%PDF-1.7\ncontent",
                    outputs=["json"],
                )
                detector.detect(
                    job_id="second",
                    output_dir=root / "second",
                    pdf_bytes=b"%PDF-1.7\ncontent",
                    outputs=["json"],
                )

            self.assertEqual(detector.model_identity, identity)
            self.assertEqual(model.predict_calls, 3)


if __name__ == "__main__":
    unittest.main()
