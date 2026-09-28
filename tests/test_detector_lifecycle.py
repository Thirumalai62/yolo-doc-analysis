import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from doc_detector import DetectorBusyError, LegalNoticeDetector, ModelError, get_detector


class FakeModel:
    names = {0: "legal_notice"}

    def __init__(self):
        self.predict_calls = 0
        self.predict_kwargs = []

    def predict(self, *args, **kwargs):
        self.predict_calls += 1
        self.predict_kwargs.append(kwargs)
        return []


def write_manifest(root: Path, model_bytes: bytes, class_names=None) -> tuple[Path, Path]:
    model_path = root / "model.pt"
    model_path.write_bytes(model_bytes)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "test": {
                        "file": "model.pt",
                        "sha256": hashlib.sha256(model_bytes).hexdigest(),
                        "class_names": class_names or ["legal_notice"],
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
                            "max_artifact_bytes": 200000,
                            "allow_private_hosts": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, model_path


class DetectorLifecycleTests(unittest.TestCase):
    def test_load_and_warmup_reuse_one_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = write_manifest(Path(temporary), b"model")
            detector = LegalNoticeDetector(model_version="test", manifest_path=manifest)
            model = FakeModel()

            with patch.object(detector, "_create_model", return_value=model) as create:
                first = detector.warmup()
                second = detector.warmup()

            self.assertTrue(detector.loaded)
            self.assertTrue(detector.warmed)
            self.assertEqual(first["model_identity"], second["model_identity"])
            self.assertEqual(model.predict_calls, 1)
            self.assertEqual(model.predict_kwargs[0]["max_det"], 300)
            create.assert_called_once_with()

    def test_get_detector_returns_same_configured_instance(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, model = write_manifest(Path(temporary), b"model")
            first = get_detector(
                model_version="test", manifest_path=manifest, model_path=model
            )
            second = get_detector(
                model_version="test", manifest_path=manifest, model_path=model
            )
            self.assertIs(first, second)

    def test_changed_model_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, model_path = write_manifest(Path(temporary), b"approved")
            model_path.write_bytes(b"changed")
            detector = LegalNoticeDetector(model_version="test", manifest_path=manifest)

            with self.assertRaisesRegex(ModelError, "checksum mismatch"):
                detector.load()

    def test_manifest_rejects_non_legal_notice_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = write_manifest(
                Path(temporary), b"model", class_names=["advertisement"]
            )
            with self.assertRaisesRegex(ModelError, "legal_notice"):
                LegalNoticeDetector(model_version="test", manifest_path=manifest)

    def test_concurrent_operation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = write_manifest(Path(temporary), b"model")
            detector = LegalNoticeDetector(model_version="test", manifest_path=manifest)
            detector._operation_lock.acquire()
            try:
                with self.assertRaises(DetectorBusyError):
                    detector.warmup()
            finally:
                detector._operation_lock.release()


if __name__ == "__main__":
    unittest.main()
