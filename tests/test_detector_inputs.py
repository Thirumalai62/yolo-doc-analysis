from http.client import HTTPMessage
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.request import Request

from PIL import Image

from doc_detector.errors import InputError, OutputError
from doc_detector.outputs import parse_outputs
from doc_detector.pdf_input import (
    _ValidatedRedirectHandler,
    download_pdf,
    render_pdf,
    resolve_pdf_source,
)


class DetectorInputTests(unittest.TestCase):
    def test_requires_exactly_one_pdf_source(self):
        with self.assertRaisesRegex(InputError, "exactly one"):
            resolve_pdf_source(
                pdf_url=None,
                pdf_path=None,
                pdf_bytes=None,
                timeout_seconds=10,
                total_timeout_seconds=20,
                max_bytes=100,
                allow_private_hosts=False,
            )
        with self.assertRaisesRegex(InputError, "exactly one"):
            resolve_pdf_source(
                pdf_url="https://files.example.com/issue.pdf",
                pdf_path="issue.pdf",
                pdf_bytes=None,
                timeout_seconds=10,
                total_timeout_seconds=20,
                max_bytes=100,
                allow_private_hosts=False,
            )

    def test_attached_pdf_is_used_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attached.pdf"
            path.write_bytes(b"%PDF-1.7\ncontent")

            source = resolve_pdf_source(
                pdf_url=None,
                pdf_path=path,
                pdf_bytes=None,
                timeout_seconds=10,
                total_timeout_seconds=20,
                max_bytes=100,
                allow_private_hosts=False,
            )

            self.assertEqual(source.kind, "path")
            self.assertEqual(source.value, path.resolve())

    def test_url_download_is_bounded_and_kept_in_memory(self):
        response = MagicMock()
        opened = response.__enter__.return_value
        opened.geturl.return_value = "https://cdn.example.com/issue.pdf"
        opened.headers = {"Content-Length": "16"}
        opened.read.side_effect = [b"%PDF-1.7\n", b"content", b""]

        with (
            patch("doc_detector.pdf_input._validate_remote_url"),
            patch("doc_detector.pdf_input._open_pdf_url", return_value=response),
        ):
            value = download_pdf(
                "https://files.example.com/issue.pdf?token=secret",
                timeout_seconds=10,
                total_timeout_seconds=20,
                max_bytes=100,
                allow_private_hosts=False,
            )

        self.assertEqual(value, b"%PDF-1.7\ncontent")

    def test_url_download_rejects_declared_oversize_response(self):
        response = MagicMock()
        opened = response.__enter__.return_value
        opened.geturl.return_value = "https://files.example.com/issue.pdf"
        opened.headers = {"Content-Length": "101"}

        with (
            patch("doc_detector.pdf_input._validate_remote_url"),
            patch("doc_detector.pdf_input._open_pdf_url", return_value=response),
        ):
            with self.assertRaisesRegex(InputError, "exceeds the 100-byte limit"):
                download_pdf(
                    "https://files.example.com/issue.pdf?token=secret",
                    timeout_seconds=10,
                    total_timeout_seconds=20,
                    max_bytes=100,
                    allow_private_hosts=False,
                )

    def test_url_download_rejects_non_pdf_content(self):
        response = MagicMock()
        opened = response.__enter__.return_value
        opened.geturl.return_value = "https://files.example.com/viewer"
        opened.headers = {}
        opened.read.side_effect = [b"<html>viewer</html>", b""]

        with (
            patch("doc_detector.pdf_input._validate_remote_url"),
            patch("doc_detector.pdf_input._open_pdf_url", return_value=response),
        ):
            with self.assertRaisesRegex(InputError, "not a valid PDF"):
                download_pdf(
                    "https://files.example.com/viewer",
                    timeout_seconds=10,
                    total_timeout_seconds=20,
                    max_bytes=100,
                    allow_private_hosts=False,
                )

    def test_private_pdf_host_is_rejected_by_default(self):
        with self.assertRaisesRegex(InputError, "non-public address"):
            download_pdf(
                "http://127.0.0.1/issue.pdf",
                timeout_seconds=10,
                total_timeout_seconds=20,
                max_bytes=100,
                allow_private_hosts=False,
            )

    def test_url_download_rejects_streamed_oversize_response(self):
        response = MagicMock()
        opened = response.__enter__.return_value
        opened.geturl.return_value = "https://files.example.com/issue.pdf"
        opened.headers = {}
        opened.read.side_effect = [b"%PDF-" + b"x" * 95, b"x", b""]

        with (
            patch("doc_detector.pdf_input._validate_remote_url"),
            patch("doc_detector.pdf_input._open_pdf_url", return_value=response),
        ):
            with self.assertRaisesRegex(InputError, "exceeds the 100-byte limit"):
                download_pdf(
                    "https://files.example.com/issue.pdf",
                    timeout_seconds=10,
                    total_timeout_seconds=20,
                    max_bytes=100,
                    allow_private_hosts=False,
                )

    def test_total_download_timeout_stops_drip_response(self):
        class DripResponse:
            headers = {}

            def __init__(self):
                self.read_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def geturl(self):
                return "https://files.example.com/issue.pdf"

            def read1(self, size):
                self.read_count += 1
                return b"%PDF-"

        response = DripResponse()
        with (
            patch("doc_detector.pdf_input._validate_remote_url"),
            patch("doc_detector.pdf_input._open_pdf_url", return_value=response),
            patch("doc_detector.pdf_input.time.monotonic", side_effect=[0, 0, 21]),
        ):
            with self.assertRaisesRegex(InputError, "20-second limit"):
                download_pdf(
                    "https://files.example.com/issue.pdf",
                    timeout_seconds=10,
                    total_timeout_seconds=20,
                    max_bytes=100,
                    allow_private_hosts=False,
                )

        self.assertEqual(response.read_count, 1)

    def test_redirect_body_is_not_drained(self):
        handler = _ValidatedRedirectHandler(
            allow_private_hosts=False,
            deadline=100,
            total_timeout_seconds=20,
        )
        handler.parent = MagicMock()
        response = MagicMock()
        request = Request("https://files.example.com/issue.pdf")
        request.timeout = 10
        headers = HTTPMessage()
        headers["location"] = "https://cdn.example.com/issue.pdf"

        with (
            patch("doc_detector.pdf_input._validate_remote_url"),
            patch("doc_detector.pdf_input.time.monotonic", return_value=0),
        ):
            handler.http_error_302(
                request,
                response,
                302,
                "Found",
                headers,
            )

        response.read.assert_not_called()
        response.close.assert_called_once_with()
        handler.parent.open.assert_called_once()

    def test_unknown_output_type_is_rejected(self):
        with self.assertRaisesRegex(OutputError, "Unsupported output"):
            parse_outputs(["json", "pdf"])

    def test_real_pdf_is_rendered_with_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "small.pdf"
            image = Image.new("RGB", (20, 20), "white")
            try:
                image.save(pdf, "PDF")
            finally:
                image.close()

            pages = render_pdf(
                pdf,
                root / "rendered",
                dpi=72,
                max_pages=1,
                max_page_pixels=10000,
                max_rendered_bytes=100000,
            )
            self.assertEqual([path.name for path in pages], ["page_0001.png"])
            self.assertTrue(pages[0].is_file())


if __name__ == "__main__":
    unittest.main()
