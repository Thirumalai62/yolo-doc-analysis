"""Focused tests for detection input handling."""

import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import main


class RemotePdfInputTests(unittest.TestCase):
    def test_download_pdf_keeps_response_in_memory(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"%PDF-1.7\ncontent"

        with patch("main.urlopen", return_value=response) as urlopen:
            pdf_data = main.download_pdf("https://files.example.com/issue.pdf?token=secret")

        self.assertEqual(pdf_data, b"%PDF-1.7\ncontent")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://files.example.com/issue.pdf?token=secret")
        self.assertEqual(request.get_header("Accept"), "application/pdf")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], main.PDF_DOWNLOAD_TIMEOUT_SECONDS)

    def test_download_pdf_rejects_non_pdf_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"<html>Not a PDF</html>"

        with patch("main.urlopen", return_value=response):
            with self.assertRaisesRegex(SystemExit, "not a valid PDF"):
                main.download_pdf("https://files.example.com/viewer")

    def test_download_errors_do_not_expose_url_query(self):
        error = HTTPError("https://files.example.com/issue.pdf", 403, "Forbidden", Message(), None)

        with patch("main.urlopen", side_effect=error):
            with self.assertRaises(SystemExit) as raised:
                main.download_pdf("https://files.example.com/issue.pdf?token=secret")

        self.assertIn("HTTP 403", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))

    def test_network_errors_are_reported_cleanly(self):
        with patch("main.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaisesRegex(SystemExit, "connection refused"):
                main.download_pdf("https://files.example.com/issue.pdf")

    def test_detection_url_renders_downloaded_bytes(self):
        destination = Path("output/run/rendered")
        expected_pages = [destination / "remote_pdf/page_0001.png"]

        with (
            patch("main.download_pdf", return_value=b"%PDF-1.7") as download,
            patch("main.render_pdf", return_value=expected_pages) as render,
        ):
            pages = main.render_detection_input(
                "https://files.example.com/issue.pdf?token=secret", 200, destination
            )

        self.assertEqual(pages, expected_pages)
        download.assert_called_once_with("https://files.example.com/issue.pdf?token=secret")
        render.assert_called_once_with(b"%PDF-1.7", destination / "remote_pdf", 200)

    def test_detection_local_input_uses_existing_flow(self):
        destination = Path("output/run/rendered")
        local_input = Path("input/issues")
        expected_pages = [destination / "issue/page_0001.png"]

        with (
            patch("main.project_path", return_value=local_input) as project_path,
            patch("main.render_input_pdfs", return_value=expected_pages) as render,
            patch("main.download_pdf") as download,
        ):
            pages = main.render_detection_input("input/issues", 200, destination)

        self.assertEqual(pages, expected_pages)
        project_path.assert_called_once_with("input/issues")
        render.assert_called_once_with(local_input, 200, destination)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
