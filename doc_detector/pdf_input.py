"""PDF input validation, in-memory URL download, and page rendering."""

from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPException
import importlib
from io import BytesIO
import ipaddress
import math
from pathlib import Path
import socket
import time
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .errors import InputError


PDF_HEADER = b"%PDF-"
READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PdfSource:
    """A validated PDF source and its non-sensitive type."""

    value: Path | bytes
    kind: str


def _validate_pdf_header(header: bytes, description: str) -> None:
    if PDF_HEADER not in header[:1024]:
        raise InputError(f"{description} is not a valid PDF.")


def _url_host(url: str) -> str:
    parsed = urlsplit(url)
    return parsed.hostname or "remote host"


def is_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _validate_remote_url(url: str, *, allow_private_hosts: bool) -> None:
    if not is_http_url(url):
        raise InputError("pdf_url must be a complete HTTP or HTTPS URL.")
    if allow_private_hosts:
        return
    parsed = urlsplit(url)
    if not parsed.hostname:
        raise InputError("pdf_url must include a hostname.")
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, ValueError) as error:
        raise InputError(f"Could not resolve the PDF host {_url_host(url)}.") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise InputError(f"The PDF host {_url_host(url)} resolves to a non-public address.")


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(
        self,
        allow_private_hosts: bool,
        deadline: float,
        total_timeout_seconds: int,
    ) -> None:
        self.allow_private_hosts = allow_private_hosts
        self.deadline = deadline
        self.total_timeout_seconds = total_timeout_seconds
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise InputError(
                f"PDF download from {_url_host(req.full_url)} exceeded the "
                f"{self.total_timeout_seconds}-second limit."
            )
        req.timeout = min(getattr(req, "timeout", remaining), remaining)
        _validate_remote_url(newurl, allow_private_hosts=self.allow_private_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    def http_error_302(self, req, fp, code, msg, headers):
        class NonDrainingResponse:
            def read(self):
                return b""

            def close(self):
                fp.close()

            def __getattr__(self, name):
                return getattr(fp, name)

        # urllib otherwise drains an unbounded redirect body before following it.
        return super().http_error_302(
            req, cast(Any, NonDrainingResponse()), code, msg, headers
        )

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _open_pdf_url(
    request: Request,
    *,
    timeout_seconds: int,
    allow_private_hosts: bool,
    deadline: float,
    total_timeout_seconds: int,
):
    opener = build_opener(
        _ValidatedRedirectHandler(
            allow_private_hosts,
            deadline,
            total_timeout_seconds,
        )
    )
    return opener.open(request, timeout=timeout_seconds)


def _set_response_timeout(response, timeout_seconds: float) -> None:
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(timeout_seconds)


def _read_response_chunk(response, size: int) -> bytes:
    # read1 performs at most one socket read, allowing the wall-clock deadline
    # to be checked even when a server continuously drip-feeds bytes.
    if callable(getattr(type(response), "read1", None)):
        return response.read1(size)
    return response.read(size)


def download_pdf(
    url: str,
    *,
    timeout_seconds: int,
    total_timeout_seconds: int,
    max_bytes: int,
    allow_private_hosts: bool,
) -> bytes:
    """Download a direct PDF URL into bounded memory without saving its source."""
    _validate_remote_url(url, allow_private_hosts=allow_private_hosts)
    if timeout_seconds <= 0 or total_timeout_seconds <= 0 or max_bytes <= 0:
        raise ValueError("Download timeouts and maximum size must be positive.")

    host = _url_host(url)
    request = Request(
        url,
        headers={"Accept": "application/pdf", "User-Agent": "legal-notice-detector/1.0"},
    )
    deadline = time.monotonic() + total_timeout_seconds
    try:
        with _open_pdf_url(
            request,
            timeout_seconds=min(timeout_seconds, total_timeout_seconds),
            allow_private_hosts=allow_private_hosts,
            deadline=deadline,
            total_timeout_seconds=total_timeout_seconds,
        ) as response:
            final_url = response.geturl()
            if final_url:
                _validate_remote_url(final_url, allow_private_hosts=allow_private_hosts)
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = 0
                if declared_size > max_bytes:
                    raise InputError(
                        f"The PDF response from {host} exceeds the {max_bytes}-byte limit."
                    )

            buffer = BytesIO()
            total = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InputError(
                        f"PDF download from {host} exceeded the {total_timeout_seconds}-second limit."
                    )
                _set_response_timeout(response, min(timeout_seconds, remaining))
                chunk = _read_response_chunk(
                    response, min(READ_CHUNK_SIZE, max_bytes - total + 1)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise InputError(
                        f"The PDF response from {host} exceeds the {max_bytes}-byte limit."
                    )
                buffer.write(chunk)
    except InputError:
        raise
    except HTTPError as error:
        raise InputError(f"PDF download failed from {host}: HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError, HTTPException) as error:
        reason = getattr(error, "reason", error)
        raise InputError(f"PDF download failed from {host}: {reason}.") from error

    pdf_data = buffer.getvalue()
    _validate_pdf_header(pdf_data, f"The response from {host}")
    return pdf_data


def resolve_pdf_source(
    *,
    pdf_url: str | None,
    pdf_path: str | Path | None,
    pdf_bytes: bytes | bytearray | memoryview | None,
    timeout_seconds: int,
    total_timeout_seconds: int,
    max_bytes: int,
    allow_private_hosts: bool,
) -> PdfSource:
    """Validate exactly one URL, attached path, or in-memory PDF source."""
    supplied = sum(value is not None for value in (pdf_url, pdf_path, pdf_bytes))
    if supplied != 1:
        raise InputError("Provide exactly one of pdf_url, pdf_path, or pdf_bytes.")

    if pdf_url is not None:
        return PdfSource(
            download_pdf(
                pdf_url,
                timeout_seconds=timeout_seconds,
                total_timeout_seconds=total_timeout_seconds,
                max_bytes=max_bytes,
                allow_private_hosts=allow_private_hosts,
            ),
            "url",
        )

    if pdf_path is not None:
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise InputError(f"Attached PDF does not exist: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise InputError(f"Attached PDF exceeds the {max_bytes}-byte limit: {path.name}")
        with path.open("rb") as file:
            _validate_pdf_header(file.read(1024), f"Attached file {path.name}")
        return PdfSource(path, "path")

    data = bytes(pdf_bytes or b"")
    if len(data) > max_bytes:
        raise InputError(f"In-memory PDF exceeds the {max_bytes}-byte limit.")
    _validate_pdf_header(data, "In-memory input")
    return PdfSource(data, "bytes")


def render_pdf(
    pdf_source: Path | bytes,
    destination: Path,
    *,
    dpi: int,
    max_pages: int,
    max_page_pixels: int,
    max_rendered_bytes: int,
) -> list[Path]:
    """Render a validated PDF path or byte buffer into numbered PNG pages."""
    if min(dpi, max_pages, max_page_pixels, max_rendered_bytes) <= 0:
        raise ValueError("PDF render limits must be positive.")

    pdfium = importlib.import_module("pypdfium2")

    document_input = str(pdf_source) if isinstance(pdf_source, Path) else pdf_source
    try:
        document = pdfium.PdfDocument(document_input)
    except Exception as error:
        raise InputError(f"PDFium could not open the supplied PDF: {error}") from error

    rendered_paths: list[Path] = []
    try:
        page_count = len(document)
        if page_count <= 0:
            raise InputError("The supplied PDF contains no pages.")
        if page_count > max_pages:
            raise InputError(
                f"The supplied PDF contains {page_count} pages; the limit is {max_pages}."
            )

        destination.mkdir(parents=True, exist_ok=True)
        scale = dpi / 72
        total_rendered_bytes = 0
        for page_index in range(page_count):
            page = document[page_index]
            bitmap = None
            image = None
            try:
                width_points, height_points = page.get_size()
                rendered_pixels = math.ceil(width_points * scale) * math.ceil(height_points * scale)
                if rendered_pixels > max_page_pixels:
                    raise InputError(
                        f"PDF page {page_index + 1} would render {rendered_pixels} pixels; "
                        f"the limit is {max_page_pixels}."
                    )
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("RGB")
                output_path = destination / f"page_{page_index + 1:04d}.png"
                image.save(output_path)
                total_rendered_bytes += output_path.stat().st_size
                if total_rendered_bytes > max_rendered_bytes:
                    raise InputError(
                        "Rendered PDF pages exceed the "
                        f"{max_rendered_bytes}-byte output limit."
                    )
                rendered_paths.append(output_path)
            finally:
                if image is not None:
                    image.close()
                if bitmap is not None:
                    bitmap.close()
                page.close()
    finally:
        document.close()
    return rendered_paths
