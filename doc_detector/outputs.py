"""Output selection and portable report helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .errors import OutputError


VISUAL_OUTPUTS = frozenset({"crops", "annotated_pages", "rendered_pages"})
SUPPORTED_OUTPUTS = VISUAL_OUTPUTS | {"json"}
DEFAULT_OUTPUTS = frozenset(SUPPORTED_OUTPUTS)


@dataclass(frozen=True)
class OutputSelection:
    crops: bool
    annotated_pages: bool
    rendered_pages: bool


@dataclass
class ArtifactBudget:
    """Track retained job artifacts against one aggregate byte limit."""

    max_bytes: int
    used_bytes: int = 0

    def add(self, size: int, description: str) -> None:
        if size < 0:
            raise ValueError("Artifact size cannot be negative.")
        if self.used_bytes + size > self.max_bytes:
            raise OutputError(
                f"Generated artifacts exceed the {self.max_bytes}-byte aggregate limit "
                f"while writing {description}."
            )
        self.used_bytes += size

    def add_file(self, path: Path) -> None:
        try:
            self.add(path.stat().st_size, path.name)
        except Exception:
            path.unlink(missing_ok=True)
            raise


def parse_outputs(outputs: Iterable[str] | None) -> OutputSelection:
    values = set(DEFAULT_OUTPUTS if outputs is None else outputs)
    unknown = values - SUPPORTED_OUTPUTS
    if unknown:
        raise OutputError(
            "Unsupported output type(s): "
            + ", ".join(sorted(unknown))
            + ". Supported values are: "
            + ", ".join(sorted(SUPPORTED_OUTPUTS))
            + "."
        )
    return OutputSelection(
        crops="crops" in values,
        annotated_pages="annotated_pages" in values,
        rendered_pages="rendered_pages" in values,
    )


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_json(path: Path, value: object, *, budget: ArtifactBudget | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    content = json.dumps(value, indent=2) + "\n"
    if budget is not None:
        budget.add(len(content.encode("utf-8")), path.name)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
