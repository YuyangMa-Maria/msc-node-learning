"""Typed access to the portable dataset catalogue.

The catalogue stores dataset provenance and the expected relative layout. It
does not contain machine-specific paths, so indexing scripts can be moved to a
new checkout without editing their source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetLicence:
    """Licence metadata that must accompany a registered external dataset."""

    name: str
    identifier: str
    url: str
    redistribution: str


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset together with its role and evidence boundary in this project."""

    id: str
    path: str
    source_url: str
    version: str
    citation: list[str]
    access_notes: str
    licence: DatasetLicence
    modalities: list[str]
    labels: list[str]
    role: str
    used_for: list[str]
    subset_used: str
    expected_layout: str
    notes: str

    def absolute_path(self, project_root: Path, dataset_root: str) -> Path:
        """Resolve the configured relative path only at the point of use."""
        return project_root / dataset_root / self.path


def load_registry(project_root: Path) -> tuple[str, list[DatasetEntry]]:
    """Load the catalogue and convert nested licence records to typed objects."""
    config_path = project_root / "configs" / "datasets.json"
    data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8-sig"))
    entries = [
        DatasetEntry(**{**item, "licence": DatasetLicence(**item["licence"])})
        for item in data["datasets"]
    ]
    return data["dataset_root"], entries
