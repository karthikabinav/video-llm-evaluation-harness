"""Dataset integration scaffold for LongVideoBench-style annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class LongVideoBenchDataset:
    """Minimal dataset wrapper for video LLM evaluation experiments."""

    def __init__(self, data_path: str, annotation_file: str) -> None:
        self.data_path = Path(data_path)
        self.annotation_file = self.data_path / annotation_file
        self.samples = self._load_annotations()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        if not self.annotation_file.exists():
            return []
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]

    def get_id(self, index: int) -> Optional[str]:
        return self.samples[index].get("id")
