"""LMM judge scaffold for video LLM evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class JudgeResult:
    sample_id: str
    score: float
    rationale: str


class LMMJudge:
    """A lightweight baseline judge interface for evaluation experiments."""

    def __init__(self, rubric: str = "Exact-match baseline") -> None:
        self.rubric = rubric

    def evaluate(self, sample_id: str, prediction: str, reference: str) -> JudgeResult:
        normalized_prediction = prediction.strip()
        normalized_reference = reference.strip()
        score = 1.0 if normalized_prediction == normalized_reference else 0.0
        rationale = (
            "Prediction matches reference exactly."
            if score == 1.0
            else "Prediction differs from the reference answer."
        )
        return JudgeResult(sample_id=sample_id, score=score, rationale=rationale)

    def batch_evaluate(self, rows: Iterable[Dict[str, str]]) -> List[JudgeResult]:
        return [
            self.evaluate(
                sample_id=row.get("id", ""),
                prediction=row.get("prediction", ""),
                reference=row.get("reference", ""),
            )
            for row in rows
        ]
