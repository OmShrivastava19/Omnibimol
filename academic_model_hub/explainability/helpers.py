"""Small explainability normalization helpers."""

from __future__ import annotations


def normalize_confidence(raw_value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    if max_value <= min_value:
        return 0.0
    clamped = min(max(raw_value, min_value), max_value)
    return (clamped - min_value) / (max_value - min_value)
