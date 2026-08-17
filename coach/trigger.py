"""Pose acceptance and temporal stability logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pose_classifier import Prediction


@dataclass(frozen=True)
class StabilityResult:
    label: str | None
    progress: float
    triggered: bool


def accepted_label(
    prediction: Prediction | None,
    *,
    min_similarity: float,
    max_distance: float,
) -> str | None:
    """Reject missing, ambiguous, or out-of-distribution classifications."""

    if prediction is None:
        return None
    if prediction.similarity < min_similarity or prediction.distance > max_distance:
        return None
    return prediction.label


class StabilityGate:
    """Require the same accepted label for a complete consecutive window."""

    def __init__(self, required_frames: int):
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        self.required_frames = required_frames
        self._labels: deque[str] = deque(maxlen=required_frames)
        self._last_label: str | None = None
        self._armed = True

    def reset(self) -> None:
        self._labels.clear()
        self._last_label = None
        self._armed = True

    def update(self, label: str | None) -> StabilityResult:
        if label is None:
            self.reset()
            return StabilityResult(None, 0.0, False)
        if label != self._last_label:
            self._labels.clear()
            self._armed = True
            self._last_label = label
        self._labels.append(label)
        progress = len(self._labels) / self.required_frames
        triggered = (
            self._armed
            and len(self._labels) == self.required_frames
            and all(value == label for value in self._labels)
        )
        if triggered:
            self._armed = False
        return StabilityResult(label, progress, triggered)
