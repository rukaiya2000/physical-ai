"""Rule-based Warrior II classification from joint angles.

The embedding classifier in ``pose_classifier.py`` measures one global
body-shape distance, where ~92% of the weight sits on raw landmark positions.
Correct and incorrect_1 differ by a single knee angle, so on any person or
camera other than the reference photos that difference drowns in positional
noise.  This module instead applies the decision table from
``docs/warrior2_pose2_to_pose3_requirements.md`` section 4: the leg group and
the arm group are judged separately, against the documented pass bands.

Angles are the same 2D image-plane measurements as ``extract_joint_angles()``
in ``notebooks/mediapipe_smoketest.ipynb``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# Landmark triplets for each measured angle: (a, vertex, c).
ANGLE_TRIPLETS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
}


def calculate_angle(a, b, c) -> float:
    """2D angle ABC in degrees, identical to the smoketest notebook."""

    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-9:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_joint_angles(landmarks: Sequence[object]) -> dict[str, float]:
    """The six angles the requirements doc reasons about."""

    def xy(index: int):
        return (float(landmarks[index].x), float(landmarks[index].y))

    return {
        name: calculate_angle(xy(a), xy(b), xy(c))
        for name, (a, b, c) in ANGLE_TRIPLETS.items()
    }


def _arms_level(angles: Mapping[str, float]) -> bool:
    """Both arms straight and in one horizontal line (doc section 1 bands)."""

    return (
        angles["left_elbow"] >= 160
        and angles["right_elbow"] >= 160
        and 85 <= angles["left_shoulder"] <= 115
        and 55 <= angles["right_shoulder"] <= 95
    )


def _legs_in_lunge(angles: Mapping[str, float]) -> bool:
    """Front (left) knee bent into the pass band, back leg long."""

    return 135 <= angles["left_knee"] <= 160 and angles["right_knee"] >= 165


def classify_by_angles(angles: Mapping[str, float]) -> str | None:
    """Doc section 4: pick a path from the leg and arm groups separately.

    Returns ``None`` when the pose matches no row — mid-transition frames,
    arms hanging down, etc.  Callers must treat ``None`` as "no decision",
    never as a fourth class.
    """

    arms = _arms_level(angles)
    legs = _legs_in_lunge(angles)

    if legs and arms:
        return "correct_pose"
    # Arms already in place but both knees straight.
    if arms and angles["left_knee"] > 165 and angles["right_knee"] >= 165:
        return "incorrect_pose_1"
    # Lunge already there but the arms are a diagonal: front high, back low.
    if legs and angles["left_shoulder"] > 115 and angles["right_shoulder"] < 55:
        return "incorrect_pose_2"
    return None
