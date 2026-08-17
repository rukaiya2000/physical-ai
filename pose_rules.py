"""Rule-based Warrior II classification from joint angles.

Correct vs incorrect 2 is a **shoulder** split (level arms vs a diagonal).
Incorrect 1 is a **knee** split (front knee straight vs bent). Elbows are
not used: both error poses already have straight arms.

Measured 2D angles match ``extract_joint_angles()`` in the smoketest notebook
and the bands in ``docs/warrior2_pose2_to_pose3_requirements.md``.
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

# Shoulders: one horizontal line at shoulder height (correct / incorrect 1).
LEFT_SHOULDER_LEVEL = (85.0, 115.0)
RIGHT_SHOULDER_LEVEL = (55.0, 95.0)
# Incorrect 2: front arm high, back arm low.
LEFT_SHOULDER_HIGH = 115.0
RIGHT_SHOULDER_LOW = 55.0
# Front (left) knee: bent vs straight. Gap 160–165 is "still moving".
FRONT_KNEE_BENT_MAX = 160.0
FRONT_KNEE_STRAIGHT_MIN = 165.0


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
    """Both arms in one horizontal line. Elbows are ignored."""

    left_lo, left_hi = LEFT_SHOULDER_LEVEL
    right_lo, right_hi = RIGHT_SHOULDER_LEVEL
    return (
        left_lo <= angles["left_shoulder"] <= left_hi
        and right_lo <= angles["right_shoulder"] <= right_hi
    )


def _arms_diagonal(angles: Mapping[str, float]) -> bool:
    """Incorrect 2: front/left arm raised, back/right arm dropped."""

    return (
        angles["left_shoulder"] > LEFT_SHOULDER_HIGH
        and angles["right_shoulder"] < RIGHT_SHOULDER_LOW
    )


def _front_knee_bent(angles: Mapping[str, float]) -> bool:
    return angles["left_knee"] <= FRONT_KNEE_BENT_MAX


def _front_knee_straight(angles: Mapping[str, float]) -> bool:
    return angles["left_knee"] > FRONT_KNEE_STRAIGHT_MIN


def classify_by_angles(angles: Mapping[str, float]) -> str | None:
    """Pick a path from shoulders first, then the front knee.

    Returns ``None`` when the pose matches no row — mid-transition frames,
    both arms hanging, mixed errors. Callers must treat ``None`` as
    "no decision", never as a fourth class.
    """

    diagonal = _arms_diagonal(angles)
    level = _arms_level(angles)
    bent = _front_knee_bent(angles)
    straight = _front_knee_straight(angles)

    # Path B: lunge can be imperfect; the tell is the shoulder diagonal.
    if diagonal and not straight:
        return "incorrect_pose_2"
    if level and bent:
        return "correct_pose"
    # Path A: arms already level, front knee still straight.
    if level and straight:
        return "incorrect_pose_1"
    return None
