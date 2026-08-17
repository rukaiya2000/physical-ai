"""Rule-based Warrior II classification from joint angles.

Correct vs incorrect 2 is a **shoulder** split (level arms vs a diagonal).
Incorrect 1 is a **knee** split (front knee straight vs bent). Elbows are
not used: both error poses already have straight arms.

Left vs right does not matter: the more bent knee is treated as the front
leg, then the same bands apply.

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
# Only flip left/right when one knee is clearly more bent.
FRONT_SIDE_MARGIN = 8.0
# Scale rule violations so shoulder and knee errors contribute comparably to
# the forced-decision fallback. These are distance scales, not pass bands.
SHOULDER_DISTANCE_SCALE = 15.0
KNEE_DISTANCE_SCALE = 5.0


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


def _distance_to_band(value: float, low: float, high: float) -> float:
    """Distance from ``value`` to a closed interval (zero when inside)."""

    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _scaled_square(distance: float, scale: float) -> float:
    return (distance / scale) ** 2


def _fallback_rule_scores(angles: Mapping[str, float]) -> dict[str, float]:
    """Return distance-to-rule scores; the smallest rule is the best fit."""

    left_level = _distance_to_band(
        angles["left_shoulder"], *LEFT_SHOULDER_LEVEL
    )
    right_level = _distance_to_band(
        angles["right_shoulder"], *RIGHT_SHOULDER_LEVEL
    )
    level_score = _scaled_square(
        left_level, SHOULDER_DISTANCE_SCALE
    ) + _scaled_square(right_level, SHOULDER_DISTANCE_SCALE)

    bent_distance = max(0.0, angles["left_knee"] - FRONT_KNEE_BENT_MAX)
    straight_distance = max(
        0.0, FRONT_KNEE_STRAIGHT_MIN - angles["left_knee"]
    )
    diagonal_left = max(
        0.0, LEFT_SHOULDER_HIGH - angles["left_shoulder"]
    )
    diagonal_right = max(
        0.0, angles["right_shoulder"] - RIGHT_SHOULDER_LOW
    )
    diagonal_knee = max(
        0.0, angles["left_knee"] - FRONT_KNEE_STRAIGHT_MIN
    )

    return {
        "correct_pose": level_score
        + _scaled_square(bent_distance, KNEE_DISTANCE_SCALE),
        "incorrect_pose_1": level_score
        + _scaled_square(straight_distance, KNEE_DISTANCE_SCALE),
        "incorrect_pose_2": (
            _scaled_square(diagonal_left, SHOULDER_DISTANCE_SCALE)
            + _scaled_square(diagonal_right, SHOULDER_DISTANCE_SCALE)
            + _scaled_square(diagonal_knee, KNEE_DISTANCE_SCALE)
        ),
    }


def _swap_sides(angles: Mapping[str, float]) -> dict[str, float]:
    """Rewrite angles as if the left leg were the front / bent leg."""

    swapped = dict(angles)
    for left_name, right_name in (
        ("left_shoulder", "right_shoulder"),
        ("left_knee", "right_knee"),
        ("left_elbow", "right_elbow"),
    ):
        swapped[left_name], swapped[right_name] = (
            angles[right_name],
            angles[left_name],
        )
    return swapped


def _with_front_leg_on_the_left(angles: Mapping[str, float]) -> Mapping[str, float]:
    """Use the more bent knee as the front leg (either Warrior II side)."""

    if angles["left_knee"] - angles["right_knee"] > FRONT_SIDE_MARGIN:
        return _swap_sides(angles)
    return angles


def _classify_left_front(angles: Mapping[str, float]) -> str | None:
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


def classify_by_angles(angles: Mapping[str, float]) -> str:
    """Pick a path from shoulders first, then the front knee.

    The front leg is whichever knee is more bent, so left-facing and
    right-facing Warrior II use the same rules.

    Exact rule matches keep their original priority. Mid-transition frames,
    mixed errors, and poses outside every pass band are forced to the nearest
    of the three rule regions, so a detected person always gets one label.
    """

    normalized = _with_front_leg_on_the_left(angles)
    exact = _classify_left_front(normalized)
    if exact is not None:
        return exact
    scores = _fallback_rule_scores(normalized)
    return min(scores, key=scores.get)
