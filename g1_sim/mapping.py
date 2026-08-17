"""Map MediaPipe pose labels to prerecorded G1 / SONIC motion names."""

from __future__ import annotations

# These are the three clips the simulated G1 can play.
MOTION_NAMES = ("correction_1", "correction_2", "correct_hold")

# Classifier labels (including aliases from the requirements doc) → motion.
MOTION_FROM_POSE: dict[str, str] = {
    "incorrect_pose_1": "correction_1",
    "incorrect_1": "correction_1",
    "wrong_pose_1": "correction_1",
    "wrong_1": "correction_1",
    "incorrect_pose_2": "correction_2",
    "incorrect_2": "correction_2",
    "wrong_pose_2": "correction_2",
    "wrong_2": "correction_2",
    "correct_pose": "correct_hold",
    "correct": "correct_hold",
    "correct_holding": "correct_hold",
}


def motion_for_pose(pose_label: str) -> str:
    """Return the motion name that should play for a classified human pose."""

    try:
        return MOTION_FROM_POSE[pose_label]
    except KeyError as error:
        known = ", ".join(sorted(MOTION_FROM_POSE))
        raise ValueError(
            f"Unknown pose label {pose_label!r}. Expected one of: {known}"
        ) from error
