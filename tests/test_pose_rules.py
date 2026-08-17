"""The rule classifier must reproduce the doc's decision table exactly."""

from __future__ import annotations

import pytest

from pose_rules import classify_by_angles

# Angles measured from the three reference images (requirements doc + notebook).
MEASURED = {
    "correct_pose": {
        "left_elbow": 172.2, "right_elbow": 174.1,
        "left_knee": 146.5, "right_knee": 170.5,
        "left_shoulder": 92.4, "right_shoulder": 73.3,
    },
    "incorrect_pose_1": {
        "left_elbow": 169.6, "right_elbow": 178.5,
        "left_knee": 175.2, "right_knee": 174.9,
        "left_shoulder": 93.9, "right_shoulder": 79.3,
    },
    "incorrect_pose_2": {
        "left_elbow": 178.6, "right_elbow": 173.4,
        "left_knee": 143.2, "right_knee": 173.0,
        "left_shoulder": 148.4, "right_shoulder": 22.2,
    },
}


@pytest.mark.parametrize("label", sorted(MEASURED))
def test_reference_poses_classify_to_their_own_label(label):
    assert classify_by_angles(MEASURED[label]) == label


def test_arms_down_with_bent_knee_is_no_decision():
    # Warrior II stance but arms hanging (both shoulders ~40): must not be
    # mistaken for incorrect_pose_2, whose signature is one arm HIGH.
    angles = dict(MEASURED["correct_pose"], left_shoulder=40.2, right_shoulder=42.6)
    assert classify_by_angles(angles) is None


def test_bent_elbows_are_no_decision():
    # pose2 of the old flow: hands on hips, knees straight, elbows bent.
    angles = dict(
        MEASURED["incorrect_pose_1"],
        left_elbow=120.1, right_elbow=116.3,
        left_shoulder=40.2, right_shoulder=42.6,
    )
    assert classify_by_angles(angles) is None


def test_half_bent_knee_is_no_decision():
    # Mid-correction on path A: knee between the bands (160-165).
    angles = dict(MEASURED["incorrect_pose_1"], left_knee=162.0)
    assert classify_by_angles(angles) is None


def test_one_joint_flips_correct_to_incorrect_1():
    angles = dict(MEASURED["correct_pose"], left_knee=175.0)
    assert classify_by_angles(angles) == "incorrect_pose_1"
