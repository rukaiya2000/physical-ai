"""The rule classifier must use shoulders for path B and knees for path A."""

from __future__ import annotations

import unittest

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


class PoseRulesTests(unittest.TestCase):
    def test_reference_poses_classify_to_their_own_label(self):
        for label, angles in MEASURED.items():
            with self.subTest(label=label):
                self.assertEqual(classify_by_angles(angles), label)

    def test_arms_down_with_bent_knee_is_no_decision(self):
        angles = dict(
            MEASURED["correct_pose"], left_shoulder=40.2, right_shoulder=42.6
        )
        self.assertIsNone(classify_by_angles(angles))

    def test_hands_on_hips_is_no_decision(self):
        angles = dict(
            MEASURED["incorrect_pose_1"],
            left_elbow=120.1,
            right_elbow=116.3,
            left_shoulder=40.2,
            right_shoulder=42.6,
        )
        self.assertIsNone(classify_by_angles(angles))

    def test_half_bent_knee_with_level_arms_is_no_decision(self):
        angles = dict(MEASURED["incorrect_pose_1"], left_knee=162.0)
        self.assertIsNone(classify_by_angles(angles))

    def test_one_joint_flips_correct_to_incorrect_1(self):
        angles = dict(MEASURED["correct_pose"], left_knee=175.0)
        self.assertEqual(classify_by_angles(angles), "incorrect_pose_1")

    def test_only_shoulders_flip_correct_to_incorrect_2(self):
        angles = dict(
            MEASURED["correct_pose"],
            left_shoulder=148.4,
            right_shoulder=22.2,
        )
        self.assertEqual(classify_by_angles(angles), "incorrect_pose_2")

    def test_bent_elbows_do_not_block_correct_if_shoulders_are_level(self):
        angles = dict(
            MEASURED["correct_pose"], left_elbow=150.0, right_elbow=148.0
        )
        self.assertEqual(classify_by_angles(angles), "correct_pose")

    def test_knee_slightly_outside_lunge_band_still_incorrect_2(self):
        angles = dict(
            MEASURED["incorrect_pose_2"], left_knee=162.0, right_knee=160.0
        )
        self.assertEqual(classify_by_angles(angles), "incorrect_pose_2")

    def test_mirrored_incorrect_2_still_incorrect_2(self):
        # test_3: right arm high, right knee bent (other Warrior II side).
        angles = {
            "left_elbow": 178.0, "right_elbow": 178.0,
            "left_knee": 169.0, "right_knee": 139.0,
            "left_shoulder": 12.0, "right_shoulder": 149.0,
        }
        self.assertEqual(classify_by_angles(angles), "incorrect_pose_2")

    def test_mirrored_level_arms_is_correct(self):
        # test_5: right knee more bent, arms roughly out.
        angles = {
            "left_elbow": 175.0, "right_elbow": 175.0,
            "left_knee": 167.0, "right_knee": 143.0,
            "left_shoulder": 65.0, "right_shoulder": 93.0,
        }
        self.assertEqual(classify_by_angles(angles), "correct_pose")

    def test_swapped_reference_incorrect_2(self):
        original = MEASURED["incorrect_pose_2"]
        mirrored = {
            "left_elbow": original["right_elbow"],
            "right_elbow": original["left_elbow"],
            "left_knee": original["right_knee"],
            "right_knee": original["left_knee"],
            "left_shoulder": original["right_shoulder"],
            "right_shoulder": original["left_shoulder"],
        }
        self.assertEqual(classify_by_angles(mirrored), "incorrect_pose_2")


if __name__ == "__main__":
    unittest.main()
