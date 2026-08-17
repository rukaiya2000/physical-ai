import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from coach.config import load_settings
from coach.presentation import humanize_label, reference_diagnostic
from coach.trigger import StabilityGate, accepted_label
from g1_sim.joints import ISAACLAB_COLUMNS_IN_MUJOCO_ORDER
from pose_classifier import Prediction


class StabilityGateTests(unittest.TestCase):
    def test_requires_the_complete_window(self):
        gate = StabilityGate(3)
        self.assertFalse(gate.update("pose_a").triggered)
        self.assertFalse(gate.update("pose_a").triggered)
        result = gate.update("pose_a")
        self.assertTrue(result.triggered)
        self.assertEqual(result.progress, 1.0)

    def test_label_change_restarts_progress(self):
        gate = StabilityGate(3)
        gate.update("pose_a")
        gate.update("pose_a")
        result = gate.update("pose_b")
        self.assertFalse(result.triggered)
        self.assertAlmostEqual(result.progress, 1 / 3)

    def test_rejects_ambiguous_and_distant_predictions(self):
        prediction = Prediction(
            "pose_a",
            0.8,
            {"pose_a": 0.8, "pose_b": 0.2},
            0.12,
            {"pose_a": 0.12, "pose_b": 0.30},
        )
        self.assertEqual(
            accepted_label(prediction, min_similarity=0.6, max_distance=0.2),
            "pose_a",
        )
        self.assertIsNone(
            accepted_label(prediction, min_similarity=0.9, max_distance=0.2)
        )
        self.assertIsNone(
            accepted_label(prediction, min_similarity=0.6, max_distance=0.1)
        )


class ClassificationPresentationTests(unittest.TestCase):
    def test_rule_label_is_presented_without_embedding_confidence(self):
        prediction = Prediction(
            "incorrect_pose_1",
            0.73,
            {"incorrect_pose_1": 0.73, "correct_pose": 0.27},
            0.12,
            {"incorrect_pose_1": 0.12, "correct_pose": 0.25},
        )
        self.assertEqual(humanize_label("incorrect_pose_1"), "Incorrect Pose 1")
        diagnostic = reference_diagnostic("incorrect_pose_1", prediction)
        self.assertEqual(
            diagnostic,
            "nearest reference agrees: Incorrect Pose 1",
        )
        self.assertNotIn("73", diagnostic)

    def test_embedding_disagreement_is_diagnostic_only(self):
        prediction = Prediction(
            "correct_pose",
            0.8,
            {"correct_pose": 0.8, "incorrect_pose_2": 0.2},
            0.10,
            {"correct_pose": 0.10, "incorrect_pose_2": 0.30},
        )
        self.assertEqual(
            reference_diagnostic("incorrect_pose_2", prediction),
            "nearest reference (diagnostic only): Correct Pose",
        )


class MotionDirectoryTests(unittest.TestCase):
    def test_sonic_csv_is_reordered_for_mujoco(self):
        import g1_sim.motions as motions

        with TemporaryDirectory() as temporary:
            recorded = Path(temporary)
            motion_dir = recorded / "correction_1"
            motion_dir.mkdir()
            with (motion_dir / "joint_pos.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([f"joint_{index}" for index in range(29)])
                writer.writerow(range(29))
            original = motions.RECORDED_DIR
            try:
                motions.RECORDED_DIR = recorded
                clip = motions.load_motion("correction_1")
            finally:
                motions.RECORDED_DIR = original

        expected = np.arange(29)[list(ISAACLAB_COLUMNS_IN_MUJOCO_ORDER)]
        np.testing.assert_array_equal(clip.joint_pos[0], expected)


class CoachConfigTests(unittest.TestCase):
    def test_ui_actions_cover_every_rule_classifier_label(self):
        project_root = Path(__file__).resolve().parents[1]
        settings = load_settings(
            project_root / "coach" / "actions.yaml",
            project_root=project_root,
        )
        self.assertEqual(
            set(settings.actions),
            {"correct_pose", "incorrect_pose_1", "incorrect_pose_2"},
        )

    def test_relative_media_paths_resolve_from_project_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "actions.yaml"
            config.write_text(
                "poses:\n"
                "  pose_a:\n"
                "    motion: demo\n"
                "    motion_dir: motions/demo\n"
                "    audio: audio/demo.wav\n"
            )
            settings = load_settings(config, project_root=root)
        action = settings.actions["pose_a"]
        self.assertEqual(action.motion_path, (root / "motions/demo").resolve())
        self.assertEqual(action.audio_path, (root / "audio/demo.wav").resolve())


if __name__ == "__main__":
    unittest.main()
