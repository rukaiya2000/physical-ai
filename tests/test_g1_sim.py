import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from g1_sim.joints import JOINT_NAMES
from g1_sim.mapping import motion_for_pose
from g1_sim.motions import CORRECT_JOINTS, RECORDED_DIR, load_motion


class MappingTests(unittest.TestCase):
    def test_pose_labels_select_distinct_motions(self):
        self.assertEqual(motion_for_pose("incorrect_pose_1"), "correction_1")
        self.assertEqual(motion_for_pose("wrong_1"), "correction_1")
        self.assertEqual(motion_for_pose("incorrect_pose_2"), "correction_2")
        self.assertEqual(motion_for_pose("correct_pose"), "correct_hold")

    def test_unknown_pose_raises(self):
        with self.assertRaises(ValueError):
            motion_for_pose("warrior_3")


class MotionClipTests(unittest.TestCase):
    def test_placeholders_end_in_the_correct_pose(self):
        for name in ("correction_1", "correct_hold"):
            clip = load_motion(name)
            np.testing.assert_allclose(clip.joint_pos[-1], CORRECT_JOINTS)
            self.assertEqual(clip.joint_pos.shape[1], len(JOINT_NAMES))
            self.assertEqual(clip.qpos_frame(0).shape, (36,))

    def test_correction_2_loads_the_sonic_clip(self):
        clip = load_motion("correction_2")
        self.assertIn("yoga_instructor_humanoid", clip.source)
        self.assertEqual(clip.n_frames, 629)
        self.assertEqual(clip.fps, 50.0)
        self.assertEqual(clip.joint_pos.shape[1], len(JOINT_NAMES))

    def test_corrections_are_not_the_same_clip(self):
        first = load_motion("correction_1")
        second = load_motion("correction_2")
        mid_first = first.joint_pos[first.n_frames // 3]
        mid_second = second.joint_pos[second.n_frames // 3]
        self.assertGreater(np.linalg.norm(mid_first - mid_second), 0.5)

    def test_recorded_npz_overrides_placeholder(self):
        fake = np.zeros((4, len(JOINT_NAMES)))
        fake[:, 3] = [0.0, 0.2, 0.4, 0.6]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "correction_1.npz"
            np.savez(path, joint_pos=fake, fps=20)
            original = RECORDED_DIR
            try:
                import g1_sim.motions as motions

                motions.RECORDED_DIR = Path(tmp)
                clip = motions.load_motion("correction_1")
            finally:
                motions.RECORDED_DIR = original
            np.testing.assert_allclose(clip.joint_pos, fake)
            self.assertEqual(clip.fps, 20)
            self.assertIn("correction_1.npz", clip.source)


class PlaybackTests(unittest.TestCase):
    def test_play_motion_sets_final_qpos(self):
        from g1_sim.fetch_model import SCENE_XML
        from g1_sim.player import play_motion

        if not SCENE_XML.exists():
            self.skipTest("Unitree G1 model is not downloaded")
        clip = play_motion("correction_1", viewer=False, realtime=False)
        np.testing.assert_allclose(clip.joint_pos[-1], CORRECT_JOINTS)


if __name__ == "__main__":
    unittest.main()
