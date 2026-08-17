import unittest
from types import SimpleNamespace

import numpy as np

from pose_classifier import PoseClassifier, pose_embedding


def make_landmarks(arm_y: float = 0.4, left_knee_x: float = 0.4):
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0)
              for _ in range(33)]
    coordinates = {
        11: (0.4, 0.3), 12: (0.6, 0.3),
        13: (0.3, arm_y), 14: (0.7, arm_y),
        15: (0.2, arm_y), 16: (0.8, arm_y),
        23: (0.45, 0.55), 24: (0.55, 0.55),
        25: (left_knee_x, 0.75), 26: (0.6, 0.75),
        27: (0.35, 0.95), 28: (0.65, 0.95),
        29: (0.34, 0.97), 30: (0.66, 0.97),
        31: (0.30, 0.98), 32: (0.70, 0.98),
    }
    for index, (x, y) in coordinates.items():
        points[index] = SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)
    return points


def transformed(landmarks, scale: float, dx: float, dy: float):
    return [
        SimpleNamespace(
            x=point.x * scale + dx,
            y=point.y * scale + dy,
            z=point.z * scale,
            visibility=point.visibility,
        )
        for point in landmarks
    ]


class PoseClassifierTests(unittest.TestCase):
    def test_embedding_is_translation_and_scale_invariant(self):
        original = make_landmarks()
        moved = transformed(original, scale=1.7, dx=-0.2, dy=0.1)
        np.testing.assert_allclose(
            pose_embedding(original), pose_embedding(moved), atol=1e-12
        )

    def test_classifier_selects_closest_pose(self):
        arms_down = make_landmarks(arm_y=0.5)
        arms_out = make_landmarks(arm_y=0.3)
        classifier = PoseClassifier(
            {"arms-down": [arms_down], "arms-out": [arms_out]}
        )

        prediction = classifier.predict(
            transformed(arms_out, scale=1.2, dx=0.1, dy=-0.05)
        )

        self.assertEqual(prediction.label, "arms-out")
        self.assertGreater(prediction.similarity, 0.5)


if __name__ == "__main__":
    unittest.main()
