"""Classify an image as one of the reference poses using MediaPipe landmarks.

The classifier is intentionally small: each labelled image is converted to a
translation- and scale-normalized landmark embedding, then a query is assigned
to the closest class prototype.  Add more images with the same label to make a
prototype represent a wider range of people and camera positions.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# Correct and incorrect examples can share similar limb angles, so face
# direction, hand points, and landmark depth are useful classification cues.
POSE_LANDMARKS = np.arange(33)


@dataclass(frozen=True)
class Prediction:
    """The best class and the comparison score for every class."""

    label: str
    similarity: float
    similarities: dict[str, float]


def _xy(landmarks: Sequence[object], index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([float(landmark.x), float(landmark.y)], dtype=np.float64)


def _xyz(landmarks: Sequence[object], index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array(
        [float(landmark.x), float(landmark.y), float(landmark.z)],
        dtype=np.float64,
    )


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC as a fraction of 180 degrees."""

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-9:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.arccos(cosine) / np.pi)


def pose_embedding(landmarks: Sequence[object]) -> np.ndarray:
    """Create a position/scale-independent feature vector for one pose.

    MediaPipe's anatomical left/right landmark identities are retained so the
    classifier can distinguish otherwise similar poses performed on each side.
    """

    if len(landmarks) < 33:
        raise ValueError(f"Expected 33 pose landmarks, received {len(landmarks)}")

    points = np.array([_xyz(landmarks, int(i)) for i in POSE_LANDMARKS])
    hip_center = (_xyz(landmarks, 23) + _xyz(landmarks, 24)) / 2.0
    hip_center_xy = (_xy(landmarks, 23) + _xy(landmarks, 24)) / 2.0
    shoulder_center = (_xy(landmarks, 11) + _xy(landmarks, 12)) / 2.0

    # Torso length plus shoulder width is stable even when arms/legs move.
    torso_length = np.linalg.norm(shoulder_center - hip_center_xy)
    shoulder_width = np.linalg.norm(_xy(landmarks, 11) - _xy(landmarks, 12))
    scale = max(float(torso_length + shoulder_width), 1e-6)
    normalized_points = (points - hip_center) / scale

    # Joint angles make the comparison less sensitive to body proportions.
    angle_joints = (
        (11, 13, 15),  # left elbow
        (12, 14, 16),  # right elbow
        (13, 11, 23),  # left shoulder
        (14, 12, 24),  # right shoulder
        (11, 23, 25),  # left hip
        (12, 24, 26),  # right hip
        (23, 25, 27),  # left knee
        (24, 26, 28),  # right knee
    )
    angles = np.array(
        [_angle(_xy(landmarks, a), _xy(landmarks, b), _xy(landmarks, c))
         for a, b, c in angle_joints]
    )

    # The multiplier gives positions and angles similar influence on distance.
    return np.concatenate([normalized_points.ravel(), angles * 1.5])


class PoseClassifier:
    """Nearest-prototype classifier for labelled pose landmarks."""

    def __init__(self, examples: Mapping[str, Sequence[Sequence[object]]]):
        if len(examples) < 2:
            raise ValueError("At least two pose classes are required")

        self.prototypes: dict[str, np.ndarray] = {}
        for label, samples in examples.items():
            embeddings = [pose_embedding(sample) for sample in samples]
            if not embeddings:
                raise ValueError(f"Pose class {label!r} has no examples")
            self.prototypes[label] = np.mean(embeddings, axis=0)

    def predict(self, landmarks: Sequence[object]) -> Prediction:
        embedding = pose_embedding(landmarks)
        labels = list(self.prototypes)
        distances = np.array(
            [np.sqrt(np.mean((embedding - self.prototypes[label]) ** 2))
             for label in labels]
        )

        # These are relative similarities, not calibrated probabilities.
        logits = -distances / 0.10
        weights = np.exp(logits - np.max(logits))
        similarities = weights / weights.sum()
        scores = {
            label: float(score) for label, score in zip(labels, similarities)
        }
        best_index = int(np.argmin(distances))
        best_label = labels[best_index]
        return Prediction(best_label, scores[best_label], scores)


def detect_landmarks(image_path: Path, landmarker: object) -> Sequence[object]:
    """Detect the most prominent person's landmarks in an image."""

    import mediapipe as mp

    result = landmarker.detect(mp.Image.create_from_file(str(image_path)))
    if not result.pose_landmarks:
        raise ValueError(f"No pose detected in {image_path}")
    return result.pose_landmarks[0]


def classifier_from_directory(reference_dir: Path, landmarker: object) -> PoseClassifier:
    """Build classes from images named Label.ext or Label_number.ext."""

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        path for path in reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not image_paths:
        raise ValueError(f"No reference images found in {reference_dir}")

    examples: dict[str, list[Sequence[object]]] = {}
    for image_path in image_paths:
        label = reference_label(image_path)
        examples.setdefault(label, []).append(detect_landmarks(image_path, landmarker))
    return PoseClassifier(examples)


def reference_label(image_path: Path) -> str:
    """Return the class label encoded in a reference-image filename.

    The current baseline filenames map to the three user-facing pose labels.
    A double-underscore numeric suffix denotes an additional example of the
    same class, e.g. ``incorrect_pose_1__2.jpg``.
    """

    label = re.sub(r"__\d+$", "", image_path.stem)
    legacy_incorrect = re.fullmatch(r"incorrect_(\d+)", label)
    if legacy_incorrect:
        return f"incorrect_pose_{legacy_incorrect.group(1)}"
    return label


def _parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="images to classify")
    parser.add_argument(
        "--references",
        type=Path,
        default=project_dir / "pose_images",
        help="directory of labelled reference images (default: pose_images)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "models" / "pose_landmarker.task",
        help="MediaPipe Pose Landmarker model",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from mediapipe.tasks.python import vision

    landmarker = vision.PoseLandmarker.create_from_model_path(str(args.model))
    try:
        classifier = classifier_from_directory(args.references, landmarker)
        for image_path in args.images:
            prediction = classifier.predict(detect_landmarks(image_path, landmarker))
            details = ", ".join(
                f"{label}={score:.1%}"
                for label, score in sorted(prediction.similarities.items())
            )
            print(
                f"{image_path}: {prediction.label} "
                f"(similarity {prediction.similarity:.1%}; {details})"
            )
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
