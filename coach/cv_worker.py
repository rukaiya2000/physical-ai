"""Background webcam capture and MediaPipe pose classification."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose_classifier import Prediction, classifier_from_directory
from pose_rules import classify_by_angles, extract_joint_angles


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    # The embedding prediction is retained only as a reference diagnostic.
    prediction: Prediction | None
    # The angle-rule label is authoritative for UI actions and demonstrations.
    rule_label: str | None
    angles: dict[str, float] | None


@dataclass(frozen=True)
class CameraStatus:
    kind: str
    message: str


def _put_latest(target: queue.Queue, value: object) -> None:
    try:
        target.put_nowait(value)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(value)


class CameraWorker(threading.Thread):
    def __init__(
        self,
        *,
        camera: int,
        reference_dir: Path,
        model_path: Path,
    ):
        super().__init__(name="pose-camera", daemon=True)
        self.camera = camera
        self.reference_dir = reference_dir
        self.model_path = model_path
        self.frames: queue.Queue[CameraFrame] = queue.Queue(maxsize=1)
        self.snapshots: queue.Queue[CameraFrame] = queue.Queue(maxsize=1)
        self.status: queue.Queue[CameraStatus] = queue.Queue()
        self._stop_event = threading.Event()
        self._snapshot_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def capture_snapshot(self) -> None:
        """Accept the next continuously detected frame as the snapshot."""

        self._snapshot_event.set()

    def run(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        landmarker = None
        capture = None
        try:
            self.status.put(
                CameraStatus("initializing", "Loading MediaPipe pose model…")
            )
            landmarker = vision.PoseLandmarker.create_from_model_path(
                str(self.model_path)
            )
            classifier = classifier_from_directory(self.reference_dir, landmarker)
            capture = cv2.VideoCapture(self.camera)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera}")
            self.status.put(CameraStatus("ready", f"Camera {self.camera} ready"))

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )
                result = landmarker.detect(mp_image)
                prediction = None
                rule_label = None
                angles = None
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    angles = extract_joint_angles(landmarks)
                    rule_label = classify_by_angles(angles)
                    prediction = classifier.predict(landmarks)
                    _draw_landmarks(rgb, landmarks)

                detected_frame = CameraFrame(
                    rgb.copy(),
                    prediction,
                    rule_label,
                    angles,
                )
                _put_latest(self.frames, detected_frame)
                if self._snapshot_event.is_set():
                    self._snapshot_event.clear()
                    _put_latest(self.snapshots, detected_frame)
        except Exception as error:  # surfaced in the UI
            self.status.put(CameraStatus("error", str(error)))
        finally:
            if capture is not None:
                capture.release()
            if landmarker is not None:
                landmarker.close()
            self.status.put(CameraStatus("stopped", "Camera stopped"))


def _draw_landmarks(rgb: np.ndarray, landmarks: list[object]) -> None:
    import cv2
    import mediapipe as mp

    height, width = rgb.shape[:2]
    for first, second in mp.solutions.pose.POSE_CONNECTIONS:
        a, b = landmarks[first], landmarks[second]
        if min(float(a.visibility), float(b.visibility)) < 0.35:
            continue
        start = (int(a.x * width), int(a.y * height))
        end = (int(b.x * width), int(b.y * height))
        cv2.line(rgb, start, end, (58, 214, 255), 2, cv2.LINE_AA)
    for landmark in landmarks:
        if float(landmark.visibility) < 0.35:
            continue
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(rgb, point, 3, (119, 255, 111), -1, cv2.LINE_AA)
