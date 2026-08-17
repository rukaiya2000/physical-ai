"""Webcam → MediaPipe → pose class → play_motion request.

Run the MuJoCo player in another process:

    python play_motion.py --serve

Then:

    python webcam_coach.py
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import zmq
from mediapipe.tasks.python import vision

from g1_sim.mapping import motion_for_pose
from pose_classifier import classifier_from_directory
from pose_rules import classify_by_angles, extract_joint_angles


def _parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--references",
        type=Path,
        default=project_dir / "pose_images",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "models" / "pose_landmarker.task",
    )
    parser.add_argument("--address", default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=8,
        help="require this many agreeing frames before sending a motion",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=8.0,
        help="seconds to wait after a motion is sent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected motion instead of sending it over ZMQ",
    )
    parser.add_argument(
        "--only-motion",
        default=None,
        help="only send this motion (e.g. correction_2); ignore other poses",
    )
    return parser.parse_args()


def _majority_label(window: deque[str]) -> str | None:
    if not window or len(window) < window.maxlen:
        return None
    label, count = Counter(window).most_common(1)[0]
    if count < len(window):
        return None
    return label


def main() -> None:
    args = _parse_args()
    landmarker = vision.PoseLandmarker.create_from_model_path(str(args.model))
    classifier = classifier_from_directory(args.references, landmarker)

    socket = None
    if not args.dry_run:
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.connect(args.address)
        socket.setsockopt(zmq.RCVTIMEO, 120_000)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    window: deque[str] = deque(maxlen=args.stable_frames)
    cooldown_until = 0.0
    print("Stand in Warrior II. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                # The angle rules decide: legs and arms are judged as separate
                # groups (doc section 4).  The embedding classifier compares
                # whole-body shape, which dilutes the single-joint differences
                # between these classes, so it is shown only as a reference.
                angles = extract_joint_angles(landmarks)
                label = classify_by_angles(angles)
                prediction = classifier.predict(landmarks)
                window.append(label)
                overlay = (
                    f"{label} → {motion_for_pose(label)} "
                    f"(embed: {prediction.label} "
                    f"{prediction.similarity:.0%})"
                )
            else:
                window.clear()
                overlay = "no person"

            cv2.putText(
                frame,
                overlay,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("webcam coach", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            stable = _majority_label(window)
            now = time.monotonic()
            if stable is None or now < cooldown_until:
                continue
            motion = motion_for_pose(stable)
            if args.only_motion and motion != args.only_motion:
                continue
            print(f"{stable} → play_motion({motion!r})")
            if socket is None:
                cooldown_until = now + args.cooldown
                continue
            socket.send_string(motion)
            reply = socket.recv_string()
            print(reply)
            cooldown_until = now + args.cooldown
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if socket is not None:
            socket.close(linger=0)


if __name__ == "__main__":
    main()
