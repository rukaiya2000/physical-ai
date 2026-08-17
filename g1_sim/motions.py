"""Load or synthesize G1 joint trajectories.

If a recorded SONIC clip exists under ``g1_sim/motions/recorded/``, that file
is used. Otherwise a kinematic Warrior II placeholder is generated so
``play_motion`` works before retargeted SONIC files arrive.

Recorded clip formats
---------------------
``name.npz`` with arrays:

- ``joint_pos``: shape ``(T, 29)`` in :data:`g1_sim.joints.JOINT_NAMES` order
- ``root_pos`` (optional): ``(T, 3)``
- ``root_quat`` (optional): ``(T, 4)`` as ``w x y z``
- ``fps`` (optional): scalar

or a directory ``name/`` containing SONIC-deploy ``joint_pos.csv`` (header is
the 29 joint names, one row per timestep). Optional ``body_pos.csv`` /
``body_quat.csv`` for the floating base.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from g1_sim.joints import JOINT_NAMES, STAND_JOINTS, STAND_ROOT_POS, STAND_ROOT_QUAT
from g1_sim.mapping import MOTION_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDED_DIR = Path(__file__).resolve().parent / "motions" / "recorded"
DEFAULT_FPS = 30.0

# Named clips that already live in the repo's sonic/ folder.
SONIC_DIRS: dict[str, Path] = {
    "correction_2": PROJECT_ROOT / "sonic" / "yoga_instructor_humanoid",
}


@dataclass(frozen=True)
class MotionClip:
    """One G1 trajectory that ``play_motion`` can execute."""

    name: str
    joint_pos: np.ndarray  # (T, 29)
    root_pos: np.ndarray  # (T, 3)
    root_quat: np.ndarray  # (T, 4) wxyz
    fps: float
    source: str

    @property
    def n_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def qpos_frame(self, index: int) -> np.ndarray:
        """Return a 36-DoF MuJoCo qpos row (freejoint + 29 joints)."""

        return np.concatenate(
            [self.root_pos[index], self.root_quat[index], self.joint_pos[index]]
        )


def joints_from_overrides(overrides: Mapping[str, float]) -> np.ndarray:
    """Build a 29-vector from the stand pose plus named joint overrides."""

    values = {name: 0.0 for name in JOINT_NAMES}
    values.update(STAND_JOINTS)
    values.update(overrides)
    return np.array([values[name] for name in JOINT_NAMES], dtype=np.float64)


def _smoothstep(n_frames: int) -> np.ndarray:
    if n_frames < 2:
        return np.array([1.0], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_frames)
    return t * t * (3.0 - 2.0 * t)


def _lerp_clip(
    start_joints: np.ndarray,
    end_joints: np.ndarray,
    start_root: np.ndarray,
    end_root: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = _smoothstep(n_frames)[:, None]
    joints = start_joints + (end_joints - start_joints) * alpha
    root = start_root + (end_root - start_root) * alpha
    return joints, root


def _concat(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    joints = np.concatenate([part[0] for part in parts], axis=0)
    roots = np.concatenate([part[1] for part in parts], axis=0)
    return joints, roots


def _hold(joints: np.ndarray, root: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.repeat(joints[None, :], n_frames, axis=0),
        np.repeat(root[None, :], n_frames, axis=0),
    )


# Warrior II on the G1: left leg is the front/bent leg (matches the CV docs).
# Hip pitch is negative for flexion in the Menagerie G1 model (see knees_bent).
_WIDE = {
    "left_hip_roll_joint": 0.42,
    "right_hip_roll_joint": -0.42,
    "left_hip_yaw_joint": 0.18,
    "right_hip_yaw_joint": -0.18,
}
_FRONT_BEND = {
    "left_hip_pitch_joint": -0.55,
    "left_knee_joint": 0.72,
    "left_ankle_pitch_joint": -0.22,
    "right_hip_pitch_joint": 0.08,
    "right_knee_joint": 0.04,
    "right_ankle_pitch_joint": -0.04,
}
_ARMS_LEVEL = {
    "left_shoulder_pitch_joint": 0.05,
    "left_shoulder_roll_joint": 1.42,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.12,
    "right_shoulder_pitch_joint": 0.05,
    "right_shoulder_roll_joint": -1.42,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.12,
}
_ARMS_DIAGONAL = {
    "left_shoulder_pitch_joint": -0.95,
    "left_shoulder_roll_joint": 1.05,
    "left_elbow_joint": 0.12,
    "right_shoulder_pitch_joint": 0.55,
    "right_shoulder_roll_joint": -0.28,
    "right_elbow_joint": 0.18,
}

STAND_JOINT_VEC = joints_from_overrides({})
WRONG1_JOINTS = joints_from_overrides({**_WIDE, **_ARMS_LEVEL})
WRONG2_JOINTS = joints_from_overrides({**_WIDE, **_FRONT_BEND, **_ARMS_DIAGONAL})
CORRECT_JOINTS = joints_from_overrides({**_WIDE, **_FRONT_BEND, **_ARMS_LEVEL})

STAND_ROOT = np.array(STAND_ROOT_POS, dtype=np.float64)
WRONG1_ROOT = np.array([0.0, 0.0, 0.78], dtype=np.float64)
LUNGE_ROOT = np.array([0.0, 0.0, 0.72], dtype=np.float64)


def _identity_quat(n_frames: int) -> np.ndarray:
    quat = np.zeros((n_frames, 4), dtype=np.float64)
    quat[:, 0] = 1.0
    return quat


def _placeholder_correction_1() -> tuple[np.ndarray, np.ndarray]:
    """Stand → wrong pose 1 (arms out, knees straight) → bend into Warrior II."""

    return _concat(
        [
            _lerp_clip(STAND_JOINT_VEC, WRONG1_JOINTS, STAND_ROOT, WRONG1_ROOT, 45),
            _hold(WRONG1_JOINTS, WRONG1_ROOT, 20),
            _lerp_clip(WRONG1_JOINTS, CORRECT_JOINTS, WRONG1_ROOT, LUNGE_ROOT, 60),
            _hold(CORRECT_JOINTS, LUNGE_ROOT, 45),
        ]
    )


def _placeholder_correction_2() -> tuple[np.ndarray, np.ndarray]:
    """Stand → wrong pose 2 (knee bent, arms diagonal) → level the arms."""

    return _concat(
        [
            _lerp_clip(STAND_JOINT_VEC, WRONG2_JOINTS, STAND_ROOT, LUNGE_ROOT, 50),
            _hold(WRONG2_JOINTS, LUNGE_ROOT, 20),
            _lerp_clip(WRONG2_JOINTS, CORRECT_JOINTS, LUNGE_ROOT, LUNGE_ROOT, 55),
            _hold(CORRECT_JOINTS, LUNGE_ROOT, 45),
        ]
    )


def _placeholder_correct_hold() -> tuple[np.ndarray, np.ndarray]:
    """Stand → correct Warrior II and hold."""

    return _concat(
        [
            _lerp_clip(STAND_JOINT_VEC, CORRECT_JOINTS, STAND_ROOT, LUNGE_ROOT, 55),
            _hold(CORRECT_JOINTS, LUNGE_ROOT, 70),
        ]
    )


_PLACEHOLDERS = {
    "correction_1": _placeholder_correction_1,
    "correction_2": _placeholder_correction_2,
    "correct_hold": _placeholder_correct_hold,
}


def _clip_from_arrays(
    name: str,
    joint_pos: np.ndarray,
    root_pos: np.ndarray | None,
    root_quat: np.ndarray | None,
    fps: float,
    source: str,
) -> MotionClip:
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"{name}: joint_pos must have shape (T, {len(JOINT_NAMES)}), "
            f"got {joint_pos.shape}"
        )
    n_frames = joint_pos.shape[0]
    if root_pos is None:
        root_pos = np.repeat(np.array(STAND_ROOT_POS)[None, :], n_frames, axis=0)
    else:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        if root_pos.shape != (n_frames, 3):
            raise ValueError(f"{name}: root_pos must have shape ({n_frames}, 3)")
    if root_quat is None:
        root_quat = _identity_quat(n_frames)
        root_quat[:] = STAND_ROOT_QUAT
    else:
        root_quat = np.asarray(root_quat, dtype=np.float64)
        if root_quat.shape != (n_frames, 4):
            raise ValueError(f"{name}: root_quat must have shape ({n_frames}, 4)")
    return MotionClip(name, joint_pos, root_pos, root_quat, fps, source)


def _load_npz(path: Path, name: str) -> MotionClip:
    with np.load(path) as data:
        fps = float(data["fps"]) if "fps" in data else DEFAULT_FPS
        return _clip_from_arrays(
            name,
            data["joint_pos"],
            data["root_pos"] if "root_pos" in data else None,
            data["root_quat"] if "root_quat" in data else None,
            fps,
            source=str(path),
        )


def _read_csv_matrix(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    body = rows[1:] if any(cell and not _is_float(cell) for cell in rows[0]) else rows
    return np.array([[float(cell) for cell in row if cell != ""] for row in body], dtype=np.float64)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _load_csv_dir(path: Path, name: str) -> MotionClip:
    joint_pos = _read_csv_matrix(path / "joint_pos.csv")
    if joint_pos.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"{path}: expected {len(JOINT_NAMES)} joint columns, got {joint_pos.shape[1]}"
        )
    root_pos = None
    root_quat = None
    body_pos_path = path / "body_pos.csv"
    body_quat_path = path / "body_quat.csv"
    if body_pos_path.exists():
        root_pos = _read_csv_matrix(body_pos_path)[:, :3]
    if body_quat_path.exists():
        root_quat = _read_csv_matrix(body_quat_path)[:, :4]
    return _clip_from_arrays(
        name, joint_pos, root_pos, root_quat, _fps_from_dir(path), str(path)
    )


def _fps_from_dir(path: Path) -> float:
    info_path = path / "info.txt"
    if info_path.exists():
        for line in info_path.read_text().splitlines():
            if line.lower().startswith("target_fps"):
                return float(line.split(":", 1)[1].strip())
    return DEFAULT_FPS


def _load_recorded(name: str) -> MotionClip | None:
    npz_path = RECORDED_DIR / f"{name}.npz"
    if npz_path.exists():
        return _load_npz(npz_path, name)
    csv_dir = RECORDED_DIR / name
    if (csv_dir / "joint_pos.csv").exists():
        return _load_csv_dir(csv_dir, name)
    sonic_dir = SONIC_DIRS.get(name)
    if sonic_dir is not None and (sonic_dir / "joint_pos.csv").exists():
        return _load_csv_dir(sonic_dir, name)
    return None


def load_motion(name: str) -> MotionClip:
    """Load a recorded clip, or fall back to the Warrior II placeholder."""

    if name not in MOTION_NAMES:
        known = ", ".join(MOTION_NAMES)
        raise ValueError(f"Unknown motion {name!r}. Expected one of: {known}")

    recorded = _load_recorded(name)
    if recorded is not None:
        return recorded

    joints, roots = _PLACEHOLDERS[name]()
    return _clip_from_arrays(
        name,
        joints,
        roots,
        None,
        DEFAULT_FPS,
        source="placeholder",
    )
