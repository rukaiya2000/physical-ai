"""Configuration for pose-to-motion and pose-to-audio actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PoseAction:
    label: str
    motion: str
    cue: str
    motion_path: Path | None = None
    audio_path: Path | None = None


@dataclass(frozen=True)
class CoachSettings:
    camera: int
    actions: dict[str, PoseAction]


def _optional_path(value: object, project_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_settings(path: Path, *, project_root: Path) -> CoachSettings:
    """Load and validate a coach action mapping."""

    raw = yaml.safe_load(path.read_text()) or {}
    raw_poses = raw.get("poses")
    if not isinstance(raw_poses, dict) or not raw_poses:
        raise ValueError(f"{path}: expected a non-empty 'poses' mapping")

    actions: dict[str, PoseAction] = {}
    for label, values in raw_poses.items():
        if not isinstance(values, dict):
            raise ValueError(f"{path}: pose {label!r} must contain a mapping")
        motion = str(values.get("motion", "")).strip()
        if not motion:
            raise ValueError(f"{path}: pose {label!r} has no motion name")
        actions[str(label)] = PoseAction(
            label=str(label),
            motion=motion,
            cue=str(values.get("cue", "")).strip(),
            motion_path=_optional_path(values.get("motion_dir"), project_root),
            audio_path=_optional_path(values.get("audio"), project_root),
        )

    return CoachSettings(
        camera=int(raw.get("camera", 0)),
        actions=actions,
    )
