"""Kinematic playback of a named motion on the simulated Unitree G1."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from g1_sim.fetch_model import ensure_g1_model
from g1_sim.motions import MotionClip, load_motion


class G1Player:
    """Keep one G1 simulation (and optional window) alive across clips."""

    def __init__(self, *, viewer: bool = True):
        self.model, self.data = _load_sim()
        self._viewer = None
        self._clip: MotionClip | None = None
        self._frame_index = 0
        self._next_frame_at = 0.0
        if viewer:
            self._viewer = _open_viewer(self.model, self.data)

    def is_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def sync(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def playing(self) -> bool:
        return self._clip is not None

    def start(self, name: str) -> MotionClip:
        """Start (or replace) a clip without blocking the caller."""

        self._clip = load_motion(name)
        self._frame_index = 0
        self._next_frame_at = time.monotonic()
        return self._clip

    def step(self) -> MotionClip | None:
        """Advance real-time playback when a frame is due.

        Returns the clip once it has completed, otherwise ``None``.
        """

        if self._clip is None or time.monotonic() < self._next_frame_at:
            return None
        clip = self._clip
        _apply_qpos(self.model, self.data, clip.qpos_frame(self._frame_index))
        self._frame_index += 1
        self._next_frame_at += 1.0 / clip.fps
        if self._frame_index >= clip.n_frames:
            self._clip = None
            return clip
        return None

    def play(self, name: str, *, realtime: bool = True) -> MotionClip:
        if not realtime:
            clip = load_motion(name)
            for _ in _advance(self.model, self.data, clip, None, []):
                if not self.is_running():
                    break
                self.sync()
            return clip
        clip = self.start(name)
        while self.playing and self.is_running():
            self.step()
            self.sync()
            time.sleep(0.001)
        return clip


def play_motion(
    name: str,
    *,
    viewer: bool = True,
    record: Path | None = None,
    realtime: bool = True,
) -> MotionClip:
    """Play a prerecorded (or placeholder) clip on the MuJoCo G1.

    This does **not** train a policy. Joint targets are applied kinematically
    so you can verify ``play_motion("correction_1")`` vs
    ``play_motion("correction_2")`` before wiring the webcam classifier.
    """

    clip = load_motion(name)
    model, data = _load_sim()
    _apply_qpos(model, data, clip.qpos_frame(0))

    renderer = None
    frames: list[np.ndarray] = []
    if record is not None:
        import mujoco

        renderer = mujoco.Renderer(model, height=480, width=640)

    if viewer:
        _play_with_viewer(model, data, clip, renderer, frames, realtime)
    else:
        _play_headless(model, data, clip, renderer, frames, realtime)

    if record is not None:
        _write_video(record, frames, clip.fps)
    return clip


def _load_sim():
    import mujoco

    scene = ensure_g1_model()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    return model, data


def _apply_qpos(model, data, qpos: np.ndarray) -> None:
    import mujoco

    data.qpos[: len(qpos)] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _advance(model, data, clip: MotionClip, renderer, frames: list[np.ndarray]):
    for index in range(clip.n_frames):
        _apply_qpos(model, data, clip.qpos_frame(index))
        if renderer is not None:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())
        yield


def _sleep(clip: MotionClip, realtime: bool) -> None:
    if realtime:
        time.sleep(1.0 / clip.fps)


def _play_headless(model, data, clip: MotionClip, renderer, frames, realtime: bool) -> None:
    for _ in _advance(model, data, clip, renderer, frames):
        _sleep(clip, realtime)


def _open_viewer(model, data):
    import mujoco.viewer

    try:
        return mujoco.viewer.launch_passive(model, data)
    except Exception as error:  # noqa: BLE001 - MuJoCo raises a generic error on macOS
        raise RuntimeError(
            "The MuJoCo window could not start. On macOS run:\n"
            "  mjpython play_motion.py --serve\n"
            "Or play one clip:\n"
            "  mjpython play_motion.py correction_2"
        ) from error


def _play_with_viewer(model, data, clip: MotionClip, renderer, frames, realtime: bool) -> None:
    with _open_viewer(model, data) as viewer:
        for _ in _advance(model, data, clip, renderer, frames):
            if not viewer.is_running():
                break
            viewer.sync()
            _sleep(clip, realtime)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError("No frames were rendered; cannot write a video")
    path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
