"""Thread-owned offscreen MuJoCo playback for embedding in the coach UI."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_sim.joints import STAND_ROOT_POS, STAND_ROOT_QUAT
from g1_sim.motions import (
    STAND_JOINT_VEC,
    MotionClip,
    load_motion,
    load_motion_path,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NVIDIA_G1_DIR = (
    PROJECT_ROOT
    / "robotics"
    / "GR00T-WholeBodyControl"
    / "gear_sonic_deploy"
    / "g1"
)


@dataclass(frozen=True)
class MotionRequest:
    name: str
    path: Path | None = None


@dataclass(frozen=True)
class SimulationFrame:
    rgb: np.ndarray
    motion: str | None
    progress: float


@dataclass(frozen=True)
class SimulationStatus:
    kind: str
    message: str
    motion: str | None = None


def _put_latest(target: queue.Queue, value: object) -> None:
    try:
        target.put_nowait(value)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(value)


class SimulationWorker(threading.Thread):
    def __init__(self, *, width: int = 640, height: int = 480):
        super().__init__(name="mujoco-renderer", daemon=True)
        self.width = width
        self.height = height
        self.frames: queue.Queue[SimulationFrame] = queue.Queue(maxsize=1)
        self.status: queue.Queue[SimulationStatus] = queue.Queue()
        self._commands: queue.Queue[MotionRequest] = queue.Queue()
        self._stop_event = threading.Event()

    def play(self, request: MotionRequest) -> None:
        self._commands.put(request)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import mujoco

        renderer = None
        try:
            self.status.put(SimulationStatus("initializing", "Loading MuJoCo G1…"))
            model, model_source = _load_model(mujoco)
            model.vis.global_.offwidth = max(
                model.vis.global_.offwidth,
                self.width,
            )
            model.vis.global_.offheight = max(
                model.vis.global_.offheight,
                self.height,
            )
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(
                model, height=self.height, width=self.width
            )
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = (0.0, 0.0, 0.72)
            camera.distance = 2.7
            camera.azimuth = 135.0
            camera.elevation = -12.0

            stand = np.concatenate(
                [STAND_ROOT_POS, STAND_ROOT_QUAT, STAND_JOINT_VEC]
            )
            _apply_qpos(model, data, stand)
            self._render(renderer, data, camera, None, 0.0)
            self.status.put(
                SimulationStatus("ready", f"MuJoCo ready · {model_source}")
            )

            clip: MotionClip | None = None
            frame_index = 0
            next_frame_at = 0.0
            while not self._stop_event.is_set():
                request = self._latest_request(timeout=0.005)
                if request is not None:
                    clip = (
                        load_motion_path(request.path, name=request.name)
                        if request.path is not None
                        else load_motion(request.name)
                    )
                    frame_index = 0
                    next_frame_at = time.monotonic()
                    self.status.put(
                        SimulationStatus("started", f"Playing {clip.name}", clip.name)
                    )

                if clip is None or time.monotonic() < next_frame_at:
                    continue
                _apply_qpos(model, data, clip.qpos_frame(frame_index))
                progress = (frame_index + 1) / clip.n_frames
                self._render(renderer, data, camera, clip.name, progress)
                frame_index += 1
                next_frame_at += 1.0 / clip.fps
                if frame_index >= clip.n_frames:
                    finished_name = clip.name
                    clip = None
                    self.status.put(
                        SimulationStatus(
                            "finished", f"Finished {finished_name}", finished_name
                        )
                    )
        except Exception as error:  # surfaced in the UI
            self.status.put(SimulationStatus("error", str(error)))
        finally:
            if renderer is not None:
                renderer.close()
            self.status.put(SimulationStatus("stopped", "MuJoCo stopped"))

    def _latest_request(self, *, timeout: float) -> MotionRequest | None:
        try:
            request = self._commands.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                request = self._commands.get_nowait()
            except queue.Empty:
                return request

    def _render(self, renderer, data, camera, motion, progress) -> None:
        renderer.update_scene(data, camera=camera)
        _put_latest(
            self.frames,
            SimulationFrame(renderer.render().copy(), motion, progress),
        )


def _apply_qpos(model, data, qpos: np.ndarray) -> None:
    import mujoco

    data.qpos[: len(qpos)] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _load_model(mujoco):
    """Use Menagerie when installed, otherwise reuse the NVIDIA G1 assets."""

    from g1_sim.fetch_model import SCENE_XML

    if SCENE_XML.exists():
        return mujoco.MjModel.from_xml_path(str(SCENE_XML)), "MuJoCo Menagerie"
    robot_xml = NVIDIA_G1_DIR / "g1_29dof_old.xml"
    scene_xml = NVIDIA_G1_DIR / "scene_empty.xml"
    if robot_xml.exists() and scene_xml.exists():
        from lxml import etree

        scene = etree.parse(str(scene_xml))
        robot = etree.parse(str(robot_xml))
        scene_asset = scene.find("asset")
        robot_asset = robot.find("asset")
        assert scene_asset is not None and robot_asset is not None
        for mesh in robot_asset.findall("mesh"):
            mesh.set("file", str((NVIDIA_G1_DIR / "meshes" / mesh.get("file")).resolve()))
            scene_asset.append(mesh)
        scene_default = scene.find("default")
        robot_default = robot.find("default")
        assert scene_default is not None and robot_default is not None
        for default in robot_default.findall("default"):
            scene_default.append(default)
        world = scene.find("worldbody")
        robot_world = robot.find("worldbody")
        assert world is not None and robot_world is not None
        world.append(robot_world.find("body"))
        xml = etree.tostring(scene, encoding="unicode")
        return mujoco.MjModel.from_xml_string(xml), "NVIDIA GEAR-SONIC"
    raise RuntimeError(
        "No G1 model found. Run 'python -m g1_sim.fetch_model' or place the "
        "NVIDIA checkout at robotics/GR00T-WholeBodyControl."
    )
