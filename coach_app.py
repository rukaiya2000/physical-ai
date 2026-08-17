"""Desktop UI: webcam pose classification → MuJoCo motion + audio cue."""

from __future__ import annotations

import argparse
import math
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from coach.audio import AudioCuePlayer
from coach.config import CoachSettings, PoseAction, load_settings
from coach.cv_worker import CameraFrame, CameraStatus, CameraWorker
from coach.presentation import humanize_label, reference_diagnostic
from coach.simulation_worker import (
    MotionRequest,
    SimulationFrame,
    SimulationStatus,
    SimulationWorker,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_IMAGE_WIDTH = 760
PANEL_IMAGE_HEIGHT = 570
SNAPSHOT_COUNTDOWN_SECONDS = 5


class CoachApp:
    def __init__(
        self,
        root: tk.Tk,
        settings: CoachSettings,
        *,
        reference_dir: Path,
        model_path: Path,
        camera: int,
    ):
        self.root = root
        self.settings = settings
        self.reference_dir = reference_dir
        self.model_path = model_path
        self.camera_index = camera
        self.audio = AudioCuePlayer()
        self.camera: CameraWorker | None = None
        self.simulation = SimulationWorker(width=800, height=600)
        self.playing = False
        self.camera_ready = False
        self.countdown_deadline: float | None = None
        self.snapshot_pending = False
        self._camera_photo = None
        self._simulation_photo = None
        self._closing = False

        self._build_ui()
        self._start_workers()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self._poll)

    def _build_ui(self) -> None:
        self.root.title("G1 Yoga Coach")
        self.root.geometry("1680x1050")
        self.root.minsize(1300, 850)
        self.root.configure(bg="#111827")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#111827")
        style.configure("Panel.TFrame", background="#1f2937")
        style.configure(
            "Title.TLabel",
            background="#111827",
            foreground="#f9fafb",
            font=("TkDefaultFont", 30, "bold"),
        )
        style.configure(
            "Heading.TLabel",
            background="#1f2937",
            foreground="#f9fafb",
            font=("TkDefaultFont", 18, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#1f2937",
            foreground="#d1d5db",
            font=("TkDefaultFont", 14),
        )
        style.configure(
            "State.TLabel",
            background="#111827",
            foreground="#67e8f9",
            font=("TkDefaultFont", 16, "bold"),
        )
        style.configure(
            "Preview.TLabel",
            background="#1f2937",
            foreground="#67e8f9",
            font=("TkDefaultFont", 14, "bold"),
        )
        style.configure(
            "Hold.Horizontal.TProgressbar",
            troughcolor="#374151",
            background="#22d3ee",
            thickness=18,
        )
        style.configure(
            "Large.TButton",
            font=("TkDefaultFont", 14, "bold"),
            padding=(16, 10),
        )
        style.configure(
            "Capture.TButton",
            font=("TkDefaultFont", 19, "bold"),
            padding=(24, 14),
        )
        style.configure(
            "Large.TCheckbutton",
            background="#111827",
            foreground="#f9fafb",
            font=("TkDefaultFont", 14),
        )

        outer = ttk.Frame(self.root, style="App.TFrame", padding=16)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="G1 Yoga Coach", style="Title.TLabel").pack(
            side="left"
        )
        self.state_var = tk.StringVar(value="INITIALIZING")
        ttk.Label(header, textvariable=self.state_var, style="State.TLabel").pack(
            side="right"
        )

        video_row = ttk.Frame(outer, style="App.TFrame")
        video_row.pack(fill="both", expand=True)
        video_row.columnconfigure(0, weight=1)
        video_row.columnconfigure(1, weight=1)
        video_row.rowconfigure(0, weight=1)

        camera_panel = ttk.Frame(video_row, style="Panel.TFrame", padding=10)
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(
            camera_panel,
            text="Live angle-rule classifier · Preview only",
            style="Heading.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        self.camera_image = tk.Label(
            camera_panel,
            text="Loading camera and MediaPipe…",
            bg="#030712",
            fg="#9ca3af",
            font=("TkDefaultFont", 16),
        )
        self.camera_image.pack(fill="both", expand=True)
        self.live_pose_var = tk.StringVar(value="Live detector initializing…")
        ttk.Label(
            camera_panel,
            textvariable=self.live_pose_var,
            style="Preview.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        simulation_panel = ttk.Frame(video_row, style="Panel.TFrame", padding=10)
        simulation_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(
            simulation_panel, text="MuJoCo demonstration", style="Heading.TLabel"
        ).pack(anchor="w", pady=(0, 8))
        self.simulation_image = tk.Label(
            simulation_panel,
            text="Loading G1 model…",
            bg="#030712",
            fg="#9ca3af",
            font=("TkDefaultFont", 16),
        )
        self.simulation_image.pack(fill="both", expand=True)

        capture_bar = ttk.Frame(outer, style="App.TFrame")
        capture_bar.pack(fill="x", pady=(14, 0))
        self.capture_button = ttk.Button(
            capture_bar,
            text="Take Pose Snapshot · 5 Second Countdown",
            command=self._start_countdown,
            style="Capture.TButton",
            state="disabled",
        )
        self.capture_button.pack(side="left")
        self.countdown_var = tk.StringVar(
            value="Use the live video to frame your full body, then take a snapshot."
        )
        ttk.Label(
            capture_bar,
            textvariable=self.countdown_var,
            style="State.TLabel",
        ).pack(side="left", padx=(20, 0))

        info = ttk.Frame(outer, style="Panel.TFrame", padding=12)
        info.pack(fill="x", pady=(12, 0))
        info.columnconfigure(1, weight=1)

        ttk.Label(info, text="Detected pose", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.pose_var = tk.StringVar(value="Waiting for camera")
        ttk.Label(info, textvariable=self.pose_var, style="Heading.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )

        ttk.Label(info, text="Snapshot timer", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.countdown_progress = ttk.Progressbar(
            info,
            maximum=100,
            style="Hold.Horizontal.TProgressbar",
        )
        self.countdown_progress.grid(
            row=1, column=1, sticky="ew", padx=(12, 0), pady=(8, 0)
        )

        ttk.Label(info, text="Coach cue", style="Body.TLabel").grid(
            row=2, column=0, sticky="nw", pady=(8, 0)
        )
        self.cue_var = tk.StringVar(value="Move into Warrior II when ready.")
        ttk.Label(
            info,
            textvariable=self.cue_var,
            style="Heading.TLabel",
            wraplength=900,
        ).grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(8, 0))

        self.detail_var = tk.StringVar(value="")
        ttk.Label(
            info,
            textvariable=self.detail_var,
            style="Body.TLabel",
            wraplength=1050,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        controls = ttk.Frame(outer, style="App.TFrame")
        controls.pack(fill="x", pady=(12, 0))
        ttk.Button(
            controls,
            text="Stop camera",
            command=self._stop_camera,
            style="Large.TButton",
        ).pack(
            side="left"
        )
        self.muted = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Mute cues",
            variable=self.muted,
            style="Large.TCheckbutton",
        ).pack(
            side="left", padx=10
        )
        ttk.Label(controls, text="Manual demonstrations:", style="State.TLabel").pack(
            side="left", padx=(18, 8)
        )
        for action in self.settings.actions.values():
            title = action.label.replace("_", " ").title()
            ttk.Button(
                controls,
                text=title,
                command=lambda selected=action: self._trigger(selected, manual=True),
                style="Large.TButton",
            ).pack(side="left", padx=3)

    def _start_workers(self) -> None:
        self.simulation.start()
        self.camera = CameraWorker(
            camera=self.camera_index,
            reference_dir=self.reference_dir,
            model_path=self.model_path,
        )
        self.camera.start()

    def _stop_camera(self) -> None:
        if self.camera is not None:
            self.camera.stop()
        self.camera_ready = False
        self.countdown_deadline = None
        self.snapshot_pending = False
        self.capture_button.configure(state="disabled")
        self.countdown_progress["value"] = 0

    def _start_countdown(self) -> None:
        if not self.camera_ready:
            self.cue_var.set("Wait for the camera to become ready.")
            return
        if self.playing:
            self.cue_var.set("Wait for the current demonstration to finish.")
            return
        if self.countdown_deadline is not None or self.snapshot_pending:
            return
        self.live_pose_var.set("Live detector active · result not submitted")
        self.countdown_deadline = time.monotonic() + SNAPSHOT_COUNTDOWN_SECONDS
        self.capture_button.configure(state="disabled")
        self.pose_var.set("Get into position")
        self.state_var.set("COUNTDOWN")
        self.cue_var.set("Hold your final pose until the snapshot is captured.")
        self.detail_var.set("Only the frame captured at zero will be classified.")
        self._update_countdown()

    def _update_countdown(self) -> None:
        if self.countdown_deadline is None:
            return
        seconds_left = max(
            0,
            math.ceil(self.countdown_deadline - time.monotonic()),
        )
        elapsed = SNAPSHOT_COUNTDOWN_SECONDS - seconds_left
        self.countdown_progress["value"] = (
            elapsed / SNAPSHOT_COUNTDOWN_SECONDS * 100
        )
        if seconds_left > 0:
            self.capture_button.configure(text=f"Snapshot in {seconds_left}…")
            self.countdown_var.set(
                f"{seconds_left} — hold your pose and look toward the camera"
            )
            return

        self.countdown_deadline = None
        self.snapshot_pending = True
        self.countdown_progress["value"] = 100
        self.capture_button.configure(text="Capturing snapshot…")
        self.countdown_var.set("Capturing and classifying one frame…")
        self.state_var.set("CAPTURING")
        if self.camera is not None:
            self.camera.capture_snapshot()

    def _poll(self) -> None:
        if self._closing:
            return
        self._poll_camera_status()
        self._update_countdown()
        self._poll_camera_frame()
        self._poll_snapshot()
        self._poll_simulation_status()
        self._poll_simulation_frame()
        self.root.after(30, self._poll)

    def _poll_camera_status(self) -> None:
        if self.camera is None:
            return
        for status in _drain(self.camera.status):
            assert isinstance(status, CameraStatus)
            if status.kind == "error":
                self.camera_ready = False
                self.state_var.set("CAMERA ERROR")
                self.cue_var.set(status.message)
                self.capture_button.configure(state="disabled")
            elif status.kind == "ready":
                self.camera_ready = True
                self.capture_button.configure(state="normal")
                self.capture_button.configure(
                    text="Take Pose Snapshot · 5 Second Countdown"
                )
                self.countdown_var.set(
                    "Frame your full body, then take a five-second snapshot."
                )
                if not self.playing:
                    self.state_var.set("READY")
            elif status.kind == "stopped":
                self.camera_ready = False
                self.capture_button.configure(state="disabled")
            if status.kind != "stopped":
                self.detail_var.set(status.message)

    def _poll_camera_frame(self) -> None:
        if self.camera is None:
            return
        frames = _drain(self.camera.frames)
        if not frames:
            return
        frame = frames[-1]
        assert isinstance(frame, CameraFrame)
        if frame.prediction is None:
            self.live_pose_var.set("Live preview: no pose · not submitted")
        elif frame.rule_label is None:
            self.live_pose_var.set(
                f"Live preview: classification unavailable · "
                f"{_angle_summary(frame.angles)} · "
                "not submitted"
            )
        else:
            self.live_pose_var.set(
                f"Live preview: {humanize_label(frame.rule_label)} · "
                "angle rules · not submitted"
            )
        self._camera_photo = _photo(
            frame.rgb,
            PANEL_IMAGE_WIDTH,
            PANEL_IMAGE_HEIGHT,
        )
        self.camera_image.configure(image=self._camera_photo, text="")

    def _poll_snapshot(self) -> None:
        if self.camera is None:
            return
        snapshots = _drain(self.camera.snapshots)
        if not snapshots:
            return
        frame = snapshots[-1]
        assert isinstance(frame, CameraFrame)
        self.snapshot_pending = False
        # Show the captured frame for this polling cycle only. The next camera
        # frame immediately resumes the live preview while the demonstration
        # plays and the snapshot result remains visible in the result panel.
        self._camera_photo = _photo(
            frame.rgb,
            PANEL_IMAGE_WIDTH,
            PANEL_IMAGE_HEIGHT,
        )
        self.camera_image.configure(image=self._camera_photo, text="")

        prediction = frame.prediction
        accepted = frame.rule_label

        if prediction is None:
            self.live_pose_var.set("Snapshot result: no person")
            self.pose_var.set("No person")
            self.detail_var.set("No pose was found in the captured snapshot.")
            self.cue_var.set("Step back, keep your full body visible, and try again.")
            self._snapshot_failed()
            return
        elif accepted is None:
            self.live_pose_var.set("Snapshot result: classification unavailable")
            self.pose_var.set("Classification unavailable")
            self.detail_var.set(
                "Pose landmarks were detected, but the rule classifier did "
                "not return a label."
            )
            self.cue_var.set(
                "Restart the coach; every detected pose should receive one "
                "of the three configured decisions."
            )
            self._snapshot_failed()
            return
        action = self.settings.actions.get(accepted)
        if action is None:
            self.live_pose_var.set("Snapshot result: configuration error")
            self.pose_var.set(f"{humanize_label(accepted)} · No action configured")
            self.detail_var.set(
                f"The angle classifier returned {accepted!r}, but actions.yaml "
                "does not define that label."
            )
            self.cue_var.set("Add the missing pose action and restart the coach.")
            self._snapshot_failed()
            return

        self.live_pose_var.set(
            f"Snapshot accepted: {humanize_label(accepted)}"
        )
        self.pose_var.set(humanize_label(accepted))
        self.detail_var.set(
            f"Angle-rule match · {_angle_summary(frame.angles)} · "
            f"{reference_diagnostic(accepted, prediction)}"
        )
        self.countdown_var.set(
            f"Snapshot detected {accepted} — starting demonstration"
        )
        self._trigger(action, manual=True)

    def _snapshot_failed(self) -> None:
        self.state_var.set("TRY AGAIN")
        self.countdown_progress["value"] = 0
        self.capture_button.configure(
            text="Retake Pose Snapshot · 5 Second Countdown",
            state="normal" if self.camera_ready else "disabled",
        )
        self.countdown_var.set("No demonstration started. Retake the snapshot.")

    def _poll_simulation_status(self) -> None:
        for status in _drain(self.simulation.status):
            assert isinstance(status, SimulationStatus)
            if status.kind == "started":
                self.playing = True
                self.state_var.set("PLAYING")
            elif status.kind == "finished":
                self.playing = False
                self.countdown_progress["value"] = 0
                self.state_var.set("READY FOR NEXT POSE")
                self.capture_button.configure(
                    text="Take Another Pose Snapshot · 5 Second Countdown",
                    state="normal" if self.camera_ready else "disabled",
                )
                self.countdown_var.set(
                    "Demonstration complete. Frame your next pose and take another snapshot."
                )
            elif status.kind == "ready" and not self.playing:
                self.state_var.set("READY")
            elif status.kind == "error":
                self.playing = False
                self.state_var.set("SIMULATION ERROR")
                self.cue_var.set(status.message)
                self.capture_button.configure(
                    state="normal" if self.camera_ready else "disabled"
                )

    def _poll_simulation_frame(self) -> None:
        frames = _drain(self.simulation.frames)
        if not frames:
            return
        frame = frames[-1]
        assert isinstance(frame, SimulationFrame)
        self._simulation_photo = _photo(
            frame.rgb,
            PANEL_IMAGE_WIDTH,
            PANEL_IMAGE_HEIGHT,
        )
        self.simulation_image.configure(image=self._simulation_photo, text="")

    def _trigger(self, action: PoseAction, *, manual: bool = False) -> None:
        if self.playing and not manual:
            return
        if action.motion_path is not None and not action.motion_path.exists():
            self.cue_var.set(f"Motion folder not found: {action.motion_path}")
            self.state_var.set("CONFIG ERROR")
            self.capture_button.configure(
                state="normal" if self.camera_ready else "disabled"
            )
            return
        self.playing = True
        self.capture_button.configure(state="disabled")
        self.state_var.set("PLAYING")
        self.cue_var.set(action.cue or action.label.replace("_", " "))
        self.simulation.play(MotionRequest(action.motion, action.motion_path))
        if not self.muted.get():
            error = self.audio.play(action.audio_path)
            if error:
                self.detail_var.set(error)

    def close(self) -> None:
        self._closing = True
        if self.camera is not None:
            self.camera.stop()
        self.simulation.stop()
        self.audio.stop()
        self.root.after(50, self.root.destroy)


def _drain(source: queue.Queue) -> list[object]:
    values: list[object] = []
    while True:
        try:
            values.append(source.get_nowait())
        except queue.Empty:
            return values


def _angle_summary(angles: dict[str, float] | None) -> str:
    if angles is None:
        return "angles unavailable"
    return (
        f"left knee {angles['left_knee']:.0f}° · "
        f"shoulders {angles['left_shoulder']:.0f}°/"
        f"{angles['right_shoulder']:.0f}°"
    )


def _photo(rgb, max_width: int, max_height: int) -> ImageTk.PhotoImage:
    image = Image.fromarray(rgb)
    scale = min(max_width / image.width, max_height / image.height)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    image = image.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "coach" / "actions.yaml",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=PROJECT_ROOT / "pose_images",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "pose_landmarker.task",
    )
    parser.add_argument("--camera", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = load_settings(args.config, project_root=PROJECT_ROOT)
    camera = settings.camera if args.camera is None else args.camera
    root = tk.Tk()
    CoachApp(
        root,
        settings,
        reference_dir=args.references.resolve(),
        model_path=args.model.resolve(),
        camera=camera,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
