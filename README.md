# physical-ai

## Setup

Do not commit `.venv`. Recreate the same environment locally:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then in Jupyter / VS Code / Cursor, select the kernel from `.venv`.

## Classify the three poses

The three files in `pose_images/` are used as labelled reference poses. Classify
a new image with:

```bash
python pose_classifier.py path/to/new_pose.jpg
```

The result includes the winning label and a relative similarity score for all
three poses. The score compares the new person's normalized MediaPipe body
landmarks and joint angles with the reference poses; it is not a calibrated
probability.

One image per class is enough for a demo, but it will mainly recognize poses
that look like those exact examples. For better results, add examples showing
different people and small pose/camera variations. Use a shared prefix for each
class, for example:

```text
pose_images/
  Pose1.jpg
  Pose1_2.jpg
  Pose1_3.jpg
  Pose2.jpg
  Pose2_2.jpg
  Pose3.jpg
  Pose3_2.jpg
```

Images such as `Pose1_2.jpg` are averaged into the `Pose1` class prototype.

## MuJoCo G1 / SONIC playback

The simulated Unitree G1 is a playback target, not a training environment. The
CV classifier selects a prerecorded motion; MuJoCo executes it.

```text
webcam → MediaPipe → pose class → play_motion(name) → MuJoCo G1
```

| Human pose | Motion |
| --- | --- |
| `incorrect_pose_1` | `correction_1` |
| `incorrect_pose_2` | `correction_2` |
| `correct_pose` | `correct_hold` |

Install MuJoCo, then fetch the G1 model (meshes stay untracked):

```bash
pip install mujoco
python -m g1_sim.fetch_model
```

Prove that two named clips are different:

```bash
python play_motion.py correction_1
python play_motion.py correction_2
```

On macOS the interactive viewer must be launched with `mjpython`:

```bash
mjpython play_motion.py correction_1
```

Headless (writes a video, no window):

```bash
python play_motion.py correction_1 --no-viewer --record outputs/correction_1.mp4
```

`correction_2` is the real SONIC clip in
`sonic/yoga_instructor_humanoid/` (629 frames at 50 fps). `correction_1` and
`correct_hold` still use kinematic placeholders until those files are added.
Drop extra clips at `g1_sim/motions/recorded/<name>.npz` or a folder with
`joint_pos.csv`.

## Live demo on a MacBook (incorrect 2 → SONIC → MuJoCo)

Run every command from the repo root. No audio. Two windows: webcam preview
+ simulated G1.

Use **Terminal.app** for the webcam process. Camera permission is more
reliable there than inside Cursor. If macOS asks, allow it under
**System Settings → Privacy & Security → Camera → Terminal**.

### 0. One-time setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m g1_sim.fetch_model
```

Optional: play the SONIC clip alone (no webcam):

```bash
mjpython play_motion.py correction_2
```

Close that window before the live test. On macOS the G1 viewer must be started
with `mjpython`, not `python`.

### 1. Start MuJoCo (leave this running)

**Terminal 1** (from the repo root):

```bash
source .venv/bin/activate
mjpython play_motion.py --serve
```

A G1 window should stay open and wait for `correction_2`.

### 2. Start the webcam classifier

**Terminal 2** (from the repo root):

```bash
source .venv/bin/activate
python webcam_coach.py --only-motion correction_2
```

A **webcam coach** window should open. When you match incorrect pose 2, the
overlay looks like `incorrect_pose_2 (xx%) → correction_2`.

`--only-motion correction_2` ignores the other two classes so they cannot
start a placeholder clip.

`webcam_coach.py --dry-run` prints the selected motion without talking to
MuJoCo.

### 3. Do the live test

1. Stand in front of the MacBook, full body in frame, hips facing the camera.
2. Hold **incorrect 2**: front **left** knee already bent, **arms on a
   diagonal** (left/front arm high, right/back arm low). Match
   `pose_images/incorrect_2.jpg`.
3. Hold still for about a second until the overlay stays on
   `incorrect_pose_2`.
4. Terminal 2 should print `incorrect_pose_2 → play_motion('correction_2')`.
5. The MuJoCo G1 should play `yoga_instructor_humanoid` (~12 s).
6. After it finishes, hold the pose again to replay (there is an 8 s
   cooldown).

Press **q** in the webcam window to quit. **Ctrl+C** in Terminal 1 to stop
MuJoCo.

### If it does not fire

- Overlay says `incorrect_pose_1` or `correct_pose`: arms are too level, or
  the front knee is too straight. Exaggerate the diagonal arms.
- Overlay says `no person`: step back so the full body is visible.
- Camera does not open: run Terminal 2 in **Terminal.app**, grant camera
  permission, then retry.
- MuJoCo never moves: confirm Terminal 1 still shows
  `G1 motion server listening` and that you used `mjpython`.
