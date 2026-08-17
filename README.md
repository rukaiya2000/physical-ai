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
