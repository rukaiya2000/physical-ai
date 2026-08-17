# Warrior II correction requirements

**Asana:** Virabhadrasana II (Warrior II)

**Videos (source of cues, in this order of authority):**

| File | Role |
| --- | --- |
| `pose_images/correction_incorrect_1.mov` | Fix **incorrect 1** → correct |
| `pose_images/correction_incorrect_2.mov` | Fix **incorrect 2** → correct |
| `pose_images/correct_holding.mov` | Success state: hold the correct pose |

**Stills (ground-truth classes used by `notebooks/mediapipe_smoketest.ipynb`):**

| Notebook image | File | Class label | Shape |
| --- | --- | --- | --- |
| `img1` / `angles_1` | `pose_images/correct_pose.jpg` | `correct_pose` | **Target** Warrior II |
| `img2` / `angles_2` | `pose_images/incorrect_1.jpg` | `incorrect_pose_1` | Arms already out; **both knees straight** |
| `img3` / `angles_3` | `pose_images/incorrect_2.jpg` | `incorrect_pose_2` | Front knee already bent; **arms not level** (front arm up, back arm down) |

**Angles:** 2D `calculate_angle(a, b, c)` via `extract_joint_angles()`. Left / right are MediaPipe body sides. In this dataset the **front / bent leg is the left knee** (person’s left, camera-right).

The coach must follow the **spoken video for that error**. Do not mix the two correction paths. Do not add extra steps.

Angles below are measured in the image plane and rounded to 0.1 degrees.

| Joint | Correct | Incorrect 1 | Incorrect 2 |
| --- | ---: | ---: | ---: |
| `left_elbow` | 172.2° | 169.6° | 178.6° |
| `right_elbow` | 174.1° | 178.5° | 173.4° |
| `left_knee` | 146.5° | 175.2° | 143.2° |
| `right_knee` | 170.5° | 174.9° | 173.0° |
| `left_shoulder` | 92.4° | 93.9° | 148.4° |
| `right_shoulder` | 73.3° | 79.3° | 22.2° |

## 1. Target: correct pose / hold

`correct_pose.jpg` and `correct_holding.mov` are the same shape. The hold video has no correction cues; the teacher confirms the pose and the student stays there.

**What “correct” looks like**

- Wide stance, hips open to the camera
- **Left (front) knee bent**, stacked over the ankle, not past the toes
- **Right (back) leg long**
- Both arms in **one horizontal line** at shoulder height, palms down
- Gaze over the **front (left) hand**
- Torso upright over the pelvis

**Measured angles (`angles_1`):**

| Joint | Correct | Demo pass band |
| --- | ---: | --- |
| `left_knee` | 146.5° | **135–160°** |
| `right_knee` | 170.5° | **165–180°** |
| `left_elbow` | 172.2° | **≥ 160°** |
| `right_elbow` | 174.1° | **≥ 160°** |
| `left_shoulder` | 92.4° | **85–115°** |
| `right_shoulder` | 73.3° | **55–95°** (back arm is foreshortened in this camera) |

A textbook 90° front knee is optional extra depth. Matching this hold is enough.

**Hold cue:** *“Okay, you’re good.”* / *“Hold.”*

---

## 2. Path A — incorrect 1 → correct

**Error:** Warrior II arms and gaze are already in place, but the **front knee is not bent**.

**Video:** `correction_incorrect_1.mov` (~9 s)

**Spoken cue (cleaned):** *“Bend your left knee while keeping your arms straight. Perfect.”*

| Joint | Incorrect 1 | Correct | What must change |
| --- | ---: | ---: | --- |
| `left_knee` | 175.2° | 146.5° | **Bend** about −29° |
| `right_knee` | 174.9° | 170.5° | Keep long |
| `left_elbow` | 169.6° | 172.2° | Keep straight |
| `right_elbow` | 178.5° | 174.1° | Keep straight |
| `left_shoulder` | 93.9° | 92.4° | Keep arms at shoulder height |
| `right_shoulder` | 79.3° | 73.3° | Keep |

### Steps (this video only)

1. **Bend the left knee.** Pass when `left_knee` is 135–160°. Knee tracks the second toe, over the ankle, not past the toes.
2. **Keep the arms straight and level.** Do not drop the hands. Elbows stay ≥ 160°. `left_shoulder` stays 85–115°.
3. **Keep the right leg long.** `right_knee` ≥ 165°.
4. **Hold.** *“Perfect.”*

**Coach queue (incorrect 1 only). First failed rule wins:**

| Priority | Fail if | Cue |
| ---: | --- | --- |
| 1 | `left_knee` > 160° | “Bend your left knee while keeping your arms straight.” |
| 2 | `left_elbow` < 160° or `right_elbow` < 160° or `left_shoulder` < 85° | “Keep your arms straight.” |
| 3 | `right_knee` < 165° | “Keep the right leg long.” |
| 4 | All pass | “Perfect. Hold.” |

Do not tell this student to lift the arms. They are already lifted.

---

## 3. Path B — incorrect 2 → correct

**Error:** The lunge is already there; the **arms are a diagonal** (front/left arm high, back/right arm low) and the gaze follows the high hand.

**Video:** `correction_incorrect_2.mov` (~13 s)

**Spoken cue (cleaned):** *“Lower your left arm so it’s straight, and lift your right arm, and hold. Perfect.”*

“Straight” here means **horizontal, in one line with the other arm**, not elbow lock (elbows are already straight).

| Joint | Incorrect 2 | Correct | What must change |
| --- | ---: | ---: | --- |
| `left_shoulder` | 148.4° | 92.4° | **Lower** the front arm about −56° |
| `right_shoulder` | 22.2° | 73.3° | **Lift** the back arm about +51° |
| `left_knee` | 143.2° | 146.5° | Keep the bend |
| `right_knee` | 173.0° | 170.5° | Keep long |
| `left_elbow` | 178.6° | 172.2° | Keep straight |
| `right_elbow` | 173.4° | 174.1° | Keep straight |

### Steps (this video only)

1. **Lower the left arm** until it is at shoulder height (`left_shoulder` 85–115°).
2. **Lift the right arm** until both arms make one horizontal line (`right_shoulder` ≥ 55°, ideally ~73°).
3. **Gaze** settles over the front hand once the arms are level (the video does not give a separate gaze cue).
4. **Hold.** *“Perfect.”*

Keep the left knee bent while the arms move. Do not straighten the front leg.

**Coach queue (incorrect 2 only). First failed rule wins:**

| Priority | Fail if | Cue |
| ---: | --- | --- |
| 1 | `left_shoulder` > 115° | “Lower your left arm so it’s straight.” |
| 2 | `right_shoulder` < 55° | “Lift your right arm.” |
| 3 | `left_knee` > 160° | “Keep the left knee bent.” |
| 4 | All pass | “Hold. Perfect.” |

Do not tell this student to bend the knee first. It is already bent.

---

## 4. How to choose a path

Detect the live pose, then pick **one** video path:

| If you see | Path |
| --- | --- |
| Elbows ≥ 160°, shoulders already ~90°, **both knees ~straight** (`left_knee` > 165°) | **Incorrect 1** |
| Front knee already bent (`left_knee` 135–160°) and **shoulders far apart** (one ≫ 115°, one ≪ 55°) | **Incorrect 2** |
| All correct bands pass | **Hold** (`correct_holding.mov`) |

Never run both arm and knee scripts at once.

---

## 5. Angle definitions (must match the notebook)

`calculate_angle(a, b, c)` is the 2D image-plane angle at vertex `b`.

| Key | MediaPipe landmarks `(a, b, c)` |
| --- | --- |
| `left_elbow` | 11 → **13** → 15 |
| `right_elbow` | 12 → **14** → 16 |
| `left_knee` | 23 → **25** → 27 |
| `right_knee` | 24 → **26** → 28 |
| `left_shoulder` | 13 → **11** → 23 |
| `right_shoulder` | 14 → **12** → 24 |

Elbow / knee **~180°** = straight. Shoulder **~90°** = arm at shoulder height. Shoulder **~150°** = arm raised. Shoulder **~20°** = arm dropped toward the hip.

## 6. Dataset naming

`pose_classifier.py` derives labels from filenames:

- `correct_pose.jpg` → `correct_pose`
- `incorrect_1.jpg` → `incorrect_pose_1`
- `incorrect_2.jpg` → `incorrect_pose_2`
- `label__2.jpg`, `label__3.jpg`, ... → additional examples for `label`

Add several people, camera positions, and small alignment variations to each
class before treating similarity scores as robust outside this demo scene.

## 7. Yoga basics (alignment, not extra steps)

These do not add cues. They constrain a pass:

- Hips stay open to the camera (Warrior II, not Warrior I).
- Torso stays vertical; do not lean over the front thigh as the knee bends.
- Front knee tracks the second toe; do not collapse inward.
- Shoulders drop away from the ears once the arms are level.
- Stop on sharp knee, hip, or shoulder pain.

---

## 8. Implementation notes

- Reuse `get_xy`, `calculate_angle`, and `extract_joint_angles` from the smoketest notebook.
- Classify against `angles_2` (incorrect 1) vs `angles_3` (incorrect 2) vs `angles_1` (correct).
- Optional: wrist `y` difference to test a level arm line on path B.
- Optional: `|knee.x − ankle.x|` on the left leg so the front knee does not pass the toes on path A.
- Camera is mostly frontal. Trust **left knee + right knee + both shoulder heights** more than matching the two shoulder numbers exactly.
