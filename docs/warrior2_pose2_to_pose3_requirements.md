# Warrior II correction requirements: Pose 2 → Pose 3

**Asana:** Virabhadrasana II (Warrior II)  
**Step sequence source:** `pose_images/pose_correction.mov` (29 s coaching take)  
**Angle source:** `notebooks/mediapipe_smoketest.ipynb` (`extract_joint_angles`, 2D `calculate_angle`)  
**Left / right:** MediaPipe body sides (the person’s left and right). In this dataset the **front / bent knee is the left knee**.

The coach must follow the **video order**. Do not invent extra steps. Pose 3 is the hold at the end of that take, not a textbook 90° lunge.

---

## 1. What the video actually does

Spoken cues from the take (tiny-Whisper transcript, cleaned):

| Time | Cue | What happens on screen |
| --- | --- | --- |
| 0–3 s | “Go in the wrong one.” | Build **Pose 2**: feet apart, hands on hips, both knees straight, look at camera |
| 3–11 s | “Now do the step a bit further away and bend your left knee.” | Widen the stance slightly, then flex the **front (left)** knee. Back (right) leg stays long. Hands still on hips |
| 11–19 s | “And now lift your arms.” | Hands leave the hips; both arms reach to shoulder height |
| 19–23 s | “And now look to the front.” | Head turns to look over the **front (left)** hand |
| 23–29 s | “Perfect. Hold it.” | **Pose 3** — Warrior II hold |

“Right left knee” in the raw audio is the teacher correcting the side: use **left** (the bent knee in Pose 3). “Look to the front” means look toward the **front leg / front hand**, not at the camera.

The stills match this:

| Label | Role in the video | Shape |
| --- | --- | --- |
| Pose 2 | “The wrong one” | Wide stance, **hands on hips**, both knees ~straight, gaze at camera |
| Pose 3 | “Hold it” | Left-leg-forward Warrior II: left knee bent, right leg straight, arms reaching, gaze over the front arm |

---

## 2. Required step list (implement this, in this order)

One spoken cue at a time. Advance only when that step’s MediaPipe check passes.

### Step 1 — Step a bit further away

**Requirement:** Increase stance width from Pose 2. Do not hop; slide one foot out.

**Check:** `|left_ankle.x − right_ankle.x|` (landmarks 27 and 28) must be **greater than Pose 2** and then stay there.

**Cue:** *“Step a bit further away.”*

### Step 2 — Bend the front (left) knee

**Requirement:** Flex the left knee; keep the right knee straight. Front knee tracks over the ankle, not past the toes.

**Check:**

| Joint | Pose 2 | Pose 3 target | Pass band |
| --- | ---: | ---: | --- |
| `left_knee` | 175.2° | 156.0° | **145–165°** |
| `right_knee` | 176.9° | 177.4° | **165–180°** |

**Cue:** *“Bend your left knee over the left ankle. Keep the right leg long.”*

Yoga safety (still required, even if the video does not say it): knee in line with the second toe; do not let it collapse inward.

### Step 3 — Lift the arms

**Requirement:** Hands off the hips. Elbows long. Arms at shoulder height. Palms down (Warrior II standard; the video lift is this line).

**Check:**

| Joint | Pose 2 | Pose 3 target | Pass band |
| --- | ---: | ---: | --- |
| `left_elbow` | 120.1° | 172.0° | **≥ 160°** |
| `right_elbow` | 116.3° | 180.0° | **≥ 160°** |
| `left_shoulder` | 40.2° | 99.8° | **85–115°** |
| `right_shoulder` | 42.6° | 68.4° | **≥ 55°** (back arm is foreshortened in this camera) |

**Cue:** *“Lift your arms to shoulder height.”*

### Step 4 — Look to the front

**Requirement:** Face turns toward the front (left) hand. Chin level.

**Check (until a neck angle exists):** nose landmark `0` sits toward the front-arm side (left wrist `15`), not centered on the chest.

**Cue:** *“Look to the front.”*

### Step 5 — Hold

**Requirement:** Steps 1–4 stay true for a short hold (“one, two”).

**Cue:** *“Perfect. Hold it.”*

---

## 3. Coach priority queue (must match the video)

Evaluate live angles. Emit **only the first failed rule**.

| Priority | Fail if | Cue |
| ---: | --- | --- |
| 1 | Stance not wider than Pose 2 | “Step a bit further away.” |
| 2 | `left_knee` > 165° | “Bend your left knee.” |
| 3 | `right_knee` < 165° | “Keep the right leg long.” |
| 4 | `left_elbow` < 160° or `right_elbow` < 160° or `left_shoulder` < 85° | “Lift your arms.” |
| 5 | Gaze still at camera | “Look to the front.” |
| 6 | All pass | “Perfect. Hold it.” |

Do not cue arms before the front knee. Do not cue gaze before the arms. That is the video.

---

## 4. Notebook angle definitions (do not change)

`calculate_angle(a, b, c)` is the 2D image-plane angle at vertex `b`.

| Key | Landmarks `(a, b, c)` |
| --- | --- |
| `left_elbow` | 11 → **13** → 15 |
| `right_elbow` | 12 → **14** → 16 |
| `left_knee` | 23 → **25** → 27 |
| `right_knee` | 24 → **26** → 28 |
| `left_shoulder` | 13 → **11** → 23 |
| `right_shoulder` | 14 → **12** → 24 |

Elbow / knee **~180°** = straight. Shoulder **~40°** = hands on hips; **~90–100°** = arms lifted.

Pose 3 is a **high** Warrior II (`left_knee` 156°, not 90°). Pass the demo bands above. A 90° front knee is optional extra depth, not required to match this video.

---

## 5. Yoga basics that still apply (do not override the video order)

These are alignment rules, not extra steps:

- Hips stay open to the camera (Warrior II, not Warrior I).
- Torso stays a vertical column; do not lean over the front thigh as the knee bends.
- Shoulders drop away from the ears once the arms are up.
- Breathe; do not lock the breath on the hold.
- Stop on sharp knee, hip, or shoulder pain.

---

## 6. Implementation notes

- Reuse `get_xy`, `calculate_angle`, and `extract_joint_angles` from the smoketest notebook.
- Compare live `angles` to Pose 3 bands; use Pose 2 as the “wrong one” baseline.
- Optional extra metric for Step 1: stance width `abs(lm[27].x - lm[28].x)`.
- Optional extra metric for Step 4: nose `x` vs mid-shoulder `x` vs left wrist `x`.
- Camera is mostly frontal. Trust **front knee + back knee + elbow straightness** more than matching the two shoulder numbers to each other.
