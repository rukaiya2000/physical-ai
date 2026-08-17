# Coaching audio

These WAV cues are extracted from the mono audio tracks in the corresponding
reference videos under `pose_images/`. They are started at the same time as the
mapped MuJoCo demonstration by `coach_app.py`.

Regenerate them with:

```bash
ffmpeg -y -i pose_images/correction_incorrect_1.mov -vn -ac 1 -ar 44100 assets/audio/correction_incorrect_1.wav
ffmpeg -y -i pose_images/correction_incorrect_2.mov -vn -ac 1 -ar 44100 assets/audio/correction_incorrect_2.wav
ffmpeg -y -i pose_images/correct_holding.mov -vn -ac 1 -ar 44100 assets/audio/correct_holding.wav
```
