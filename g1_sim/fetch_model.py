"""Download the MuJoCo Menagerie Unitree G1 model (meshes + MJCF)."""

from __future__ import annotations

import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"
MENAGERIE_DIR = THIRD_PARTY / "mujoco_menagerie"
SCENE_XML = MENAGERIE_DIR / "unitree_g1" / "scene.xml"


def ensure_g1_model() -> Path:
    """Clone a sparse Menagerie checkout if the G1 scene is missing."""

    if SCENE_XML.exists():
        return SCENE_XML

    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if not MENAGERIE_DIR.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                MENAGERIE_URL,
                str(MENAGERIE_DIR),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "unitree_g1"],
            cwd=MENAGERIE_DIR,
            check=True,
        )
    elif not SCENE_XML.exists():
        subprocess.run(
            ["git", "sparse-checkout", "set", "unitree_g1"],
            cwd=MENAGERIE_DIR,
            check=True,
        )

    if not SCENE_XML.exists():
        raise FileNotFoundError(
            f"Failed to fetch Unitree G1 model at {SCENE_XML}. "
            "Clone google-deepmind/mujoco_menagerie and sparse-checkout unitree_g1."
        )
    return SCENE_XML


if __name__ == "__main__":
    path = ensure_g1_model()
    print(path)
