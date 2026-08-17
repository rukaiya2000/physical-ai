"""CLI: play_motion("correction_1") on the MuJoCo Unitree G1.

Examples
--------
python play_motion.py correction_1
python play_motion.py correction_2 --no-viewer --record outputs/correction_2.mp4
python play_motion.py --serve
"""

from __future__ import annotations

import argparse
from pathlib import Path

from g1_sim.mapping import MOTION_NAMES
from g1_sim.player import play_motion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "motion",
        nargs="?",
        choices=MOTION_NAMES,
        help="prerecorded motion to play on the simulated G1",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run without a MuJoCo window (use with --record)",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="write an mp4 of kinematic playback",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="listen for motion names over ZMQ (default tcp://127.0.0.1:5555)",
    )
    parser.add_argument(
        "--bind",
        default="tcp://127.0.0.1:5555",
        help="ZMQ bind address for --serve",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.serve:
        from g1_sim.zmq_server import serve_motions

        serve_motions(bind=args.bind, viewer=not args.no_viewer)
        return
    if args.motion is None:
        raise SystemExit(
            "Pass a motion name (correction_1, correction_2, correct_hold) "
            "or --serve"
        )
    clip = play_motion(
        args.motion,
        viewer=not args.no_viewer,
        record=args.record,
    )
    print(
        f"played {clip.name} ({clip.n_frames} frames @ {clip.fps:.0f} fps, "
        f"source={clip.source})"
    )


if __name__ == "__main__":
    main()
