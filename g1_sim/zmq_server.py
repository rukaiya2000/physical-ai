"""ZMQ server: receive a motion name, play it on the MuJoCo G1.

Keep one G1 window open. The webcam process can stay tiny:

    socket.send_string("correction_2")
    print(socket.recv_string())
"""

from __future__ import annotations

from g1_sim.mapping import MOTION_NAMES
from g1_sim.player import G1Player


def serve_motions(bind: str = "tcp://127.0.0.1:5555", viewer: bool = True) -> None:
    import zmq

    player = G1Player(viewer=viewer)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(bind)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(f"G1 motion server listening on {bind}")
    print(f"Send one of: {', '.join(MOTION_NAMES)}")
    try:
        while player.is_running():
            events = dict(poller.poll(timeout=20))
            player.sync()
            if socket not in events:
                continue
            name = socket.recv_string().strip()
            try:
                clip = player.play(name)
                reply = f"ok {clip.name} frames={clip.n_frames} source={clip.source}"
            except Exception as error:  # noqa: BLE001 - send the error to the client
                reply = f"error {error}"
            socket.send_string(reply)
            print(reply)
    except KeyboardInterrupt:
        print("\nstopping motion server")
    finally:
        socket.close(linger=0)
        player.close()
