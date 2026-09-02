"""Relay the robot RealSense view to PICO without starting robot control.

The camera server exposes RGB, stereo IR and depth over ZMQ.  The current
XRoboToolkit Remote Vision client first connects to this process on port 13579
and sends an OPEN_CAMERA request containing its video receive address.  This
process bridges those two networks and deliberately imports no DDS controller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import signal
import socket
import struct
import threading
import time

import cv2
import numpy as np
import zmq


def should_start_embedded_video(pico_enabled: bool, external_video: bool) -> bool:
    """Keep Pico tracking enabled while an external relay owns video."""
    return bool(pico_enabled and not external_video)


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    enable_hevc: bool
    render_mode: int
    port: int
    camera: str
    ip: str


def _read_compact_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("missing compact string length")
    length = data[offset]
    offset += 1
    if offset + length > len(data):
        raise ValueError("truncated compact string")
    return data[offset : offset + length].decode("utf-8"), offset + length


def parse_control_message(packet: bytes) -> tuple[str, CameraRequest | None]:
    """Parse the length-prefixed protocol used by the official sender."""
    if len(packet) < 4:
        raise ValueError("control packet missing wrapper length")
    body_length = struct.unpack(">I", packet[:4])[0]
    body = packet[4:]
    if len(body) != body_length:
        raise ValueError("control packet wrapper length mismatch")

    if len(body) < 8:
        raise ValueError("control protocol is truncated")
    command_length = struct.unpack("<i", body[:4])[0]
    if command_length < 0 or 4 + command_length + 4 > len(body):
        raise ValueError("invalid command length")
    command_end = 4 + command_length
    command = body[4:command_end].rstrip(b"\0").decode("utf-8")
    data_length = struct.unpack("<i", body[command_end : command_end + 4])[0]
    data = body[command_end + 4 :]
    if data_length < 0 or len(data) != data_length:
        raise ValueError("control data length mismatch")
    if command != "OPEN_CAMERA":
        return command, None

    if len(data) < 31 or data[:2] != b"\xca\xfe":
        raise ValueError("invalid camera request magic")
    if data[2] != 1:
        raise ValueError(f"unsupported camera protocol version {data[2]}")
    values = struct.unpack("<7i", data[3:31])
    camera, offset = _read_compact_string(data, 31)
    ip, offset = _read_compact_string(data, offset)
    if offset != len(data):
        raise ValueError("unexpected camera request trailing data")
    return command, CameraRequest(
        width=values[0],
        height=values[1],
        fps=values[2],
        bitrate=values[3],
        enable_hevc=bool(values[4]),
        render_mode=values[5],
        port=values[6],
        camera=camera,
        ip=ip,
    )


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("control connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_control_message(sock: socket.socket) -> tuple[str, CameraRequest | None]:
    header = _recv_exact(sock, 4)
    body_length = struct.unpack(">I", header)[0]
    if body_length > 64 * 1024:
        raise ValueError(f"control packet too large: {body_length}")
    return parse_control_message(header + _recv_exact(sock, body_length))


def wait_for_open_camera(
    host: str, port: int, stop_event: threading.Event
) -> tuple[socket.socket, CameraRequest]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    server.settimeout(0.5)
    print(f"[PICO_VIEW] waiting for XRoboToolkit on {host}:{port}", flush=True)
    try:
        while not stop_event.is_set():
            try:
                connection, peer = server.accept()
            except socket.timeout:
                continue
            print(f"[PICO_VIEW] control client connected from {peer[0]}", flush=True)
            try:
                command, config = recv_control_message(connection)
                if command == "OPEN_CAMERA" and config is not None:
                    print(f"[PICO_VIEW] OPEN_CAMERA {config}", flush=True)
                    return connection, config
                connection.close()
            except (ConnectionError, OSError, ValueError) as exc:
                print(f"[PICO_VIEW] invalid control request: {exc}", flush=True)
                connection.close()
    finally:
        server.close()
    raise InterruptedError("stopped before OPEN_CAMERA")


def control_connection_closed(connection: socket.socket | None) -> bool:
    if connection is None:
        return False
    try:
        data = connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return False
    except OSError:
        return True
    return data == b""


def initialize_gstreamer(gst) -> None:
    """Initialize PyGObject GStreamer bindings with a portable argv value."""

    gst.init([])


def create_pyav_encoder(width: int, height: int, fps: int, bitrate: int):
    """Create the low-latency Annex-B H.264 encoder available in conda sonic."""
    import av

    codec = av.CodecContext.create("libx264", "w")
    codec.width = int(width)
    codec.height = int(height)
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, int(fps))
    codec.framerate = Fraction(int(fps), 1)
    codec.bit_rate = int(bitrate)
    codec.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "profile": "baseline",
        "g": "15",
        "x264-params": "annexb=1:repeat-headers=1",
    }
    codec.open()
    return codec


def encode_pyav_frame(codec, frame_bgr: np.ndarray, frame_id: int, force_keyframe: bool):
    import av

    frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
    frame.pts = int(frame_id)
    if force_keyframe:
        frame.pict_type = av.video.frame.PictureType.I
    return [bytes(packet) for packet in codec.encode(frame)]


def prepare_sonic_ego_frame(
    frame_rgb: np.ndarray,
    output_width: int | None = None,
    output_height: int | None = None,
) -> np.ndarray:
    """Convert SONIC RGB into uncropped, aspect-preserving stereo BGR."""
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"ego_view must be HxWx3, got {frame.shape}")
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if output_width is None or output_height is None:
        return np.hstack((bgr, bgr))
    if output_width <= 0 or output_height <= 0 or output_width % 2:
        raise ValueError(
            f"stereo output must have positive even width and height, got "
            f"{output_width}x{output_height}"
        )

    eye_width = output_width // 2
    source_height, source_width = bgr.shape[:2]
    scale = min(eye_width / source_width, output_height / source_height)
    fitted_width = max(1, int(round(source_width * scale)))
    fitted_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    fitted = cv2.resize(
        bgr, (fitted_width, fitted_height), interpolation=interpolation
    )
    eye = np.zeros((output_height, eye_width, 3), dtype=np.uint8)
    x_offset = (eye_width - fitted_width) // 2
    y_offset = (output_height - fitted_height) // 2
    eye[
        y_offset : y_offset + fitted_height,
        x_offset : x_offset + fitted_width,
    ] = fitted
    return np.hstack((eye, eye))


class PicoVideoStreamer:
    """Encode BGR frames as low-latency H.264 and send them to PICO."""

    def __init__(
        self,
        pico_ip: str,
        port: int,
        width=1280,
        height=720,
        fps=30,
        bitrate=4_000_000,
        encoder="gstreamer",
    ):
        self.pico_ip = pico_ip
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.encoder = encoder
        self._running = False
        self._connected = False
        self._sock = None
        self._pipeline = None
        self._appsrc = None
        self._frame_id = 0
        self._latest_frame = None
        self._lock = threading.Lock()
        self._force_keyframe = True
        self._av_codec = None
        self._ever_connected = False
        self._connection_lost = threading.Event()

    def start(self):
        if self.encoder == "pyav":
            self._av_codec = create_pyav_encoder(
                self.width, self.height, self.fps, self.bitrate
            )
        elif self.encoder == "gstreamer":
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            self._gst = Gst
            initialize_gstreamer(Gst)
            pipe_str = (
                f"appsrc name=src is-live=True format=time ! "
                f"video/x-raw,format=BGR,width={self.width},height={self.height},"
                f"framerate={self.fps}/1 ! videoconvert ! "
                "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 "
                f"bitrate={max(1, self.bitrate // 1000)} byte-stream=true aud=true ! "
                "video/x-h264,profile=baseline ! h264parse config-interval=-1 ! "
                "video/x-h264,stream-format=byte-stream,alignment=au ! "
                "appsink name=sink emit-signals=True sync=False"
            )
            self._pipeline = Gst.parse_launch(pipe_str)
            self._appsrc = self._pipeline.get_by_name("src")
            self._pipeline.get_by_name("sink").connect(
                "new-sample", self._on_encoded_frame
            )
            self._pipeline.set_state(Gst.State.PLAYING)
        else:
            raise ValueError(f"unsupported encoder: {self.encoder}")
        self._running = True
        threading.Thread(target=self._connection_loop, daemon=True).start()
        threading.Thread(target=self._push_loop, daemon=True).start()
        print(
            f"[PICO_VIEW] H.264 sender started for {self.pico_ip}:{self.port}",
            flush=True,
        )

    def submit_frame(self, frame_bgr: np.ndarray):
        with self._lock:
            self._latest_frame = frame_bgr.copy()

    def _connection_loop(self):
        while self._running:
            if not self._connected:
                if self._sock is not None:
                    self._sock.close()
                    self._sock = None
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2.0)
                    sock.connect((self.pico_ip, self.port))
                    sock.settimeout(None)
                    self._sock = sock
                    self._connected = True
                    self._ever_connected = True
                    self._force_keyframe = True
                    print("[PICO_VIEW] connected to PICO", flush=True)
                except OSError:
                    pass
            time.sleep(1.0)

    def _on_encoded_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(self._gst.MapFlags.READ)
        if ok:
            if self._connected and self._sock is not None:
                self._send_packet(info.data)
            buf.unmap(info)
        return self._gst.FlowReturn.OK

    def _send_packet(self, payload: bytes):
        if not self._connected or self._sock is None:
            return
        try:
            self._sock.sendall(struct.pack(">I", len(payload)) + payload)
        except OSError:
            self._connected = False
            if self._ever_connected:
                self._connection_lost.set()
            print("[PICO_VIEW] PICO connection lost; retrying", flush=True)

    def connection_lost(self) -> bool:
        return self._connection_lost.is_set()

    def _push_loop(self):
        interval = 1.0 / self.fps
        while self._running:
            started = time.monotonic()
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is not None:
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                frame = np.ascontiguousarray(frame)
                if self.encoder == "pyav" and self._av_codec is not None:
                    packets = encode_pyav_frame(
                        self._av_codec,
                        frame,
                        self._frame_id,
                        self._force_keyframe,
                    )
                    self._force_keyframe = False
                    for packet in packets:
                        self._send_packet(packet)
                elif self._appsrc is not None:
                    gst_buf = self._gst.Buffer.new_wrapped(frame.tobytes())
                    gst_buf.pts = self._frame_id * (self._gst.SECOND // self.fps)
                    gst_buf.duration = self._gst.SECOND // self.fps
                    self._appsrc.emit("push-buffer", gst_buf)
                self._frame_id += 1
            stop_wait = interval - (time.monotonic() - started)
            if stop_wait > 0:
                time.sleep(stop_wait)

    def stop(self):
        self._running = False
        self._connected = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
        self._av_codec = None


def decode_view(parts: list[bytes], view: str) -> np.ndarray:
    if len(parts) != 3:
        raise ValueError(f"camera server returned {len(parts)} parts, expected 3")

    encoded = parts[1] if view == "ir" else parts[0]
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"failed to decode {view} frame")

    # The PICO ZEDMINI Remote Vision layout expects left/right images side by
    # side.  RealSense IR is already stereo; RGB is monocular, so duplicate it
    # to avoid splitting one image into two mismatched eye views.
    if view == "rgb":
        frame = np.hstack((frame, frame))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream the robot camera to PICO without sending motor commands"
    )
    parser.add_argument("--camera-ip", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5556)
    parser.add_argument(
        "--camera-protocol",
        choices=("legacy", "sonic"),
        default="legacy",
        help="legacy get_frame multipart or official SONIC composed camera",
    )
    parser.add_argument("--camera-key", default="ego_view")
    parser.add_argument(
        "--encoder",
        choices=("auto", "gstreamer", "pyav"),
        default="auto",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=13579)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="use the legacy direct-to-headset protocol instead of OPEN_CAMERA",
    )
    parser.add_argument("--pico-ip", default="192.168.50.178")
    parser.add_argument("--pico-port", type=int, default=12345)
    parser.add_argument(
        "--view",
        choices=("ir", "rgb"),
        default="rgb",
        help="duplicated live RGB (default) or stereo IR",
    )
    args = parser.parse_args()

    stop_event = threading.Event()

    def stop(_signum=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    control_connection = None
    if args.direct:
        config = CameraRequest(
            width=1280,
            height=720,
            fps=30,
            bitrate=4_000_000,
            enable_hevc=False,
            render_mode=1,
            port=args.pico_port,
            camera="ZED",
            ip=args.pico_ip,
        )
    else:
        try:
            control_connection, config = wait_for_open_camera(
                args.listen_host, args.listen_port, stop_event
            )
        except InterruptedError:
            return 0

    if config.enable_hevc:
        if control_connection is not None:
            control_connection.close()
        raise RuntimeError("Pico requested HEVC; select H.264 in Remote Vision")

    encoder = args.encoder
    if encoder == "auto":
        encoder = "pyav" if args.camera_protocol == "sonic" else "gstreamer"

    context = None
    camera = None
    if args.camera_protocol == "legacy":
        context = zmq.Context()

        def connect_legacy_camera():
            socket_ = context.socket(zmq.REQ)
            socket_.setsockopt(zmq.RCVTIMEO, 2000)
            socket_.setsockopt(zmq.SNDTIMEO, 2000)
            socket_.setsockopt(zmq.LINGER, 0)
            socket_.connect(f"tcp://{args.camera_ip}:{args.camera_port}")
            return socket_

        camera = connect_legacy_camera()
    else:
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        def connect_sonic_camera():
            return ComposedCameraClientSensor(
                server_ip=args.camera_ip, port=args.camera_port
            )

        camera = connect_sonic_camera()

    streamer = PicoVideoStreamer(
        pico_ip=config.ip,
        port=config.port,
        width=config.width,
        height=config.height,
        fps=config.fps,
        bitrate=config.bitrate,
        encoder=encoder,
    )
    streamer.start()
    print(
        f"[PICO_VIEW] camera={args.camera_ip}:{args.camera_port} "
        f"protocol={args.camera_protocol} "
        f"view={args.camera_key if args.camera_protocol == 'sonic' else args.view} "
        f"encoder={encoder} -> pico={config.ip}:{config.port}",
        flush=True,
    )
    print("[PICO_VIEW] video only: DDS and robot motors are not initialized", flush=True)

    frames = 0
    report_time = time.monotonic()
    last_camera_frame_time = time.monotonic()
    last_camera_timestamp = None
    try:
        while not stop_event.is_set():
            frame_started = time.monotonic()
            if control_connection_closed(control_connection):
                print(
                    "[PICO_VIEW] Remote Vision control connection closed; restarting handshake",
                    flush=True,
                )
                break
            if streamer.connection_lost():
                print(
                    "[PICO_VIEW] video socket lost; restarting OPEN_CAMERA handshake",
                    flush=True,
                )
                break
            try:
                if args.camera_protocol == "legacy":
                    camera.send(b"get_frame")
                    parts = camera.recv_multipart()
                    streamer.submit_frame(decode_view(parts, args.view))
                    frames += 1
                    last_camera_frame_time = time.monotonic()
                else:
                    message = camera.read(blocking=False)
                    if message is not None:
                        image = message.get("images", {}).get(args.camera_key)
                        timestamp = message.get("timestamps", {}).get(args.camera_key)
                        if image is not None and timestamp != last_camera_timestamp:
                            streamer.submit_frame(
                                prepare_sonic_ego_frame(
                                    image,
                                    output_width=streamer.width,
                                    output_height=streamer.height,
                                )
                            )
                            last_camera_timestamp = timestamp
                            last_camera_frame_time = time.monotonic()
                            frames += 1
                    if time.monotonic() - last_camera_frame_time > 1.0:
                        raise TimeoutError("official camera stream is stale")
            except zmq.Again:
                print("[PICO_VIEW] camera timeout; reconnecting", flush=True)
                camera.close()
                camera = connect_legacy_camera()
                last_camera_frame_time = time.monotonic()
            except TimeoutError as exc:
                print(f"[PICO_VIEW] {exc}; reconnecting", flush=True)
                camera.close()
                camera = connect_sonic_camera()
                last_camera_timestamp = None
                last_camera_frame_time = time.monotonic()
            except ValueError as exc:
                print(f"[PICO_VIEW] {exc}", flush=True)

            now = time.monotonic()
            if now - report_time >= 5.0:
                print(
                    f"[PICO_VIEW] camera_fps={frames / (now - report_time):.1f} "
                    f"pico_connected={streamer._connected}",
                    flush=True,
                )
                frames = 0
                report_time = now
            remaining = (1.0 / streamer.fps) - (time.monotonic() - frame_started)
            if remaining > 0:
                stop_event.wait(remaining)
    finally:
        streamer.stop()
        if control_connection is not None:
            control_connection.close()
        if camera is not None:
            camera.close()
        if context is not None:
            context.term()
        print("[PICO_VIEW] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
