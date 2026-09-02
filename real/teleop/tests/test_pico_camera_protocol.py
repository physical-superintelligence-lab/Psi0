"""Regression tests for XRoboToolkit Remote Vision control messages."""

from __future__ import annotations

import os
import socket
import struct
import sys
import unittest

import numpy as np


TELEOP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TELEOP_ROOT not in sys.path:
    sys.path.insert(0, TELEOP_ROOT)

import pico_camera_view  # noqa: E402


def compact_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return bytes([len(encoded)]) + encoded


def open_camera_packet() -> bytes:
    camera_data = (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 1280, 720, 30, 4_000_000, 0, 1, 12345)
        + compact_string("ZED")
        + compact_string("192.168.50.178")
    )
    command = b"OPEN_CAMERA"
    protocol = (
        struct.pack("<i", len(command))
        + command
        + struct.pack("<i", len(camera_data))
        + camera_data
    )
    return struct.pack(">I", len(protocol)) + protocol


class PicoCameraProtocolTest(unittest.TestCase):
    def test_initializes_gstreamer_with_argv_list(self) -> None:
        calls = []

        class FakeGst:
            @staticmethod
            def init(argv):
                calls.append(argv)

        pico_camera_view.initialize_gstreamer(FakeGst)

        self.assertEqual(calls, [[]])

    def test_external_video_does_not_disable_pico_tracking_mode(self) -> None:
        self.assertFalse(
            pico_camera_view.should_start_embedded_video(
                pico_enabled=True, external_video=True
            )
        )
        self.assertTrue(
            pico_camera_view.should_start_embedded_video(
                pico_enabled=True, external_video=False
            )
        )

    def test_parses_official_open_camera_request(self) -> None:
        command, config = pico_camera_view.parse_control_message(
            open_camera_packet()
        )

        self.assertEqual(command, "OPEN_CAMERA")
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.height, 720)
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.bitrate, 4_000_000)
        self.assertFalse(config.enable_hevc)
        self.assertEqual(config.camera, "ZED")
        self.assertEqual(config.ip, "192.168.50.178")
        self.assertEqual(config.port, 12345)

    def test_rejects_bad_camera_magic(self) -> None:
        packet = bytearray(open_camera_packet())
        camera_magic = packet.index(b"\xca\xfe")
        packet[camera_magic] = 0

        with self.assertRaisesRegex(ValueError, "magic"):
            pico_camera_view.parse_control_message(bytes(packet))

    def test_prepares_official_rgb_as_duplicated_bgr(self) -> None:
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255

        stereo = pico_camera_view.prepare_sonic_ego_frame(rgb)

        self.assertEqual(stereo.shape, (2, 6, 3))
        np.testing.assert_array_equal(stereo[:, :3, 2], 255)
        np.testing.assert_array_equal(stereo[:, 3:, 2], 255)
        np.testing.assert_array_equal(stereo[:, :, :2], 0)

    def test_letterboxes_complete_four_by_three_frame_per_eye(self) -> None:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :, 0] = 10
        rgb[:, :, 1] = 20
        rgb[:, :, 2] = 30
        rgb[0, 0] = [255, 0, 0]
        rgb[-1, -1] = [0, 255, 0]

        stereo = pico_camera_view.prepare_sonic_ego_frame(
            rgb, output_width=2560, output_height=720
        )

        self.assertEqual(stereo.shape, (720, 2560, 3))
        left = stereo[:, :1280]
        right = stereo[:, 1280:]
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(left[:, :160], 0)
        np.testing.assert_array_equal(left[:, 1120:], 0)
        self.assertGreater(np.count_nonzero(left[:, 160:1120]), 0)
        self.assertEqual(tuple(left[0, 160]), (0, 0, 255))
        self.assertEqual(tuple(left[-1, 1119]), (0, 255, 0))

    def test_video_socket_loss_requests_fresh_open_camera_handshake(self) -> None:
        class BrokenSocket:
            def sendall(self, _payload):
                raise ConnectionResetError

        streamer = pico_camera_view.PicoVideoStreamer(
            pico_ip="127.0.0.1", port=12345, encoder="pyav"
        )
        streamer._connected = True
        streamer._ever_connected = True
        streamer._sock = BrokenSocket()

        streamer._send_packet(b"frame")

        self.assertFalse(streamer._connected)
        self.assertTrue(streamer.connection_lost())

    def test_pyav_encoder_emits_annex_b_keyframe(self) -> None:
        encoder = pico_camera_view.create_pyav_encoder(64, 48, 30, 300_000)
        packets = pico_camera_view.encode_pyav_frame(
            encoder, np.zeros((48, 64, 3), dtype=np.uint8), frame_id=0, force_keyframe=True
        )

        self.assertTrue(packets)
        payload = b"".join(packets)
        self.assertTrue(payload.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01")))
        self.assertIn(b"\x00\x00\x00\x01\x67", payload)

    def test_detects_remote_vision_control_disconnect(self) -> None:
        server, client = socket.socketpair()
        try:
            self.assertFalse(pico_camera_view.control_connection_closed(server))
            client.close()
            self.assertTrue(pico_camera_view.control_connection_closed(server))
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
