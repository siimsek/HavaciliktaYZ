"""End-to-end smoke and API contract guards for NetworkManager."""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from src.network import NetworkManager, FrameFetchStatus, SendResultStatus
from src.payload_schema import CompetitionPayloadSchema


class _Response:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = content

    def json(self):
        return self._payload


class TestNetworkSmokeAndContract(unittest.TestCase):
    def test_network_http_smoke_flow_acks_result(self):
        net = NetworkManager(base_url="http://mock.local", simulation_mode=False)
        net._sleep_with_backoff = lambda attempt: None

        frame_payload = {
            "id": "frame-1",
            "url": "/frames/1",
            "image_url": "/images/1.jpg",
            "translation_x": "1.5",
            "translation_y": "2.5",
            "translation_z": "10.0",
            "gps_health_status": 1,
        }

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", img)
        self.assertTrue(ok)

        def fake_get(url, timeout=None):
            if url == "http://mock.local":
                return _Response(200, payload={"status": "ok"})
            if url == "http://mock.local/next_frame":
                return _Response(200, payload=frame_payload)
            if url == "http://mock.local/images/1.jpg":
                return _Response(200, content=enc.tobytes())
            raise AssertionError(f"Unexpected GET URL: {url}")

        net.session = Mock()
        net.session.get = Mock(side_effect=fake_get)
        net.session.post = Mock(return_value=_Response(200, payload={"status": "ok"}))

        self.assertTrue(net.start_session())

        frame_result = net.get_frame()
        self.assertEqual(frame_result.status, FrameFetchStatus.OK)
        self.assertEqual(frame_result.frame_data["frame_id"], "frame-1")

        frame = net.download_image(frame_result.frame_data)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (16, 16))

        status = net.send_result(
            frame_id=frame_result.frame_data["frame_id"],
            detected_objects=[
                {
                    "cls": "0",
                    "landing_status": "-1",
                    "movement_status": "1",
                    "top_left_x": 1,
                    "top_left_y": 1,
                    "bottom_right_x": 8,
                    "bottom_right_y": 8,
                }
            ],
            detected_translation={"translation_x": 0, "translation_y": 0, "translation_z": 0},
            frame_data=frame_result.frame_data,
            frame_shape=frame.shape,
        )
        self.assertEqual(status, SendResultStatus.ACKED)

    def test_contract_frame_schema_normalizes_aliases_and_numeric_fields(self):
        net = NetworkManager(base_url="http://mock.local", simulation_mode=False)
        payload = {
            "id": 99,
            "image_url": "/images/99.jpg",
            "translation_x": "NaN",
            "translation_y": "invalid",
            "translation_z": "12.5",
            "gps_health_status": "unknown",
        }

        self.assertTrue(net._validate_frame_data(payload))
        self.assertEqual(payload["frame_id"], 99)
        self.assertEqual(payload["frame_url"], "/images/99.jpg")
        self.assertEqual(payload["image_url"], "/images/99.jpg")
        self.assertEqual(payload["gps_health"], 0)
        self.assertEqual(payload["translation_x"], 0.0)
        self.assertEqual(payload["translation_y"], 0.0)
        self.assertEqual(payload["translation_z"], 12.5)

    def test_contract_submit_payload_has_required_fields_and_canonical_motion(self):
        payload = NetworkManager.build_competition_payload(
            frame_id="f-9",
            detected_objects=[
                {
                    "cls": "1",
                    "landing_status": "-1",
                    "movement_status": "0",
                    "top_left_x": 10,
                    "top_left_y": 12,
                    "bottom_right_x": 40,
                    "bottom_right_y": 55,
                }
            ],
            detected_translation={"translation_x": 1, "translation_y": 2, "translation_z": 3},
            frame_data={"id": "f-9", "user": "Takim_ID", "url": "/frames/9"},
            frame_shape=(100, 100, 3),
        )

        CompetitionPayloadSchema.validate_top_level_payload(payload)
        self.assertEqual(len(payload["detected_objects"]), 1)
        obj = payload["detected_objects"][0]
        self.assertIn("motion_status", obj)
        self.assertNotIn("movement_status", obj)


if __name__ == "__main__":
    unittest.main()
