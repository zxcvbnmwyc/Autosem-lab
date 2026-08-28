import base64
import json
import unittest
from unittest.mock import patch

import numpy as np

from grounding import (
    QwenGrounder,
    GroundingError,
    _data_url_for_image,
    _model_request,
    _model_response_text,
    parse_grounding_payload,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, _type, _value, _traceback) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class GroundingTests(unittest.TestCase):
    def test_found_box_maps_to_original_image_coordinates(self) -> None:
        proposal = parse_grounding_payload(
            {
                "status": "found",
                "boxes": [
                    {
                        "x0": 100,
                        "y0": 200,
                        "x1": 900,
                        "y1": 800,
                        "confidence": 0.75,
                        "label": "cup",
                    }
                ],
                "note": "one object",
            }
        )
        public = proposal.as_public(1001, 501)
        self.assertEqual(public["candidate"]["box_xyxy"], [100.0, 100.0, 900.0, 400.0])
        self.assertEqual(public["candidate"]["box_1000"], [100.0, 200.0, 900.0, 800.0])

    def test_representative_point_maps_to_original_image_coordinates(self) -> None:
        proposal = parse_grounding_payload(
            {
                "status": "found",
                "boxes": [
                    {
                        "x0": 100,
                        "y0": 200,
                        "x1": 900,
                        "y1": 800,
                        "point": {"x": 400, "y": 500},
                        "confidence": 0.75,
                        "label": "cup",
                    }
                ],
                "note": "one object",
            }
        )
        public = proposal.as_public(1001, 501)
        self.assertEqual(public["candidate"]["point_1000"], [400.0, 500.0])
        self.assertEqual(public["candidate"]["point_xy"], [400.0, 250.0])

    def test_representative_point_outside_box_is_rejected(self) -> None:
        with self.assertRaises(GroundingError):
            parse_grounding_payload(
                {
                    "status": "found",
                    "boxes": [
                        {
                            "x0": 100,
                            "y0": 200,
                            "x1": 900,
                            "y1": 800,
                            "point": {"x": 50, "y": 500},
                            "confidence": 0.75,
                        }
                    ],
                }
            )

    def test_not_found_requires_no_boxes(self) -> None:
        proposal = parse_grounding_payload(
            {"status": "not_found", "boxes": [], "note": None}
        )
        self.assertIsNone(proposal.as_public(20, 20)["candidate"])

    def test_invalid_box_is_rejected(self) -> None:
        with self.assertRaises(GroundingError):
            parse_grounding_payload(
                {
                    "status": "found",
                    "boxes": [
                        {
                            "x0": -1,
                            "y0": 0,
                            "x1": 4,
                            "y1": 4,
                            "confidence": 0.5,
                            "label": None,
                        }
                    ],
                    "note": None,
                }
            )

    def test_numeric_strings_from_model_are_accepted(self) -> None:
        proposal = parse_grounding_payload(
            {
                "status": "found",
                "boxes": [
                    {
                        "x0": "10",
                        "y0": "20.5",
                        "x1": "30",
                        "y1": "40",
                        "confidence": "0.9",
                    }
                ],
            }
        )
        self.assertEqual(proposal.candidates[0].y0, 20.5)

    def test_request_uses_qwen_image_json_and_non_thinking_fields(self) -> None:
        request = _model_request("data:image/jpeg;base64,abc", "blue cup", "qwen3-vl-flash")
        self.assertFalse(request["enable_thinking"])
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIn("JSON", request["messages"][0]["content"])
        self.assertIn('"point"', request["messages"][0]["content"])
        parts = request["messages"][1]["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/jpeg;base64,abc")
        self.assertEqual(parts[1]["text"], "Target description: blue cup")

    def test_image_is_encoded_as_a_jpeg_data_url(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        result = _data_url_for_image(image)
        prefix = "data:image/jpeg;base64,"
        self.assertTrue(result.startswith(prefix))
        self.assertGreater(len(base64.b64decode(result.removeprefix(prefix))), 20)

    def test_model_response_text_accepts_openai_content(self) -> None:
        content = _model_response_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"status":"not_found","boxes":[]}'
                        }
                    }
                ]
            }
        )
        self.assertEqual(content, '{"status":"not_found","boxes":[]}')

    def test_config_without_key_is_disabled(self) -> None:
        self.assertFalse(QwenGrounder(api_key="").configured)

    def test_accepts_workspace_openai_compatible_endpoint(self) -> None:
        endpoint = QwenGrounder(
            api_key="not-a-real-key",
            base_url=(
                "https://ws-jezurpmuo05q16c9.cn-beijing.maas.aliyuncs.com/"
                "compatible-mode/v1"
            ),
        )._base_url_value()
        self.assertEqual(
            endpoint,
            "https://ws-jezurpmuo05q16c9.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1",
        )

    def test_rejects_non_alibaba_endpoint(self) -> None:
        with self.assertRaises(GroundingError):
            QwenGrounder(
                api_key="not-a-real-key",
                base_url="https://example.com/compatible-mode/v1",
            )._base_url_value()

    def test_rejects_lookalike_alibaba_endpoint(self) -> None:
        with self.assertRaises(GroundingError):
            QwenGrounder(
                api_key="not-a-real-key",
                base_url=(
                    "https://ws-jezurpmuo05q16c9.cn-beijing.maas.aliyuncs.com."
                    "evil.example/compatible-mode/v1"
                ),
            )._base_url_value()

    def test_uses_bearer_header_without_putting_key_in_url(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"status":"found","boxes":[{"x0":1,'
                                    '"y0":2,"x1":30,"y1":40,'
                                    '"confidence":0.9}]}'
                                )
                            }
                        }
                    ]
                }
            )

        grounder = QwenGrounder(
            api_key="test-key",
            base_url=(
                "https://ws-jezurpmuo05q16c9.cn-beijing.maas.aliyuncs.com/"
                "compatible-mode/v1"
            ),
            model="qwen3-vl-flash",
            timeout_seconds=5,
        )
        with patch("grounding.urllib.request.urlopen", fake_urlopen):
            proposal = grounder.ground(np.zeros((20, 30, 3), dtype=np.uint8), "cup")

        self.assertEqual(proposal.status, "found")
        self.assertEqual(
            captured["url"],
            "https://ws-jezurpmuo05q16c9.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertNotIn("test-key", str(captured["url"]))
        self.assertEqual(captured["timeout"], 5)
        payload = captured["payload"]
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertTrue(
            payload["messages"][1]["content"][0]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )


if __name__ == "__main__":
    unittest.main()
