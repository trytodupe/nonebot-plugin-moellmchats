import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

module_path = (
    Path(__file__).resolve().parents[1]
    / "nonebot_plugin_moellmchats"
    / "response_utils.py"
)
spec = spec_from_file_location("response_utils", module_path)
response_utils = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(response_utils)

build_image_reference = response_utils.build_image_reference
detect_image_media_type = response_utils.detect_image_media_type
extract_image_generation_calls = response_utils.extract_image_generation_calls
extract_response_output_text = response_utils.extract_response_output_text
is_image_generation_sse_event = response_utils.is_image_generation_sse_event
parse_sse_event_chunk = response_utils.parse_sse_event_chunk
parse_response_json_text = response_utils.parse_response_json_text
replace_image_placeholders = response_utils.replace_image_placeholders


class ResponseUtilsTest(unittest.TestCase):
    def test_replace_image_placeholders_in_order(self):
        text = "look [图片] then [图片]"
        replaced = replace_image_placeholders(
            text,
            [
                build_image_reference("first image"),
                build_image_reference("second image"),
            ],
        )
        self.assertEqual(
            replaced,
            "look [image:first image] then [image:second image]",
        )

    def test_replace_image_placeholders_appends_when_missing(self):
        text = "hello"
        replaced = replace_image_placeholders(
            text,
            [build_image_reference("fallback image")],
        )
        self.assertEqual(replaced, "hello\n[image:fallback image]")

    def test_parse_response_json_text(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"assistant_reply":"ok","image_memories":[]}',
                        }
                    ],
                }
            ]
        }
        self.assertEqual(
            parse_response_json_text(response),
            {"assistant_reply": "ok", "image_memories": []},
        )
        self.assertEqual(
            extract_response_output_text(response),
            '{"assistant_reply":"ok","image_memories":[]}',
        )

    def test_extract_image_generation_calls(self):
        response = {
            "output": [
                {
                    "type": "image_generation_call",
                    "id": "imggen_123",
                    "result": ["ZmFrZV9pbWFnZV8x", "ZmFrZV9pbWFnZV8y"],
                }
            ]
        }
        self.assertEqual(
            extract_image_generation_calls(response),
            [
                {
                    "result": "ZmFrZV9pbWFnZV8x",
                    "image_id": "imggen_123",
                    "action": None,
                },
                {
                    "result": "ZmFrZV9pbWFnZV8y",
                    "image_id": "imggen_123",
                    "action": None,
                },
            ],
        )

    def test_parse_response_json_text_with_empty_reply_shell(self):
        response = {
            "output": [
                {
                    "type": "image_generation_call",
                    "id": "imggen_123",
                    "result": "ZmFrZV9pbWFnZQ==",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"assistant_reply":"","image_memories":[]}',
                        }
                    ],
                },
            ]
        }
        self.assertEqual(
            parse_response_json_text(response),
            {"assistant_reply": "", "image_memories": []},
        )
        self.assertEqual(
            extract_response_output_text(response),
            '{"assistant_reply":"","image_memories":[]}',
        )

    def test_parse_sse_event_chunk(self):
        chunk = (
            b'event: response.image_generation_call.in_progress\n'
            b'data: {"type":"response.image_generation_call.in_progress","id":"img_123"}\n\n'
        )
        self.assertEqual(
            parse_sse_event_chunk(chunk),
            [
                {
                    "event": "response.image_generation_call.in_progress",
                    "data": {
                        "type": "response.image_generation_call.in_progress",
                        "id": "img_123",
                    },
                }
            ],
        )

    def test_is_image_generation_sse_event(self):
        self.assertTrue(
            is_image_generation_sse_event(
                "response.image_generation_call.in_progress",
                {"type": "response.image_generation_call.in_progress"},
            )
        )
        self.assertFalse(
            is_image_generation_sse_event(
                "response.output_text.delta",
                {"type": "response.output_text.delta"},
            )
        )

    def test_detect_image_media_type_from_magic_bytes(self):
        self.assertEqual(
            detect_image_media_type(b"\xff\xd8\xff\xdb\x00\x43"),
            "image/jpeg",
        )
        self.assertEqual(
            detect_image_media_type(b"\x89PNG\r\n\x1a\nrest"),
            "image/png",
        )
        self.assertEqual(
            detect_image_media_type(b"GIF89arest"),
            "image/gif",
        )
        self.assertEqual(
            detect_image_media_type(b"RIFF1234WEBPrest"),
            "image/webp",
        )

    def test_detect_image_media_type_fallback(self):
        self.assertEqual(
            detect_image_media_type(b"not-an-image", "image/png; charset=binary"),
            "image/png",
        )
        self.assertIsNone(detect_image_media_type(b"not-an-image", "application/octet-stream"))


if __name__ == "__main__":
    unittest.main()
