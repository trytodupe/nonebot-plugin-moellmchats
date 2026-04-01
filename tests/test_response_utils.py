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
extract_response_output_text = response_utils.extract_response_output_text
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


if __name__ == "__main__":
    unittest.main()
