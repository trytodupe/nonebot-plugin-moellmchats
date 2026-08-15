# ruff: noqa: PT009

import base64
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import struct
import sys
import unittest

module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "image_generation.py"
spec = spec_from_file_location("moellmchats_image_generation_test_module", module_path)
image_generation = module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = image_generation
spec.loader.exec_module(image_generation)

actual_image_scale = image_generation.actual_image_scale
build_vertex_image_payload = image_generation.build_vertex_image_payload
extract_vertex_images = image_generation.extract_vertex_images
normalize_openai_image_size = image_generation.normalize_openai_image_size
normalize_vertex_image_request = image_generation.normalize_vertex_image_request


class ImageGenerationTest(unittest.TestCase):
    def test_normalizes_openai_size_constraints(self):
        self.assertEqual(normalize_openai_image_size("1536x1024"), "1536x1024")
        self.assertEqual(normalize_openai_image_size("1537x1024"), "auto")
        self.assertEqual(normalize_openai_image_size("4096x1024"), "auto")
        self.assertEqual(normalize_openai_image_size("tiny"), "auto")

    def test_builds_flash_lite_payload_with_reference(self):
        request = normalize_vertex_image_request(
            {
                "prompt": "keep the subject and make it blue",
                "model": "gemini-3.1-flash-lite-image",
                "aspect_ratio": "16:9",
                "image_size": "4K",
                "image_ids": ["img_1"],
            }
        )
        self.assertIsNotNone(request)
        payload = build_vertex_image_payload(
            request,
            {
                "img_1": {
                    "bytes": b"image",
                    "mime_type": "image/png",
                }
            },
        )

        generation_config = payload["generationConfig"]
        self.assertEqual(generation_config["imageConfig"], {"aspectRatio": "16:9", "imageSize": "1K"})
        self.assertEqual(generation_config["candidateCount"], 1)
        self.assertEqual(
            payload["contents"][0]["parts"][0]["inlineData"]["data"],
            base64.b64encode(b"image").decode(),
        )

    def test_omits_image_size_for_gemini_25(self):
        request = normalize_vertex_image_request(
            {
                "prompt": "draw it",
                "model": "gemini-2.5-flash-image",
                "aspect_ratio": "21:9",
                "image_size": "4K",
                "image_ids": ["one", "two", "three", "four"],
            }
        )
        payload = build_vertex_image_payload(request, {})

        self.assertEqual(payload["generationConfig"]["imageConfig"], {"aspectRatio": "21:9"})
        self.assertEqual(payload["generationConfig"]["topK"], 64)
        self.assertEqual(request["image_ids"], ["one", "two", "three"])

    def test_extracts_vertex_inline_image(self):
        self.assertEqual(
            extract_vertex_images(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": base64.b64encode(b"png").decode(),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            [b"png"],
        )

    def test_reports_actual_png_scale(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 1254, 836)
        self.assertEqual(actual_image_scale(png_header, "auto"), "1254x836")


if __name__ == "__main__":
    unittest.main()
