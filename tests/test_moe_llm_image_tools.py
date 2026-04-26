import asyncio
import unittest
import json
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

package = ModuleType("nonebot_plugin_moellmchats")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats")]
Path("/tmp/moellmchats-test-config").mkdir(parents=True, exist_ok=True)
Path("/tmp/moellmchats-test-data").mkdir(parents=True, exist_ok=True)
module_path = (
    Path(__file__).resolve().parents[1]
    / "nonebot_plugin_moellmchats"
    / "moe_llm.py"
)
spec = spec_from_file_location(
    "nonebot_plugin_moellmchats.moe_llm",
    module_path,
)
moe_llm = module_from_spec(spec)
assert spec.loader is not None

class FakeMessage:
    def __init__(self, segments=None):
        self.segments = list(segments or [])

    def __add__(self, other):
        if isinstance(other, FakeMessage):
            return FakeMessage(self.segments + other.segments)
        return FakeMessage(self.segments + [other])


class FakeMessageSegment:
    def __init__(self, seg_type, data):
        self.type = seg_type
        self.data = data

    def __add__(self, other):
        if isinstance(other, FakeMessage):
            return FakeMessage([self] + other.segments)
        return FakeMessage([self, other])

    @staticmethod
    def reply(message_id):
        return FakeMessageSegment("reply", {"id": message_id})

    @staticmethod
    def text(text):
        return FakeMessageSegment("text", {"text": text})

    @staticmethod
    def image(file):
        return FakeMessageSegment("image", {"file": file})


with patch.dict(
    "sys.modules",
    {
        "aiohttp": SimpleNamespace(),
        "httpx": SimpleNamespace(),
        "nonebot": SimpleNamespace(),
        "nonebot.adapters.onebot.v11": SimpleNamespace(MessageSegment=FakeMessageSegment),
        "nonebot.log": SimpleNamespace(logger=SimpleNamespace(warning=SimpleNamespace(), info=SimpleNamespace(), error=SimpleNamespace())),
        "nonebot_plugin_localstore": SimpleNamespace(
            get_plugin_config_dir=lambda: Path("/tmp/moellmchats-test-config"),
            get_plugin_data_dir=lambda: Path("/tmp/moellmchats-test-data"),
            get_plugin_data_file=lambda name: Path("/tmp/moellmchats-test-data") / name,
        ),
        "openai": SimpleNamespace(AsyncOpenAI=object),
        "ujson": json,
        "nonebot_plugin_moellmchats": package,
    },
):
    spec.loader.exec_module(moe_llm)

MoeLlm = moe_llm.MoeLlm


class MoeLlmImageToolsTest(unittest.TestCase):
    def build_llm(self):
        llm = MoeLlm(
            bot=SimpleNamespace(),
            event=SimpleNamespace(user_id=1, group_id=2, message_id=42),
            format_message_dict={},
        )
        llm.messages_handler = SimpleNamespace(user_refs=[])
        return llm

    def test_builds_image_generation_and_edit_tools(self):
        llm = self.build_llm()
        tools, include = llm._build_responses_tools(external_image_generation=True)
        tool_names = [tool.get("name") or tool.get("type") for tool in tools]

        self.assertEqual(include, [])
        self.assertIn("get_imagegen_instructions", tool_names)
        self.assertIn("image_generation", tool_names)
        self.assertIn("image_edit", tool_names)

    def test_exposes_imagegen_instructions_without_external_image_tools(self):
        llm = self.build_llm()
        tools, include = llm._build_responses_tools(external_image_generation=False)
        tool_names = [tool.get("name") or tool.get("type") for tool in tools]

        self.assertEqual(include, [])
        self.assertIn("get_imagegen_instructions", tool_names)
        self.assertNotIn("image_generation", tool_names)
        self.assertNotIn("image_edit", tool_names)

    def test_extracts_generation_and_legacy_generate_image_args(self):
        llm = self.build_llm()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "image_generation",
                    "arguments": '{"prompt":"a cat","size":"1024x1024","n":2}',
                },
                {
                    "type": "function_call",
                    "name": "generate_image",
                    "arguments": '{"prompt":"a dog","size":"1536x1024","n":9}',
                },
            ]
        }

        self.assertEqual(
            llm._extract_image_generation_args(response),
            [
                {"prompt": "a cat", "size": "1024x1024", "n": 2},
                {"prompt": "a dog", "size": "1536x1024", "n": 4},
            ],
        )

    def test_merges_streamed_function_calls_into_empty_final_response(self):
        llm = self.build_llm()
        response = {"id": "resp_1", "output": []}
        streamed_function_calls = [
            {
                "type": "function_call",
                "name": "get_imagegen_instructions",
                "arguments": "{}",
            }
        ]

        merged = llm._merge_streamed_function_calls(response, streamed_function_calls)

        self.assertEqual(merged["output"], streamed_function_calls)
        self.assertEqual(
            llm._extract_function_args(merged, "get_imagegen_instructions"),
            [{}],
        )

    def test_does_not_duplicate_streamed_function_calls(self):
        llm = self.build_llm()
        function_call = {
            "type": "function_call",
            "name": "get_imagegen_instructions",
            "arguments": "{}",
        }
        response = {"id": "resp_1", "output": [function_call.copy()]}

        merged = llm._merge_streamed_function_calls(response, [function_call])

        self.assertEqual(merged["output"], [function_call])

    def test_prepare_current_images_uses_raw_images_only_after_imagegen_instructions(self):
        llm = self.build_llm()
        llm.messages_handler.current_images = [{"source_url": "https://example.com/a.png"}]
        llm._prepare_images = AsyncMock(return_value=[])

        asyncio.run(llm._prepare_current_images(session=object()))
        self.assertFalse(llm._prepare_images.call_args.kwargs["include_known_images"])

        llm.imagegen_instructions_provided = True
        asyncio.run(llm._prepare_current_images(session=object()))
        self.assertTrue(llm._prepare_images.call_args.kwargs["include_known_images"])

    def test_extracts_image_edit_args(self):
        llm = self.build_llm()
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "image_edit",
                    "arguments": '{"prompt":"make it rainy","image_ids":["img_1","img_2"],"size":"1024x1536","n":2}',
                }
            ]
        }

        self.assertEqual(
            llm._extract_image_edit_args(response),
            [
                {
                    "prompt": "make it rainy",
                    "image_ids": ["img_1", "img_2"],
                    "size": "1024x1536",
                    "n": 2,
                }
            ],
        )

    def test_derives_edit_url_from_generation_url(self):
        llm = self.build_llm()
        llm.model_info = {
            "external_image_generation": {
                "generation_url": "https://api.example.com/v1/images/generations",
            }
        }

        self.assertEqual(
            llm._image_edit_url(),
            "https://api.example.com/v1/images/edits",
        )

    def test_preserves_upstream_image_api_error_body(self):
        llm = self.build_llm()

        self.assertEqual(
            llm._format_image_api_error("图片生成", 400, '{"error":{"message":"bad prompt"}}'),
            '图片生成 请求失败：HTTP 400 {"error":{"message":"bad prompt"}}',
        )

    def test_detects_redundant_image_completion_reply(self):
        llm = self.build_llm()

        self.assertTrue(llm._is_redundant_image_completion_reply("好了。"))
        self.assertTrue(llm._is_redundant_image_completion_reply("done"))
        self.assertFalse(llm._is_redundant_image_completion_reply("这张图里保留了原本的水彩画风。"))

    def test_build_reply_message_prefixes_reply_segment_without_at(self):
        llm = self.build_llm()

        message = llm.build_reply_message("hello")

        self.assertIsInstance(message, FakeMessage)
        self.assertEqual(
            [(segment.type, segment.data) for segment in message.segments],
            [
                ("reply", {"id": 42}),
                ("text", {"text": "hello"}),
            ],
        )

    def test_send_generation_notice_event_once(self):
        bot = SimpleNamespace(call_api=AsyncMock())
        llm = MoeLlm(
            bot=bot,
            event=SimpleNamespace(user_id=1, group_id=2, message_id=42),
            format_message_dict={},
        )
        llm.messages_handler = SimpleNamespace(user_refs=[])

        asyncio.run(llm.send_generation_notice_event_once())
        asyncio.run(llm.send_generation_notice_event_once())

        bot.call_api.assert_awaited_once_with(
            "set_msg_emoji_like",
            message_id=42,
            emoji_id=moe_llm.IMAGE_GENERATION_NOTICE_EMOJI_ID,
        )

    def test_send_reply_message_wraps_image_with_reply_segment(self):
        bot = SimpleNamespace(send=AsyncMock())
        llm = MoeLlm(
            bot=bot,
            event=SimpleNamespace(user_id=1, group_id=2, message_id=42),
            format_message_dict={},
        )
        llm.messages_handler = SimpleNamespace(user_refs=[])

        asyncio.run(llm.send_reply_message(FakeMessageSegment.image(b"img")))

        bot.send.assert_awaited_once()
        sent_event, sent_message = bot.send.await_args.args
        self.assertEqual(sent_event.message_id, 42)
        self.assertEqual(
            [(segment.type, segment.data) for segment in sent_message.segments],
            [
                ("reply", {"id": 42}),
                ("image", {"file": b"img"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
