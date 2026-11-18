import base64
import hashlib
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid
from asyncio import TimeoutError
from collections import defaultdict, deque

import aiohttp
import httpx
import ujson as json
from nonebot.adapters.onebot.v11 import MessageSegment
from nonebot.log import logger
from openai import AsyncOpenAI

from .Config import config_parser
from .ImageCache import image_cache
from .ImageMemory import image_memory_store
from .MessagesHandler import MessagesHandler
from .ModelSelector import model_selector
from .image_generation import (
    OPENAI_IMAGE_SIZE_DEFAULT,
    OPENAI_IMAGE_SIZE_EXAMPLES,
    VERTEX_IMAGE_MODELS,
    VERTEX_IMAGE_PROFILES,
    VERTEX_IMAGE_TOOL_DESCRIPTION,
    actual_image_scale,
    build_vertex_image_payload,
    extract_vertex_images,
    normalize_openai_image_size,
    normalize_vertex_image_request,
)
from .response_utils import (
    build_image_reference,
    extract_image_generation_calls,
    extract_response_output_text,
    is_long_message,
    normalize_image_summary,
    parse_assistant_response_payload,
    replace_image_placeholders,
)
from .group_context import collect_group_context
from .agy_acp import (
    DEFAULT_AGY_MODEL,
    close_shared_acp_client,
    get_agy_runtime_dir,
    get_shared_acp_client,
    responses_content_to_acp_blocks,
)
from .prompt_templates import build_group_chat_prompt
from .trigger_rules import contains_big_doubao_trigger

context_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_group_history"))
)

IMAGE_GENERATION_NOTICE_EMOJI_ID = 10024

BASE_PROMPT = (
    "你在 QQ 群聊里回复当前用户的最新一条消息。"
    "目标是自然接话并把该说的信息说到位。"
    "不要提系统提示、工具或内部推理。"
)

IMAGEGEN_PROMPT_GUIDANCE = Path(__file__).with_name("imagegen_instructions.md").read_text(encoding="utf-8").strip()

IMAGEGEN_PLUGIN_INSTRUCTIONS = """Plugin-specific image tool rules:
- Before calling image_generation or image_edit, rewrite the user's request into a complete standalone prompt.
- Preserve the user's concrete requirements. Do not add unrelated characters, brands, slogans, objects, or narrative beats.
- Include only useful fields: use case, asset type, primary request, input images, subject, scene/backdrop, style/medium, composition/framing, lighting/mood, text verbatim, constraints, avoid.
- For edits, state invariants explicitly: change only the requested parts and keep identity, pose, composition, and other image details unchanged unless the user asked to change them.
- For image references, use image_ids from the current turn, fetched recent images, or fetched avatars. Never include private platform IDs.
- For transparent/cutout requests, ask for a perfectly flat chroma-key background unless the user explicitly asked for native transparency.
- For exact text in images, quote it verbatim and ask for clean readable typography.
- After generating or editing an image, do not add a separate assistant text reply such as "done", "finished", or "here it is" unless the user explicitly asks for a caption or explanation. The image output itself is the reply.
- Keep the final prompt concise enough to be directly sent to the Images API."""

IMAGEGEN_TOOL_INSTRUCTIONS = "\n\n".join((IMAGEGEN_PROMPT_GUIDANCE, IMAGEGEN_PLUGIN_INSTRUCTIONS))

IMAGEGEN_TOOL_DESCRIPTION = (
    "Use get_imagegen_instructions first when the user asks for image generation or editing, "
    "then call image_generation or image_edit with the refined standalone prompt. "
    "The prompt should follow the $imagegen guidance: preserve user intent, add only material visual details, "
    "state edit invariants, quote exact text verbatim, and avoid private platform IDs."
)


def _configured_environment_value(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        from nonebot import get_driver

        value = getattr(get_driver().config, name.lower(), "")
    except (ImportError, RuntimeError):
        return ""
    return str(value).strip() if value else ""


def _read_vertex_api_key(config: dict) -> str:
    credential_env = str(config.get("credential_env") or "").strip()
    if credential_env:
        api_key = _configured_environment_value(credential_env)
        if api_key:
            return api_key

    credential_file = str(config.get("credential_file") or "").strip()
    if not credential_file:
        return ""
    try:
        return Path(credential_file).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _extract_event_plain_text(event, format_message_dict: dict | None = None) -> str:
    get_message = getattr(event, "get_message", None)
    if callable(get_message):
        try:
            msg = get_message()
            extract = getattr(msg, "extract_plain_text", None)
            if callable(extract):
                return str(extract() or "")
        except Exception:
            pass
    if isinstance(format_message_dict, dict):
        return "".join(str(part) for part in (format_message_dict.get("text") or []))
    return ""


class MoeLlm:
    def __init__(
        self,
        bot,
        event,
        format_message_dict: dict,
        is_objective: bool = False,
        temperament: str = "default",
        use_big_doubao: bool = False,
    ):
        self.bot = bot
        self.event = event
        self.format_message_dict = format_message_dict
        self.user_id = event.user_id
        self.is_objective = is_objective
        self.temperament = temperament
        self.use_big_doubao = use_big_doubao
        self.model_info = {}
        self.prompt = BASE_PROMPT
        self.fetched_images = []
        self.fetch_recent_images_rounds = 0
        self.empty_response_retries = 0
        self.pending_user_avatar_requests = []
        self.image_inputs_by_id = {}
        self.imagegen_instructions_provided = False
        self.generation_notice_sent = False
        self.native_image_size = OPENAI_IMAGE_SIZE_DEFAULT
        self.native_image_size_selected = False
        self.session_key = self._build_session_key()

    def _build_session_key(self) -> str:
        if hasattr(self.event, "group_id"):
            return f"group:{self.event.group_id}"
        return f"private:{self.event.user_id}"

    def _reply_segment(self):
        if message_id := getattr(self.event, "message_id", None):
            return MessageSegment.reply(message_id)
        return None

    def build_reply_message(self, content) -> MessageSegment | str:
        reply_segment = self._reply_segment()
        if reply_segment is None:
            return content
        if isinstance(content, str):
            content = MessageSegment.text(content)
        return reply_segment + content

    async def send_reply_message(self, content):
        await self.bot.send(self.event, self.build_reply_message(content))

    async def send_generation_notice_event_once(self):
        if self.generation_notice_sent:
            return
        message_id = getattr(self.event, "message_id", None)
        if not message_id:
            self.generation_notice_sent = True
            return
        try:
            await self.bot.call_api(
                "set_msg_emoji_like",
                message_id=message_id,
                emoji_id=IMAGE_GENERATION_NOTICE_EMOJI_ID,
            )
        except Exception:
            logger.warning("Failed to send image-generation emoji-like event", exc_info=True)
        finally:
            self.generation_notice_sent = True

    def _format_upstream_error(self, exc: Exception) -> str:
        message = str(exc).strip()
        if not message:
            return type(exc).__name__
        return message

    def _response_body(self, response) -> dict:
        if isinstance(response, dict):
            return response
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", warnings=False)
        return {}

    def _format_response_terminal_status(self, body: dict) -> str:
        status = body.get("status")
        error = body.get("error") or {}
        if status == "failed":
            code = str(error.get("code") or "").strip()
            message = str(error.get("message") or "").strip()
            detail = ": ".join(part for part in (code, message) if part)
            return f"模型请求失败：{detail}" if detail else "模型请求失败，请稍后重试。"

        if status == "incomplete":
            details = body.get("incomplete_details") or {}
            reason = str(details.get("reason") or "").strip()
            if reason:
                return f"模型响应未完成：{reason}。"
            return "模型响应未完成，请稍后重试。"

        return "模型响应流未完成，请稍后重试。"

    def _is_redundant_image_completion_reply(self, text: str) -> bool:
        normalized = str(text).strip().strip("。.!！~～ ")
        return normalized in {
            "好了",
            "好啦",
            "完成了",
            "已完成",
            "生成好了",
            "生成完成",
            "图好了",
            "图片好了",
            "done",
            "finished",
        }

    async def _check_400_error(self, response) -> str | None:
        if response.status == 400:
            error_content = await response.text()
            logger.warning(f"API request rejected: {error_content}")

            sensitive_keywords = [
                "DataInspectionFailed",
                "content_filter",
                "sensitive",
                "safety",
                "violation",
                "audit",
                "prohibited",
            ]
            if any(k.lower() in error_content.lower() for k in sensitive_keywords):
                return "请求被内容审核拦截。"
            return "API 请求被拒绝。"
        return None

    def _use_responses_api(self) -> bool:
        api_style = self.model_info.get("api_style")
        if api_style:
            return api_style == "responses"
        return self.model_info.get("url", "").rstrip("/").endswith("/responses")

    def _use_agy_acp(self) -> bool:
        return self.model_info.get("api_style") == "agy_acp"

    def _use_native_web_search(self) -> bool:
        return bool(self.model_info.get("use_native_web_search"))

    def _use_native_image_generation(self) -> bool:
        return bool(self.model_info.get("use_native_image_generation"))

    def _use_external_image_generation(self) -> bool:
        return bool(self.model_info.get("use_external_image_generation"))

    def _native_image_model_label(self) -> str:
        return str(
            self.model_info.get("native_image_generation_model") or "gpt-image-2"
        ).strip()

    def _vertex_image_generation_config(self) -> dict:
        return config_parser.get_config("vertex_image_generation") or {}

    def _can_use_vertex_image_generation(self) -> bool:
        config = self._vertex_image_generation_config()
        if not config.get("enabled") or not hasattr(self.event, "group_id"):
            return False
        allowed_group_ids = {
            str(group_id).strip()
            for group_id in config.get("allowed_group_ids") or []
            if str(group_id).strip()
        }
        return str(self.event.group_id) in allowed_group_ids

    def _external_image_generation_config(self) -> dict:
        return self.model_info.get("external_image_generation") or {}

    def _provider_extra_headers(self) -> dict[str, str]:
        return model_selector.build_extra_headers(self.model_info)

    def _merge_headers(self, *parts: dict[str, str]) -> dict[str, str]:
        merged: dict[str, str] = {}
        for part in parts:
            for key, value in part.items():
                if value is not None:
                    merged[str(key)] = str(value)
        return merged

    def _image_generation_url(self) -> str:
        config = self._external_image_generation_config()
        return config.get("generation_url") or config.get("url") or "https://api.jucode.cn/v1/images/generations"

    def _image_edit_url(self) -> str:
        config = self._external_image_generation_config()
        generation_url = self._image_generation_url()
        return config.get("edit_url") or generation_url.replace("/generations", "/edits")

    def _supports_image_input(self) -> bool:
        return bool(
            self.model_info.get("is_vision")
            or self._use_responses_api()
            or self._use_agy_acp()
        )

    def prompt_handler(self):
        recent_context = list(context_dict[self.session_key])[:-1]
        safeguards_enabled = config_parser.get_config("safeguards_enabled") is not False
        self.prompt = build_group_chat_prompt(
            BASE_PROMPT,
            recent_context,
            enable_safeguards=safeguards_enabled,
            instruction_profile="minimal",
        )
        if runtime_context := collect_group_context(self.bot, self.event):
            self.prompt += "\n\n群运行时补充信息：\n" + runtime_context

    async def _send_text_response(self, text: str):
        if not is_long_message(text) or not hasattr(self.event, "group_id"):
            await self.send_reply_message(text)
            return

        nodes = [
            {
                "type": "node",
                "data": {
                    "name": "OwO",
                    "uin": str(self.bot.self_id),
                    "content": text,
                },
            }
        ]
        try:
            await self.bot.call_api(
                "send_group_forward_msg",
                group_id=self.event.group_id,
                messages=nodes,
            )
        except Exception:
            logger.error(traceback.format_exc())
            await self.send_reply_message(text)

    async def stream_llm_chat(
        self, session, url, headers, data, proxy, is_segment=False
    ) -> bool | str:
        buffer = []
        async with session.post(url, headers=headers, json=data, proxy=proxy) as response:
            if error_msg := await self._check_400_error(response):
                return error_msg
            if response.status != 200:
                logger.warning(f"Warning: {response}")
                return False
            async for line in response.content:
                if not line or line.startswith(b"data: [DONE]") or line.startswith(b"[DONE]"):
                    break
                decoded = (
                    line[5:].decode("utf-8")
                    if line.startswith(b"data:")
                    else line.decode("utf-8")
                )
                if not decoded.strip() or decoded.startswith(":"):
                    continue
                json_data = json.loads(decoded)
                choices = json_data.get("choices", [{}])
                if not choices:
                    continue
                message = choices[0].get("message", {}) or choices[0].get("delta", {})
                content = message.get("content", "")
                if content:
                    buffer.append(content)
        result = "".join(buffer).strip()
        result, image_memories = self._extract_structured_assistant_output(result)
        self._apply_image_memory_updates(image_memories)
        if not result:
            return False
        if not self.is_objective:
            self.messages_handler.post_process(result)
        await self._send_text_response(result)
        return True

    async def none_stream_llm_chat(self, session, url, headers, data, proxy) -> bool | str:
        async with session.post(
            url=url,
            data=data,
            headers=headers,
            ssl=False,
            proxy=proxy,
        ) as resp:
            if error_msg := await self._check_400_error(resp):
                return error_msg
            response = await resp.json()
            if resp.status != 200 or not response:
                logger.warning(response)
                return False
        choices = response.get("choices")
        if not choices:
            logger.warning(response)
            return False
        result, image_memories = self._extract_structured_assistant_output(
            choices[0]["message"].get("content") or ""
        )
        self._apply_image_memory_updates(image_memories)
        if not result:
            return False
        if not self.is_objective:
            self.messages_handler.post_process(result)
        await self._send_text_response(result)
        return True

    def _get_response_schema(self) -> dict:
        return {
            "type": "json_schema",
            "name": "group_chat_turn",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "assistant_reply": {"type": "string"},
                    "image_memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "client_image_id": {"type": "string"},
                                "summary": {"type": "string"},
                            },
                            "required": ["client_image_id", "summary"],
                        },
                    },
                },
                "required": ["assistant_reply", "image_memories"],
            },
        }

    def _build_user_ref_map(self) -> dict[str, str]:
        result = {}
        for item in self.messages_handler.user_refs:
            ref = item.get("ref")
            user_id = item.get("user_id")
            if ref and user_id:
                result[str(ref)] = str(user_id)
        return result

    def _safe_user_ref_summary(self) -> list[dict[str, str]]:
        result = []
        for item in self.messages_handler.user_refs:
            ref = item.get("ref")
            display_name = item.get("display_name")
            relation = item.get("relation")
            if ref and display_name:
                result.append(
                    {
                        "ref": str(ref),
                        "display_name": str(display_name),
                        "relation": str(relation or "user"),
                    }
                )
        return result

    def _build_user_ref_content(self) -> list[dict]:
        user_refs = self._safe_user_ref_summary()
        if not user_refs:
            return []
        return [
            {
                "type": "input_text",
                "text": "User refs for this turn, without platform IDs: "
                + json.dumps(user_refs, ensure_ascii=False, separators=(",", ":")),
            }
        ]

    def _build_responses_tools(
        self,
        *,
        native_web_search: bool = False,
        native_image_generation: bool = False,
        external_image_generation: bool = False,
        local_image_cache: bool = False,
    ) -> tuple[list[dict], list[str]]:
        tools = []
        include = []
        if local_image_cache:
            tools.append(
                {
                    "type": "function",
                    "name": "fetch_recent_images",
                    "description": "Fetch recent QQ images cached by the plugin for the current user. Use this conservatively when the user asks to reference, edit, combine, redraw, or generate from recently sent images. Prefer fetching several images at once because QQ reply can attach only one image and the intended references may span multiple recent messages. If the fetched images do not include the intended references, call this tool again with a larger offset to fetch older cached images.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of cached images to fetch. Prefer 4 to 8 when unsure, bounded by plugin configuration.",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "How many newest cached images to skip. Use 0 first. If the first batch is not enough, call again with offset equal to the number already inspected to fetch older images.",
                            },
                        },
                        "required": ["limit", "offset"],
                    },
                    "strict": True,
                }
            )
        if self.messages_handler.user_refs:
            tools.append(
                {
                    "type": "function",
                    "name": "fetch_user_avatar",
                    "description": "Fetch a QQ user's avatar for this turn by temporary user_ref, never by QQ number. Use when the user asks to generate an image containing themselves or a mentioned user, or asks what they / a mentioned user look like as a person. Use current_user for 'me' and mentioned_user_N for mentioned users.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "user_ref": {
                                "type": "string",
                                "description": "Temporary user reference from this turn, such as current_user or mentioned_user_1.",
                            }
                        },
                        "required": ["user_ref"],
                    },
                    "strict": True,
                }
            )
        if native_web_search:
            tools.append({"type": "web_search"})
            include.append("web_search_call.action.sources")
        if native_image_generation:
            if not self.native_image_size_selected:
                tools.append(
                    {
                        "type": "function",
                        "name": "set_openai_image_size",
                        "description": "Select the native OpenAI image output size before calling image_generation when the user requests a size, resolution, aspect ratio, portrait, or landscape output. The provider may approximate the requested pixels. Skip this function when automatic sizing is acceptable.",
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "size": {
                                    "type": "string",
                                    "description": "Use auto or WIDTHxHEIGHT. Both edges must be multiples of 16, each edge at most 3840, aspect ratio from 1:3 through 3:1, and total pixels from 655360 through 8294400. Common choices: "
                                    + ", ".join(OPENAI_IMAGE_SIZE_EXAMPLES),
                                }
                            },
                            "required": ["size"],
                        },
                        "strict": True,
                    }
                )
            native_image_tool = {"type": "image_generation"}
            if self.native_image_size_selected:
                native_image_tool["size"] = self.native_image_size
            tools.append(native_image_tool)
        if not self.imagegen_instructions_provided:
            tools.append(
                {
                    "type": "function",
                    "name": "get_imagegen_instructions",
                    "description": "Return the $imagegen prompt-refinement instructions. Call this before using any native or external image generation/editing tool unless the same instructions were already fetched in this turn.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {},
                        "required": [],
                    },
                    "strict": True,
                }
            )
        if external_image_generation:
            tools.append(
                {
                    "type": "function",
                    "name": "image_generation",
                    "description": "POST /v1/images/generations. Generate a new image with an external image generation service. " + IMAGEGEN_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Complete standalone prompt for the image generator, including style, composition, text to render, and any needed non-private visual references.",
                            },
                            "size": {
                                "type": "string",
                                "enum": ["1024x1024", "1024x1536", "1536x1024"],
                                "description": "Output size. Use 1024x1024 by default; portrait posters usually use 1024x1536; landscape banners use 1536x1024.",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of images to generate, normally 1.",
                            },
                        },
                        "required": ["prompt", "size", "n"],
                    },
                    "strict": True,
                }
            )
            tools.append(
                {
                    "type": "function",
                    "name": "image_edit",
                    "description": "POST /v1/images/edits. Edit one or more existing images with an external image generation service. Use this when the user asks to modify, redraw, restyle, compose, or use images/avatars as visual references. " + IMAGEGEN_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Complete standalone edit prompt. State what to change and what must remain unchanged.",
                            },
                            "image_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Image IDs to send to the image edit API. Use IDs from current attachments, fetched recent images, or fetched avatars. Include 1 to 16 images.",
                            },
                            "size": {
                                "type": "string",
                                "enum": ["1024x1024", "1024x1536", "1536x1024"],
                                "description": "Output size. Use 1024x1024 by default; portrait posters usually use 1024x1536; landscape banners use 1536x1024.",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of edited images to return, normally 1.",
                            },
                        },
                        "required": ["prompt", "image_ids", "size", "n"],
                    },
                    "strict": True,
                }
            )
        if self._can_use_vertex_image_generation():
            all_aspect_ratios = sorted(
                {
                    ratio
                    for profile in VERTEX_IMAGE_PROFILES.values()
                    for ratio in profile.aspect_ratios
                }
            )
            tools.append(
                {
                    "type": "function",
                    "name": "generate_image_with_gemini",
                    "description": VERTEX_IMAGE_TOOL_DESCRIPTION + " " + IMAGEGEN_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Complete standalone generation or edit prompt.",
                            },
                            "model": {
                                "type": "string",
                                "enum": list(VERTEX_IMAGE_MODELS),
                            },
                            "aspect_ratio": {
                                "type": "string",
                                "enum": all_aspect_ratios,
                            },
                            "image_size": {
                                "type": "string",
                                "enum": ["512", "1K", "2K", "4K"],
                                "description": "Requested scale. For gemini-2.5-flash-image this value is ignored because that model selects its own size.",
                            },
                            "image_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Zero to 14 image IDs from current attachments, fetched recent images, or fetched avatars.",
                            },
                        },
                        "required": [
                            "prompt",
                            "model",
                            "aspect_ratio",
                            "image_size",
                            "image_ids",
                        ],
                    },
                    "strict": True,
                }
            )
        return tools, include

    def _build_chat_tools(
        self,
        *,
        external_image_generation: bool = False,
        local_image_cache: bool = False,
    ) -> list[dict]:
        tools = []
        if local_image_cache:
            tools.append(
                {
                    "type": "function",
                    "name": "fetch_recent_images",
                    "description": "Fetch recent QQ images cached by the plugin for the current user. Use this conservatively when the user asks to reference, edit, combine, redraw, or generate from recently sent images. Prefer fetching several images at once because QQ reply can attach only one image and the intended references may span multiple recent messages. If the fetched images do not include the intended references, call this tool again with a larger offset to fetch older cached images.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of cached images to fetch. Prefer 4 to 8 when unsure, bounded by plugin configuration.",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "How many newest cached images to skip. Use 0 first. If the first batch is not enough, call again with offset equal to the number already inspected to fetch older cached images.",
                            },
                        },
                        "required": ["limit", "offset"],
                    },
                    "strict": True,
                }
            )
        if self.messages_handler.user_refs:
            tools.append(
                {
                    "type": "function",
                    "name": "fetch_user_avatar",
                    "description": "Fetch a QQ user's avatar for this turn by temporary user_ref, never by QQ number. Use when the user asks to generate an image containing themselves or a mentioned user, or asks what they / a mentioned user look like as a person. Use current_user for 'me' and mentioned_user_N for mentioned users.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "user_ref": {
                                "type": "string",
                                "description": "Temporary user reference from this turn, such as current_user or mentioned_user_1.",
                            }
                        },
                        "required": ["user_ref"],
                    },
                    "strict": True,
                }
            )
        if not self.imagegen_instructions_provided:
            tools.append(
                {
                    "type": "function",
                    "name": "get_imagegen_instructions",
                    "description": "Return the $imagegen prompt-refinement instructions. Call this before using any image generation/editing tool unless the same instructions were already fetched in this turn.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {},
                        "required": [],
                    },
                    "strict": True,
                }
            )
        if external_image_generation:
            tools.append(
                {
                    "type": "function",
                    "name": "image_generation",
                    "description": "POST /v1/images/generations. Generate a new image with an external image generation service. " + IMAGEGEN_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Complete standalone prompt for the image generator, including style, composition, text to render, and any needed non-private visual references.",
                            },
                            "size": {
                                "type": "string",
                                "enum": ["1024x1024", "1024x1536", "1536x1024"],
                                "description": "Output size. Use 1024x1024 by default; portrait posters usually use 1024x1536; landscape banners use 1536x1024.",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of images to generate, normally 1.",
                            },
                        },
                        "required": ["prompt", "size", "n"],
                    },
                    "strict": True,
                }
            )
            tools.append(
                {
                    "type": "function",
                    "name": "image_edit",
                    "description": "POST /v1/images/edits. Edit one or more existing images with an external image generation service. Use this when the user asks to modify, redraw, restyle, compose, or use images/avatars as visual references. " + IMAGEGEN_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Complete standalone edit prompt. State what to change and what must remain unchanged.",
                            },
                            "image_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Image IDs to send to the image edit API. Use IDs from current attachments, fetched recent images, or fetched avatars. Include 1 to 16 images.",
                            },
                            "size": {
                                "type": "string",
                                "enum": ["1024x1024", "1024x1536", "1536x1024"],
                                "description": "Output size. Use 1024x1024 by default; portrait posters usually use 1024x1536; landscape banners use 1536x1024.",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of edited images to return, normally 1.",
                            },
                        },
                        "required": ["prompt", "image_ids", "size", "n"],
                    },
                    "strict": True,
                }
            )
        return tools

    def _extract_function_args(self, response: dict, function_name: str) -> list[dict]:
        args_list = []
        for item in response.get("output", []):
            if item.get("type") != "function_call" or item.get("name") != function_name:
                continue
            arguments = item.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
            except ValueError:
                parsed = {}
            args_list.append(parsed)
        return args_list

    def _merge_streamed_function_calls(self, response: dict, streamed_function_calls: list[dict]) -> dict:
        if not streamed_function_calls:
            return response
        output = response.setdefault("output", [])
        existing_function_calls = {
            (
                item.get("name"),
                item.get("arguments"),
            )
            for item in output
            if item.get("type") == "function_call"
        }
        for item in streamed_function_calls:
            key = (item.get("name"), item.get("arguments"))
            if key not in existing_function_calls:
                output.append(item)
                existing_function_calls.add(key)
        return response

    def _extract_image_generation_args(self, response: dict) -> list[dict]:
        args_list = []
        parsed_calls = [
            *self._extract_function_args(response, "image_generation"),
            *self._extract_function_args(response, "generate_image"),
        ]
        for parsed in parsed_calls:
            prompt = str(parsed.get("prompt") or "").strip()
            if not prompt:
                continue
            size = str(parsed.get("size") or "1024x1024").strip()
            n = parsed.get("n") or 1
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 1
            args_list.append(
                {
                    "prompt": prompt,
                    "size": size,
                    "n": max(1, min(n, 4)),
                }
            )
        return args_list

    def _extract_image_edit_args(self, response: dict) -> list[dict]:
        args_list = []
        for parsed in self._extract_function_args(response, "image_edit"):
            prompt = str(parsed.get("prompt") or "").strip()
            image_ids = parsed.get("image_ids") or []
            if isinstance(image_ids, str):
                image_ids = [image_ids]
            image_ids = [str(image_id).strip() for image_id in image_ids if str(image_id).strip()]
            if not prompt or not image_ids:
                continue
            size = str(parsed.get("size") or "1024x1024").strip()
            n = parsed.get("n") or 1
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 1
            args_list.append(
                {
                    "prompt": prompt,
                    "image_ids": image_ids[:16],
                    "size": size,
                    "n": max(1, min(n, 4)),
                }
            )
        return args_list

    def _extract_chat_tool_calls(self, response: dict) -> list[dict]:
        choices = response.get("choices") or []
        if not choices:
            return []
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        parsed_tool_calls = []
        for index, item in enumerate(tool_calls):
            function = item.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            parsed_tool_calls.append(
                {
                    "id": item.get("id") or f"chat_tool_call_{index}",
                    "type": "function",
                    "name": name,
                    "arguments": function.get("arguments") or "{}",
                }
            )
        function_call = message.get("function_call") or {}
        if function_call and not parsed_tool_calls:
            name = function_call.get("name")
            if name:
                parsed_tool_calls.append(
                    {
                        "id": "chat_tool_call_0",
                        "type": "function",
                        "name": name,
                        "arguments": function_call.get("arguments") or "{}",
                    }
                )
        return parsed_tool_calls

    def _merge_streamed_chat_tool_calls(self, response: dict, streamed_tool_calls: list[dict]) -> dict:
        if not streamed_tool_calls:
            return response
        choices = response.setdefault("choices", [])
        if not choices:
            choices.append({"message": {"role": "assistant"}})
        message = choices[0].setdefault("message", {})
        existing_tool_calls = {
            (
                item.get("id"),
                (item.get("function") or {}).get("name"),
                (item.get("function") or {}).get("arguments"),
            )
            for item in message.get("tool_calls") or []
        }
        tool_calls = message.setdefault("tool_calls", [])
        for item in streamed_tool_calls:
            key = (
                item.get("id"),
                item.get("name"),
                item.get("arguments"),
            )
            if key in existing_tool_calls:
                continue
            tool_calls.append(
                {
                    "id": item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    },
                }
            )
            existing_tool_calls.add(key)
        return response

    def _convert_responses_content_to_chat(self, content):
        if not isinstance(content, list):
            return content
        converted = []
        for item in content:
            item_type = item.get("type")
            if item_type == "input_text":
                converted.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "input_image":
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item.get("image_url")},
                    }
                )
        return converted

    async def _build_chat_messages(self, session, send_message_list: list[dict]) -> list[dict]:
        response_input = await self._build_responses_input(session, send_message_list)
        return [
            {
                "role": item["role"],
                "content": self._convert_responses_content_to_chat(item["content"]),
            }
            for item in response_input
        ]

    def _strip_think_tags(self, content: str) -> str:
        start_tag = "<think>"
        end_tag = "</think>"
        start = content.find(start_tag)
        end = content.find(end_tag)
        if start == -1 and end != -1:
            end += len(end_tag)
            start = 0
            return content[:start] + content[end:]
        if start != -1 and end != -1:
            end += len(end_tag)
            return content[:start] + content[end:]
        return content

    def _extract_structured_assistant_output(
        self, *candidates: str | None
    ) -> tuple[str, list[dict]]:
        normalized_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            normalized = self._strip_think_tags(candidate).strip()
            if not normalized:
                continue
            normalized_candidates.append(normalized)
            parsed = parse_assistant_response_payload(normalized)
            if parsed:
                return (
                    str(parsed.get("assistant_reply") or "").strip(),
                    parsed.get("image_memories") or [],
                )

        for normalized in normalized_candidates:
            if normalized:
                return normalized, []
        return "", []

    def _image_api_headers(self, api_key: str, *, multipart: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept-Encoding": "identity",
        }
        if not multipart:
            headers["Content-Type"] = "application/json"
        return headers

    def _format_image_api_error(self, action: str, status: int, body: str) -> str:
        text = body.strip()
        if len(text) > 500:
            text = text[:500] + "..."
        return f"{action} 请求失败：HTTP {status} {text}".strip()

    async def _send_image_api_response(self, session, body: dict, proxy: str | None) -> int:
        sent_count = 0
        for item in body.get("data") or []:
            image_bytes = None
            if b64_json := item.get("b64_json"):
                try:
                    image_bytes = base64.b64decode(b64_json)
                except Exception:
                    logger.warning("Failed to decode external generated image")
            elif image_url := item.get("url"):
                async with session.get(image_url, proxy=proxy, ssl=False) as image_response:
                    if image_response.status == 200:
                        image_bytes = await image_response.read()
            if image_bytes:
                await self.send_reply_message(MessageSegment.image(image_bytes))
                sent_count += 1
        return sent_count

    async def _generate_external_images(self, requests: list[dict]) -> int | str:
        if not requests:
            return 0
        config = self._external_image_generation_config()
        url = self._image_generation_url()
        model = config.get("model") or "gpt-image-2"
        api_key_env = config.get("api_key_env") or "CODEX_API_KEY"
        api_key = _configured_environment_value(api_key_env)
        if not api_key:
            return f"图片生成 API key 未配置：{api_key_env}。"
        headers = self._image_api_headers(api_key)
        proxy = config.get("proxy") or self.model_info.get("proxy")
        timeout = int(config.get("timeout") or 300)
        sent_count = 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            for request in requests:
                payload = {
                    "model": model,
                    "prompt": request["prompt"],
                    "n": request.get("n") or 1,
                    "size": request.get("size") or config.get("size") or "1024x1024",
                }
                async with session.post(url, headers=headers, json=payload, proxy=proxy, ssl=False) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        logger.warning(
                            {
                                "event": "external_image_generation_failed",
                                "status": response.status,
                                "body": response_text[:1000],
                            }
                        )
                        return self._format_image_api_error("图片生成", response.status, response_text)
                    try:
                        body = json.loads(response_text)
                    except ValueError:
                        logger.warning(
                            {
                                "event": "external_image_generation_bad_json",
                                "status": response.status,
                            }
                        )
                        return "图片生成响应解析失败。"
                    sent_count += await self._send_image_api_response(session, body, proxy)
        return sent_count

    async def _edit_external_images(self, requests: list[dict]) -> int | str:
        if not requests:
            return 0
        config = self._external_image_generation_config()
        url = self._image_edit_url()
        model = config.get("model") or "gpt-image-2"
        api_key_env = config.get("api_key_env") or "CODEX_API_KEY"
        api_key = _configured_environment_value(api_key_env)
        if not api_key:
            return f"图片编辑 API key 未配置：{api_key_env}。"
        headers = self._image_api_headers(api_key, multipart=True)
        proxy = config.get("proxy") or self.model_info.get("proxy")
        timeout = int(config.get("timeout") or 300)
        sent_count = 0
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            for request in requests:
                form = aiohttp.FormData()
                form.add_field("model", model)
                form.add_field("prompt", request["prompt"])
                form.add_field("n", str(request.get("n") or 1))
                form.add_field("size", request.get("size") or config.get("size") or "1024x1024")
                missing_image_ids = []
                for index, image_id in enumerate(request["image_ids"]):
                    image_input = self.image_inputs_by_id.get(image_id)
                    if not image_input:
                        missing_image_ids.append(image_id)
                        continue
                    form.add_field(
                        "image[]",
                        image_input["bytes"],
                        filename=image_input.get("filename") or f"image_{index}.png",
                        content_type=image_input.get("mime_type") or "image/png",
                    )
                if missing_image_ids:
                    return "图片编辑缺少可用输入图：" + ", ".join(missing_image_ids)
                async with session.post(url, headers=headers, data=form, proxy=proxy, ssl=False) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        logger.warning(
                            {
                                "event": "external_image_edit_failed",
                                "status": response.status,
                                "body": response_text[:1000],
                            }
                        )
                        return self._format_image_api_error("图片编辑", response.status, response_text)
                    try:
                        body = json.loads(response_text)
                    except ValueError:
                        logger.warning(
                            {
                                "event": "external_image_edit_bad_json",
                                "status": response.status,
                            }
                        )
                        return "图片编辑响应解析失败。"
                    sent_count += await self._send_image_api_response(session, body, proxy)
        return sent_count

    def _extract_fetch_recent_images_args(self, response: dict) -> dict | None:
        max_limit = int(config_parser.get_config("fetch_recent_images_max_limit") or 6)
        default_limit = int(config_parser.get_config("fetch_recent_images_default_limit") or 3)
        for parsed in self._extract_function_args(response, "fetch_recent_images"):
            limit = parsed.get("limit") or default_limit
            offset = parsed.get("offset") or 0
            return {
                "limit": max(1, min(int(limit), max_limit)),
                "offset": max(0, int(offset)),
            }
        return None

    async def _send_image_with_metadata(
        self,
        image_bytes: bytes,
        *,
        model: str,
        scale: str,
    ) -> None:
        actual_scale = actual_image_scale(image_bytes, scale)
        message = MessageSegment.image(image_bytes) + MessageSegment.text(
            f"\n模型：{model} · Scale：{actual_scale}"
        )
        await self.send_reply_message(message)

    async def _send_generated_images(
        self,
        image_calls: list[dict],
        *,
        model: str | None = None,
        scale: str | None = None,
    ) -> int:
        sent_count = 0
        for image_call in image_calls:
            image_base64 = image_call.get("result")
            if not image_base64:
                continue
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception:
                logger.warning("Failed to decode generated image")
                continue
            await self._send_image_with_metadata(
                image_bytes,
                model=model or self._native_image_model_label(),
                scale=scale or self.native_image_size,
            )
            sent_count += 1
        return sent_count

    def _extract_vertex_image_requests(self, response: dict) -> list[dict]:
        requests = []
        for arguments in self._extract_function_args(
            response, "generate_image_with_gemini"
        ):
            if request := normalize_vertex_image_request(arguments):
                requests.append(request)
        return requests

    async def _generate_vertex_images(self, requests: list[dict]) -> int | str:
        if not requests:
            return 0
        if not self._can_use_vertex_image_generation():
            return "当前会话不能使用 Vertex 图片生成。"
        config = self._vertex_image_generation_config()
        api_key = _read_vertex_api_key(config)
        if not api_key:
            return "Vertex 图片生成凭据不可用。"

        base_url = str(
            config.get("base_url")
            or "https://aiplatform.googleapis.com/v1/publishers/google/models"
        ).rstrip("/")
        timeout = int(config.get("timeout") or 300)
        sent_count = 0
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            for request in requests:
                missing_image_ids = [
                    image_id
                    for image_id in request["image_ids"]
                    if image_id not in self.image_inputs_by_id
                ]
                if missing_image_ids:
                    return "Vertex 图片生成缺少可用输入图：" + ", ".join(
                        missing_image_ids
                    )
                payload = build_vertex_image_payload(
                    request, self.image_inputs_by_id
                )
                url = f"{base_url}/{request['model']}:generateContent"
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    ssl=False,
                ) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        logger.warning(
                            {
                                "event": "vertex_image_generation_failed",
                                "model": request["model"],
                                "status": response.status,
                                "body": response_text[:1000],
                            }
                        )
                        return self._format_image_api_error(
                            "Vertex 图片生成", response.status, response_text
                        )
                    try:
                        body = json.loads(response_text)
                    except ValueError:
                        return "Vertex 图片生成响应解析失败。"
                    images = extract_vertex_images(body)
                    if not images:
                        return "Vertex 图片生成未返回图片。"
                    requested_scale = request["image_size"]
                    if requested_scale == "MODEL_SELECTED":
                        requested_scale = request["aspect_ratio"]
                    for image_bytes in images:
                        await self._send_image_with_metadata(
                            image_bytes,
                            model=request["model"],
                            scale=requested_scale,
                        )
                        sent_count += 1
        return sent_count

    async def _handle_chat_tool_calls(
        self,
        tool_calls: list[dict],
        *,
        external_image_generation: bool,
        local_image_cache: bool,
    ) -> tuple[bool, int | str]:
        rerun_requested = False
        sent_images = 0
        for tool_call in tool_calls:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments") or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except ValueError:
                parsed_arguments = {}
            if name == "fetch_recent_images" and local_image_cache:
                if not self._can_use_group_image_cache():
                    continue
                max_limit = int(config_parser.get_config("fetch_recent_images_max_limit") or 6)
                default_limit = int(config_parser.get_config("fetch_recent_images_default_limit") or 3)
                limit = parsed_arguments.get("limit") or default_limit
                offset = parsed_arguments.get("offset") or 0
                self.fetch_recent_images_rounds += 1
                fetched_images = await image_cache.get_recent_group_images(
                    group_id=self.event.group_id,
                    limit=max(1, min(int(limit), max_limit)),
                    offset=max(0, int(offset)),
                )
                known_image_ids = {image.get("image_id") for image in self.fetched_images}
                self.fetched_images.extend(
                    image
                    for image in fetched_images
                    if image.get("image_id") not in known_image_ids
                )
                if fetched_images:
                    rerun_requested = True
            elif name == "fetch_user_avatar":
                user_ref = str(parsed_arguments.get("user_ref") or "").strip()
                if user_ref:
                    known_avatar_refs = {
                        str(item.get("user_ref"))
                        for item in self.pending_user_avatar_requests
                        if item.get("user_ref")
                    }
                    if user_ref not in known_avatar_refs:
                        self.pending_user_avatar_requests.append({"user_ref": user_ref})
                        rerun_requested = True
            elif name == "get_imagegen_instructions":
                if not self.imagegen_instructions_provided:
                    self.imagegen_instructions_provided = True
                    rerun_requested = True
            elif name in {"image_generation", "generate_image"} and external_image_generation:
                generated_requests = self._extract_image_generation_args(
                    {"output": [{"type": "function_call", "name": name, "arguments": arguments}]}
                )
                generated = await self._generate_external_images(generated_requests)
                if isinstance(generated, str):
                    return False, generated
                sent_images += generated
            elif name == "image_edit" and external_image_generation:
                edited_requests = self._extract_image_edit_args(
                    {"output": [{"type": "function_call", "name": name, "arguments": arguments}]}
                )
                edited = await self._edit_external_images(edited_requests)
                if isinstance(edited, str):
                    return False, edited
                sent_images += edited
        return rerun_requested, sent_images

    async def _chat_completions_once(
        self,
        session,
        url,
        headers,
        data,
        proxy,
    ) -> dict | str | bool:
        if self.model_info.get("stream"):
            streamed_text_chunks = []
            streamed_tool_calls = {}
            async with session.post(url, headers=headers, json=data, proxy=proxy) as response:
                if error_msg := await self._check_400_error(response):
                    return error_msg
                if response.status != 200:
                    logger.warning(f"Warning: {response}")
                    return False
                async for line in response.content:
                    if not line or line.startswith(b"data: [DONE]") or line.startswith(b"[DONE]"):
                        break
                    decoded = (
                        line[5:].decode("utf-8")
                        if line.startswith(b"data:")
                        else line.decode("utf-8")
                    )
                    if not decoded.strip() or decoded.startswith(":"):
                        continue
                    json_data = json.loads(decoded)
                    choices = json_data.get("choices", [{}])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content", "")
                    if content:
                        streamed_text_chunks.append(content)
                    for tool_call in delta.get("tool_calls") or []:
                        index = tool_call.get("index", 0)
                        current = streamed_tool_calls.setdefault(
                            index,
                            {
                                "id": tool_call.get("id"),
                                "name": None,
                                "arguments": "",
                            },
                        )
                        if tool_call.get("id"):
                            current["id"] = tool_call.get("id")
                        function = tool_call.get("function") or {}
                        if function.get("name"):
                            current["name"] = function.get("name")
                        if function.get("arguments"):
                            current["arguments"] += function.get("arguments")
            ordered_tool_calls = [
                {
                    "id": item.get("id") or f"chat_tool_call_{index}",
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                }
                for index, item in sorted(streamed_tool_calls.items())
                if item.get("name")
            ]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "".join(streamed_text_chunks).strip(),
                            "tool_calls": ordered_tool_calls or None,
                        }
                    }
                ]
            }
        async with session.post(
            url=url,
            json=data,
            headers=headers,
            ssl=False,
            proxy=proxy,
        ) as resp:
            if error_msg := await self._check_400_error(resp):
                return error_msg
            response = await resp.json()
            if resp.status != 200 or not response:
                logger.warning(response)
                return False
        return response

    async def chat_completions_llm_chat(
        self,
        session,
        url,
        headers,
        send_message_list,
        proxy,
        external_image_generation=False,
        local_image_cache=False,
    ) -> bool | str:
        max_tool_rounds = 5
        for _ in range(max_tool_rounds):
            chat_messages = await self._build_chat_messages(session, send_message_list)
            chat_messages.insert(0, {"role": "system", "content": self.prompt})
            payload = {
                "model": self.model_info["model"],
                "messages": chat_messages,
                "max_tokens": self.model_info.get("max_tokens"),
                "temperature": self.model_info.get("temperature"),
                "top_p": self.model_info.get("top_p"),
                "stream": self.model_info.get("stream", False),
            }
            if self.model_info.get("top_k"):
                payload["top_k"] = self.model_info.get("top_k")
            tools = self._build_chat_tools(
                external_image_generation=external_image_generation,
                local_image_cache=local_image_cache,
            )
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            response = await self._chat_completions_once(session, url, headers, payload, proxy)
            if isinstance(response, str):
                return response
            if response is False:
                return False

            tool_calls = self._extract_chat_tool_calls(response)
            if tool_calls:
                rerun_requested, sent_images = await self._handle_chat_tool_calls(
                    tool_calls,
                    external_image_generation=external_image_generation,
                    local_image_cache=local_image_cache,
                )
                if isinstance(sent_images, str):
                    return sent_images
                if rerun_requested and sent_images == 0:
                    continue
                if sent_images > 0:
                    return True

            message = (response.get("choices") or [{}])[0].get("message") or {}
            assistant_reply, image_memories = self._extract_structured_assistant_output(
                str(message.get("content") or "")
            )
            self._apply_image_memory_updates(image_memories)
            if not assistant_reply:
                return False
            if not self.is_objective:
                self.messages_handler.post_process(assistant_reply)
            await self._send_text_response(assistant_reply)
            return True
        return "工具调用轮次过多。"

    async def _prepare_images(
        self,
        session,
        images: list[dict],
        *,
        include_known_images: bool = False,
        remember_in_group: bool = True,
    ) -> list[dict]:
        prepared = []
        proxy = self.model_info.get("proxy")
        group = getattr(self.event, "group_id", f"private:{self.user_id}") if remember_in_group else f"avatar:{self.user_id}"
        for source_index, image in enumerate(images):
            downloaded = await image_cache.read_image(
                image, group_id=group, user_id=self.user_id,
                message_id=getattr(self.event, "message_id", 0), proxy=proxy,
            )
            if downloaded is None:
                continue
            image_bytes, mime_type = downloaded
            filename = "image." + mime_type.split("/", 1)[-1]
            digest = hashlib.sha256(image_bytes).hexdigest()
            image_id = f"img_sha256_{digest[:16]}"
            summary = image_memory_store.get_summary(image_id)
            image["image_id"] = image_id
            image["mime_type"] = mime_type
            self.image_inputs_by_id[image_id] = {
                "bytes": image_bytes,
                "mime_type": mime_type,
                "filename": filename,
            }
            if summary and not include_known_images:
                image["summary"] = summary
                prepared.append(
                    {
                        "image_id": image_id,
                        "summary": summary,
                        "known": True,
                        "source_index": source_index,
                    }
                )
                continue
            data_url = f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode()
            prepared.append(
                {
                    "image_id": image_id,
                    "mime_type": mime_type,
                    "data_url": data_url,
                    "known": False,
                    "source_index": source_index,
                }
            )
        return prepared

    async def _prepare_current_images(self, session) -> list[dict]:
        return await self._prepare_images(
            session,
            self.messages_handler.current_images,
            include_known_images=self.imagegen_instructions_provided,
        )

    async def _prepare_user_avatar(self, session, user_ref: str) -> dict | None:
        user_id = self._build_user_ref_map().get(str(user_ref))
        if not user_id:
            return None
        source_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
        prepared = await self._prepare_images(
            session,
            [{"source_url": source_url}],
            include_known_images=True,
            remember_in_group=False,
        )
        if not prepared:
            return None
        image = prepared[0]
        return {
            "user_ref": str(user_ref),
            "image_id": image["image_id"],
            "data_url": image["data_url"],
        }

    async def _build_user_avatar_content(self, session) -> list[dict]:
        content = []
        if not self._supports_image_input():
            return content
        seen_refs = set()
        for parsed in self.pending_user_avatar_requests:
            user_ref = str(parsed.get("user_ref") or "")
            if not user_ref or user_ref in seen_refs:
                continue
            seen_refs.add(user_ref)
            avatar = await self._prepare_user_avatar(session, user_ref)
            if not avatar:
                continue
            content.append(
                {
                    "type": "input_text",
                    "text": f"Avatar fetched for {avatar['user_ref']} [image:{avatar['image_id']}].",
                }
            )
            content.append({"type": "input_image", "image_url": avatar["data_url"]})
        return content

    async def _build_fetched_image_content(self, session) -> list[dict]:
        content = []
        if not (self._supports_image_input() and self.fetched_images):
            return content
        prepared_images = await self._prepare_images(
            session,
            self.fetched_images,
            include_known_images=True,
        )
        if prepared_images:
            content.append(
                {
                    "type": "input_text",
                    "text": "Images fetched from recent QQ cache for this turn.",
                }
            )
        for image in prepared_images:
            content.append(
                {
                    "type": "input_text",
                    "text": f"Attachment for fetched recent image [image:{image['image_id']}]",
                }
            )
            content.append({"type": "input_image", "image_url": image["data_url"]})
        return content

    async def _build_responses_input(self, session, send_message_list: list[dict]) -> list[dict]:
        input_items = []
        fetched_image_content = await self._build_fetched_image_content(session)
        if fetched_image_content:
            input_items.append(
                {
                    "role": "user",
                    "content": fetched_image_content,
                }
            )
        for message in send_message_list[:-1]:
            input_items.append({"role": message["role"], "content": message["content"]})

        user_ref_content = self._build_user_ref_content()
        if user_ref_content:
            input_items.append({"role": "user", "content": user_ref_content})

        avatar_content = await self._build_user_avatar_content(session)
        if avatar_content:
            input_items.append({"role": "user", "content": avatar_content})

        if self.imagegen_instructions_provided:
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Tool result from get_imagegen_instructions:\n"
                            + IMAGEGEN_TOOL_INSTRUCTIONS,
                        }
                    ],
                }
            )

        current_message = send_message_list[-1]
        current_content = await self._build_current_message_content(
            session, str(current_message.get("content") or "")
        )
        input_items.append({"role": "user", "content": current_content})
        return input_items

    async def _build_current_message_content(
        self, session, current_text: str
    ) -> str | list[dict]:
        if not (self._supports_image_input() and self.messages_handler.current_images):
            return current_text

        prepared_images = await self._prepare_current_images(session)
        markers = ["[图片暂不可用]"] * len(self.messages_handler.current_images)
        new_images = []
        for image in prepared_images:
            source_index = image["source_index"]
            if image["known"]:
                markers[source_index] = build_image_reference(image["summary"])
            else:
                markers[source_index] = f"[image:{image['image_id']}]"
                new_images.append(image)

        current_text = replace_image_placeholders(current_text, markers)
        current_content: list[dict] = [{"type": "input_text", "text": current_text}]
        for image in new_images:
            current_content.append(
                {
                    "type": "input_text",
                    "text": f"Attachment for [image:{image['image_id']}]",
                }
            )
            current_content.append(
                {
                    "type": "input_image",
                    "image_url": image["data_url"],
                }
            )
        return current_content

    def _apply_image_memory_updates(self, image_memories: list[dict]):
        summary_map = {}
        for item in image_memories:
            image_id = item.get("client_image_id")
            summary = normalize_image_summary(item.get("summary", ""))
            if image_id and summary:
                summary_map[image_id] = summary

        for image in self.messages_handler.current_images:
            if image.get("summary"):
                continue
            image_id = image.get("image_id")
            summary = summary_map.get(image_id)
            if summary:
                image["summary"] = summary
                image_memory_store.set_summary(
                    image_id, summary, mime_type=image.get("mime_type")
                )

        self.messages_handler.update_current_user_message_with_image_summaries()

    def _sync_group_context_with_current_user_message(self):
        session_messages = context_dict[self.session_key]
        if not session_messages:
            return
        sender_name = self.event.sender.card or self.event.sender.nickname
        session_messages[-1] = {
            "speaker_name": sender_name,
            "content": self.messages_handler.new_user_msg["content"],
            "images": self.messages_handler.current_images,
        }

    def _can_use_group_image_cache(self) -> bool:
        return hasattr(self.event, "group_id")

    def _extract_stream_text_delta(self, event) -> str:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            return delta
        text = getattr(event, "text", None)
        if isinstance(text, str):
            return text
        return ""

    def _looks_like_recent_image_request(self) -> bool:
        text = self.messages_handler.current_text
        image_words = ("图", "图片", "照片", "脸", "logo", "p", "P", "生成", "合成", "参考", "照着", "换", "改")
        reference_words = ("刚才", "最近", "上面", "前面", "后面", "前几张", "后几张", "这几张", "那几张")
        return any(word in text for word in image_words) and any(word in text for word in reference_words)

    async def _prefetch_recent_images_if_needed(self):
        if not self._can_use_group_image_cache():
            return
        if self.messages_handler.current_images or self.fetched_images:
            return
        if not self._looks_like_recent_image_request():
            return
        limit = int(config_parser.get_config("fetch_recent_images_default_limit") or 6)
        max_limit = int(config_parser.get_config("fetch_recent_images_max_limit") or 10)
        self.fetched_images = await image_cache.get_recent_group_images(
            group_id=self.event.group_id,
            limit=max(1, min(limit, max_limit)),
            offset=0,
        )
        if self.fetched_images:
            logger.info(f"Prefetched {len(self.fetched_images)} recent cached images")

    def _log_responses_summary(
        self,
        body: dict,
        tools: list[dict],
        event_counts: dict[str, int],
        streamed_text_chars: int,
    ):
        output = body.get("output") or []
        output_summary = []
        for item in output:
            summary = {"type": item.get("type")}
            if item.get("type") == "function_call":
                summary["name"] = item.get("name")
                summary["arguments"] = item.get("arguments")
            elif item.get("type") == "image_generation_call":
                summary["id"] = item.get("id") or item.get("image_id")
                summary["has_result"] = bool(item.get("result"))
            elif item.get("type") == "message":
                summary["role"] = item.get("role")
                summary["content_types"] = [content.get("type") for content in item.get("content", [])]
            output_summary.append(summary)
        logger.info(
            {
                "event": "responses_summary",
                "response_id": body.get("id"),
                "status": body.get("status"),
                "tools": [tool.get("name") or tool.get("type") for tool in tools],
                "output": output_summary,
                "usage": body.get("usage"),
                "tool_usage": body.get("tool_usage"),
                "event_counts": event_counts,
                "streamed_text_chars": streamed_text_chars,
                "output_text_chars": len(body.get("output_text") or ""),
                "current_image_count": len(self.messages_handler.current_images),
                "fetched_image_count": len(self.fetched_images),
            }
        )

    async def responses_llm_chat(
        self,
        url,
        headers,
        send_message_list,
        proxy,
        native_web_search=False,
        external_image_generation=False,
        local_image_cache=False,
    ) -> bool | str:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as input_session:
            response_input = await self._build_responses_input(input_session, send_message_list)

        text_payload = {"format": self._get_response_schema()}
        if verbosity := self.model_info.get("verbosity"):
            text_payload["verbosity"] = verbosity

        payload = {
            "model": self.model_info["model"],
            "store": False,
            "instructions": self.prompt,
            "input": response_input,
            "text": text_payload,
        }
        if max_tokens := self.model_info.get("max_tokens"):
            payload["max_output_tokens"] = max_tokens
        if reasoning_effort := self.model_info.get("reasoning_effort"):
            payload["reasoning"] = {"effort": reasoning_effort}
        tools, include = self._build_responses_tools(
            native_web_search=native_web_search,
            native_image_generation=self._use_native_image_generation(),
            external_image_generation=external_image_generation,
            local_image_cache=local_image_cache,
        )
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if include:
            payload["include"] = include

        api_key = self.model_info["key"].removeprefix("Bearer").strip()
        base_url = url.rsplit("/", 1)[0]
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": 300,
            "default_headers": self._provider_extra_headers(),
        }
        http_client = None
        if proxy:
            http_client = httpx.AsyncClient(proxy=proxy, timeout=300)
            client_kwargs["http_client"] = http_client
        client = AsyncOpenAI(**client_kwargs)

        streamed_text_chunks = []
        streamed_image_calls = {}
        partial_image_calls = {}
        sent_stream_image_ids = set()
        sent_stream_image_count = 0
        stream_event_counts = defaultdict(int)
        streamed_function_calls = []
        final_response = None
        terminal_response = None
        stream_started_at = time.monotonic()

        def log_generation_candidate_event(event, event_type: str):
            item = getattr(event, "item", None)
            item_type = getattr(item, "type", None) if item is not None else None
            item_id = (
                getattr(item, "id", None)
                if item is not None
                else getattr(event, "item_id", None)
            )
            logger.info(
                {
                    "event": "responses_image_generation_candidate_event",
                    "event_type": event_type,
                    "item_type": item_type,
                    "item_id": item_id,
                    "output_index": getattr(event, "output_index", None),
                    "content_index": getattr(event, "content_index", None),
                    "sequence_number": getattr(event, "sequence_number", None),
                    "status": getattr(item, "status", None) if item is not None else getattr(event, "status", None),
                    "has_result": bool(getattr(item, "result", None)) if item is not None else False,
                    "partial_image_b64_len": len(getattr(event, "partial_image_b64", None) or ""),
                    "notice_sent": self.generation_notice_sent,
                    "elapsed_seconds": round(time.monotonic() - stream_started_at, 3),
                }
            )

        try:
            if not self.model_info.get("stream", False):
                final_response = await client.responses.create(**payload)
            else:
                stream = await client.responses.create(**payload, stream=True)
                async for event in stream:
                        event_type = getattr(event, "type", "")
                        stream_event_counts[event_type] += 1
                        # log_generation_candidate_event(event, event_type)
                        if event_type == "response.output_text.delta":
                            if delta := self._extract_stream_text_delta(event):
                                streamed_text_chunks.append(delta)
                        elif event_type == "response.output_item.added":
                            item = getattr(event, "item", None)
                            item_type = getattr(item, "type", "") if item is not None else ""
                            if item_type == "image_generation_call":
                                await self.send_generation_notice_event_once()
                        elif event_type == "response.output_item.done":
                            item = getattr(event, "item", None)
                            item_type = getattr(item, "type", "") if item is not None else ""
                            if item_type == "function_call":
                                function_name = getattr(item, "name", None)
                                streamed_function_calls.append(
                                    {
                                        "type": "function_call",
                                        "name": function_name,
                                        "arguments": getattr(item, "arguments", None),
                                    }
                                )
                                if function_name in {
                                    "get_imagegen_instructions",
                                    "set_openai_image_size",
                                    "generate_image_with_gemini",
                                }:
                                    await self.send_generation_notice_event_once()
                            elif (
                                item is not None
                                and item_type == "image_generation_call"
                                and getattr(item, "result", None)
                            ):
                                item_id = getattr(item, "id", None) or f"output_{getattr(event, 'output_index', 0)}"
                                streamed_image_calls[item_id] = {
                                    "result": item.result,
                                    "image_id": item_id,
                                }
                        elif event_type == "response.image_generation_call.partial_image":
                            item_id = getattr(event, "item_id", None)
                            partial_image_b64 = getattr(event, "partial_image_b64", None)
                            if item_id and partial_image_b64:
                                image_call = {
                                    "result": partial_image_b64,
                                    "image_id": item_id,
                                }
                                partial_image_calls[item_id] = image_call
                        if (
                            not self.generation_notice_sent
                            and "image_generation_call" in event_type
                        ):
                            await self.send_generation_notice_event_once()
                        if event_type == "response.completed":
                            final_response = getattr(event, "response", None)
                        elif event_type in {"response.failed", "response.incomplete"}:
                            terminal_response = getattr(event, "response", None)
                if final_response is None:
                    logger.warning(
                        {
                            "event": "responses_stream_incomplete",
                            "error": "response.completed was not received",
                            "event_counts": dict(stream_event_counts),
                            "function_calls": streamed_function_calls,
                        }
                    )
        except Exception as exc:
            logger.error(traceback.format_exc())
            if partial_image_calls:
                await self._send_generated_images(
                    list(partial_image_calls.values())
                )
                return True
            return self._format_upstream_error(exc)
        finally:
            await client.close()
            if http_client is not None:
                await http_client.aclose()

        body = self._response_body(final_response if final_response is not None else terminal_response)
        if not body:
            body = {"id": None, "status": "stream_incomplete", "output": streamed_function_calls}
        body = self._merge_streamed_function_calls(body, streamed_function_calls)
        self._log_responses_summary(
            body,
            tools,
            dict(stream_event_counts),
            sum(len(chunk) for chunk in streamed_text_chunks),
        )
        if body.get("status") in {"failed", "incomplete"}:
            return self._format_response_terminal_status(body)
        if final_response is None:
            return self._format_response_terminal_status(body)
        rerun_with_imagegen_instructions = (
            not self.imagegen_instructions_provided
            and self._extract_function_args(body, "get_imagegen_instructions")
        )
        if rerun_with_imagegen_instructions:
            await self.send_generation_notice_event_once()
            self.imagegen_instructions_provided = True
        size_requests = self._extract_function_args(body, "set_openai_image_size")
        rerun_with_native_image_size = bool(size_requests) and not self.native_image_size_selected
        if rerun_with_native_image_size:
            await self.send_generation_notice_event_once()
            self.native_image_size = normalize_openai_image_size(
                size_requests[-1].get("size")
            )
            self.native_image_size_selected = True
        avatar_args = self._extract_function_args(body, "fetch_user_avatar")
        known_avatar_refs = {
            str(item.get("user_ref"))
            for item in self.pending_user_avatar_requests
            if item.get("user_ref")
        }
        new_avatar_args = [
            item
            for item in avatar_args
            if item.get("user_ref") and str(item.get("user_ref")) not in known_avatar_refs
        ]
        if new_avatar_args:
            self.pending_user_avatar_requests.extend(new_avatar_args)
            return await self.responses_llm_chat(
                url,
                headers,
                send_message_list,
                proxy,
                native_web_search=native_web_search,
                external_image_generation=external_image_generation,
                local_image_cache=local_image_cache,
            )
        max_fetch_rounds = int(config_parser.get_config("fetch_recent_images_max_rounds") or 3)
        if local_image_cache:
            fetch_args = self._extract_fetch_recent_images_args(body)
            if fetch_args:
                if not self._can_use_group_image_cache():
                    return False
                if self.fetch_recent_images_rounds < max_fetch_rounds:
                    self.fetch_recent_images_rounds += 1
                    fetched_images = await image_cache.get_recent_group_images(
                        group_id=self.event.group_id,
                        limit=fetch_args["limit"],
                        offset=fetch_args["offset"],
                    )
                    known_image_ids = {image.get("image_id") for image in self.fetched_images}
                    self.fetched_images.extend(
                        image
                        for image in fetched_images
                        if image.get("image_id") not in known_image_ids
                    )
                    if fetched_images:
                        return await self.responses_llm_chat(
                            url,
                            headers,
                            send_message_list,
                            proxy,
                            native_web_search=native_web_search,
                            external_image_generation=external_image_generation,
                            local_image_cache=True,
                        )
                logger.info(
                    {
                        "event": "responses_fetch_final_answer",
                        "fetch_rounds": self.fetch_recent_images_rounds,
                        "max_fetch_rounds": max_fetch_rounds,
                        "fetched_image_count": len(self.fetched_images),
                    }
                )
                return await self.responses_llm_chat(
                    url,
                    headers,
                    send_message_list,
                    proxy,
                    native_web_search=native_web_search,
                    external_image_generation=external_image_generation,
                    local_image_cache=False,
                )
        if rerun_with_imagegen_instructions or rerun_with_native_image_size:
            return await self.responses_llm_chat(
                url,
                headers,
                send_message_list,
                proxy,
                native_web_search=native_web_search,
                external_image_generation=external_image_generation,
                local_image_cache=local_image_cache,
            )
        generate_image_args = self._extract_image_generation_args(body)
        edit_image_args = self._extract_image_edit_args(body)
        vertex_image_requests = self._extract_vertex_image_requests(body)
        external_sent_images = 0
        if generate_image_args or edit_image_args or vertex_image_requests:
            await self.send_generation_notice_event_once()
        if vertex_image_requests:
            generated = await self._generate_vertex_images(vertex_image_requests)
            if isinstance(generated, str):
                return generated
            external_sent_images += generated
        if edit_image_args:
            edited = await self._edit_external_images(edit_image_args)
            if isinstance(edited, str):
                return edited
            external_sent_images += edited
        if generate_image_args:
            generated = await self._generate_external_images(generate_image_args)
            if isinstance(generated, str):
                return generated
            external_sent_images += generated

        image_calls = extract_image_generation_calls(body)
        if not image_calls:
            image_calls = list(streamed_image_calls.values())
        if not image_calls:
            image_calls = list(partial_image_calls.values())
        image_calls = [
            image_call
            for image_call in image_calls
            if image_call.get("image_id") not in sent_stream_image_ids
        ]

        streamed_reply = "".join(streamed_text_chunks).strip()
        body_output_text = extract_response_output_text(body)
        output_text_obj = getattr(final_response, "output_text", None) if final_response is not None else None
        assistant_reply, image_memories = self._extract_structured_assistant_output(
            output_text_obj if isinstance(output_text_obj, str) else None,
            streamed_reply,
            body_output_text,
        )

        if not assistant_reply and not image_calls and sent_stream_image_count == 0 and external_sent_images == 0:
            if body.get("status") == "completed" and self.empty_response_retries < 1:
                self.empty_response_retries += 1
                logger.warning(
                    {
                        "event": "responses_empty_output_retry",
                        "response_id": body.get("id"),
                        "streamed_text_chars": len(streamed_reply),
                    }
                )
                return await self.responses_llm_chat(
                    url,
                    headers,
                    send_message_list,
                    proxy,
                    native_web_search=native_web_search,
                    external_image_generation=external_image_generation,
                    local_image_cache=local_image_cache,
                )
            if self.fetched_images and self._looks_like_recent_image_request():
                return f"这轮模型没返回文本或图片；我这边只取到了 {len(self.fetched_images)} 张群缓存图。"
            return "模型返回了空结果，请再试一次。"

        self._apply_image_memory_updates(image_memories)
        self._sync_group_context_with_current_user_message()
        if not self.is_objective and assistant_reply:
            self.messages_handler.post_process(assistant_reply)

        sent_images = external_sent_images + sent_stream_image_count + await self._send_generated_images(image_calls)
        if sent_images > 0 and self._is_redundant_image_completion_reply(assistant_reply):
            assistant_reply = ""
        if assistant_reply:
            await self._send_text_response(assistant_reply)
        elif sent_images == 0:
            return False
        return True

    def _stage_agy_input_image(
        self, turn_inputs_dir: Path | None, image_id: str
    ) -> str | None:
        if turn_inputs_dir is None or not image_id:
            return None
        info = self.image_inputs_by_id.get(image_id)
        if not info or not info.get("bytes"):
            return None
        mime_type = str(info.get("mime_type") or "image/png").lower()
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        ext = ext_map.get(mime_type, ".png")
        turn_inputs_dir.mkdir(parents=True, exist_ok=True)
        file_path = turn_inputs_dir / f"{image_id}{ext}"
        try:
            if not file_path.exists():
                file_path.write_bytes(info["bytes"])
            return str(file_path)
        except OSError:
            return None

    async def _handle_agy_mcp_tool_call(
        self,
        session,
        turn_inputs_dir: Path,
        tool_name: str,
        arguments: dict,
    ) -> list[dict] | str:
        normalized = tool_name.strip()
        if normalized.startswith("mcp_qq_context_"):
            normalized = normalized[len("mcp_qq_context_") :]

        if normalized == "get_imagegen_instructions":
            self.imagegen_instructions_provided = True
            await self.send_generation_notice_event_once()
            return IMAGEGEN_TOOL_INSTRUCTIONS

        if normalized == "fetch_user_avatar":
            user_ref = str(arguments.get("user_ref") or "").strip()
            if not user_ref:
                return json.dumps(
                    {"error": "缺少 user_ref 参数。"}, ensure_ascii=False
                )
            avatar = await self._prepare_user_avatar(session, user_ref)
            if not avatar:
                return json.dumps(
                    {"error": f"未找到用户引用 {user_ref} 的头像。"},
                    ensure_ascii=False,
                )
            image_id = avatar["image_id"]
            file_path = self._stage_agy_input_image(turn_inputs_dir, image_id)
            result_payload = {
                "user_ref": avatar["user_ref"],
                "image_id": image_id,
                "file_path": file_path,
                "note": "如需将该头像作为生图/改图参考，请将 file_path 传入 generate_image 的 ImagePaths 参数。",
            }
            content_blocks: list[dict] = [
                {
                    "type": "text",
                    "text": json.dumps(result_payload, ensure_ascii=False),
                }
            ]
            info = self.image_inputs_by_id.get(image_id)
            if info and info.get("bytes"):
                content_blocks.append(
                    {
                        "type": "image",
                        "mimeType": str(info.get("mime_type") or "image/jpeg"),
                        "data": base64.b64encode(info["bytes"]).decode("ascii"),
                    }
                )
            return content_blocks

        if normalized == "fetch_recent_images":
            if not self._can_use_group_image_cache():
                return json.dumps(
                    {"images": [], "message": "当前会话不支持群图片缓存。"},
                    ensure_ascii=False,
                )
            max_fetch_rounds = int(
                config_parser.get_config("fetch_recent_images_max_rounds") or 3
            )
            if self.fetch_recent_images_rounds >= max_fetch_rounds:
                return json.dumps(
                    {
                        "images": [],
                        "message": f"已达到本轮最大取图轮次（{max_fetch_rounds}）。",
                    },
                    ensure_ascii=False,
                )
            max_limit = int(
                config_parser.get_config("fetch_recent_images_max_limit") or 6
            )
            default_limit = int(
                config_parser.get_config("fetch_recent_images_default_limit") or 3
            )
            try:
                limit = int(arguments.get("limit") or default_limit)
            except (TypeError, ValueError):
                limit = default_limit
            try:
                offset = int(arguments.get("offset") or 0)
            except (TypeError, ValueError):
                offset = 0
            limit = max(1, min(limit, max_limit))
            offset = max(0, offset)

            self.fetch_recent_images_rounds += 1
            fetched = await image_cache.get_recent_group_images(
                group_id=self.event.group_id,
                limit=limit,
                offset=offset,
            )
            known_ids = {img.get("image_id") for img in self.fetched_images}
            self.fetched_images.extend(
                img for img in fetched if img.get("image_id") not in known_ids
            )
            prepared = await self._prepare_images(
                session, fetched, include_known_images=True
            )
            items = []
            image_blocks: list[dict] = []
            for img in prepared:
                image_id = img["image_id"]
                file_path = self._stage_agy_input_image(turn_inputs_dir, image_id)
                summary = image_memory_store.get_summary(image_id) or ""
                items.append(
                    {
                        "image_id": image_id,
                        "file_path": file_path,
                        "summary": summary,
                    }
                )
                info = self.image_inputs_by_id.get(image_id)
                if info and info.get("bytes"):
                    image_blocks.append(
                        {
                            "type": "image",
                            "mimeType": str(info.get("mime_type") or "image/jpeg"),
                            "data": base64.b64encode(info["bytes"]).decode("ascii"),
                        }
                    )
            return [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "images": items,
                            "note": "如需引用或编辑某张图，请将其 file_path 传入 generate_image 的 ImagePaths 参数。",
                        },
                        ensure_ascii=False,
                    ),
                },
                *image_blocks,
            ]

        raise ValueError(f"未知的 MCP 工具: {tool_name}")

    async def _build_agy_acp_prompt_blocks(
        self,
        session,
        send_message_list: list[dict],
        *,
        turn_inputs_dir: Path | None = None,
    ) -> list[dict]:
        schema_json = json.dumps(
            self._get_response_schema()["schema"], ensure_ascii=False
        )
        blocks: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"{self.prompt}\n\n"
                    "可用工具说明：\n"
                    "- 需要联网检索或查实时信息时，可使用 search_web。\n"
                    "- 需要获取群内最近发送的历史图片时，可调用 mcp_qq_context_fetch_recent_images。\n"
                    "- 需要获取发言人（current_user）或被提及群友（mentioned_user_N）的 QQ 头像时，可调用 mcp_qq_context_fetch_user_avatar。\n"
                    "- 需要生成或编辑图片前，请先调用 mcp_qq_context_get_imagegen_instructions 获取提示词优化规范，再直接调用 generate_image 工具生成图片；"
                    "若需基于已有图片或头像进行改图/合成/参考，请将其本地文件路径（file_path）传入 generate_image 的 ImagePaths 参数。"
                    "系统会自动把生成的图片发送给群聊用户，请勿声称自己无法生成或发送图片。\n"
                    "- 请勿调用文件读写或终端命令工具。\n\n"
                    "请严格只输出一个符合以下 JSON Schema 的 JSON 对象，不要输出任何额外解释或 markdown 代码块：\n"
                    f"{schema_json}"
                ),
            }
        ]

        fetched_image_content = await self._build_fetched_image_content(session)
        if fetched_image_content:
            blocks.extend(responses_content_to_acp_blocks(fetched_image_content))

        for message in send_message_list[:-1]:
            role = str(message.get("role") or "user")
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                blocks.append({"type": "text", "text": f"[{role}] {content}"})
            elif isinstance(content, list):
                converted = responses_content_to_acp_blocks(content)
                if converted:
                    blocks.append({"type": "text", "text": f"[{role}]"})
                    blocks.extend(converted)

        user_ref_content = self._build_user_ref_content()
        if user_ref_content:
            blocks.extend(responses_content_to_acp_blocks(user_ref_content))

        avatar_content = await self._build_user_avatar_content(session)
        if avatar_content:
            blocks.extend(responses_content_to_acp_blocks(avatar_content))

        current_message = send_message_list[-1]
        current_content = await self._build_current_message_content(
            session, str(current_message.get("content") or "")
        )
        blocks.extend(responses_content_to_acp_blocks(current_content))

        if turn_inputs_dir is not None and self.image_inputs_by_id:
            staged_lines = []
            for image_id in self.image_inputs_by_id:
                staged_path = self._stage_agy_input_image(turn_inputs_dir, image_id)
                if staged_path:
                    staged_lines.append(f"- [image:{image_id}]: {staged_path}")
            if staged_lines:
                blocks.append(
                    {
                        "type": "text",
                        "text": (
                            "已就绪的输入图片本地路径（供 generate_image 的 ImagePaths 参数使用）：\n"
                            + "\n".join(staged_lines)
                        ),
                    }
                )
        return blocks

    async def agy_acp_llm_chat(
        self, session, send_message_list: list[dict]
    ) -> str | bool:
        turn_inputs_dir = (
            get_agy_runtime_dir() / "inputs" / uuid.uuid4().hex
        )
        try:
            prompt_blocks = await self._build_agy_acp_prompt_blocks(
                session,
                send_message_list,
                turn_inputs_dir=turn_inputs_dir,
            )
            client = get_shared_acp_client()

            async def mcp_handler(tool_name: str, arguments: dict):
                return await self._handle_agy_mcp_tool_call(
                    session, turn_inputs_dir, tool_name, arguments
                )

            def get_input_hashes() -> set[str]:
                return {
                    hashlib.sha256(info["bytes"]).hexdigest()
                    for info in self.image_inputs_by_id.values()
                    if info and info.get("bytes")
                }

            turn_result = await client.run_turn(
                prompt_blocks=prompt_blocks,
                model_info=self.model_info,
                on_image_generation_start=self.send_generation_notice_event_once,
                mcp_tool_handler=mcp_handler,
                extra_input_hashes=get_input_hashes,
            )
        finally:
            if turn_inputs_dir.exists():
                shutil.rmtree(turn_inputs_dir, ignore_errors=True)

        raw_steps = turn_result.text_steps or (
            [turn_result.text] if turn_result.text else []
        )
        assistant_reply = ""
        merged_memories: dict[str, dict] = {}
        for step_text in raw_steps:
            step_reply, step_memories = self._extract_structured_assistant_output(
                step_text
            )
            if step_reply:
                assistant_reply = step_reply
            for item in step_memories:
                image_id = item.get("client_image_id")
                if image_id and item.get("summary"):
                    merged_memories[image_id] = item
        image_memories = list(merged_memories.values())
        generated_images = turn_result.images or []

        if not assistant_reply and not generated_images:
            if self.fetched_images and self._looks_like_recent_image_request():
                return f"这轮模型没返回文本或图片；我这边只取到了 {len(self.fetched_images)} 张群缓存图。"
            return "模型返回了空结果，请再试一次。"

        self._apply_image_memory_updates(image_memories)
        self._sync_group_context_with_current_user_message()
        if not self.is_objective and assistant_reply:
            self.messages_handler.post_process(assistant_reply)

        sent_images = 0
        image_model_label = str(
            self.model_info.get("image_generation_model") or "agy-image-generator"
        ).strip()
        for img_bytes in generated_images:
            await self._send_image_with_metadata(
                img_bytes,
                model=image_model_label,
                scale="auto",
            )
            sent_images += 1

        if sent_images > 0 and self._is_redundant_image_completion_reply(assistant_reply):
            assistant_reply = ""
        if assistant_reply:
            await self._send_text_response(assistant_reply)
        elif sent_images == 0:
            return False
        return True

    async def get_llm_chat(self) -> str | bool:
        self.messages_handler = MessagesHandler(self.user_id)
        self.messages_handler.pre_process(self.format_message_dict)
        is_big_doubao = self.use_big_doubao or contains_big_doubao_trigger(
            _extract_event_plain_text(self.event, self.format_message_dict)
        )
        self.model_info = (
            model_selector.get_big_doubao_model()
            if is_big_doubao
            else model_selector.get_model("selected_model")
        )
        if not self.model_info:
            return "未找到可用模型配置。"

        logger.info(f"模型选择为：{self.model_info.get('model', DEFAULT_AGY_MODEL)}")
        self.prompt_handler()
        send_message_list = self.messages_handler.get_send_message_list()

        await self._prefetch_recent_images_if_needed()

        use_native_web_search = (
            model_selector.get_web_search() and self._use_native_web_search()
        )
        use_external_image_generation = self._use_external_image_generation()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            try:
                if self._use_agy_acp():
                    return await self.agy_acp_llm_chat(session, send_message_list)
                headers = self._merge_headers({
                    "Authorization": self.model_info["key"],
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                }, self._provider_extra_headers())
                if self._use_responses_api():
                    return await self.responses_llm_chat(
                        self.model_info["url"],
                        headers,
                        send_message_list,
                        self.model_info.get("proxy"),
                        native_web_search=use_native_web_search,
                        external_image_generation=use_external_image_generation,
                        local_image_cache=True,
                    )
                return await self.chat_completions_llm_chat(
                    session,
                    self.model_info["url"],
                    headers,
                    send_message_list,
                    self.model_info.get("proxy"),
                    external_image_generation=use_external_image_generation,
                    local_image_cache=True,
                )
            except RuntimeError as exc:
                return str(exc)
            except TimeoutError:
                return "请求超时。"
            except Exception as exc:
                logger.warning(
                    {
                        "event": "llm_request_failed",
                        "model": self.model_info.get("model"),
                        "use_responses_api": self._use_responses_api(),
                        "message_count": len(send_message_list),
                        "current_image_count": len(self.messages_handler.current_images),
                        "fetched_image_count": len(self.fetched_images),
                    }
                )
                logger.error(traceback.format_exc())
                return self._format_upstream_error(exc)
