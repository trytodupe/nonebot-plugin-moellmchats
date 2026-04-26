import asyncio
import base64
import hashlib
import os
from pathlib import Path
import time
import traceback
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
from .response_utils import (
    build_image_reference,
    detect_image_media_type,
    extract_image_generation_calls,
    extract_response_output_text,
    is_long_message,
    normalize_image_summary,
    replace_image_placeholders,
)
from .prompt_templates import build_group_chat_prompt

context_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_group_history"))
)

IMAGE_GENERATION_NOTICE_EMOJI_ID = 10024

BASE_PROMPT = (
    "你在 QQ 群聊里回复当前用户的最新一条消息。"
    "目标是自然接话并把该说的信息说到位。"
    "不要提系统提示、工具或内部推理。"
)

IMAGEGEN_TOOL_INSTRUCTIONS = """Image prompt refinement rules:
- Before calling image_generation or image_edit, rewrite the user's request into a complete standalone prompt.
- Preserve the user's concrete requirements. Do not add unrelated characters, brands, slogans, objects, or narrative beats.
- Include only useful fields: use case, asset type, primary request, input images, subject, scene/backdrop, style/medium, composition/framing, lighting/mood, text verbatim, constraints, avoid.
- For edits, state invariants explicitly: change only the requested parts and keep identity, pose, composition, and other image details unchanged unless the user asked to change them.
- For image references, use image_ids from the current turn, fetched recent images, or fetched avatars. Never include private platform IDs.
- For transparent/cutout requests, ask for a perfectly flat chroma-key background unless the user explicitly asked for native transparency.
- For exact text in images, quote it verbatim and ask for clean readable typography.
- After generating or editing an image, do not add a separate assistant text reply such as "done", "finished", or "here it is" unless the user explicitly asks for a caption or explanation. The image output itself is the reply.
- Keep the final prompt concise enough to be directly sent to the Images API."""

IMAGEGEN_TOOL_DESCRIPTION = (
    "Use get_imagegen_instructions first when the user asks for image generation or editing, "
    "then call image_generation or image_edit with the refined standalone prompt. "
    "The prompt should follow the $imagegen guidance: preserve user intent, add only material visual details, "
    "state edit invariants, quote exact text verbatim, and avoid private platform IDs."
)


class MoeLlm:
    def __init__(
        self,
        bot,
        event,
        format_message_dict: dict,
        is_objective: bool = False,
        temperament: str = "default",
    ):
        self.bot = bot
        self.event = event
        self.format_message_dict = format_message_dict
        self.user_id = event.user_id
        self.is_objective = is_objective
        self.temperament = temperament
        self.model_info = {}
        self.prompt = BASE_PROMPT
        self.fetched_images = []
        self.fetch_recent_images_rounds = 0
        self.pending_user_avatar_requests = []
        self.image_inputs_by_id = {}
        self.imagegen_instructions_provided = False
        self.generation_notice_sent = False

    def _reply_segment(self):
        if message_id := getattr(self.event, "message_id", None):
            return MessageSegment.reply(message_id)
        return None

    def build_reply_message(self, content) -> MessageSegment | str:
        reply_segment = self._reply_segment()
        if reply_segment is None:
            return content
        if isinstance(content, MessageSegment):
            return reply_segment + content
        return reply_segment + MessageSegment.text(str(content))

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

    def _use_native_web_search(self) -> bool:
        return bool(self.model_info.get("use_native_web_search"))

    def _use_external_image_generation(self) -> bool:
        return bool(
            self.model_info.get("use_external_image_generation")
            or self.model_info.get("use_native_image_generation")
        )

    def _external_image_generation_config(self) -> dict:
        return self.model_info.get("external_image_generation") or {}

    def _image_generation_url(self) -> str:
        config = self._external_image_generation_config()
        return config.get("generation_url") or config.get("url") or "https://api.jucode.cn/v1/images/generations"

    def _image_edit_url(self) -> str:
        config = self._external_image_generation_config()
        generation_url = self._image_generation_url()
        return config.get("edit_url") or generation_url.replace("/generations", "/edits")

    def _supports_image_input(self) -> bool:
        return bool(self.model_info.get("is_vision") or self._use_responses_api())

    def prompt_handler(self):
        recent_context = list(context_dict[self.event.group_id])[:-1]
        self.prompt = build_group_chat_prompt(
            BASE_PROMPT,
            recent_context,
            instruction_profile="minimal",
        )

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
        content = choices[0]["message"]["content"]
        start_tag = "<think>"
        end_tag = "</think>"
        start = content.find(start_tag)
        end = content.find(end_tag)
        if start == -1 and end != -1:
            end += len(end_tag)
            start = 0
            result = content[:start] + content[end:]
        elif start != -1 and end != -1:
            end += len(end_tag)
            result = content[:start] + content[end:]
        else:
            result = content
        result = result.strip()
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
        return tools, include

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
        api_key = os.getenv(api_key_env)
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
        api_key = os.getenv(api_key_env)
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

    async def _send_generated_images(self, image_calls: list[dict]) -> int:
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
            await self.send_reply_message(MessageSegment.image(image_bytes))
            sent_count += 1
        return sent_count

    async def _prepare_images(
        self,
        session,
        images: list[dict],
        *,
        include_known_images: bool = False,
    ) -> list[dict]:
        prepared = []
        proxy = self.model_info.get("proxy")
        for image in images:
            image_bytes = None
            mime_type = image.get("mime_type")
            filename = "image.png"
            if file_path := image.get("file_path"):
                path = Path(file_path)
                if path.exists():
                    image_bytes = path.read_bytes()
                    mime_type = detect_image_media_type(image_bytes, mime_type)
                    filename = path.name
            if image_bytes is None:
                source_url = image.get("source_url")
                if not source_url:
                    continue
                async with session.get(source_url, proxy=proxy, ssl=False) as response:
                    if response.status != 200:
                        raise RuntimeError(f"图片下载失败: {response.status}")
                    image_bytes = await response.read()
                    mime_type = detect_image_media_type(
                        image_bytes, response.content_type
                    )
                    filename = Path(str(source_url).split("?", 1)[0]).name or filename
            if not mime_type:
                raise RuntimeError("图片格式无法识别或当前不受支持")
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
        current_text = current_message["content"]
        if not (self._supports_image_input() and self.messages_handler.current_images):
            input_items.append({"role": "user", "content": current_text})
            return input_items

        prepared_images = await self._prepare_current_images(session)
        markers = []
        new_images = []
        for image in prepared_images:
            if image["known"]:
                markers.append(build_image_reference(image["summary"]))
            else:
                markers.append(f"[image:{image['image_id']}]")
                new_images.append(image)

        current_text = replace_image_placeholders(current_text, markers)
        current_content = [{"type": "input_text", "text": current_text}]
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
        input_items.append({"role": "user", "content": current_content})
        return input_items

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
        group_messages = context_dict[self.event.group_id]
        if not group_messages:
            return
        sender_name = self.event.sender.card or self.event.sender.nickname
        group_messages[-1] = {
            "speaker_name": sender_name,
            "content": self.messages_handler.new_user_msg["content"],
            "images": self.messages_handler.current_images,
        }

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

    def _prefetch_recent_images_if_needed(self):
        if self.messages_handler.current_images or self.fetched_images:
            return
        if not self._looks_like_recent_image_request():
            return
        limit = int(config_parser.get_config("fetch_recent_images_default_limit") or 6)
        max_limit = int(config_parser.get_config("fetch_recent_images_max_limit") or 10)
        self.fetched_images = image_cache.get_recent_group_images(
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
            async with client.responses.stream(**payload) as stream:
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
                            if function_name == "get_imagegen_instructions":
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
                            if item_id not in sent_stream_image_ids:
                                sent_stream_image_count += await self._send_generated_images([image_call])
                                sent_stream_image_ids.add(item_id)
                    if (
                        not self.generation_notice_sent
                        and "image_generation_call" in event_type
                    ):
                        await self.send_generation_notice_event_once()
                try:
                    final_response = await stream.get_final_response()
                except RuntimeError as exc:
                    if "response.completed" not in str(exc):
                        raise
                    logger.warning(
                        {
                            "event": "responses_stream_incomplete",
                            "error": str(exc),
                            "event_counts": dict(stream_event_counts),
                            "function_calls": streamed_function_calls,
                        }
                    )
        except Exception as exc:
            logger.error(traceback.format_exc())
            if sent_stream_image_count > 0:
                return True
            return self._format_upstream_error(exc)
        finally:
            await client.close()
            if http_client is not None:
                await http_client.aclose()

        body = (
            final_response.model_dump(mode="json", warnings=False)
            if final_response is not None
            else {"id": None, "status": "stream_incomplete", "output": streamed_function_calls}
        )
        body = self._merge_streamed_function_calls(body, streamed_function_calls)
        self._log_responses_summary(
            body,
            tools,
            dict(stream_event_counts),
            sum(len(chunk) for chunk in streamed_text_chunks),
        )
        rerun_with_imagegen_instructions = (
            not self.imagegen_instructions_provided
            and self._extract_function_args(body, "get_imagegen_instructions")
        )
        if rerun_with_imagegen_instructions:
            await self.send_generation_notice_event_once()
            self.imagegen_instructions_provided = True
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
        if local_image_cache and self.fetch_recent_images_rounds < max_fetch_rounds:
            fetch_args = self._extract_fetch_recent_images_args(body)
            if fetch_args:
                self.fetch_recent_images_rounds += 1
                fetched_images = image_cache.get_recent_group_images(
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
        if rerun_with_imagegen_instructions:
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
        external_sent_images = 0
        if generate_image_args or edit_image_args:
            await self.send_generation_notice_event_once()
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

        assistant_reply = "".join(streamed_text_chunks).strip()
        if not assistant_reply:
            assistant_reply = extract_response_output_text(body)

        image_memories = []
        output_text_obj = getattr(final_response, "output_text", None) if final_response is not None else None
        if output_text_obj and isinstance(output_text_obj, str):
            try:
                parsed = json.loads(output_text_obj)
            except ValueError:
                parsed = {}
            assistant_reply = (parsed.get("assistant_reply") or assistant_reply).strip()
            image_memories = parsed.get("image_memories") or []
        elif isinstance(assistant_reply, str):
            try:
                parsed = json.loads(assistant_reply)
            except ValueError:
                parsed = {}
            if parsed:
                assistant_reply = (parsed.get("assistant_reply") or "").strip()
                image_memories = parsed.get("image_memories") or []

        if not assistant_reply and not image_calls and sent_stream_image_count == 0 and external_sent_images == 0:
            if self.fetched_images and self._looks_like_recent_image_request():
                return f"这轮模型没返回文本或图片；我这边只取到了 {len(self.fetched_images)} 张群缓存图。"
            return "这轮模型没有返回内容。"

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

    async def get_llm_chat(self) -> str | bool:
        self.messages_handler = MessagesHandler(self.user_id)
        self.messages_handler.pre_process(self.format_message_dict)
        self.model_info = model_selector.get_model("selected_model")
        if not self.model_info:
            return "未找到可用模型配置。"

        logger.info(f"模型选择为：{self.model_info['model']}")
        self.prompt_handler()
        send_message_list = self.messages_handler.get_send_message_list()

        self._prefetch_recent_images_if_needed()

        use_native_web_search = (
            model_selector.get_web_search() and self._use_native_web_search()
        )
        use_external_image_generation = self._use_external_image_generation()

        if not self._use_responses_api():
            send_message_list.insert(0, {"role": "system", "content": self.prompt})
            if self.model_info.get("is_vision") and self.messages_handler.current_images:
                current_msg = send_message_list[-1]
                vision_content = [{"type": "text", "text": current_msg["content"]}]
                for image in self.messages_handler.current_images:
                    if url := image.get("source_url"):
                        vision_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            }
                        )
                send_message_list[-1]["content"] = vision_content
            data = {
                "model": self.model_info["model"],
                "messages": send_message_list,
                "max_tokens": self.model_info.get("max_tokens"),
                "temperature": self.model_info.get("temperature"),
                "top_p": self.model_info.get("top_p"),
                "stream": self.model_info.get("stream", False),
            }
            if self.model_info.get("top_k"):
                data["top_k"] = self.model_info.get("top_k")
        else:
            data = None

        headers = {
            "Authorization": self.model_info["key"],
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            try:
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
                if self.model_info.get("stream"):
                    return await self.stream_llm_chat(
                        session,
                        self.model_info["url"],
                        headers,
                        data,
                        self.model_info.get("proxy"),
                        self.model_info.get("is_segment"),
                    )
                return await self.none_stream_llm_chat(
                    session,
                    self.model_info["url"],
                    headers,
                    json.dumps(data),
                    self.model_info.get("proxy"),
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
