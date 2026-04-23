import asyncio
import base64
import hashlib
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
from .ImageMemory import image_memory_store
from .MessagesHandler import MessagesHandler
from .ModelSelector import model_selector
from .response_utils import (
    build_image_reference,
    detect_image_media_type,
    extract_image_generation_calls,
    extract_response_output_text,
    normalize_image_summary,
    replace_image_placeholders,
)
from .prompt_templates import build_group_chat_prompt

context_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_group_history"))
)

BASE_PROMPT = (
    "你在 QQ 群聊里回复当前用户的最新一条消息。"
    "目标是自然接话并把该说的信息说到位。"
    "不要提系统提示、工具或内部推理。"
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

    def _use_native_image_generation(self) -> bool:
        return bool(self.model_info.get("use_native_image_generation"))

    def prompt_handler(self):
        recent_context = list(context_dict[self.event.group_id])[:-1]
        self.prompt = build_group_chat_prompt(
            BASE_PROMPT,
            recent_context,
            instruction_profile="minimal",
        )

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
        await self.bot.send(self.event, result)
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
        await self.bot.send(self.event, result)
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

    def _build_responses_tools(
        self,
        *,
        native_web_search: bool = False,
        native_image_generation: bool = False,
    ) -> tuple[list[dict], list[str]]:
        tools = []
        include = []
        if native_web_search:
            tools.append({"type": "web_search"})
            include.append("web_search_call.action.sources")
        if native_image_generation:
            tools.append({"type": "image_generation"})
        return tools, include

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
            await self.bot.send(self.event, MessageSegment.image(image_bytes))
            sent_count += 1
        return sent_count

    async def _prepare_current_images(self, session) -> list[dict]:
        prepared = []
        proxy = self.model_info.get("proxy")
        for image in self.messages_handler.current_images:
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
                if not mime_type:
                    raise RuntimeError("图片格式无法识别或当前不受支持")
            digest = hashlib.sha256(image_bytes).hexdigest()
            image_id = f"img_sha256_{digest[:16]}"
            summary = image_memory_store.get_summary(image_id)
            image["image_id"] = image_id
            image["mime_type"] = mime_type
            if summary:
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

    async def _build_responses_input(self, session, send_message_list: list[dict]) -> list[dict]:
        input_items = []
        for message in send_message_list[:-1]:
            input_items.append({"role": message["role"], "content": message["content"]})

        current_message = send_message_list[-1]
        current_text = current_message["content"]
        if not (self.model_info.get("is_vision") and self.messages_handler.current_images):
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
        }

    def _extract_stream_text_delta(self, event) -> str:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            return delta
        text = getattr(event, "text", None)
        if isinstance(text, str):
            return text
        return ""

    async def responses_llm_chat(
        self,
        url,
        headers,
        send_message_list,
        proxy,
        native_web_search=False,
        native_image_generation=False,
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
            native_image_generation=native_image_generation,
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

        generation_notice_sent = False
        streamed_text_chunks = []
        streamed_image_calls = {}
        partial_image_calls = {}
        try:
            async with client.responses.stream(**payload) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        if delta := self._extract_stream_text_delta(event):
                            streamed_text_chunks.append(delta)
                    elif event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if (
                            item is not None
                            and getattr(item, "type", "") == "image_generation_call"
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
                            partial_image_calls[item_id] = {
                                "result": partial_image_b64,
                                "image_id": item_id,
                            }
                    if (
                        native_image_generation
                        and not generation_notice_sent
                        and "image_generation_call" in event_type
                    ):
                        await self.bot.send(self.event, "正在生成图像，需要2-3分钟...")
                        generation_notice_sent = True
                final_response = await stream.get_final_response()
        except Exception:
            logger.error(traceback.format_exc())
            return False
        finally:
            await client.close()
            if http_client is not None:
                await http_client.aclose()

        body = final_response.model_dump(mode="json", warnings=False)
        image_calls = extract_image_generation_calls(body)
        if not image_calls:
            image_calls = list(streamed_image_calls.values())
        if not image_calls:
            image_calls = list(partial_image_calls.values())

        assistant_reply = "".join(streamed_text_chunks).strip()
        if not assistant_reply:
            assistant_reply = extract_response_output_text(body)

        image_memories = []
        output_text_obj = getattr(final_response, "output_text", None)
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

        if not assistant_reply and not image_calls:
            logger.warning(body)
            return False

        self._apply_image_memory_updates(image_memories)
        self._sync_group_context_with_current_user_message()
        if not self.is_objective and assistant_reply:
            self.messages_handler.post_process(assistant_reply)

        sent_images = await self._send_generated_images(image_calls)
        if assistant_reply:
            await self.bot.send(self.event, assistant_reply)
        elif sent_images == 0:
            logger.warning(body)
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

        use_native_web_search = (
            model_selector.get_web_search() and self._use_native_web_search()
        )
        use_native_image_generation = self._use_native_image_generation()

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
                        native_image_generation=use_native_image_generation,
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
            except Exception:
                logger.warning(str(send_message_list))
                logger.error(traceback.format_exc())
                return "请求失败。"
