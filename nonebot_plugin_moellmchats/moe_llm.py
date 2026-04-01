import asyncio
import base64
import hashlib
import random
import traceback
from asyncio import TimeoutError
from collections import defaultdict, deque

import aiohttp
import ujson as json
from nonebot.log import logger

from .Categorize import Categorize
from .Config import config_parser
from .ImageMemory import image_memory_store
from .MessagesHandler import MessagesHandler
from .ModelSelector import model_selector
from .Search import Search
from .TemperamentManager import temperament_manager
from .response_utils import (
    build_image_reference,
    extract_response_output_text,
    normalize_image_summary,
    parse_response_json_text,
    replace_image_placeholders,
)
from .utils import get_emotion, get_emotions_names, parse_emotion

context_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_group_history"))
)


class MoeLlm:
    def __init__(
        self,
        bot,
        event,
        format_message_dict: dict,
        is_objective: bool = False,
        temperament="默认",
    ):
        self.bot = bot
        self.event = event
        self.format_message_dict = format_message_dict
        self.user_id = event.user_id
        self.is_objective = is_objective
        self.temperament = temperament
        self.model_info = {}
        self.emotion_flag = False
        self.prompt = f"{temperament_manager.get_temperament_prompt(temperament)}。我的id是{event.sender.card or event.sender.nickname}"

    async def send_emotion_message(self, content: str) -> str:
        if self.emotion_flag:
            content, emotion_names_list = parse_emotion(content)
            if content:
                await self.bot.send(self.event, content)
            for emotion_name in emotion_names_list:
                if emotion := get_emotion(emotion_name):
                    await self.bot.send(self.event, emotion)
        else:
            await self.bot.send(self.event, content)
        return content

    async def _check_400_error(self, response) -> str | None:
        if response.status == 400:
            error_content = await response.text()
            logger.warning(f"API请求400错误: {error_content}")

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
                return "图片或内容可能包含敏感信息，被AI审核拦截了喵 >_<"
            return "API请求被拒绝 (400)，请检查后台日志。"
        return None

    def _use_responses_api(self) -> bool:
        api_style = self.model_info.get("api_style")
        if api_style:
            return api_style == "responses"
        return self.model_info.get("url", "").rstrip("/").endswith("/responses")

    def _use_native_web_search(self) -> bool:
        return bool(self.model_info.get("use_native_web_search"))

    async def stream_llm_chat(
        self, session, url, headers, data, proxy, is_segment=False
    ) -> bool | str:
        buffer = []
        assistant_result = []
        punctuation_buffer = ""
        is_second_send = False
        async with session.post(url, headers=headers, json=data, proxy=proxy) as response:
            if error_msg := await self._check_400_error(response):
                return error_msg
            if response.status == 200:
                max_segments = self.model_info.get("max_segments", 5)
                current_segment = 0
                jump_out = False
                current_content = ""
                async for line in response.content:
                    if (
                        not line
                        or line.startswith(b"data: [DONE]")
                        or line.startswith(b"[DONE]")
                        or jump_out
                    ):
                        break
                    if line.startswith(b"data:"):
                        decoded = line[5:].decode("utf-8")
                    else:
                        decoded = line.decode("utf-8")
                    if not decoded.strip() or decoded.startswith(":"):
                        continue
                    json_data = json.loads(decoded)
                    content = ""
                    choices = json_data.get("choices", [{}])
                    if not choices:
                        continue
                    if message := choices[0].get("message", {}):
                        content = message.get("content", "")
                    elif message := choices[0].get("delta", {}):
                        content = message.get("content", "")
                    if not content:
                        continue
                    if is_segment and self.temperament != "ai助手":
                        for char in content:
                            if char in ["。", "？", "！", "—", "\n"]:
                                punctuation_buffer += char
                            else:
                                if punctuation_buffer:
                                    current_content = (
                                        "".join(buffer) + punctuation_buffer
                                    ).strip()
                                    if current_content:
                                        if current_segment >= max_segments:
                                            buffer = ["太长了，不发了"]
                                            jump_out = True
                                            break
                                        if is_second_send:
                                            await asyncio.sleep(
                                                2 + len(current_content) / 3
                                            )
                                        else:
                                            is_second_send = True
                                        current_content = await self.send_emotion_message(
                                            current_content
                                        )
                                        current_segment += 1
                                        assistant_result.append(current_content)
                                    buffer = []
                                    punctuation_buffer = ""
                                buffer.append(char)
                    else:
                        buffer.append(content)
                result = "".join(buffer) if jump_out else "".join(buffer) + punctuation_buffer
                if is_second_send and current_content:
                    await asyncio.sleep(2 + len(current_content) / 3)
                elif result.strip():
                    is_second_send = True
                if result := result.strip():
                    result = await self.send_emotion_message(result)
                    if not self.is_objective:
                        self.messages_handler.post_process(
                            "".join(assistant_result) + result
                        )
                    return True
                if is_second_send:
                    if not self.is_objective:
                        self.messages_handler.post_process("".join(assistant_result))
                    return True
            else:
                logger.warning(f"Warning: {response}")
        return False

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
        if choices := response.get("choices"):
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
        else:
            logger.warning(response)
            return False
        if not self.is_objective:
            self.messages_handler.post_process(result.strip())
        await self.bot.send(self.event, result.strip())
        return True

    def prompt_handler(self):
        if self.temperament != "ai助手":
            if (
                config_parser.get_config("emotions_enabled")
                and self.model_info.get("is_segment")
                and self.model_info.get("stream")
                and random.random() < config_parser.get_config("emotion_rate")
            ):
                self.emotion_flag = True
                emotion_prompt = (
                    "。回复时根据回答内容，发送表情包，每次回复最多发一个表情包，格式为中括号+表情包名字，如：[表情包名字]。"
                    f"可选表情有{get_emotions_names()}"
                )
            else:
                emotion_prompt = ""
            self.prompt += (
                f"。现在你在一个qq群中,你只需回复我{emotion_prompt}。"
                "群里近期聊天内容，冒号前面是id，后面是内容：\n"
            )
            context_dict_ = list(context_dict[self.event.group_id])[:-1]
            self.prompt += "\n".join(context_dict_)

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
                mime_type = response.content_type or "application/octet-stream"
            digest = hashlib.sha256(image_bytes).hexdigest()
            image_id = f"img_sha256_{digest[:16]}"
            summary = image_memory_store.get_summary(image_id)
            image["image_id"] = image_id
            image["mime_type"] = mime_type
            if summary:
                image["summary"] = summary
                prepared.append({
                    "image_id": image_id,
                    "summary": summary,
                    "known": True,
                })
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
        group_messages[-1] = (
            f"{sender_name}:{self.messages_handler.new_user_msg['content']}"
        )

    async def responses_llm_chat(
        self,
        session,
        url,
        headers,
        send_message_list,
        proxy,
        native_web_search=False,
    ) -> bool | str:
        response_input = await self._build_responses_input(session, send_message_list)
        payload = {
            "model": self.model_info["model"],
            "store": False,
            "instructions": self.prompt,
            "input": response_input,
            "text": {"format": self._get_response_schema()},
        }
        if max_tokens := self.model_info.get("max_tokens"):
            payload["max_output_tokens"] = max_tokens
        if reasoning_effort := self.model_info.get("reasoning_effort"):
            payload["reasoning"] = {"effort": reasoning_effort}
        if native_web_search:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
            payload["include"] = ["web_search_call.action.sources"]

        async with session.post(
            url,
            headers=headers,
            json=payload,
            proxy=proxy,
            ssl=False,
        ) as response:
            if error_msg := await self._check_400_error(response):
                return error_msg
            body = await response.json()
            if response.status != 200:
                logger.warning(body)
                return False

        try:
            structured = parse_response_json_text(body)
        except Exception:
            logger.warning(traceback.format_exc())
            structured = {}

        assistant_reply = (structured.get("assistant_reply") or "").strip()
        if not assistant_reply:
            assistant_reply = extract_response_output_text(body)
        if not assistant_reply:
            logger.warning(body)
            return False

        image_memories = structured.get("image_memories") or []
        self._apply_image_memory_updates(image_memories)
        self._sync_group_context_with_current_user_message()
        if not self.is_objective:
            self.messages_handler.post_process(assistant_reply)
        await self.bot.send(self.event, assistant_reply)
        return True

    async def get_llm_chat(self) -> str | bool:
        self.messages_handler = MessagesHandler(self.user_id)
        plain = self.messages_handler.pre_process(self.format_message_dict)
        internet_required = False
        key_word = ""
        if model_selector.get_moe() or model_selector.get_web_search():
            category = Categorize(plain)
            category_result = await category.get_category()
            if isinstance(category_result, str):
                return category_result
            if isinstance(category_result, tuple):
                difficulty, internet_required, key_word, vision_required = category_result
                logger.info(
                    f"难度：{difficulty}, 联网：{internet_required}, 关键词：{key_word}, 视觉：{vision_required}"
                )
                if model_selector.get_moe():
                    if vision_required and self.messages_handler.current_images:
                        vision_model_key = model_selector.model_config.get("vision_model")
                        if vision_model_key:
                            self.model_info = model_selector.get_model("vision_model")
                            logger.info(
                                f"触发视觉任务，切换至视觉模型: {self.model_info['model']}"
                            )
                        else:
                            logger.info(
                                "触发视觉任务，但配置文件 model_config.json 缺少 vision_model 字段，退回普通模型"
                            )
                    else:
                        self.model_info = model_selector.get_moe_current_model(difficulty)
        if not self.model_info:
            self.model_info = model_selector.get_model("selected_model")
        logger.info(f"模型选择为：{self.model_info['model']}")
        self.prompt_handler()
        send_message_list = self.messages_handler.get_send_message_list()

        use_native_web_search = (
            internet_required and model_selector.get_web_search() and self._use_native_web_search()
        )
        if internet_required and model_selector.get_web_search() and not use_native_web_search:
            search = Search(key_word)
            await self.bot.send(self.event, "検索中...検索中...=￣ω￣=")
            if search_result := await search.get_search():
                self.messages_handler.search_message_handler(search_result)
                send_message_list = self.messages_handler.get_send_message_list()
            elif isinstance(search_result, bool):
                await self.bot.send(self.event, "没搜到，可能没有相关内容")
            else:
                await self.bot.send(self.event, "搜索失败，请检查日志输出")
        elif use_native_web_search:
            await self.bot.send(self.event, "検索中...検索中...=￣ω￣=")

        if not self._use_responses_api():
            send_message_list.insert(0, {"role": "system", "content": self.prompt})
            if self.model_info.get("is_vision") and self.messages_handler.current_images:
                logger.info(
                    f"检测到多模态模型 {self.model_info['model']} 且存在图片，正在构建多模态请求..."
                )
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
            max_retry_times = config_parser.get_config("max_retry_times") or 3
            result = ""
            for retry_times in range(max_retry_times):
                if retry_times > 0:
                    await self.bot.send(
                        self.event,
                        f"api又卡了呐！第 {retry_times+1} 次尝试，请勿多次发送~",
                    )
                    await asyncio.sleep(2 ** (retry_times + 1))
                try:
                    if self._use_responses_api():
                        result = await self.responses_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            send_message_list,
                            self.model_info.get("proxy"),
                            native_web_search=use_native_web_search,
                        )
                    elif self.model_info.get("stream"):
                        result = await self.stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            data,
                            self.model_info.get("proxy"),
                            self.model_info.get("is_segment"),
                        )
                    else:
                        result = await self.none_stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            json.dumps(data),
                            self.model_info.get("proxy"),
                        )
                    if result:
                        return result
                except RuntimeError as exc:
                    return str(exc)
                except TimeoutError:
                    return "网络超时呐，多半是api反应太慢（"
                except Exception:
                    logger.warning(str(send_message_list))
                    logger.error(traceback.format_exc())
                    continue
            return "api寄！"
