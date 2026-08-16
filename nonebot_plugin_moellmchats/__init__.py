from typing import Optional

import aiohttp
from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11 import GROUP, Bot, GroupMessageEvent, Message, MessageEvent, PrivateMessageEvent
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, require
from nonebot.plugin.on import on_message
from nonebot.rule import Rule, to_me

require("nonebot_plugin_localstore")

from . import moe_llm as llm
from .access_control import evaluate_private_access, is_private_acl_exempt_user
from .Config import config_parser
from .group_access import configure_group_gate, event_access_allowed, group_has_superuser
from .ImageCache import image_cache
from .request_registry import PendingRequest, RequestSnapshot, request_registry
from .trigger_rules import contains_doubao_help_trigger, should_trigger_group_chat
from .utils import format_message

__plugin_meta__ = PluginMetadata(
    name="MoEllm聊天",
    description="Minimal QQ group chat bridge with context stitching and model tools.",
    usage="艾特 bot 进行对话。",
    type="application",
    homepage="https://github.com/Elflare/nonebot-plugin-moellmchats",
    supported_adapters={"~onebot.v11"},
)


@get_driver().on_startup
def _configure_group_gate():
    configure_group_gate(config_parser.get_config("group_gate_mode") or "auto")


def _session_key(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group:{event.group_id}"
    return f"private:{event.user_id}"


def _user_name(event: MessageEvent) -> str:
    return event.sender.card or event.sender.nickname or str(event.user_id)


def _prompt_preview(format_message_dict: dict, limit: int = 60) -> str:
    text = "".join(format_message_dict.get("text") or []).strip()
    text = " ".join(text.split())
    image_count = len(format_message_dict.get("images") or [])
    if not text:
        text = "[图片]" if image_count else "[空消息]"
    elif image_count:
        text = f"{text} [图片 x{image_count}]"
    if len(text) > limit:
        return f"{text[: limit - 1]}…"
    return text


def _format_elapsed(seconds: int) -> str:
    minutes, seconds = divmod(max(0, seconds), 60)
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def _message_id_from_send_result(result) -> Optional[int]:
    if isinstance(result, dict):
        message_id = result.get("message_id")
    else:
        message_id = getattr(result, "message_id", None)
    try:
        return int(message_id) if message_id is not None else None
    except (TypeError, ValueError):
        return None


def _reply_message_id(event: MessageEvent) -> Optional[int]:
    reply = getattr(event, "reply", None)
    message_id = getattr(reply, "message_id", None) if reply is not None else None
    try:
        return int(message_id) if message_id is not None else None
    except (TypeError, ValueError):
        return None


def _confirmation_text(active: RequestSnapshot, new_preview: str, current_scope: str) -> str:
    elapsed = _format_elapsed(active.elapsed_seconds(request_registry.now()))
    active_preview = active.prompt_preview if active.scope == current_scope else "其他会话中的请求"
    return (
        f"你已有请求 #{active.request_id} 正在运行（已用时 {elapsed}）：\n"
        f"“{active_preview}”\n\n"
        "是否同时开始新请求：\n"
        f"“{new_preview}”\n\n"
        "回复本消息“确认”以继续。"
    )


def _pending_exists_text(active: RequestSnapshot) -> str:
    elapsed = _format_elapsed(active.elapsed_seconds(request_registry.now()))
    return (
        f"你已有请求 #{active.request_id} 正在运行（已用时 {elapsed}），"
        "并且还有一个新请求等待确认。\n"
        "请先回复上一条确认消息，或等待 2 分钟后重试。"
    )


message_matcher = on_message(
    rule=Rule(group_has_superuser),
    permission=GROUP,
    priority=1,
    block=False,
)


async def cache_message_images(event: GroupMessageEvent, message_dict: dict):
    images = message_dict.get("images") or []
    if not images:
        return []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        return await image_cache.cache_images(
            session,
            group_id=event.group_id,
            user_id=event.user_id,
            images=images,
        )


@message_matcher.handle()
async def context_dict_func(bot: Bot, event: MessageEvent):
    if not isinstance(event, GroupMessageEvent):
        return
    if message_dict := await format_message(event, bot):
        if message_dict["text"] or message_dict["images"]:
            cached_images = await cache_message_images(event, message_dict)
            if cached_images:
                message_dict["images"] = cached_images
            sender_name = event.sender.card or event.sender.nickname
            llm.context_dict[_session_key(event)].append(
                {
                    "speaker_name": sender_name,
                    "content": "".join(message_dict["text"]),
                    "images": message_dict["images"],
                }
            )


async def handle_llm(
    bot: Bot,
    event: MessageEvent,
    format_message_dict: dict,
    *,
    allow_concurrent: bool = False,
):
    user_id = str(event.user_id)
    scope = _session_key(event)
    preview = _prompt_preview(format_message_dict)
    begin = request_registry.begin(
        user_id=user_id,
        user_name=_user_name(event),
        scope=scope,
        prompt_preview=preview,
        allow_concurrent=allow_concurrent,
    )
    if not begin.started:
        llm_sender = llm.MoeLlm(bot, event, format_message_dict)
        if begin.pending_exists:
            await llm_sender.send_reply_message(_pending_exists_text(begin.active_request))
            return

        request_registry.add_pending(
            user_id=user_id,
            scope=scope,
            prompt_preview=preview,
            payload={"bot": bot, "event": event, "format_message_dict": format_message_dict},
        )
        try:
            send_result = await bot.send(
                event,
                llm_sender.build_reply_message(_confirmation_text(begin.active_request, preview, scope)),
            )
        except Exception:
            request_registry.remove_pending(user_id)
            raise
        confirmation_message_id = _message_id_from_send_result(send_result)
        if confirmation_message_id is None:
            request_registry.remove_pending(user_id)
            await llm_sender.send_reply_message("无法创建确认请求，请稍后重试。")
            return
        request_registry.bind_confirmation(user_id, confirmation_message_id)
        return

    try:
        llm_chat = llm.MoeLlm(
            bot,
            event,
            format_message_dict,
        )
        result = await llm_chat.get_llm_chat()
        if isinstance(result, str):
            await llm_chat.send_reply_message(result)
    finally:
        request_registry.finish(begin.request.request_id)


llm_status_matcher = on_command(
    "llm",
    rule=to_me(),
    priority=5,
    block=True,
)


@llm_status_matcher.handle()
async def handle_llm_status(event: MessageEvent, args: Message = CommandArg()):
    if args.extract_plain_text().strip().lower() != "status":
        await llm_status_matcher.finish("用法：ttd llm status")
        return

    requests = request_registry.snapshot(_session_key(event))
    if not requests:
        await llm_status_matcher.finish("当前没有正在运行的请求。")
        return

    now = request_registry.now()
    lines = ["当前正在运行的请求："]
    for request in requests:
        elapsed = _format_elapsed(request.elapsed_seconds(now))
        lines.append(f"#{request.request_id} · {request.user_name} · {elapsed}\n“{request.prompt_preview}”")
    await llm_status_matcher.finish("\n\n".join(lines))


async def valid_confirmation(event: MessageEvent) -> bool:
    if event.get_message().extract_plain_text().strip() != "确认":
        return False
    reply_message_id = _reply_message_id(event)
    if reply_message_id is None:
        return False
    return request_registry.has_valid_confirmation(
        user_id=str(event.user_id),
        scope=_session_key(event),
        reply_message_id=reply_message_id,
    )


confirmation_matcher = on_message(
    rule=Rule(valid_confirmation),
    priority=4,
    block=True,
)


@confirmation_matcher.handle()
async def handle_confirmation(event: MessageEvent):
    reply_message_id = _reply_message_id(event)
    if reply_message_id is None:
        return
    pending = request_registry.take_confirmed(
        user_id=str(event.user_id),
        scope=_session_key(event),
        reply_message_id=reply_message_id,
    )
    if pending is None:
        return
    await _start_pending_request(pending)


async def _start_pending_request(pending: PendingRequest) -> None:
    payload = pending.payload
    await handle_llm(
        payload["bot"],
        payload["event"],
        payload["format_message_dict"],
        allow_concurrent=True,
    )


async def at_me_only(bot: Bot, event: MessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    for seg in event.original_message:
        if seg.type == "at" and str(seg.data.get("qq")) == str(bot.self_id):
            return True
    return False


def _contains_doubao_help_trigger(message) -> bool:
    return contains_doubao_help_trigger(message.extract_plain_text())


async def group_chat_trigger(bot: Bot, event: MessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    should_trigger = should_trigger_group_chat(
        await at_me_only(bot, event),
        event.get_message().extract_plain_text(),
    )
    return should_trigger and await group_has_superuser(bot, event)


async def private_message_only(bot: Bot, event: MessageEvent) -> bool:
    return (
        isinstance(event, PrivateMessageEvent)
        and await event_access_allowed(bot, event)
        and not is_private_acl_exempt_user(event.user_id)
    )


llm_matcher = on_message(
    rule=Rule(group_chat_trigger),
    permission=GROUP,
    priority=99,
    block=True,
)


@llm_matcher.handle()
async def _(bot: Bot, event: MessageEvent):
    if not isinstance(event, GroupMessageEvent):
        return
    format_message_dict = await format_message(event, bot)
    if not format_message_dict["text"] and not format_message_dict["images"]:
        return
    cached_images = await cache_message_images(event, format_message_dict)
    if cached_images:
        format_message_dict["images"] = cached_images
    await handle_llm(bot, event, format_message_dict)


private_llm_matcher = on_message(
    rule=Rule(private_message_only),
    priority=99,
    block=True,
)


@private_llm_matcher.handle()
async def handle_private_llm(bot: Bot, event: MessageEvent):
    if not isinstance(event, PrivateMessageEvent):
        return
    format_message_dict = await format_message(event, bot)
    plain_text = "".join(format_message_dict.get("text") or []).strip()
    decision = await evaluate_private_access(bot, event, plain_text)
    if decision.reply_text:
        llm_sender = llm.MoeLlm(bot, event, format_message_dict)
        await llm_sender.send_reply_message(decision.reply_text)
    if not decision.allowed:
        await private_llm_matcher.finish()
        return
    if not format_message_dict["text"] and not format_message_dict["images"]:
        return
    await handle_llm(bot, event, format_message_dict)
