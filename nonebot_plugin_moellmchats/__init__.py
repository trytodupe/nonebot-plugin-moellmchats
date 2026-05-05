import asyncio
from collections import defaultdict

import aiohttp
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageEvent, PrivateMessageEvent, GROUP
from nonebot.plugin import PluginMetadata, require
from nonebot.plugin.on import on_message
from nonebot.rule import Rule

require("nonebot_plugin_localstore")

from . import moe_llm as llm
from .Config import config_parser
from .ImageCache import image_cache
from .access_control import evaluate_private_access, is_access_request_plugin_available, is_private_acl_exempt_user
from .utils import format_message


__plugin_meta__ = PluginMetadata(
    name="MoEllm聊天",
    description="Minimal QQ group chat bridge with context stitching and model tools.",
    usage='艾特 bot 进行对话。',
    type="application",
    homepage="https://github.com/Elflare/nonebot-plugin-moellmchats",
    supported_adapters={"~onebot.v11"},
)

cd = defaultdict(int)
is_repeat_ask_dict = defaultdict(bool)


def _session_key(event: MessageEvent) -> str:
    if isinstance(event, GroupMessageEvent):
        return f"group:{event.group_id}"
    return f"private:{event.user_id}"

message_matcher = on_message(permission=GROUP, priority=1, block=False)


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
    matcher,
    format_message_dict: dict,
):
    user_id = event.sender.user_id
    cd_seconds = config_parser.get_config("cd_seconds")
    remaining = cd_seconds - (event.time - cd[user_id])
    if remaining > 0:
        sender_name = event.sender.card or event.sender.nickname
        notice = f"{sender_name} 的上一轮请求仍在处理中，约 {remaining} 秒后继续。"
        llm_sender = llm.MoeLlm(
            bot,
            event,
            format_message_dict,
        )
        if is_repeat_ask_dict[user_id]:
            await llm_sender.send_reply_message(notice)
            await matcher.finish()
            return
        await llm_sender.send_reply_message(notice)
        is_repeat_ask_dict[user_id] = True
        await asyncio.sleep(max(0, remaining))

    cd[user_id] = event.time
    llm_chat = llm.MoeLlm(
        bot,
        event,
        format_message_dict,
    )
    result = await llm_chat.get_llm_chat()
    is_repeat_ask_dict[user_id] = False
    if isinstance(result, str):
        cd[user_id] = 0
        await llm_chat.send_reply_message(result)
        await matcher.finish()
        return
    elif not result:
        cd[user_id] = 0


async def at_me_only(bot: Bot, event: MessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    for seg in event.original_message:
        if seg.type == "at" and str(seg.data.get("qq")) == str(bot.self_id):
            return True
    return False


def private_message_only(event: MessageEvent) -> bool:
    return isinstance(event, PrivateMessageEvent) and not is_private_acl_exempt_user(event.user_id)


llm_matcher = on_message(
    rule=Rule(at_me_only),
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
    await handle_llm(bot, event, llm_matcher, format_message_dict)


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
    await handle_llm(bot, event, private_llm_matcher, format_message_dict)
