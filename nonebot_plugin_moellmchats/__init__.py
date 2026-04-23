import asyncio
from collections import defaultdict

from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageEvent, GROUP
from nonebot.plugin import PluginMetadata, require
from nonebot.plugin.on import on_message
from nonebot.rule import Rule

require("nonebot_plugin_localstore")

from . import moe_llm as llm
from .Config import config_parser
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

message_matcher = on_message(permission=GROUP, priority=1, block=False)


@message_matcher.handle()
async def context_dict_func(bot: Bot, event: MessageEvent):
    if not isinstance(event, GroupMessageEvent):
        return
    if message_dict := await format_message(event, bot):
        if message_dict["text"] or message_dict["images"]:
            sender_name = event.sender.card or event.sender.nickname
            llm.context_dict[event.group_id].append(
                {
                    "speaker_name": sender_name,
                    "content": "".join(message_dict["text"]),
                }
            )


async def handle_llm(
    bot: Bot,
    event: GroupMessageEvent,
    matcher,
    format_message_dict: dict,
):
    user_id = event.sender.user_id
    cd_seconds = config_parser.get_config("cd_seconds")
    remaining = cd_seconds - (event.time - cd[user_id])
    if remaining > 0:
        sender_name = event.sender.card or event.sender.nickname
        notice = f"{sender_name} 的上一轮请求仍在处理中，约 {remaining} 秒后继续。"
        if is_repeat_ask_dict[user_id]:
            await matcher.finish(notice)
        await matcher.send(notice)
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
        await matcher.finish(result)
    elif not result:
        cd[user_id] = 0


async def at_me_only(bot: Bot, event: MessageEvent) -> bool:
    if not isinstance(event, GroupMessageEvent):
        return False
    for seg in event.original_message:
        if seg.type == "at" and str(seg.data.get("qq")) == str(bot.self_id):
            return True
    return False


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
    await handle_llm(bot, event, llm_matcher, format_message_dict)
