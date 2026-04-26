import nonebot
from nonebot.adapters.onebot.v11 import Message
from nonebot.log import logger


# 消息格式转换
async def format_message(event, bot) -> dict:
    text_message = []
    reply_text = ""
    image_urls = []
    user_refs = []
    mentioned_user_index = 0

    sender_name = event.sender.card or event.sender.nickname or str(event.user_id)
    user_refs.append(
        {
            "ref": "current_user",
            "display_name": str(sender_name),
            "relation": "current_speaker",
            "user_id": str(event.user_id),
        }
    )

    # 1. 处理回复消息中的图片 (使用你提供的逻辑)
    if reply := event.reply:
        # 手动遍历回复内容，保留 [图片] 占位符
        reply_segments = []
        for seg in event.reply.message:
            if seg.type == "text":
                reply_segments.append(seg.data.get("text", ""))
            elif seg.type == "image":
                reply_segments.append("[图片]")
            # 可以在这里加 elif seg.type == "face": 处理表情等其他类型
        reply_text = "".join(reply_segments).strip()
        text_message.append(
            f"[回复 {event.reply.sender.card or event.reply.sender.nickname} 的消息 [{reply_text}]]"
        )

        try:
            # 获取原消息详情以提取图片
            quoted_message = await bot.get_msg(message_id=reply.message_id)
            message_list = quoted_message["message"]
            if isinstance(message_list, str):  # gocq是str
                message_image = Message(message_list)
                # 查找是否有图片段
                for seg in message_image:
                    if seg.type == "image":
                        if url := seg.data.get("url"):
                            image_urls.append({"source_url": url})
            else:  # shamrock是list
                for message in message_list:
                    if message.get("type") == "image":
                        if url := message.get("data").get("url"):
                            image_urls.append({"source_url": url})
        except Exception:
            logger.warning("获取回复消息图片失败")

    # 2. 处理当前消息
    for msgseg in event.get_message():
        if msgseg.type == "at":
            qq = msgseg.data.get("qq")
            if qq != nonebot.get_bot().self_id:
                name = await get_member_name(event.group_id, qq, bot)
                mentioned_user_index += 1
                ref = f"mentioned_user_{mentioned_user_index}"
                user_refs.append(
                    {
                        "ref": ref,
                        "display_name": name,
                        "relation": "mentioned_user",
                        "user_id": str(qq),
                    }
                )
                text_message.append(f"{name}({ref})")
        elif msgseg.type == "image":
            text_message.append("[图片]")
            if url := msgseg.data.get("url"):
                image_urls.append({"source_url": url})
        elif msgseg.type == "face":
            pass
        elif msgseg.type == "text":
            if plain := msgseg.data.get("text", ""):
                text_message.append(plain)

    return {"text": text_message, "reply": reply_text, "images": image_urls, "user_refs": user_refs}


async def get_member_name(group: int, sender_id: int, bot) -> str:  # 将QQ号转换成昵称
    try:
        member_info = await bot.get_group_member_info(
            group_id=group, user_id=sender_id, no_cache=False
        )
        name = member_info.get("card") or member_info.get("nickname")
    except Exception:
        name = sender_id
        logger.warning("获取成员info失败")
    return str(name)
