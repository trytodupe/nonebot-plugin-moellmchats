def contains_doubao_help_trigger(text: str) -> bool:
    doubao_index = text.find("豆包")
    if doubao_index < 0:
        return False
    return text.find("帮我", doubao_index + len("豆包")) >= 0


def should_trigger_group_chat(has_at_mention: bool, text: str) -> bool:
    return has_at_mention or contains_doubao_help_trigger(text)
