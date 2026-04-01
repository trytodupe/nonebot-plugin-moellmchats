import json


def normalize_image_summary(summary: str) -> str:
    return " ".join(summary.strip().split())


def build_image_reference(summary: str) -> str:
    return f"[image:{normalize_image_summary(summary)}]"


def replace_image_placeholders(
    text: str, replacements: list[str], placeholder: str = "[图片]"
) -> str:
    if placeholder not in text:
        if replacements:
            return f"{text}\n" + "\n".join(replacements)
        return text

    parts = text.split(placeholder)
    result = [parts[0]]
    for index, replacement in enumerate(replacements):
        result.append(replacement)
        if index + 1 < len(parts):
            result.append(parts[index + 1])

    if len(replacements) + 1 < len(parts):
        result.extend(parts[len(replacements) + 1 :])
    return "".join(result)


def extract_response_output_text(response: dict) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        texts = []
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
        if texts:
            return "".join(texts).strip()
    return (response.get("output_text") or "").strip()


def parse_response_json_text(response: dict) -> dict:
    text = extract_response_output_text(response)
    if not text:
        return {}
    return json.loads(text)
