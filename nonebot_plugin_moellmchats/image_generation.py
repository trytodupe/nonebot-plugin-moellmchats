import base64
from dataclasses import dataclass
import re
import struct

OPENAI_IMAGE_SIZE_DEFAULT = "auto"
OPENAI_IMAGE_SIZE_EXAMPLES = (
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
)

VERTEX_IMAGE_MODELS = (
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-2.5-flash-image",
)

_VERTEX_COMMON_ASPECT_RATIOS = (
    "1:1",
    "3:2",
    "2:3",
    "3:4",
    "1:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "1:8",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
)


@dataclass(frozen=True)
class VertexImageProfile:
    sizes: tuple[str, ...]
    aspect_ratios: tuple[str, ...]
    max_reference_images: int = 14
    fixed_top_k: int | None = None
    omit_image_size: bool = False


VERTEX_IMAGE_PROFILES = {
    "gemini-3-pro-image": VertexImageProfile(
        sizes=("1K", "2K", "4K"),
        aspect_ratios=(*_VERTEX_COMMON_ASPECT_RATIOS, "9:21"),
    ),
    "gemini-3.1-flash-image": VertexImageProfile(
        sizes=("512", "1K", "2K", "4K"),
        aspect_ratios=(*_VERTEX_COMMON_ASPECT_RATIOS, "9:21"),
    ),
    "gemini-3.1-flash-lite-image": VertexImageProfile(
        sizes=("1K",),
        aspect_ratios=_VERTEX_COMMON_ASPECT_RATIOS,
    ),
    "gemini-2.5-flash-image": VertexImageProfile(
        sizes=(),
        aspect_ratios=(
            "1:1",
            "3:2",
            "2:3",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ),
        max_reference_images=3,
        fixed_top_k=64,
        omit_image_size=True,
    ),
}


VERTEX_IMAGE_TOOL_DESCRIPTION = (
    "Generate or edit one image with Vertex AI Gemini. Use this tool only when "
    "the user explicitly requests Google, Gemini, or a named Gemini image model. "
    "Otherwise use the native OpenAI image_generation tool. Current standard image "
    "prices: gemini-3-pro-image costs $0.134 at 1K/2K and $0.24 at 4K; "
    "gemini-3.1-flash-image costs $0.045 at 512, $0.067 at 1K, $0.101 at 2K, "
    "and $0.15 at 4K; gemini-3.1-flash-lite-image costs $0.034 at its fixed 1K; "
    "gemini-2.5-flash-image costs approximately $0.0387 at its model-selected size. "
    "Pro is the highest-quality option. 3.1 Flash supports 512 through 4K and is "
    "the default balanced option. 3.1 Flash-Lite is the cheapest and supports only "
    "1K. Pro, 3.1 Flash, and 3.1 Flash-Lite accept up to 14 reference images; "
    "2.5 Flash accepts up to 3 and supports only aspect-ratio selection. Video "
    "input is not available through this QQ integration. Reference images may be "
    "supplied with image_ids."
)


def normalize_openai_image_size(value: object) -> str:
    size = str(value or OPENAI_IMAGE_SIZE_DEFAULT).strip().lower()
    if size == "auto":
        return size
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        return OPENAI_IMAGE_SIZE_DEFAULT
    width, height = (int(part) for part in match.groups())
    short_edge = min(width, height)
    long_edge = max(width, height)
    pixels = width * height
    if width % 16 or height % 16 or long_edge > 3840 or long_edge > short_edge * 3 or pixels < 655_360 or pixels > 8_294_400:
        return OPENAI_IMAGE_SIZE_DEFAULT
    return f"{width}x{height}"


def normalize_vertex_image_request(arguments: dict) -> dict | None:
    prompt = str(arguments.get("prompt") or "").strip()
    model = str(arguments.get("model") or "gemini-3.1-flash-image").strip()
    aspect_ratio = str(arguments.get("aspect_ratio") or "1:1").strip()
    image_size = str(arguments.get("image_size") or "1K").strip().upper()
    if image_size == "512":
        image_size = "512"
    if not prompt or model not in VERTEX_IMAGE_PROFILES:
        return None
    profile = VERTEX_IMAGE_PROFILES[model]
    if aspect_ratio not in profile.aspect_ratios:
        aspect_ratio = "1:1"
    if profile.omit_image_size:
        image_size = "MODEL_SELECTED"
    elif image_size not in profile.sizes:
        image_size = profile.sizes[0]
    image_ids = arguments.get("image_ids") or []
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    return {
        "prompt": prompt,
        "model": model,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "image_ids": [str(item).strip() for item in image_ids if str(item).strip()][
            : profile.max_reference_images
        ],
    }


def build_vertex_image_payload(request: dict, image_inputs_by_id: dict) -> dict:
    profile = VERTEX_IMAGE_PROFILES[request["model"]]
    parts = []
    for image_id in request["image_ids"]:
        image_input = image_inputs_by_id.get(image_id)
        if not image_input:
            continue
        parts.append(
            {
                "inlineData": {
                    "mimeType": image_input["mime_type"],
                    "data": base64.b64encode(image_input["bytes"]).decode(),
                }
            }
        )
    parts.append({"text": request["prompt"]})
    generation_config = {
        "responseModalities": ["TEXT", "IMAGE"],
        "temperature": 1.0,
        "topP": 0.95,
        "candidateCount": 1,
        "imageConfig": {"aspectRatio": request["aspect_ratio"]},
    }
    if not profile.omit_image_size:
        generation_config["imageConfig"]["imageSize"] = request["image_size"]
    if profile.fixed_top_k is not None:
        generation_config["topK"] = profile.fixed_top_k
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def extract_vertex_images(response: dict) -> list[bytes]:
    images = []
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            encoded = inline_data.get("data")
            if not encoded:
                continue
            try:
                images.append(base64.b64decode(encoded, validate=True))
            except (ValueError, TypeError):
                continue
    return images


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            return None
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if segment_length < 7:
                return None
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return width, height
        offset += segment_length
    return None


def actual_image_scale(data: bytes, fallback: str) -> str:
    dimensions = image_dimensions(data)
    if not dimensions:
        return fallback
    return f"{dimensions[0]}x{dimensions[1]}"
