import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Awaitable, Callable

try:
    from nonebot.log import logger
except ImportError:  # pragma: no cover - fallback for standalone unit tests
    import logging

    logger = logging.getLogger(__name__)


DEFAULT_AGY_MODEL = "gemini-4-hydrogen-medium"
DEFAULT_AGY_AGENT_NAME = "doubao"
DEFAULT_ACP_TIMEOUT_SECONDS = 300.0
MCP_SERVER_NAME = "qq_context"
_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
_SAVED_IMAGE_PATH_RE = re.compile(r"Generated image is saved at\s+([^\r\n]+?)(?:\.\s*$|$)")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

def _build_doubao_agent_md(mcp_url: str | None = None) -> str:
    mcp_block = ""
    if mcp_url:
        mcp_block = (
            "mcpServers:\n"
            f"  - name: {MCP_SERVER_NAME}\n"
            f"    url: {mcp_url}\n"
        )
    return (
        "---\n"
        f"name: {DEFAULT_AGY_AGENT_NAME}\n"
        "description: QQ group chat assistant with web search, QQ context tools, and direct image generation\n"
        "mainAgent: true\n"
        "subagent: false\n"
        "inheritMcp: true\n"
        f"{mcp_block}"
        "tools:\n"
        "  - search_web\n"
        "  - generate_image\n"
        "---\n\n"
        "# Instructions\n"
        "You are a QQ group chat assistant.\n"
        "- Use `search_web` for live web information.\n"
        f"- Use MCP server `{MCP_SERVER_NAME}` (via `call_mcp_tool` with `ServerName=\"{MCP_SERVER_NAME}\"`) for QQ context tools:\n"
        "  - `fetch_recent_images` (`Arguments={\"limit\": 4, \"offset\": 0}`): Fetch recent QQ group images when the user asks to reference, edit, combine, or redraw recently sent group images that are not already attached.\n"
        "  - `fetch_user_avatar` (`Arguments={\"user_ref\": \"current_user\"}` or `\"mentioned_user_1\"`): Fetch a QQ user's avatar when the user asks to generate or edit an image containing themselves or a mentioned user, or asks what their avatar looks like.\n"
        "  - `get_imagegen_instructions` (`Arguments={}`): Fetch `$imagegen` prompt-refinement rules before calling `generate_image`.\n"
        "- Call `generate_image` directly to generate or edit images. When editing or referencing existing images or avatars, pass their local file paths in `ImagePaths`.\n"
    )


_DOUBAO_AGENT_MD = _build_doubao_agent_md()

MCP_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "fetch_recent_images",
        "description": (
            "Fetch recent QQ images cached by the plugin for the current group. "
            "Use conservatively when the user asks to reference, edit, combine, redraw, or generate from recently sent images. "
            "Returns each image's image_id, local file_path (to pass in generate_image ImagePaths), and cached summary."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of cached images to fetch. Prefer 4 to 8 when unsure.",
                },
                "offset": {
                    "type": "integer",
                    "description": "How many newest cached images to skip. Use 0 first.",
                },
            },
            "required": ["limit", "offset"],
        },
    },
    {
        "name": "fetch_user_avatar",
        "description": (
            "Fetch a QQ user's avatar for this turn by temporary user_ref (e.g. current_user or mentioned_user_1), never by QQ number. "
            "Returns user_ref, image_id, and local file_path (to pass in generate_image ImagePaths)."
        ),
        "inputSchema": {
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
    },
    {
        "name": "get_imagegen_instructions",
        "description": (
            "Return the $imagegen prompt-refinement instructions. "
            "Call this before using generate_image unless already fetched in this turn."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
    },
]

_ALLOWED_DIRECT_TOOLS = {
    "search_web",
    "generate_image",
    "fetch_recent_images",
    "fetch_user_avatar",
    "get_imagegen_instructions",
    f"mcp_{MCP_SERVER_NAME}_fetch_recent_images",
    f"mcp_{MCP_SERVER_NAME}_fetch_user_avatar",
    f"mcp_{MCP_SERVER_NAME}_get_imagegen_instructions",
}
_IMAGE_NOTICE_TRIGGER_TOOLS = {
    "generate_image",
    "get_imagegen_instructions",
    f"mcp_{MCP_SERVER_NAME}_get_imagegen_instructions",
}
_DENIED_TOOLS = {
    "run_command",
    "read_file",
    "write_file",
    "edit_file",
    "invoke_subagent",
    "define_subagent",
    "manage_subagents",
    "manage_task",
    "run_workflow",
    "swarm",
    "schedule",
    "delete_file",
    "view_file",
    "list_dir",
    "grep_search",
    "find_by_name",
}


@dataclass
class AcpTurnResult:
    text: str
    images: list[bytes] = field(default_factory=list)
    session_id: str | None = None
    text_steps: list[str] = field(default_factory=list)


def get_agy_runtime_dir() -> Path:
    shm_dir = Path("/dev/shm")
    uid = os.getuid() if hasattr(os, "getuid") else 0
    if shm_dir.exists() and os.access(shm_dir, os.W_OK):
        return shm_dir / f"moellmchats_agy_acp_{uid}"
    return Path(tempfile.gettempdir()) / f"moellmchats_agy_acp_{uid}"


def prepare_agy_runtime(
    runtime_dir: Path | None = None,
    oauth_token_source: Path | None = None,
    mcp_url: str | None = None,
) -> tuple[Path, Path, Path]:
    base_dir = Path(runtime_dir) if runtime_dir is not None else get_agy_runtime_dir()
    gemini_dir = base_dir / "gemini"
    app_data_dir = gemini_dir / "antigravity-cli"
    cwd_dir = base_dir / "cwd"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    (cwd_dir / ".git").mkdir(parents=True, exist_ok=True)

    agents_dir = cwd_dir / ".agents" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    agent_file = agents_dir / f"{DEFAULT_AGY_AGENT_NAME}.md"
    agent_md = _build_doubao_agent_md(mcp_url)
    if not agent_file.exists() or agent_file.read_text(encoding="utf-8") != agent_md:
        agent_file.write_text(agent_md, encoding="utf-8")

    src_token = (
        Path(oauth_token_source)
        if oauth_token_source is not None
        else Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    )
    dst_token = app_data_dir / "antigravity-oauth-token"
    if src_token.exists() and not dst_token.exists() and not dst_token.is_symlink():
        try:
            os.symlink(src_token, dst_token)
        except FileExistsError:
            pass

    return base_dir, gemini_dir, cwd_dir


def cleanup_session_files(gemini_dir: Path, session_id: str | None) -> None:
    if not session_id:
        return
    app_data_dir = Path(gemini_dir) / "antigravity-cli"
    conversations_dir = app_data_dir / "conversations"
    for suffix in (".db", ".db-wal", ".db-shm"):
        db_file = conversations_dir / f"{session_id}{suffix}"
        if db_file.exists() or db_file.is_symlink():
            try:
                db_file.unlink()
            except OSError:
                pass

    brain_dir = app_data_dir / "brain" / session_id
    if brain_dir.exists():
        shutil.rmtree(brain_dir, ignore_errors=True)


def data_url_to_acp_image(data_url: str) -> dict[str, str] | None:
    if not isinstance(data_url, str):
        return None
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        return None
    mime_type, b64_data = match.group(1).strip(), match.group(2).strip()
    if not mime_type or not b64_data:
        return None
    return {
        "type": "image",
        "mimeType": mime_type,
        "data": b64_data,
    }


def responses_content_to_acp_blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "text", "text": content}] if text else []

    blocks: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return blocks

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in ("input_text", "text"):
            text = str(item.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif item_type in ("input_image", "image"):
            if "image_url" in item:
                image_block = data_url_to_acp_image(str(item.get("image_url") or ""))
                if image_block:
                    blocks.append(image_block)
            elif item.get("data") and item.get("mimeType"):
                blocks.append(
                    {
                        "type": "image",
                        "mimeType": str(item["mimeType"]),
                        "data": str(item["data"]),
                    }
                )
    return blocks


def _normalize_tool_identifier(raw_name: Any) -> str:
    if not isinstance(raw_name, str):
        return ""
    cleaned = raw_name.strip().lower()
    for prefix in ("run ", "mcp tool "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    if not cleaned:
        return ""
    token = re.split(r"[\s:(\[]+", cleaned, maxsplit=1)[0]
    return token


def _resolve_tool_identifier(
    tool_obj: dict[str, Any],
    fallback_obj: dict[str, Any] | None = None,
) -> str:
    fallback = fallback_obj if isinstance(fallback_obj, dict) else {}
    raw_names = [
        tool_obj.get("toolName"),
        tool_obj.get("name"),
        tool_obj.get("tool"),
        fallback.get("toolName"),
        fallback.get("name"),
        fallback.get("tool"),
        tool_obj.get("title"),
    ]
    tool_name = ""
    for raw in raw_names:
        normalized = _normalize_tool_identifier(raw)
        if normalized:
            tool_name = normalized
            break

    if tool_name == "call_mcp_tool":
        for payload_key in ("args", "rawInput"):
            payload = tool_obj.get(payload_key)
            if not isinstance(payload, dict):
                continue
            server_name = (
                str(
                    payload.get("ServerName")
                    or payload.get("serverName")
                    or payload.get("server_name")
                    or ""
                )
                .strip()
                .lower()
            )
            inner_tool = _normalize_tool_identifier(
                payload.get("ToolName")
                or payload.get("toolName")
                or payload.get("tool_name")
                or payload.get("name")
            )
            if server_name and server_name != MCP_SERVER_NAME:
                return f"mcp_{server_name}_{inner_tool or 'unknown'}"
            if inner_tool:
                return inner_tool
        title_tool = _normalize_tool_identifier(tool_obj.get("title"))
        if title_tool and title_tool != "call_mcp_tool":
            return title_tool

    return tool_name


def _pick_permission_option_id(options: Any, *, allow: bool) -> str:
    if not isinstance(options, list):
        return "allow_once" if allow else "reject_once"

    preferred_kinds = (
        ("allow_once", "allow_always") if allow else ("reject_once", "reject_always")
    )
    for target_kind in preferred_kinds:
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") == target_kind:
                option_id = opt.get("optionId")
                if option_id:
                    return str(option_id)

    prefixes = ("allow",) if allow else ("reject", "deny")
    for opt in options:
        if isinstance(opt, dict):
            option_id = str(opt.get("optionId") or "")
            if option_id.lower().startswith(prefixes):
                return option_id

    return "allow_once" if allow else "reject_once"


def evaluate_permission_request(params: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(params, dict):
        return False, "reject_once"

    tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
    tool_name = _resolve_tool_identifier(tool_call, params)

    allowed = tool_name not in _DENIED_TOOLS and tool_name in _ALLOWED_DIRECT_TOOLS
    option_id = _pick_permission_option_id(params.get("options"), allow=allowed)
    return allowed, option_id


def _extract_saved_image_paths_from_update(update: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    locations = update.get("locations")
    if isinstance(locations, list):
        for loc in locations:
            if isinstance(loc, dict) and isinstance(loc.get("path"), str):
                paths.append(Path(loc["path"]))

    content = update.get("content")
    items = content if isinstance(content, list) else ([content] if content else [])
    for item in items:
        if not isinstance(item, dict):
            continue
        inner = item.get("content") if item.get("type") == "content" else item
        if isinstance(inner, dict) and inner.get("type") == "text":
            text = str(inner.get("text") or "")
            for match in _SAVED_IMAGE_PATH_RE.finditer(text):
                raw_path = match.group(1).strip()
                if raw_path:
                    paths.append(Path(raw_path))
    return paths


def _extract_images_from_dir(
    root_dir: Path,
    *,
    ignore_paths: set[Path] | None = None,
    ignore_dir_names: set[str] | None = None,
) -> tuple[list[bytes], list[Path]]:
    if not root_dir.exists():
        return [], []
    images: list[bytes] = []
    found_paths: list[Path] = []
    ignored = ignore_paths or set()
    ignored_dirs = ignore_dir_names or set()
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if ignored_dirs and any(
            part in ignored_dirs for part in path.relative_to(root_dir).parts[:-1]
        ):
            continue
        if path in ignored:
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            images.append(path.read_bytes())
            found_paths.append(path)
        except OSError:
            continue
    return images, found_paths


class AgyAcpClient:
    def __init__(
        self,
        *,
        runtime_dir: Path | None = None,
        oauth_token_source: Path | None = None,
        process_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = DEFAULT_ACP_TIMEOUT_SECONDS,
    ):
        self.runtime_dir = runtime_dir
        self.oauth_token_source = oauth_token_source
        self._process_factory = process_factory
        self.timeout_seconds = timeout_seconds

        self._lock = asyncio.Lock()
        self._proc: Any = None
        self._reader_task: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future] = {}
        self._session_text_steps: dict[str, list[list[str]]] = {}
        self._session_images: dict[str, list[bytes]] = {}
        self._session_notice_callbacks: dict[str, Callable[[], Awaitable[None]]] = {}
        self._session_notice_fired: set[str] = set()
        self._background_tasks: set[asyncio.Task] = set()
        self._initialized = False
        self._gemini_dir: Path | None = None
        self._cwd_dir: Path | None = None
        self._mcp_server: asyncio.AbstractServer | None = None
        self._mcp_url: str | None = None
        self._active_mcp_handler: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None
        self._active_mcp_tools: list[dict[str, Any]] = list(MCP_TOOL_DEFINITIONS)

    async def _ensure_mcp_server(self) -> None:
        if self._mcp_server is not None and self._mcp_url:
            return
        server = await asyncio.start_server(
            self._handle_mcp_connection, "127.0.0.1", 0
        )
        sockets = server.sockets or []
        if not sockets:
            server.close()
            await server.wait_closed()
            raise RuntimeError("无法启动内置 MCP HTTP 服务。")
        port = sockets[0].getsockname()[1]
        self._mcp_server = server
        self._mcp_url = f"http://127.0.0.1:{port}/mcp"

    async def _stop_mcp_server(self) -> None:
        self._mcp_url = None
        if self._mcp_server is not None:
            try:
                self._mcp_server.close()
                await self._mcp_server.wait_closed()
            except Exception:
                pass
            self._mcp_server = None

    async def _handle_mcp_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    break
                parts = request_line.decode("latin-1", errors="replace").strip().split()
                if len(parts) < 2:
                    break
                http_method = parts[0].upper()

                content_length = 0
                connection_close = False
                while True:
                    header_line = await reader.readline()
                    if not header_line or header_line in (b"\r\n", b"\n"):
                        break
                    decoded = header_line.decode("latin-1", errors="replace").strip()
                    if ":" not in decoded:
                        continue
                    k, v = decoded.split(":", 1)
                    k_lower = k.strip().lower()
                    v_clean = v.strip()
                    if k_lower == "content-length":
                        try:
                            content_length = int(v_clean)
                        except ValueError:
                            content_length = 0
                    elif k_lower == "connection" and v_clean.lower() == "close":
                        connection_close = True

                raw_body = (
                    await reader.readexactly(content_length)
                    if content_length > 0
                    else b""
                )
                if http_method != "POST":
                    status, resp_body = 405, b""
                else:
                    status, resp_body = await self._dispatch_mcp_payload(raw_body)

                reason = {
                    200: "OK",
                    202: "Accepted",
                    400: "Bad Request",
                    405: "Method Not Allowed",
                }.get(status, "OK")
                headers = [
                    f"HTTP/1.1 {status} {reason}",
                    f"Content-Length: {len(resp_body)}",
                ]
                if resp_body:
                    headers.append("Content-Type: application/json; charset=utf-8")
                if connection_close:
                    headers.append("Connection: close")
                writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("latin-1") + resp_body)
                await writer.drain()
                if connection_close:
                    break
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.warning(f"agy_acp MCP HTTP connection error: {exc}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch_mcp_payload(self, raw_body: bytes) -> tuple[int, bytes]:
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            return 400, b""
        if not isinstance(body, dict):
            return 400, b""

        method = body.get("method")
        req_id = body.get("id")
        if req_id is None or method in (
            "notifications/initialized",
            "notifications/roots/list_changed",
        ):
            return 202, b""

        if method == "initialize":
            return 200, json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": MCP_SERVER_NAME, "version": "1.0.0"},
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8")

        if method == "ping":
            return 200, json.dumps(
                {"jsonrpc": "2.0", "id": req_id, "result": {}}
            ).encode("utf-8")

        if method == "tools/list":
            return 200, json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": self._active_mcp_tools},
                },
                ensure_ascii=False,
            ).encode("utf-8")

        if method == "tools/call":
            params = body.get("params") or {}
            tool_name = str(params.get("name") or "").strip()
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            handler = self._active_mcp_handler
            if handler is None:
                return 200, json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": "当前轮次未绑定 MCP 工具处理器。"}],
                            "isError": True,
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            try:
                raw_result = await handler(tool_name, arguments)
                if isinstance(raw_result, list):
                    content_blocks = raw_result
                elif isinstance(raw_result, str):
                    content_blocks = [{"type": "text", "text": raw_result}]
                else:
                    content_blocks = [
                        {
                            "type": "text",
                            "text": json.dumps(raw_result, ensure_ascii=False),
                        }
                    ]
                return 200, json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"content": content_blocks},
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            except Exception as exc:
                logger.warning(f"agy_acp MCP tool '{tool_name}' failed: {exc}")
                return 200, json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": f"工具执行失败: {exc}"}],
                            "isError": True,
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8")

        # Return HTTP 400 for legacy discovery probes (e.g. server/discover) so agy
        # immediately falls back to standard MCP 2024-11-05 Streamable HTTP.
        return 400, b""

    async def _spawn_process(self) -> Any:
        await self._ensure_mcp_server()
        _, gemini_dir, cwd_dir = prepare_agy_runtime(
            runtime_dir=self.runtime_dir,
            oauth_token_source=self.oauth_token_source,
            mcp_url=self._mcp_url,
        )
        self._gemini_dir = gemini_dir
        self._cwd_dir = cwd_dir

        cmd = [
            "agy",
            "acp",
            "--agent",
            DEFAULT_AGY_AGENT_NAME,
            "--disable-telemetry",
            "--gemini-dir",
            str(gemini_dir),
            "--app-data-dir",
            "antigravity-cli",
            "--log-file",
            "/dev/null",
        ]

        if self._process_factory is not None:
            return await self._process_factory(cmd, cwd=str(cwd_dir))

        try:
            return await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(cwd_dir),
                limit=16 * 1024 * 1024,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 agy 命令，无法使用大豆包（agy_acp）。") from exc

    async def _ensure_started(self) -> None:
        if (
            self._proc is not None
            and getattr(self._proc, "returncode", None) is None
            and self._initialized
        ):
            await self._ensure_mcp_server()
            return

        await self._reset_process()
        await self._ensure_mcp_server()
        self._proc = await self._spawn_process()
        self._reader_task = asyncio.create_task(self._reader_loop())

        await self._send_request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "nonebot-plugin-moellmchats",
                    "version": "1.0.0",
                },
            },
        )
        self._initialized = True

    async def _reset_process(self) -> None:
        self._initialized = False
        self._active_mcp_handler = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._proc is not None:
            try:
                if getattr(self._proc, "returncode", None) is None:
                    self._proc.terminate()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=3.0)
                    except asyncio.TimeoutError:
                        self._proc.kill()
                        await self._proc.wait()
            except Exception:
                pass
            self._proc = None

        await self._stop_mcp_server()

        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("agy acp 进程已退出。"))
        self._pending.clear()

    async def close(self) -> None:
        async with self._lock:
            await self._reset_process()

    async def _write_json(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("agy acp 标准输入不可用。")
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

    async def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future
        try:
            await self._write_json(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": method,
                    "params": params,
                }
            )
            return await asyncio.wait_for(future, timeout=self.timeout_seconds)
        finally:
            self._pending.pop(req_id, None)

    def _trigger_image_notice(self, session_id: str) -> None:
        if not session_id or session_id in self._session_notice_fired:
            return
        callback = self._session_notice_callbacks.get(session_id)
        if callback is None:
            return
        self._session_notice_fired.add(session_id)

        async def _run_notice() -> None:
            try:
                await callback()
            except Exception:
                logger.warning("agy_acp image generation notice callback failed")

        task = asyncio.create_task(_run_notice())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"agy acp 返回了非 JSON 行: {text[:200]}")
                    continue

                if "id" in msg and "method" in msg:
                    await self._handle_server_request(msg)
                elif "id" in msg:
                    self._handle_response(msg)
                elif msg.get("method") == "session/update":
                    self._handle_session_update(msg.get("params") or {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"agy acp reader loop error: {exc}")
        finally:
            self._initialized = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("agy acp 连接已断开。"))

    def _handle_response(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        future = self._pending.get(req_id)
        if future is None or future.done():
            return
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            future.set_exception(RuntimeError(f"agy acp 请求失败: {message}"))
        else:
            future.set_result(msg.get("result"))

    async def _handle_server_request(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            allowed, option_id = evaluate_permission_request(params)
            tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
            resolved_tool = _resolve_tool_identifier(tool_call, params)
            tool_label = (
                resolved_tool
                or tool_call.get("toolName")
                or tool_call.get("name")
                or tool_call.get("title")
            )
            if allowed and resolved_tool in _IMAGE_NOTICE_TRIGGER_TOOLS:
                req_sid = str(params.get("sessionId") or "")
                if req_sid:
                    self._trigger_image_notice(req_sid)
                else:
                    for active_sid in self._session_text_steps:
                        self._trigger_image_notice(active_sid)
            logger.info(
                {
                    "event": "agy_acp_permission",
                    "allowed": allowed,
                    "option_id": option_id,
                    "tool": tool_label,
                }
            )
            await self._write_json(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": option_id,
                        }
                    },
                }
            )
            return

        await self._write_json(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not supported: {method}",
                },
            }
        )

    def _handle_session_update(self, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        update = params.get("update")
        if not session_id or not isinstance(update, dict):
            return

        update_type = update.get("sessionUpdate") or update.get("type")
        if update_type in ("tool_call", "tool_call_update"):
            tool_name = _resolve_tool_identifier(update)
            if tool_name in _IMAGE_NOTICE_TRIGGER_TOOLS:
                self._trigger_image_notice(session_id)
            if tool_name == "generate_image" and update.get("status") == "completed":
                for img_path in _extract_saved_image_paths_from_update(update):
                    if img_path.is_file():
                        try:
                            self._session_images.setdefault(session_id, []).append(
                                img_path.read_bytes()
                            )
                            img_path.unlink(missing_ok=True)
                        except OSError:
                            pass
            steps = self._session_text_steps.setdefault(session_id, [[]])
            if steps and steps[-1]:
                steps.append([])
            return

        if update_type not in ("agent_message_chunk", "agent_message"):
            return

        content = update.get("content")
        items = content if isinstance(content, list) else [content]
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    steps = self._session_text_steps.setdefault(session_id, [[]])
                    steps[-1].append(text)
            elif item_type == "image":
                b64_data = item.get("data")
                if isinstance(b64_data, str) and b64_data:
                    try:
                        self._session_images.setdefault(session_id, []).append(
                            base64.b64decode(b64_data)
                        )
                    except Exception:
                        pass

    async def run_turn(
        self,
        *,
        prompt_blocks: list[dict[str, Any]],
        model_info: dict[str, Any] | None = None,
        on_image_generation_start: Callable[[], Awaitable[None]] | None = None,
        mcp_tool_handler: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        mcp_tools: list[dict[str, Any]] | None = None,
        extra_input_hashes: set[str] | Callable[[], set[str]] | None = None,
    ) -> AcpTurnResult:
        model_info = model_info or {}
        async with self._lock:
            await self._ensure_started()
            assert self._gemini_dir is not None and self._cwd_dir is not None

            existing_cwd_files = {
                p for p in self._cwd_dir.rglob("*") if p.is_file()
            }

            self._active_mcp_handler = mcp_tool_handler
            self._active_mcp_tools = (
                list(mcp_tools) if mcp_tools is not None else list(MCP_TOOL_DEFINITIONS)
            )
            mcp_servers = (
                [{"name": MCP_SERVER_NAME, "url": self._mcp_url}]
                if (self._mcp_url and mcp_tool_handler is not None)
                else []
            )

            session_res = await self._send_request(
                "session/new",
                {
                    "cwd": str(self._cwd_dir),
                    "mcpServers": mcp_servers,
                },
            )
            session_id = str((session_res or {}).get("sessionId") or "")
            if not session_id:
                self._active_mcp_handler = None
                raise RuntimeError("agy acp 未返回有效的 sessionId。")

            self._session_text_steps[session_id] = [[]]
            self._session_images[session_id] = []
            if on_image_generation_start is not None:
                self._session_notice_callbacks[session_id] = on_image_generation_start

            try:
                model_value = str(model_info.get("model") or DEFAULT_AGY_MODEL).strip()
                if model_value:
                    await self._send_request(
                        "session/set_config_option",
                        {
                            "sessionId": session_id,
                            "configId": "model",
                            "value": model_value,
                        },
                    )
                for config_key in ("thought_level", "mode"):
                    config_val = model_info.get(config_key)
                    if config_val:
                        await self._send_request(
                            "session/set_config_option",
                            {
                                "sessionId": session_id,
                                "configId": config_key,
                                "value": str(config_val),
                            },
                        )

                await self._send_request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": prompt_blocks,
                    },
                )

                input_hashes: set[str] = set()
                if callable(extra_input_hashes):
                    input_hashes.update(extra_input_hashes())
                elif isinstance(extra_input_hashes, set):
                    input_hashes.update(extra_input_hashes)
                for block in prompt_blocks:
                    if isinstance(block, dict) and block.get("type") == "image":
                        b64_data = block.get("data")
                        if isinstance(b64_data, str) and b64_data:
                            try:
                                input_hashes.add(
                                    hashlib.sha256(base64.b64decode(b64_data)).hexdigest()
                                )
                            except Exception:
                                pass

                session_brain_dir = (
                    self._gemini_dir / "antigravity-cli" / "brain" / session_id
                )
                brain_images, _ = _extract_images_from_dir(
                    session_brain_dir,
                    ignore_dir_names={
                        ".user_uploaded",
                        ".system_generated",
                        "scratch",
                        "inputs",
                    },
                )
                cwd_images, cwd_image_paths = _extract_images_from_dir(
                    self._cwd_dir,
                    ignore_paths=existing_cwd_files,
                )

                collected_images: list[bytes] = []
                seen_hashes = set(input_hashes)
                for img in (
                    self._session_images.get(session_id, [])
                    + brain_images
                    + cwd_images
                ):
                    digest = hashlib.sha256(img).hexdigest()
                    if digest not in seen_hashes:
                        seen_hashes.add(digest)
                        collected_images.append(img)

                for p in cwd_image_paths:
                    try:
                        p.unlink()
                    except OSError:
                        pass

                raw_steps = self._session_text_steps.get(session_id, [])
                text_steps = [
                    "".join(chunks)
                    for chunks in raw_steps
                    if "".join(chunks).strip()
                ]
                text_reply = text_steps[-1] if text_steps else ""

                return AcpTurnResult(
                    text=text_reply,
                    images=collected_images,
                    session_id=session_id,
                    text_steps=text_steps,
                )
            finally:
                self._active_mcp_handler = None
                self._session_text_steps.pop(session_id, None)
                self._session_images.pop(session_id, None)
                self._session_notice_callbacks.pop(session_id, None)
                self._session_notice_fired.discard(session_id)
                try:
                    if self._proc is not None and getattr(self._proc, "returncode", None) is None:
                        await self._send_request(
                            "session/close",
                            {"sessionId": session_id},
                        )
                except Exception:
                    pass
                cleanup_session_files(self._gemini_dir, session_id)


_shared_acp_client: AgyAcpClient | None = None


def get_shared_acp_client() -> AgyAcpClient:
    global _shared_acp_client
    if _shared_acp_client is None:
        _shared_acp_client = AgyAcpClient()
    return _shared_acp_client


async def close_shared_acp_client() -> None:
    global _shared_acp_client
    if _shared_acp_client is not None:
        await _shared_acp_client.close()
        _shared_acp_client = None
