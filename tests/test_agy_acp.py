import asyncio
import base64
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest


module_path = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats" / "agy_acp.py"
spec = spec_from_file_location("nonebot_plugin_moellmchats.agy_acp", module_path)
agy_acp = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(agy_acp)


class _FakeStdin:
    def __init__(self, process: "_FakeAcpProcess"):
        self._process = process
        self._buffer = bytearray()

    def write(self, data: bytes) -> None:
        self._buffer.extend(data)
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = bytearray(rest)
            text = line.decode("utf-8").strip()
            if text:
                self._process.handle_client_message(json.loads(text))

    async def drain(self) -> None:
        return None


class _FakeAcpProcess:
    def __init__(
        self,
        *,
        runtime_dir: Path,
        session_id: str = "sess-1",
        emit_permission_checks: bool = False,
        create_brain_image: bool = False,
    ):
        self.runtime_dir = runtime_dir
        self.session_id = session_id
        self.emit_permission_checks = emit_permission_checks
        self.create_brain_image = create_brain_image
        self.returncode: int | None = None
        self.stdin = _FakeStdin(self)
        self.stdout = asyncio.StreamReader()
        self.received_methods: list[str] = []
        self.config_options_set: list[tuple[str, str]] = []
        self.permission_outcomes: list[tuple[str, str]] = []
        self._pending_prompt_id: int | str | None = None

    def _push(self, payload: dict) -> None:
        self.stdout.feed_data((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

    def handle_client_message(self, msg: dict) -> None:
        if "method" in msg:
            method = msg["method"]
            req_id = msg.get("id")
            params = msg.get("params") or {}
            self.received_methods.append(method)

            if method == "initialize":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1}})
            elif method == "session/new":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": self.session_id}})
            elif method == "session/set_config_option":
                self.config_options_set.append((params["configId"], params["value"]))
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {"configOptions": []}})
            elif method == "session/prompt":
                if self.emit_permission_checks:
                    self._pending_prompt_id = req_id
                    self._push(
                        {
                            "jsonrpc": "2.0",
                            "id": 9001,
                            "method": "session/request_permission",
                            "params": {
                                "sessionId": self.session_id,
                                "toolCall": {"title": "search_web"},
                                "options": [
                                    {"optionId": "allow_once", "kind": "allow_once"},
                                    {"optionId": "reject_once", "kind": "reject_once"},
                                ],
                            },
                        }
                    )
                else:
                    self._finish_prompt(req_id)
            elif method == "session/close":
                self._push({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif "id" in msg:
            req_id = msg["id"]
            outcome = ((msg.get("result") or {}).get("outcome") or {}).get("optionId", "")
            if req_id == 9001:
                self.permission_outcomes.append(("search_web", outcome))
                self._push(
                    {
                        "jsonrpc": "2.0",
                        "id": 9002,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": self.session_id,
                            "toolCall": {"title": "run_command"},
                            "options": [
                                {"optionId": "allow_once", "kind": "allow_once"},
                                {"optionId": "reject_once", "kind": "reject_once"},
                            ],
                        },
                    }
                )
            elif req_id == 9002:
                self.permission_outcomes.append(("run_command", outcome))
                assert self._pending_prompt_id is not None
                self._finish_prompt(self._pending_prompt_id)

    def _finish_prompt(self, req_id: int | str) -> None:
        self._push(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": "internal thinking"},
                    },
                },
            }
        )
        if self.create_brain_image:
            self._push(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": self.session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {
                                "type": "text",
                                "text": '{"assistant_reply":"","image_memories":[{"client_image_id":"img1","summary":"cat"}]}',
                            },
                        },
                    },
                }
            )
            brain_dir = (
                self.runtime_dir
                / "gemini"
                / "antigravity-cli"
                / "brain"
                / self.session_id
            )
            brain_dir.mkdir(parents=True, exist_ok=True)
            img_path = brain_dir / "generated.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            conv_dir = self.runtime_dir / "gemini" / "antigravity-cli" / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            (conv_dir / f"{self.session_id}.db").write_bytes(b"sqlite")
            self._push(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": self.session_id,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "name": "generate_image",
                            "status": "in_progress",
                        },
                    },
                }
            )
            self._push(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": self.session_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "name": "generate_image",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "content",
                                    "content": {
                                        "type": "text",
                                        "text": f"Generated image is saved at {img_path}.",
                                    },
                                }
                            ],
                        },
                    },
                }
            )
        self._push(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": self.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {
                            "type": "text",
                            "text": '{"assistant_reply":"你好","image_memories":[]}',
                        },
                    },
                },
            }
        )

        self._push({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0
            self.stdout.feed_eof()

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class AgyAcpTest(unittest.TestCase):
    def test_permission_allow_and_deny_rules(self):
        options = [
            {"optionId": "allow_once", "kind": "allow_once"},
            {"optionId": "reject_once", "kind": "reject_once"},
        ]

        for allowed_tool in (
            "search_web",
            "generate_image",
            "mcp_qq_context_fetch_recent_images",
            "mcp_qq_context_fetch_user_avatar",
            "mcp_qq_context_get_imagegen_instructions",
            "Run mcp_qq_context_fetch_user_avatar",
            "MCP Tool fetch_user_avatar",
        ):
            allowed, opt = agy_acp.evaluate_permission_request(
                {"toolCall": {"name": allowed_tool}, "options": options}
            )
            self.assertTrue(allowed, msg=allowed_tool)
            self.assertEqual(opt, "allow_once")

        allowed_mcp_call, opt = agy_acp.evaluate_permission_request(
            {
                "toolCall": {
                    "name": "call_mcp_tool",
                    "args": {
                        "ServerName": "qq_context",
                        "ToolName": "fetch_user_avatar",
                        "Arguments": {"user_ref": "current_user"},
                    },
                    "title": "MCP Tool fetch_user_avatar",
                },
                "options": options,
            }
        )
        self.assertTrue(allowed_mcp_call)
        self.assertEqual(opt, "allow_once")

        for denied_mcp_payload in (
            {"ServerName": "other_server", "ToolName": "fetch_user_avatar"},
            {"ServerName": "qq_context", "ToolName": "run_command"},
        ):
            allowed_bad, opt_bad = agy_acp.evaluate_permission_request(
                {
                    "toolCall": {
                        "name": "call_mcp_tool",
                        "args": denied_mcp_payload,
                    },
                    "options": options,
                }
            )
            self.assertFalse(allowed_bad)
            self.assertEqual(opt_bad, "reject_once")

        for denied_tool in (
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
            "unknown_custom_tool",
        ):
            allowed, opt = agy_acp.evaluate_permission_request(
                {"toolCall": {"title": denied_tool}, "options": options}
            )
            self.assertFalse(allowed, msg=denied_tool)
            self.assertEqual(opt, "reject_once")

    def test_data_url_and_responses_content_conversion(self):
        raw_b64 = base64.b64encode(b"hello-png").decode("ascii")
        block = agy_acp.data_url_to_acp_image(f"data:image/png;base64,{raw_b64}")
        self.assertEqual(
            block,
            {"type": "image", "mimeType": "image/png", "data": raw_b64},
        )
        self.assertIsNone(agy_acp.data_url_to_acp_image("https://example.com/a.png"))

        blocks = agy_acp.responses_content_to_acp_blocks(
            [
                {"type": "input_text", "text": "看这张图"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{raw_b64}"},
            ]
        )
        self.assertEqual(
            blocks,
            [
                {"type": "text", "text": "看这张图"},
                {"type": "image", "mimeType": "image/jpeg", "data": raw_b64},
            ],
        )

    def test_runtime_setup_and_session_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            token_src = tmp_path / "source_token"
            token_src.write_text("oauth-secret", encoding="utf-8")
            runtime_dir = tmp_path / "runtime"

            _, gemini_dir, cwd_dir = agy_acp.prepare_agy_runtime(
                runtime_dir=runtime_dir,
                oauth_token_source=token_src,
                mcp_url="http://127.0.0.1:54321/mcp",
            )
            self.assertTrue((cwd_dir / ".git").is_dir())
            agent_file = cwd_dir / ".agents" / "agents" / "doubao.md"
            self.assertTrue(agent_file.is_file())
            agent_text = agent_file.read_text(encoding="utf-8")
            self.assertIn("generate_image", agent_text)
            self.assertIn("http://127.0.0.1:54321/mcp", agent_text)
            token_dst = gemini_dir / "antigravity-cli" / "antigravity-oauth-token"
            self.assertTrue(token_dst.is_symlink())
            self.assertEqual(token_dst.read_text(encoding="utf-8"), "oauth-secret")

            sid = "abc-123"
            conv_dir = gemini_dir / "antigravity-cli" / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            db_file = conv_dir / f"{sid}.db"
            db_file.write_bytes(b"db")
            brain_dir = gemini_dir / "antigravity-cli" / "brain" / sid
            brain_dir.mkdir(parents=True, exist_ok=True)
            (brain_dir / "out.png").write_bytes(b"png")

            agy_acp.cleanup_session_files(gemini_dir, sid)
            self.assertFalse(db_file.exists())
            self.assertFalse(brain_dir.exists())

    def test_client_lifecycle_permissions_images_and_restart(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as tmp:
                runtime_dir = Path(tmp) / "runtime"
                spawned: list[_FakeAcpProcess] = []
                spawned_cmds: list[list[str]] = []
                notice_calls = 0

                async def on_notice():
                    nonlocal notice_calls
                    notice_calls += 1

                async def factory(cmd, cwd):
                    idx = len(spawned) + 1
                    spawned_cmds.append(list(cmd))
                    proc = _FakeAcpProcess(
                        runtime_dir=runtime_dir,
                        session_id=f"sess-{idx}",
                        emit_permission_checks=(idx == 1),
                        create_brain_image=(idx == 1),
                    )
                    spawned.append(proc)
                    return proc

                client = agy_acp.AgyAcpClient(
                    runtime_dir=runtime_dir,
                    process_factory=factory,
                    timeout_seconds=5.0,
                )
                try:
                    res1 = await client.run_turn(
                        prompt_blocks=[{"type": "text", "text": "hello"}],
                        model_info={"thought_level": "medium", "mode": "default"},
                        on_image_generation_start=on_notice,
                    )
                    self.assertIn("--agent", spawned_cmds[0])
                    self.assertIn("doubao", spawned_cmds[0])
                    self.assertEqual(res1.text, '{"assistant_reply":"你好","image_memories":[]}')
                    self.assertEqual(
                        res1.text_steps,
                        [
                            '{"assistant_reply":"","image_memories":[{"client_image_id":"img1","summary":"cat"}]}',
                            '{"assistant_reply":"你好","image_memories":[]}',
                        ],
                    )
                    self.assertEqual(res1.images, [b"\x89PNG\r\n\x1a\nfake"])
                    self.assertEqual(notice_calls, 1)
                    self.assertEqual(
                        spawned[0].received_methods,
                        [
                            "initialize",
                            "session/new",
                            "session/set_config_option",
                            "session/set_config_option",
                            "session/set_config_option",
                            "session/prompt",
                            "session/close",
                        ],
                    )
                    self.assertEqual(
                        spawned[0].config_options_set,
                        [
                            ("model", "gemini-4-hydrogen-medium"),
                            ("thought_level", "medium"),
                            ("mode", "default"),
                        ],
                    )
                    self.assertEqual(
                        spawned[0].permission_outcomes,
                        [("search_web", "allow_once"), ("run_command", "reject_once")],
                    )
                    self.assertFalse(
                        (runtime_dir / "gemini" / "antigravity-cli" / "brain" / "sess-1").exists()
                    )
                    self.assertFalse(
                        (
                            runtime_dir
                            / "gemini"
                            / "antigravity-cli"
                            / "conversations"
                            / "sess-1.db"
                        ).exists()
                    )

                    spawned[0].terminate()
                    res2 = await client.run_turn(
                        prompt_blocks=[{"type": "text", "text": "second"}],
                        model_info={"model": "gemini-4-carbon"},
                    )
                    self.assertEqual(len(spawned), 2)
                    self.assertEqual(res2.text, '{"assistant_reply":"你好","image_memories":[]}')
                finally:
                    await client.close()

        asyncio.run(run_test())

    def test_ignores_user_uploaded_and_input_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain_dir = Path(tmp) / "brain" / "sess-x"
            (brain_dir / ".user_uploaded").mkdir(parents=True)
            (brain_dir / ".system_generated").mkdir(parents=True)
            (brain_dir / ".user_uploaded" / "input.png").write_bytes(b"user-input")
            (brain_dir / ".system_generated" / "sys.png").write_bytes(b"sys-img")
            (brain_dir / "generated.png").write_bytes(b"gen-img")

            images, _ = agy_acp._extract_images_from_dir(
                brain_dir,
                ignore_dir_names={".user_uploaded", ".system_generated"},
            )
            self.assertEqual(images, [b"gen-img"])

    def test_in_process_mcp_http_server_lifecycle(self):
        async def post_raw(port: int, body: dict) -> tuple[int, bytes]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            raw = json.dumps(body).encode("utf-8")
            req = (
                f"POST /mcp HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(raw)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("latin-1") + raw
            writer.write(req)
            await writer.drain()
            status_line = (await reader.readline()).decode("latin-1").strip()
            status_code = int(status_line.split()[1])
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
            resp_body = await reader.read()
            writer.close()
            await writer.wait_closed()
            return status_code, resp_body

        async def run_test():
            client = agy_acp.AgyAcpClient()
            await client._ensure_mcp_server()
            try:
                assert client._mcp_url is not None
                port = int(client._mcp_url.split(":")[2].split("/")[0])

                # Legacy server/discover probe should return 400
                status, _ = await post_raw(
                    port, {"jsonrpc": "2.0", "id": 1, "method": "server/discover"}
                )
                self.assertEqual(status, 400)

                # Standard initialize -> 200
                status, body = await post_raw(
                    port, {"jsonrpc": "2.0", "id": 2, "method": "initialize"}
                )
                self.assertEqual(status, 200)
                init_res = json.loads(body.decode("utf-8"))
                self.assertEqual(init_res["result"]["serverInfo"]["name"], "qq_context")

                # Notification -> 202
                status, _ = await post_raw(
                    port, {"jsonrpc": "2.0", "method": "notifications/initialized"}
                )
                self.assertEqual(status, 202)

                # tools/list -> 200
                status, body = await post_raw(
                    port, {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
                )
                self.assertEqual(status, 200)
                tools_res = json.loads(body.decode("utf-8"))
                tool_names = [t["name"] for t in tools_res["result"]["tools"]]
                self.assertEqual(
                    tool_names,
                    [
                        "fetch_recent_images",
                        "fetch_user_avatar",
                        "get_imagegen_instructions",
                    ],
                )

                # tools/call with handler bound
                async def fake_handler(name: str, args: dict):
                    return {"tool": name, "user_ref": args.get("user_ref")}

                client._active_mcp_handler = fake_handler
                status, body = await post_raw(
                    port,
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "fetch_user_avatar",
                            "arguments": {"user_ref": "current_user"},
                        },
                    },
                )
                self.assertEqual(status, 200)
                call_res = json.loads(body.decode("utf-8"))
                self.assertEqual(
                    json.loads(call_res["result"]["content"][0]["text"]),
                    {"tool": "fetch_user_avatar", "user_ref": "current_user"},
                )
            finally:
                await client._stop_mcp_server()
                self.assertIsNone(client._mcp_url)

        asyncio.run(run_test())

    def test_idle_timeout_reaps_process_and_respawns_on_next_turn(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as tmp:
                runtime_dir = Path(tmp) / "runtime"
                spawned: list[_FakeAcpProcess] = []

                async def factory(cmd, cwd):
                    idx = len(spawned) + 1
                    proc = _FakeAcpProcess(
                        runtime_dir=runtime_dir,
                        session_id=f"idle-{idx}",
                    )
                    spawned.append(proc)
                    return proc

                client = agy_acp.AgyAcpClient(
                    runtime_dir=runtime_dir,
                    process_factory=factory,
                    timeout_seconds=5.0,
                    idle_timeout_seconds=0.05,
                )
                try:
                    await client.run_turn(
                        prompt_blocks=[{"type": "text", "text": "turn 1"}],
                    )
                    self.assertEqual(len(spawned), 1)
                    self.assertIsNotNone(client._proc)

                    await asyncio.sleep(0.12)
                    self.assertIsNone(client._proc)
                    self.assertIsNone(client._mcp_url)

                    await client.run_turn(
                        prompt_blocks=[{"type": "text", "text": "turn 2"}],
                    )
                    self.assertEqual(len(spawned), 2)
                finally:
                    await client.close()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
