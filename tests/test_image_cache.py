"""Exercise the public cache interface with real SQLite and a local HTTP server."""

import ast
import asyncio
from collections import defaultdict, deque
from importlib import import_module
import logging
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

PACKAGE_PATH = Path(__file__).resolve().parents[1] / "nonebot_plugin_moellmchats"
package = ModuleType("image_cache_tests_package")
package.__path__ = [str(PACKAGE_PATH)]
config = ModuleType("image_cache_tests_package.Config")
config.config_parser = SimpleNamespace(get_config=lambda _: None)
with patch.dict(
    "sys.modules",
    {
        package.__name__: package,
        config.__name__: config,
        "nonebot_plugin_localstore": SimpleNamespace(get_plugin_data_dir=lambda: Path("/tmp/unused-image-cache")),
        "nonebot": SimpleNamespace(),
        "nonebot.log": SimpleNamespace(logger=logging.getLogger("image-cache-test")),
        "nonebot.adapters.onebot.v11": SimpleNamespace(Message=list),
    },
):
    cache_module = import_module("image_cache_tests_package.ImageCache")
    utils = import_module("image_cache_tests_package.utils")

ImageCache = cache_module.ImageCache
PNG = b"\x89PNG\r\n\x1a\n"


class ImageCacheTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.now = 1000.0
        self.requests = []
        self.blocked = {}
        self.started = defaultdict(asyncio.Event)
        self.active = 0
        self.peak = 0
        app = web.Application()
        app.router.add_get("/{key}", self.serve)
        self.server = TestServer(app)
        await self.server.start_server()
        self.caches = []
        self.tasks = []
        self.cache = await self.make_cache()

    async def asyncTearDown(self):
        for event in self.blocked.values():
            event.set()
        for cache in self.caches:
            await cache.close()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.server.close()
        self.temp.cleanup()

    async def make_cache(self, **kwargs):
        cache = ImageCache(Path(self.temp.name) / str(len(self.caches)), clock=lambda: self.now, **kwargs)
        await cache.start()
        self.caches.append(cache)
        return cache

    def url(self, key="same"):
        return str(self.server.make_url("/" + key))

    def observe(self, key="same", *, cache=None, group=1, user=2, message=3):
        return (cache or self.cache).observe(
            group_id=group, user_id=user, message_id=message, images=[{"source_url": self.url(key)}]
        )

    async def serve(self, request):
        key = request.match_info["key"]
        self.requests.append(key)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started[key].set()
        try:
            if key in self.blocked:
                await self.blocked[key].wait()
            if key == "error":
                return web.Response(status=503)
            if key == "oversized":
                response = web.StreamResponse()
                await response.prepare(request)
                try:
                    for _ in range(8):
                        await response.write(PNG + b"x" * 64)
                except ConnectionResetError:
                    pass
                return response
            if key.startswith("alias"):
                return web.Response(body=PNG + b"identical")
            return web.Response(body=PNG + key.encode())
        finally:
            self.active -= 1

    async def wait_started(self, key):
        await asyncio.wait_for(self.started[key].wait(), 2)

    async def test_context_collection_returns_while_download_is_stalled(self):
        self.blocked["same"] = asyncio.Event()

        class GroupEvent(SimpleNamespace):
            def get_message(inner):
                return [SimpleNamespace(type="image", data={"url": self.url(), "file": "same.png"})]

        source = ast.parse((PACKAGE_PATH / "__init__.py").read_text())
        functions = []
        for node in source.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "_session_key",
                "cache_message_images",
                "context_dict_func",
            }:
                node.decorator_list = []
                functions.append(node)
        contexts = defaultdict(lambda: deque(maxlen=30))
        namespace = {
            "GroupMessageEvent": GroupEvent,
            "MessageEvent": object,
            "Bot": object,
            "format_message": utils.format_message,
            "image_cache": self.cache,
            "llm": SimpleNamespace(context_dict=contexts),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(PACKAGE_PATH / "__init__.py"), "exec"), namespace)
        event = GroupEvent(group_id=1, user_id=2, message_id=3, reply=None, sender=SimpleNamespace(card="", nickname="sender"))
        await asyncio.wait_for(namespace["context_dict_func"](SimpleNamespace(), event), 1)
        assert contexts["group:1"][0]["content"] == "[图片]"
        await self.wait_started("same")
        assert not self.blocked["same"].is_set()
        self.blocked["same"].set()
        await self.cache.drain()

    async def test_concurrent_demand_and_prefetch_share_one_download(self):
        self.blocked["same"] = asyncio.Event()
        self.observe()
        await self.wait_started("same")
        tasks = [asyncio.create_task(self.cache.read_image({"source_url": self.url()}, group_id=1, user_id=2)) for _ in range(20)]
        self.tasks.extend(tasks)
        self.blocked["same"].set()
        results = await asyncio.gather(*tasks)
        assert self.requests.count("same") == 1
        assert all(item and item[0] == PNG + b"same" for item in results)

    async def test_expiring_one_group_keeps_other_group_image_alive(self):
        self.observe("alias-one", group=1)
        await self.cache.drain()
        self.now += 100
        self.observe("alias-two", group=2)
        await self.cache.drain()
        second = await self.cache.get_recent_group_images(group_id=2)
        self.now += 3501
        await self.cache.cleanup()
        assert await self.cache.get_recent_group_images(group_id=1) == []
        remaining = await self.cache.get_recent_group_images(group_id=2)
        assert remaining[0]["image_id"] == second[0]["image_id"]
        assert await self.cache.read_image(remaining[0]) is not None
        assert len(list((Path(self.temp.name) / "0").glob("img_sha256_*"))) == 1

    async def test_all_url_aliases_expire_and_can_be_downloaded_again(self):
        for key in ("alias-one", "alias-two"):
            self.observe(key)
            await self.cache.drain()
        assert len(list((Path(self.temp.name) / "0").glob("img_sha256_*"))) == 1
        self.now += 3601
        await self.cache.cleanup()
        assert await self.cache.get_recent_group_images(group_id=1) == []
        assert len(list((Path(self.temp.name) / "0").glob("img_sha256_*"))) == 0
        for key in ("alias-one", "alias-two"):
            assert await self.cache.read_image({"source_url": self.url(key)}, group_id=1) is not None
        assert self.requests.count("alias-one") == 2
        assert self.requests.count("alias-two") == 2

    async def test_queue_and_per_group_limits_contain_flood(self):
        cache = await self.make_cache(workers=2, queue_size=6, per_group_pending=2)
        for key in ("blocked-a", "blocked-b"):
            self.blocked[key] = asyncio.Event()
        self.observe("blocked-a", cache=cache, group=1)
        self.observe("blocked-b", cache=cache, group=2)
        await self.wait_started("blocked-a")
        await self.wait_started("blocked-b")
        accepted = sum(self.observe(f"flood-{n}", cache=cache, group=1) for n in range(100))
        assert accepted == 1
        for group in range(3, 20):
            self.observe(f"group-{group}", cache=cache, group=group)
        assert cache.snapshot()["pending"] <= 6
        assert cache.snapshot()["dropped"] > 90
        for event in self.blocked.values():
            event.set()
        await cache.drain()
        assert self.peak <= 2

    async def test_explicit_request_preempts_queued_background_work(self):
        cache = await self.make_cache(workers=1, queue_size=3, per_group_pending=3)
        self.blocked["blocked"] = asyncio.Event()
        self.observe("blocked", cache=cache)
        await self.wait_started("blocked")
        self.observe("background-a", cache=cache, group=2)
        self.observe("background-b", cache=cache, group=3)
        task = asyncio.create_task(cache.read_image({"source_url": self.url("demand")}, group_id=4))
        self.tasks.append(task)
        await asyncio.sleep(0)
        self.blocked["blocked"].set()
        assert await task is not None
        await cache.drain()
        assert self.requests[:2] == ["blocked", "demand"]
        assert cache.snapshot()["dropped"] == 1

    async def test_streaming_size_limit_and_failure_retry(self):
        cache = await self.make_cache(max_image_bytes=32)
        assert await cache.read_image({"source_url": self.url("oversized")}) is None
        assert await cache.read_image({"source_url": self.url("error")}) is None
        assert await cache.read_image({"source_url": self.url("error")}) is None
        assert self.requests.count("error") == 1
        self.now += 31
        assert await cache.read_image({"source_url": self.url("error")}) is None
        assert self.requests.count("error") == 2
        assert await cache.get_recent_group_images(group_id="__inputs__") == []

    async def test_timeout_does_not_stop_later_images(self):
        cache = await self.make_cache(timeout=0.5)
        self.blocked["slow"] = asyncio.Event()
        assert await cache.read_image({"source_url": self.url("slow")}) is None
        assert await cache.read_image({"source_url": self.url("healthy")}) is not None

    async def test_disk_budget_evicts_oldest_and_references(self):
        cache = await self.make_cache(max_cache_bytes=24)
        self.observe("one", cache=cache)
        await cache.drain()
        self.now += 1
        self.observe("two", cache=cache)
        await cache.drain()
        self.now += 1
        self.observe("three", cache=cache)
        await cache.drain()
        recent = await cache.get_recent_group_images(group_id=1)
        assert [record["source_url"] for record in recent] == [self.url("three"), self.url("two")]
        assert sum(p.stat().st_size for p in (Path(self.temp.name) / "1").glob("img_sha256_*")) <= 24

    async def test_sqlite_writer_contention_does_not_block_event_loop(self):
        connection = sqlite3.connect(Path(self.temp.name) / "0" / "index.sqlite3")
        connection.execute("BEGIN IMMEDIATE")
        try:
            self.observe("disk-contention")
            await self.wait_started("disk-contention")
            heartbeat = asyncio.Event()
            start = perf_counter()
            asyncio.get_running_loop().call_later(0.1, heartbeat.set)
            await heartbeat.wait()
            assert perf_counter() - start < 1.0
        finally:
            connection.rollback()
            connection.close()
        await self.cache.drain()
        assert len(await self.cache.get_recent_group_images(group_id=1)) == 1

    async def test_cached_read_survives_restart_without_json_index(self):
        self.observe()
        await self.cache.drain()
        await self.cache.close()
        await self.cache.start()
        recent = await self.cache.get_recent_group_images(group_id=1)
        assert await self.cache.read_image(recent[0]) is not None
        assert len(self.requests) == 1
        assert not (Path(self.temp.name) / "0" / "image_cache_index.json").exists()

    async def test_prefetch_can_be_disabled_without_disabling_explicit_images(self):
        cache = await self.make_cache(prefetch=False)
        assert self.observe(cache=cache) == 0
        assert self.requests == []
        assert await cache.read_image({"source_url": self.url()}) is not None

    async def test_close_finishes_waiters_and_stops_downloads(self):
        self.blocked["same"] = asyncio.Event()
        task = asyncio.create_task(self.cache.read_image({"source_url": self.url()}))
        self.tasks.append(task)
        await self.wait_started("same")
        await self.cache.close()
        assert await task is None
        assert self.cache.snapshot()["pending"] == 0


if __name__ == "__main__":
    unittest.main()
