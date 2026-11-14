"""Bounded image prefetch and demand downloads, with SQLite work off the event loop."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
import time
from urllib.parse import urlsplit

import aiohttp
from nonebot.log import logger
import nonebot_plugin_localstore as store

from .Config import config_parser
from .image_cache_store import ImageStore


@dataclass
class _Job:
    url: str
    proxy: str | None
    owner: str
    future: asyncio.Future
    demand: bool = False
    running: bool = False
    refs: dict = field(default_factory=dict)


class ImageCache:
    """observe() never waits; all downloads, disk access and eviction live here.

    start()/close() belong to bot lifecycle. read_image() is the bounded demand
    path for current images, quoted images, recent images and avatars.
    """

    def __init__(
        self,
        directory=None,
        *,
        workers=4,
        queue_size=64,
        per_group_pending=8,
        max_image_bytes=10 * 1024 * 1024,
        max_cache_bytes=256 * 1024 * 1024,
        expire_seconds=3600,
        max_records=20,
        timeout=15,
        retry_seconds=30,
        prefetch=True,
        session_factory=None,
        clock=time.time,
    ):
        directory = directory or store.get_plugin_data_dir() / "image_cache_v2"
        self.store = ImageStore(directory, ttl=expire_seconds, max_bytes=max_cache_bytes, max_records=max_records)
        self.workers = max(1, int(workers))
        self.queue_size = max(self.workers, int(queue_size))
        self.per_group_pending = max(1, int(per_group_pending))
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.timeout = max(0.1, float(timeout))
        self.retry_seconds = max(0, float(retry_seconds))
        self.prefetch = prefetch
        self._clock = clock
        self._session_factory = session_factory
        self._session = None
        self._tasks = []
        self._jobs = OrderedDict()
        self._finishing = set()
        self._failures = OrderedDict()
        self._active_groups = set()
        self._wake = asyncio.Event()
        self._io_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._dropped = 0
        self._failed = 0

    async def _io(self, function, *args):
        # Finish an in-progress transaction before releasing the lock on cancellation.
        async with self._io_lock:
            future = asyncio.get_running_loop().run_in_executor(None, partial(function, *args))
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                await asyncio.shield(future)
                raise

    async def start(self):
        async with self._lifecycle_lock:
            if self._started:
                return
            await self._io(self.store.initialize, self._clock())
            if self._session_factory is None:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    connector=aiohttp.TCPConnector(limit=self.workers, limit_per_host=self.workers),
                )
            else:
                self._session = self._session_factory()
            self._closing = False
            self._started = True
            self._tasks = [asyncio.create_task(self._worker(), name="image-cache-worker") for _ in range(self.workers)]
            self._tasks.append(asyncio.create_task(self._maintenance(), name="image-cache-cleanup"))

    def _detach(self, key, job):
        if self._jobs.get(key) is job:
            del self._jobs[key]

    def _submit(self, url, *, group_id, user_id, message_id, proxy=None, demand=False):
        if not self._started or self._closing or not isinstance(url, str) or len(url) > 8192:
            return None
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return None
        except ValueError:
            return None
        key = (url, proxy)
        now = self._clock()
        if self._failures.get(key, 0) > now:
            return None
        group, user = str(group_id), str(user_id)
        job = self._jobs.get(key)
        ref_key = (group, user)
        if job is None or ref_key not in job.refs:
            if sum(len(pending.refs) for pending in self._jobs.values()) >= self.queue_size * self.per_group_pending:
                self._dropped += 1
                return None
            count = sum(sum(g == group for g, _ in pending.refs) for pending in self._jobs.values())
            if count >= self.per_group_pending + int(demand):
                self._dropped += 1
                return None
        if job is None:
            if len(self._jobs) + len(self._finishing) >= self.queue_size:
                # Reserve progress for explicit requests by discarding queued prefetch.
                victim = next(((k, j) for k, j in self._jobs.items() if not j.running and not j.demand), None)
                if not demand or victim is None:
                    self._dropped += 1
                    return None
                victim_key, victim_job = victim
                self._detach(victim_key, victim_job)
                victim_job.future.set_result(None)
                self._dropped += 1
            job = _Job(url, proxy, group, asyncio.get_running_loop().create_future(), demand=demand)
            self._jobs[key] = job
        job.demand = job.demand or demand
        # Repeated messages from one sender replace metadata rather than growing a waiter list.
        job.refs[ref_key] = (group, user, str(message_id), now)
        self._wake.set()
        return job.future

    def observe(self, *, group_id, user_id, images, message_id=0):
        """Best-effort background prefetch; bounded synchronous queue admission only."""
        if not self.prefetch:
            return 0
        accepted = 0
        for image in images[: self.per_group_pending]:
            if self._submit(image.get("source_url"), group_id=group_id, user_id=user_id, message_id=message_id) is not None:
                accepted += 1
        return accepted

    def snapshot(self):
        return {
            "pending": len(self._jobs) + len(self._finishing),
            "active": len(self._active_groups),
            "dropped": self._dropped,
            "failed": self._failed,
        }

    async def drain(self):
        """Wait for currently admitted work, useful for orderly callers and tests."""
        while self._jobs or self._finishing:
            futures = {job.future for job in self._jobs.values()} | self._finishing
            await asyncio.gather(*(asyncio.shield(future) for future in futures))

    async def _next_job(self):
        while not self._closing:
            for demand in (True, False):
                for key, job in self._jobs.items():
                    if not job.running and job.demand == demand and job.owner not in self._active_groups:
                        job.running = True
                        self._active_groups.add(job.owner)
                        return key, job
            self._wake.clear()
            await self._wake.wait()
        return None

    async def _download(self, job):
        # The outer timeout also bounds streaming if a custom connector ignores session timeouts.
        async def receive():
            async with self._session.get(job.url, proxy=job.proxy) as response:
                if response.status != 200:
                    return None
                if response.content_length is not None and response.content_length > self.max_image_bytes:
                    return None
                data = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if len(data) + len(chunk) > self.max_image_bytes:
                        return None
                    data.extend(chunk)
                return bytes(data)

        return await asyncio.wait_for(receive(), timeout=self.timeout)

    async def _worker(self):
        while not self._closing:
            item = await self._next_job()
            if item is None:
                return
            key, job = item
            result = None
            try:
                result = await self._io(self.store.lookup, job.url, self._clock())
                if result is None:
                    data = await self._download(job)
                    if data is not None:
                        result = await self._io(self.store.save, job.url, data, list(job.refs.values()), self._clock())
                # New arrivals now see the stored blob, while this job finishes its reference transaction.
                self._finishing.add(job.future)
                self._detach(key, job)
                if result is not None:
                    result = await self._io(self.store.attach, result, job.url, list(job.refs.values()), self._clock())
                if result is None:
                    self._failed += 1
                    self._failures[key] = self._clock() + self.retry_seconds
                    self._failures.move_to_end(key)
                    while len(self._failures) > self.queue_size * 4:
                        self._failures.popitem(last=False)
            except asyncio.CancelledError:
                result = None
                raise
            except Exception as exc:
                result = None
                # Do not log signed URLs or a traceback for every failed image.
                self._failed += 1
                self._failures[key] = self._clock() + self.retry_seconds
                while len(self._failures) > self.queue_size * 4:
                    self._failures.popitem(last=False)
                logger.warning(f"Image cache request failed ({type(exc).__name__}); retry delayed")
            finally:
                self._detach(key, job)
                self._active_groups.discard(job.owner)
                self._finishing.discard(job.future)
                if not job.future.done():
                    job.future.set_result(result)
                self._wake.set()

    async def _maintenance(self):
        while True:
            await asyncio.sleep(60)
            try:
                await self.cleanup()
            except Exception as exc:
                logger.warning(f"Image cache cleanup failed ({type(exc).__name__})")

    async def cleanup(self):
        await self._io(self.store.cleanup, self._clock())

    async def _recent(self, group_id, user_id, limit, offset):
        if self._closing:
            return []
        try:
            return await self._io(
                self.store.recent, str(group_id), user_id, max(1, min(int(limit), 20)), max(0, int(offset)), self._clock()
            )
        except Exception as exc:
            logger.warning(f"Recent image lookup unavailable ({type(exc).__name__})")
            return []

    async def get_recent_group_images(self, *, group_id, limit=3, offset=0):
        return await self._recent(group_id, None, limit, offset)

    async def get_recent_images(self, *, group_id, user_id, limit=3, offset=0):
        return await self._recent(group_id, str(user_id), limit, offset)

    async def read_image(self, image, *, group_id="__inputs__", user_id=0, message_id=0, proxy=None):
        """Return bounded image bytes or None; cache failures never discard text."""
        if self._closing:
            return None
        try:
            return await self._read_image(image, group_id=group_id, user_id=user_id, message_id=message_id, proxy=proxy)
        except Exception as exc:
            logger.warning(f"Image input unavailable ({type(exc).__name__})")
            return None

    async def _read_image(self, image, *, group_id="__inputs__", user_id=0, message_id=0, proxy=None):
        await self.start()
        if self._closing:
            return None
        if image.get("image_id"):
            result = await self._io(self.store.read, image["image_id"], self.max_image_bytes, self._clock())
            if result is not None:
                return result
        future = self._submit(
            image.get("source_url"), group_id=group_id, user_id=user_id, message_id=message_id, proxy=proxy, demand=True
        )
        if future is None:
            return None
        try:
            record = await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout * 2)
        except asyncio.TimeoutError:
            return None
        if record is None:
            return None
        return await self._io(self.store.read, record["image_id"], self.max_image_bytes, self._clock())

    async def close(self):
        async with self._lifecycle_lock:
            self._closing = True
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            for job in self._jobs.values():
                if not job.future.done():
                    job.future.set_result(None)
            self._jobs.clear()
            self._tasks.clear()
            if self._session is not None:
                await self._session.close()
            self._session = None
            self._started = False


def _configured_cache():
    def setting(name, default):
        value = config_parser.get_config(name)
        return default if value is None else value

    return ImageCache(
        workers=setting("image_cache_workers", 4),
        queue_size=setting("image_cache_queue_size", 64),
        per_group_pending=setting("image_cache_per_group_pending", 8),
        max_image_bytes=setting("image_cache_max_image_bytes", 10 * 1024 * 1024),
        max_cache_bytes=setting("image_cache_max_bytes", 256 * 1024 * 1024),
        expire_seconds=setting("image_cache_expire_seconds", 3600),
        max_records=setting("image_cache_max_records_per_user", 20),
        timeout=setting("image_cache_download_timeout", 15),
        prefetch=setting("image_cache_prefetch", True),
    )


image_cache = _configured_cache()
