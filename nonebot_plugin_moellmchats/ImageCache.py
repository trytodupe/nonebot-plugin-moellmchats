from __future__ import annotations

import hashlib
from pathlib import Path
import time

import nonebot_plugin_localstore as store
import ujson as json

from .Config import config_parser
from .response_utils import detect_image_media_type

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}

cache_dir: Path = store.get_plugin_data_dir() / "image_cache"
index_file: Path = store.get_plugin_data_file("image_cache_index.json")


class ImageCache:
    def __init__(self):
        self.cache_dir = cache_dir
        self.index_file = index_file
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = self._load()

    def _load(self) -> dict:
        try:
            with open(self.index_file, "r", encoding="utf-8") as file:
                index = json.load(file)
        except FileNotFoundError:
            index = {}
        index.setdefault("records", {})
        index.setdefault("groups", {})
        index.setdefault("source_urls", {})
        return index

    def _save(self):
        with open(self.index_file, "w", encoding="utf-8") as file:
            json.dump(self.index, file, indent=4, ensure_ascii=False)

    def _group_key(self, group_id: int | str, user_id: int | str) -> str:
        return f"{group_id}:{user_id}"

    def _room_key(self, group_id: int | str) -> str:
        return f"{group_id}:*"

    def _max_records_per_user(self) -> int:
        return int(config_parser.get_config("image_cache_max_records_per_user") or 20)

    def _expire_seconds(self) -> int:
        return int(config_parser.get_config("image_cache_expire_seconds") or 3600)

    def _trim_group_records(self, group_key: str):
        record_ids = self.index["groups"].get(group_key, [])
        expire_before = time.time() - self._expire_seconds()
        kept_record_ids = []
        for record_id in record_ids:
            record = self.index["records"].get(record_id)
            if not record:
                continue
            if record.get("created_at", 0) < expire_before:
                self._delete_record(record_id)
                continue
            kept_record_ids.append(record_id)
        max_records = self._max_records_per_user()
        expired_by_limit = kept_record_ids[:-max_records]
        for record_id in expired_by_limit:
            self._delete_record(record_id)
        self.index["groups"][group_key] = kept_record_ids[-max_records:]

    def _delete_record(self, record_id: str):
        record = self.index["records"].pop(record_id, None)
        if not record:
            return
        source_url = record.get("source_url")
        if source_url and self.index["source_urls"].get(source_url) == record_id:
            self.index["source_urls"].pop(source_url, None)
        file_path = record.get("file_path")
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass

    async def cache_images(
        self,
        session,
        *,
        group_id: int | str,
        user_id: int | str,
        images: list[dict],
        proxy: str | None = None,
    ) -> list[dict]:
        cached_records = []
        group_key = self._group_key(group_id, user_id)
        room_key = self._room_key(group_id)
        for image in images:
            source_url = image.get("source_url")
            if not source_url:
                continue
            existing_record_id = self.index["source_urls"].get(source_url)
            existing_record = self.index["records"].get(existing_record_id)
            if existing_record and Path(existing_record["file_path"]).exists():
                group_records = self.index["groups"].setdefault(group_key, [])
                if existing_record_id in group_records:
                    group_records.remove(existing_record_id)
                group_records.append(existing_record_id)
                room_records = self.index["groups"].setdefault(room_key, [])
                if existing_record_id in room_records:
                    room_records.remove(existing_record_id)
                room_records.append(existing_record_id)
                cached_records.append(existing_record)
                continue
            async with session.get(source_url, proxy=proxy, ssl=False) as response:
                if response.status != 200:
                    continue
                image_bytes = await response.read()
                mime_type = detect_image_media_type(image_bytes, response.content_type)
                if not mime_type:
                    continue
            digest = hashlib.sha256(image_bytes).hexdigest()
            record_id = f"img_sha256_{digest[:16]}"
            suffix = IMAGE_EXTENSIONS.get(mime_type, ".img")
            file_path = self.cache_dir / f"{record_id}{suffix}"
            if not file_path.exists():
                file_path.write_bytes(image_bytes)
            record = {
                "image_id": record_id,
                "group_id": str(group_id),
                "user_id": str(user_id),
                "source_url": source_url,
                "file_path": str(file_path),
                "mime_type": mime_type,
                "created_at": time.time(),
            }
            self.index["records"][record_id] = record
            self.index["source_urls"][source_url] = record_id
            group_records = self.index["groups"].setdefault(group_key, [])
            if record_id in group_records:
                group_records.remove(record_id)
            group_records.append(record_id)
            room_records = self.index["groups"].setdefault(room_key, [])
            if record_id in room_records:
                room_records.remove(record_id)
            room_records.append(record_id)
            cached_records.append(record)
        self._trim_group_records(group_key)
        self._trim_group_records(room_key)
        self._save()
        return cached_records

    def get_recent_images(
        self,
        *,
        group_id: int | str,
        user_id: int | str,
        limit: int = 3,
        offset: int = 0,
    ) -> list[dict]:
        group_key = self._group_key(group_id, user_id)
        self._trim_group_records(group_key)
        self._save()
        all_record_ids = list(reversed(self.index["groups"].get(group_key, [])))
        record_ids = all_record_ids[offset : offset + limit]
        records = []
        for record_id in record_ids:
            record = self.index["records"].get(record_id)
            if not record:
                continue
            if not Path(record["file_path"]).exists():
                continue
            records.append(record)
        return records

    def get_recent_group_images(
        self,
        *,
        group_id: int | str,
        limit: int = 3,
        offset: int = 0,
    ) -> list[dict]:
        room_key = self._room_key(group_id)
        self._trim_group_records(room_key)

        records_by_id = {}
        room_record_ids = list(reversed(self.index["groups"].get(room_key, [])))
        for record_id in room_record_ids:
            record = self.index["records"].get(record_id)
            if record:
                records_by_id[record_id] = record

        for record_id, record in self.index["records"].items():
            if str(record.get("group_id")) == str(group_id):
                records_by_id.setdefault(record_id, record)

        records = sorted(
            records_by_id.values(),
            key=lambda record: record.get("created_at", 0),
            reverse=True,
        )
        records = [
            record
            for record in records
            if record.get("file_path") and Path(record["file_path"]).exists()
        ]
        self.index["groups"][room_key] = [record["image_id"] for record in reversed(records)]
        self._save()
        return records[offset : offset + limit]


image_cache = ImageCache()
