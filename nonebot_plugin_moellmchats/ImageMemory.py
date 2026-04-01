from pathlib import Path

import nonebot_plugin_localstore as store
import ujson as json

data_file: Path = store.get_plugin_data_file("image_memories.json")


class ImageMemoryStore:
    def __init__(self):
        self.filepath = data_file
        self.memories = self._load()

    def _load(self) -> dict:
        try:
            with open(self.filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as file:
            json.dump(self.memories, file, indent=4, ensure_ascii=False)

    def get_summary(self, image_id: str) -> str | None:
        record = self.memories.get(image_id)
        if record:
            return record.get("summary")
        return None

    def set_summary(self, image_id: str, summary: str, mime_type: str | None = None):
        self.memories[image_id] = {"summary": summary, "mime_type": mime_type}
        self._save()


image_memory_store = ImageMemoryStore()
