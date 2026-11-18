"""SQLite metadata and content-addressed files; called only from the cache's I/O worker."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sqlite3

from .response_utils import detect_image_media_type


class ImageStore:
    def __init__(self, directory, *, ttl, max_bytes, max_records, max_entries=4096):
        self.directory = Path(directory)
        self.database = self.directory / "index.sqlite3"
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.max_records = max_records
        self.max_entries = max_entries
        self._last_cleanup_at: float | None = None

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self, now):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL, size INTEGER NOT NULL, accessed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS urls (
                    url TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images ON DELETE CASCADE,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refs (
                    group_id TEXT NOT NULL, user_id TEXT NOT NULL, image_id TEXT NOT NULL REFERENCES images ON DELETE CASCADE,
                    message_id TEXT NOT NULL, source_url TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
                    PRIMARY KEY (group_id, user_id, image_id)
                );
                CREATE INDEX IF NOT EXISTS refs_recent ON refs (group_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS refs_expiry ON refs (expires_at);
                CREATE INDEX IF NOT EXISTS refs_image ON refs (image_id);
                CREATE INDEX IF NOT EXISTS urls_image ON urls (image_id);
                CREATE INDEX IF NOT EXISTS urls_expiry ON urls (expires_at);
            """)
        self.cleanup(now)
        # Recover files left by an interrupted write, without touching the legacy cache.
        with self._connect() as db:
            known = {Path(row[0]).name for row in db.execute("SELECT file_path FROM images")}
        for path in self.directory.glob("img_sha256_*"):
            if path.name not in known:
                path.unlink(missing_ok=True)

    def lookup(self, url, now):
        with self._connect() as db:
            row = db.execute(
                "SELECT images.* FROM urls JOIN images USING(image_id) WHERE url=? AND expires_at>?", (url, now)
            ).fetchone()
            if row is None:
                return None
            if not Path(row["file_path"]).is_file():
                db.execute("DELETE FROM images WHERE image_id=?", (row["image_id"],))
                return None
            return dict(row)

    def save(self, url, data, refs, now):
        mime = detect_image_media_type(data)
        if mime is None or len(data) > self.max_bytes:
            return None
        image_id = "img_sha256_" + hashlib.sha256(data).hexdigest()[:16]
        path = self.directory / image_id
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        with self._connect() as db:
            db.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?) ON CONFLICT(image_id) DO UPDATE SET accessed_at=excluded.accessed_at",
                (image_id, str(path), mime, len(data), now),
            )
            db.execute(
                "INSERT INTO urls VALUES (?, ?, ?) ON CONFLICT(url) DO UPDATE SET "
                "image_id=excluded.image_id, expires_at=excluded.expires_at",
                (url, image_id, now + self.ttl),
            )
            self._attach(db, image_id, url, refs)
        return {"image_id": image_id, "file_path": str(path), "mime_type": mime}

    def _attach(self, db, image_id, url, refs):
        for ref in refs:
            group, user, message, seen = ref
            db.execute(
                """INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id,user_id,image_id) DO UPDATE SET
                message_id=excluded.message_id, source_url=excluded.source_url,
                created_at=MAX(refs.created_at,excluded.created_at), expires_at=MAX(refs.expires_at,excluded.expires_at)""",
                (group, user, image_id, message, url, seen, seen + self.ttl),
            )
            db.execute(
                """DELETE FROM refs WHERE group_id=? AND user_id=? AND image_id IN
                (SELECT image_id FROM refs WHERE group_id=? AND user_id=?
                 ORDER BY created_at DESC, image_id LIMIT -1 OFFSET ?)""",
                (group, user, group, user, self.max_records),
            )
            db.execute(
                """DELETE FROM refs WHERE group_id=? AND image_id IN
                (SELECT image_id FROM refs WHERE group_id=? GROUP BY image_id
                 ORDER BY MAX(created_at) DESC, image_id LIMIT -1 OFFSET ?)""",
                (group, group, self.max_records),
            )

    def attach(self, record, url, refs, now):
        needs_cleanup = (
            self._last_cleanup_at is None or (now - self._last_cleanup_at) >= 30.0
        )
        with self._connect() as db:
            if db.execute("SELECT 1 FROM images WHERE image_id=?", (record["image_id"],)).fetchone() is None:
                return None
            self._attach(db, record["image_id"], url, refs)
            db.execute("UPDATE urls SET expires_at=? WHERE url=?", (now + self.ttl, url))
            db.execute("UPDATE images SET accessed_at=? WHERE image_id=?", (now, record["image_id"]))
            if not needs_cleanup:
                total = db.execute("SELECT COALESCE(SUM(size),0) FROM images").fetchone()[0]
                needs_cleanup = total > self.max_bytes
        if needs_cleanup:
            self.cleanup(now)
        return self.lookup(url, now)

    def cleanup(self, now):
        self._last_cleanup_at = now
        deleted = []
        with self._connect() as db:
            db.execute("DELETE FROM refs WHERE expires_at<=?", (now,))
            db.execute("DELETE FROM urls WHERE expires_at<=?", (now,))
            db.execute(
                "DELETE FROM urls WHERE url IN (SELECT url FROM urls ORDER BY expires_at DESC, url LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )
            db.execute(
                "DELETE FROM refs WHERE rowid IN (SELECT rowid FROM refs ORDER BY created_at DESC, rowid LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )
            orphans = list(
                db.execute(
                    "SELECT image_id,file_path FROM images WHERE NOT EXISTS "
                    "(SELECT 1 FROM refs WHERE refs.image_id=images.image_id)"
                )
            )
            for row in orphans:
                deleted.append(row["file_path"])
                db.execute("DELETE FROM images WHERE image_id=?", (row["image_id"],))
            total = db.execute("SELECT COALESCE(SUM(size),0) FROM images").fetchone()[0]
            for row in list(db.execute("SELECT image_id,file_path,size FROM images ORDER BY accessed_at, image_id")):
                if total <= self.max_bytes:
                    break
                total -= row["size"]
                deleted.append(row["file_path"])
                db.execute("DELETE FROM images WHERE image_id=?", (row["image_id"],))
        for name in deleted:
            Path(name).unlink(missing_ok=True)

    def recent(self, group, user, limit, offset, now):
        self.cleanup(now)
        with self._connect() as db:
            sql = """SELECT images.image_id,images.file_path,images.mime_type,refs.group_id,refs.user_id,
                refs.message_id,refs.source_url,refs.created_at FROM refs JOIN images USING(image_id)
                WHERE refs.group_id=?"""
            args = [group]
            if user is not None:
                sql += " AND refs.user_id=?"
                args.append(user)
            sql += " ORDER BY refs.created_at DESC, refs.rowid DESC"
            rows = db.execute(sql, args)
            result, seen = [], set()
            for row in rows:
                if row["image_id"] in seen or not Path(row["file_path"]).is_file():
                    continue
                seen.add(row["image_id"])
                result.append(dict(row))
            return result[offset : offset + limit]

    def read(self, image_id, max_bytes, now):
        with self._connect() as db:
            row = db.execute("SELECT * FROM images WHERE image_id=?", (image_id,)).fetchone()
            if row is None:
                return None
            path = Path(row["file_path"])
            try:
                if path.stat().st_size > max_bytes:
                    return None
                data = path.read_bytes()
            except FileNotFoundError:
                db.execute("DELETE FROM images WHERE image_id=?", (image_id,))
                return None
            db.execute("UPDATE images SET accessed_at=? WHERE image_id=?", (now, image_id))
            return data, row["mime_type"]
