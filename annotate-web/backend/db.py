import json
import os
from pathlib import Path

import aiosqlite

_db: aiosqlite.Connection | None = None
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _db_path() -> Path:
    # Single source of truth: DATABASE_PATH env var (default lives next to the
    # image storage so the whole annotate stack is one volume).
    return Path(os.environ.get("DATABASE_PATH", "/data/annotate.db"))


async def init_db() -> None:
    global _db
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _db = await aiosqlite.connect(path)
    _db.row_factory = aiosqlite.Row
    # WAL gives concurrent readers a chance during writes and is the standard
    # choice for an app-owned sqlite file on a persistent disk.
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")
    for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
        await _db.executescript(migration.read_text())
    await _db.commit()


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None


def conn() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("db not initialised")
    return _db


async def insert_submission(sub_id: str, ip_hash: str, metadata: dict, storage_path: str) -> None:
    await conn().execute(
        """
        INSERT INTO submissions (id, ip_hash, status, metadata, storage_path)
        VALUES (?, ?, 'base', ?, ?)
        """,
        (sub_id, ip_hash, json.dumps(metadata), storage_path),
    )
    await conn().commit()


async def mark_refined(sub_id: str) -> bool:
    cur = await conn().execute(
        "UPDATE submissions SET status = 'refined' WHERE id = ?",
        (sub_id,),
    )
    await conn().commit()
    return cur.rowcount > 0


async def get_storage_path(sub_id: str) -> str | None:
    async with conn().execute(
        "SELECT storage_path FROM submissions WHERE id = ?", (sub_id,)
    ) as cur:
        row = await cur.fetchone()
        return row["storage_path"] if row else None
