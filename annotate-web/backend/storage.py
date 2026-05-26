import hashlib
import json
import os
import uuid
from datetime import date
from pathlib import Path

STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "/data"))
IP_HASH_SALT = os.environ.get("IP_HASH_SALT", "change-me-in-prod")


def hash_ip(ip: str) -> str:
    return hashlib.sha256((ip + IP_HASH_SALT).encode()).hexdigest()


def new_submission_path() -> tuple[str, Path, str]:
    sub_id = str(uuid.uuid4())
    today = date.today().isoformat()
    rel = f"{today}/{sub_id}"
    abs_path = STORAGE_DIR / rel
    abs_path.mkdir(parents=True, exist_ok=True)
    return sub_id, abs_path, rel


def submission_dir(rel_path: str) -> Path:
    return STORAGE_DIR / rel_path


async def write_bytes(dest: Path, data: bytes) -> None:
    dest.write_bytes(data)


def write_metadata_json(dir_path: Path, metadata: dict) -> None:
    (dir_path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
