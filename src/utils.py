import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("tomato")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Unsafe path: " + str(relative))
    return p


def files_under(root):
    return sorted(
        p for p in Path(root).rglob("*") if p.is_file() and p.name != ".gitkeep"
    )


def require_not_exposed(c):
    from .config_loader import path_for

    if (path_for(c, "test_lock_manifest").parent / "final_exposure.json").exists():
        raise RuntimeError(
            "Final evaluation has begun. Training/re-splitting is blocked in this experiment. "
            "Start a separately documented experiment with genuinely unseen test sources; "
            "reshuffling previously seen images does not restore an untouched holdout."
        )
