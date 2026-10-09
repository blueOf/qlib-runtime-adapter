from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import (CONFIG_ROOT, PROJECT_ROOT, RUNS_ROOT, WORKSPACE_ROOT,
                    data_root)

STACK = PROJECT_ROOT
WORKSPACE = WORKSPACE_ROOT
DATA_ROOT = data_root()
MARKET_ROOT = DATA_ROOT
PROVIDER_ROOT = DATA_ROOT / "provider"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def instrument(code):
    code = str(code)
    return code.lower() if len(code) == 8 else ("sh" if code.startswith("6") else "sz") + code
