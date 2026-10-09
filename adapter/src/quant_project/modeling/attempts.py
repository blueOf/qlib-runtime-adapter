"""Append-only attempt history, independent of successful run publication."""
from __future__ import annotations

import json
from pathlib import Path

from ..common import utc_now
from ..research_registry import _registry_lock
from .contracts import ConfigError, sha256_json


class ModelAttemptLedger:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def events(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, identity, status, **details):
        attempt_id = sha256_json(identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _registry_lock(self.path.with_suffix(".lock")):
            history = [event for event in self.events() if event["attempt_id"] == attempt_id]
            if status == "started" and history:
                raise ConfigError("attempt_id already exists; use a new run identity for another invocation")
            if status != "started" and (not history or history[-1]["status"] != "started"):
                raise ConfigError("attempt can finish only after a single started event")
            record = {"schema": "quant-project-model-attempt-v1", "attempt_id": attempt_id,
                      "identity": identity, "status": status, "recorded_at": utc_now(), **details}
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        return record

    def summary(self, run_id=None):
        events = self.events()
        if run_id:
            events = [event for event in events if event["identity"].get("run_id") == run_id]
        latest = {event["attempt_id"]: event for event in events}
        return {"attempts": len(latest), "fit_attempts": sum(bool(event["identity"].get("counts_as_fit"))
                                                         for event in latest.values()),
                "succeeded": sum(event["status"] == "succeeded" for event in latest.values()),
                "failed": sum(event["status"] == "failed" for event in latest.values()),
                "unfinished": sum(event["status"] == "started" for event in latest.values())}
