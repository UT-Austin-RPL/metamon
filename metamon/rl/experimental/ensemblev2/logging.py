from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class FeatureLogger:
    """Append finished battles (one JSON line each) to a JSONL file.

    Each line is one battle: ``{"members": [...], "num_steps": N, "steps": [...]}``
    where every step holds all gathered features (per-member per-gamma probs and
    Q-values on canonical action indices), the legal actions, and the chosen
    action. This is the on-disk substrate for training future learned deciders.

    Enabled only when ``METAMON_ENSEMBLEV2_LOG_DIR`` is set. Disabled loggers are
    cheap no-ops.
    """

    def __init__(
        self,
        log_dir: Optional[str],
        member_descriptions: list[dict[str, Any]],
        meta: Optional[dict[str, Any]] = None,
    ):
        self.member_descriptions = member_descriptions
        self.meta = meta or {}
        self.enabled = bool(log_dir)
        self._lock = threading.Lock()
        self._battles_written = 0
        self._path: Optional[str] = None
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._path = os.path.join(
                log_dir, f"ensemblev2_{os.getpid()}_{stamp}.jsonl"
            )

    @classmethod
    def from_env(
        cls,
        member_descriptions: list[dict[str, Any]],
        meta: Optional[dict[str, Any]] = None,
    ) -> "FeatureLogger":
        return cls(
            log_dir=os.environ.get("METAMON_ENSEMBLEV2_LOG_DIR"),
            member_descriptions=member_descriptions,
            meta=meta,
        )

    @property
    def path(self) -> Optional[str]:
        return self._path

    def log_battle(
        self, steps: list[dict[str, Any]], won: Optional[bool] = None
    ) -> None:
        if not self.enabled or not steps:
            return
        record = {
            "meta": self.meta,
            "members": self.member_descriptions,
            "num_steps": len(steps),
            "won": won,
            "steps": steps,
        }
        line = json.dumps(record)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._battles_written += 1
