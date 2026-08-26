"""Shared training-progress interfaces for baseline runners.

The runner owns the output directory and fold numbering. Models only expose a
small configuration interface and emit the same epoch event schema, so a new
dataset runner does not need model-specific environment variables.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

EpochProgressCallback = Callable[[dict[str, object]], None]


def make_epoch_progress_callback(
    directory: str | Path | None,
    fold_id: int | None,
) -> EpochProgressCallback | None:
    """Create a flushed per-fold JSONL sink.

    ``N2P3NET_EPOCH_PROGRESS_DIR`` remains a read-only compatibility fallback
    for older standalone callers and tests. New runners pass ``directory``
    explicitly through :class:`Baseline`.
    """

    if directory is None:
        directory = os.environ.get("N2P3NET_EPOCH_PROGRESS_DIR")
    if directory is None or fold_id is None:
        return None
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    fold = int(fold_id)
    target = target_dir / f"fold_{fold}.jsonl"
    # Creating a new fold sink marks a fresh attempt even if training fails
    # before epoch 0. Do not leave stale rows visible to the monitor.
    target.write_text("", encoding="utf-8")

    def write(event: dict[str, object]) -> None:
        row = {
            **event,
            "type": "epoch",
            "fold": fold,
            "pid": os.getpid(),
            "ts": datetime.now(UTC).isoformat(),
        }
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    return write
