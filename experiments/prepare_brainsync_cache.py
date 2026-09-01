"""Validate BrainSync BIDS raw sessions and generate an attested epoch cache."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.brainsync import (  # noqa: E402
    InvalidSessionPolicy,
    load_brainsync_sessions_resilient,
)
from data.epochs import save_epoch_dataset  # noqa: E402


def _environment_path(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        action="append",
        default=None,
        help="Repeat for multiple decisions; defaults to BRAIN_SYNC_SESSION_DIR.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="NPZ output; defaults to BRAIN_SYNC_OUTPUT_DIR/brainsync_epochs.npz.",
    )
    parser.add_argument(
        "--invalid-session",
        choices=tuple(policy.value for policy in InvalidSessionPolicy),
        default=InvalidSessionPolicy.ERROR.value,
    )
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()

    session_dirs = args.session_dir or [_environment_path("BRAIN_SYNC_SESSION_DIR")]
    if not session_dirs or any(value is None for value in session_dirs):
        parser.error("--session-dir or BRAIN_SYNC_SESSION_DIR is required")
    output_value = args.output
    if output_value is None:
        output_dir = _environment_path("BRAIN_SYNC_OUTPUT_DIR")
        if output_dir is None:
            parser.error("--output or BRAIN_SYNC_OUTPUT_DIR is required")
        output_value = str(Path(output_dir) / "brainsync_epochs.npz")

    print("PROGRESS 5", flush=True)
    result = load_brainsync_sessions_resilient(
        [str(value) for value in session_dirs],
        invalid_session_policy=args.invalid_session,
    )
    print("PROGRESS 85", flush=True)
    output = save_epoch_dataset(
        output_value,
        result.dataset,
        compressed=not args.uncompressed,
    )
    for failure in result.failures:
        print(
            f"[skipped-session] {failure.session_dir}: "
            f"{failure.error_type}: {failure.message}",
            flush=True,
        )
    print(f"[prepared] {output} X={result.dataset.X.shape}", flush=True)
    print(f"[record] {output.with_suffix('.record.json')}", flush=True)
    print("PROGRESS 100", flush=True)


if __name__ == "__main__":
    main()
