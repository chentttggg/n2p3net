"""Run an ordered queue of existing dataset science commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research.dataset_queue import (  # noqa: E402
    DatasetQueue,
    execute_queue_entry,
    plan_from_record,
    supervise_queue_entry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", type=Path)
    modes.add_argument("--supervise-entry", type=Path)
    modes.add_argument("--execute-entry", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if args.supervise_entry is not None or args.execute_entry is not None:
        if args.request_sha256 is None:
            parser.error("internal supervision requires a bound request SHA")
        if args.supervise_entry is not None:
            return supervise_queue_entry(args.supervise_entry, args.request_sha256)
        return execute_queue_entry(args.execute_entry, args.request_sha256)
    if args.output is None:
        parser.error("queue execution requires --output")
    record = json.loads(args.plan.read_bytes())
    plan = plan_from_record(record, output_root=args.output)
    result = DatasetQueue(plan, args.output, poll_seconds=args.poll_seconds).run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
