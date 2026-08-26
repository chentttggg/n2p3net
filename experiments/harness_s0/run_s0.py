"""Run the S0 counterexample harness and write a machine-readable report.

Usage:
    .venv\\Scripts\\python.exe experiments\\harness_s0\\run_s0.py --out tmp\\s0_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.harness_s0.checks import run_all_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="S0 counterexample harness")
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = {
        "schema": "n2p3net_s0_harness_report/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "checks": run_all_checks(),
    }
    for name, check in report["checks"].items():
        if not bool(check["passed"]):
            report["passed"] = False
            print(f"[s0] FAIL {name}: {check}", flush=True)
        else:
            print(f"[s0] PASS {name}", flush=True)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[s0] report: {path}", flush=True)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
