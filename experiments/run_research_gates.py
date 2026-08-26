"""Evaluate the morphology recovery gate from one evidence JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.research_gates import evaluate_morphology_recovery  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="pre-registered Neural-RIDE research gates")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evidence_path = Path(args.evidence)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    morphology = evaluate_morphology_recovery(
        np.asarray(evidence["true_latency_ms"]),
        np.asarray(evidence["predicted_latency_ms"]),
        np.asarray(evidence["true_width_ms"]),
        np.asarray(evidence["predicted_width_ms"]),
        np.asarray(evidence["interval_lower_ms"]),
        np.asarray(evidence["interval_upper_ms"]),
    )
    payload = {
        "schema": "neural_ride_research_gates/2",
        "morphology": morphology.to_dict(),
        "all_passed": morphology.passed,
    }
    output = Path(args.output) if args.output else evidence_path.with_name("research_gates.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"morphology={morphology.passed}")
    print(output)


if __name__ == "__main__":
    main()
