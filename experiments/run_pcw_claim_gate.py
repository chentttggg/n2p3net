"""Evaluate the PCW routing-window diagnostic gate from evidence JSON.

The gate can only promote the PCW ``tau`` description between
``fold-calibrated routing window`` and
``fold-calibrated discriminative routing window``. It never grants a
physiological single-trial latency claim; that claim belongs exclusively to
``measurement.LatencyMeasurement``.

Expected input keys:
  true_latency_ms, predicted_latency_ms,
  tau_gradient_norms, reference_gradient_norms, tau0_before_ms, tau0_after_ms,
  split_half_first_ms, split_half_second_ms,
  pcw_score, fixed_window_score, mean_pool_score.
"""

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

from models.pcw_validation import (  # noqa: E402
    ablation_evidence,
    evaluate_pcw_claim_gate,
    gradient_health,
    latency_recovery,
    split_half_stability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCW routing-window diagnostic gate")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    report = evaluate_pcw_claim_gate(
        latency_recovery(
            np.asarray(evidence["true_latency_ms"]),
            np.asarray(evidence["predicted_latency_ms"]),
        ),
        gradient_health(
            np.asarray(evidence["tau_gradient_norms"]),
            np.asarray(evidence["reference_gradient_norms"]),
            np.asarray(evidence["tau0_before_ms"]),
            np.asarray(evidence["tau0_after_ms"]),
        ),
        split_half_stability(
            np.asarray(evidence["split_half_first_ms"]),
            np.asarray(evidence["split_half_second_ms"]),
        ),
        ablation_evidence(
            evidence["pcw_score"],
            evidence["fixed_window_score"],
            evidence["mean_pool_score"],
        ),
    )
    payload = report.to_dict()
    output = (
        Path(args.output) if args.output else Path(args.evidence).with_name("pcw_claim_gate.json")
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={report.passed} allowed_claim={report.allowed_claim}")
    if report.failures:
        print(f"failures={','.join(report.failures)}")
    print(output)


if __name__ == "__main__":
    main()
