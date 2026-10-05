"""Fit dimensionless EA features under a caller-supplied, hash-pinned fit plan.

The hash binds the selected plan bytes; it is not a digital signature or proof
of scientific validity. The experiment/split owner must supply the correct plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.ea_artifact import (  # noqa: E402
    EAFitPlan,
    fit_ea_feature_artifact,
    save_ea_feature_artifact,
)
from data.epochs import load_epoch_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-plan", type=Path, required=True,
                        help="Plan supplied by the experiment/split owner; EA never chooses its own fit rows.")
    parser.add_argument("--fit-plan-sha256", required=True,
                        help="Expected SHA-256 supplied independently by the caller.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve(strict=False)
    if any(path.exists() for path in (output, output.with_suffix(".record.json"))):
        raise FileExistsError("EA cache derivation requires a new output path.")
    plan_bytes = args.fit_plan.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != args.fit_plan_sha256:
        raise ValueError("EA fit plan bytes disagree with caller's expected SHA-256")
    plan = EAFitPlan.from_record(json.loads(plan_bytes))
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=False, validation="attested")
    artifact = fit_ea_feature_artifact(
        dataset, expected_plan=plan, name=args.name,
    )
    saved = save_ea_feature_artifact(
        output, artifact, parent=dataset, expected_plan=plan,
        compressed=not args.uncompressed,
    )
    print(json.dumps({
        "output": str(saved), "artifact_record": str(saved.with_suffix(".record.json")),
        "fit_scope": plan.authorization.fit_scope, "reference_mode": plan.reference_mode,
        "shape": list(artifact.X.shape), "authorization_digest": plan.authorization.record()["authorization_digest"],
        "fit_plan_id": plan.plan_id, "fit_plan_sha256": args.fit_plan_sha256,
    }), flush=True)


if __name__ == "__main__":
    main()
