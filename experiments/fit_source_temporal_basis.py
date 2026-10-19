"""Freeze a supervised time basis from explicit source cache rows.

Plan n2p3_source_temporal_plan/1: rank, excluded_target_studies, sources
[{cache, rows}] where rows is an integer list or the explicit token "all".
All selected source participants must be outside the declared held-out studies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.temporal_transfer import TemporalBasis  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402
from data.identity import source_dataset_ids_from_identity  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def fit_plan(plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "n2p3_source_temporal_plan/1":
        raise ValueError("unsupported source temporal plan schema")
    excluded = plan["excluded_target_studies"]
    if not isinstance(excluded, list) or not excluded or any(not isinstance(x, str) or not x for x in excluded):
        raise ValueError("source temporal fit requires explicit excluded target studies")
    if output.exists():
        raise ValueError("source temporal output already exists; do not overwrite frozen state")
    scopes = []
    for item in plan["sources"]:
        path = Path(item["cache"])
        path = path if path.is_absolute() else plan_path.resolve().parent / path
        dataset = load_epoch_dataset(path, validation="attested", require_labels=True)
        rows = np.arange(dataset.n_epochs) if item["rows"] == "all" else np.asarray(item["rows"])
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows) or np.any(rows < 0) or np.any(rows >= dataset.n_epochs):
            raise ValueError("source temporal rows must be explicit and valid")
        table = dataset.identity_table.subset(np.unique(dataset.subject_ids[rows]).astype(str).tolist())
        if set(source_dataset_ids_from_identity(table)) & set(excluded):
            raise ValueError("source temporal fit includes an excluded target study")
        scopes.append((dataset, rows))
    basis = TemporalBasis.fit_source(scopes, rank=plan["rank"])
    basis.verify_source(scopes)
    atomic_write_json(output, basis.record())
    restored = TemporalBasis.from_record(json.loads(output.read_text(encoding="utf-8")))
    restored.verify_source(scopes)
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    fit_plan(args.plan, args.output)


if __name__ == "__main__":
    main()
