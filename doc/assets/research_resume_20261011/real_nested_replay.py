"""Accept the recovered nested/dictionary package chain on attested native C.

This is a public-data software diagnostic, not an efficacy test or a comparison
of equal-capacity dictionaries. The candidate roster is fixed before fitting.
Source and MAT identities are saved before execution. Replay uses the original
independent-prefix owner to prepare a label-free future queue beyond C800.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from data.epochs import PreprocessingSpec  # noqa: E402
from research.calibration_selection import (  # noqa: E402
    CalibrationProgramCandidate,
    FittedCalibrationSelection,
    fit_nested_calibration_selection,
)
from research.contracts import semantic_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.ensemble import EnsembleProgram  # noqa: E402
from transfer.ensemble_fit import CalibrationExpertProgram  # noqa: E402
from transfer.ensemble_views import (  # noqa: E402
    prepare_bnci_calibration_groups,
    prepare_bnci_independent_prefixes,
)
from transfer.response_dictionary import ResponseDictionarySpec  # noqa: E402


def declared_candidates():
    baseline = CalibrationExpertProgram("raw", "continuous_ar", noise_order=0,
        dictionary=ResponseDictionarySpec("gaussian", 1.2, 2))
    raised = CalibrationExpertProgram("raw", "continuous_ar", noise_order=0,
        dictionary=ResponseDictionarySpec("raised_cosine", 1.2, 3, "hard_zero"))
    program = EnsembleProgram.equal_weights("raw_nested_software", ("raw",))
    return (
        CalibrationProgramCandidate("gaussian_fixed", (baseline,), program),
        CalibrationProgramCandidate("gaussian_temperature", (baseline,), program, "grouped_temperature"),
        CalibrationProgramCandidate("raised_fixed", (raised,), program),
    )


def run(mat_path, mat_sha256, output, *, producer_directory=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").rglob("*.py"))}
    source[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    candidates = declared_candidates()
    spec = {"schema": "n2p3_native_nested_replay/1", "evidence_scope": "public_calibration_software_diagnostic",
        "mat": {"path": str(Path(mat_path).resolve()), "sha256": mat_sha256},
        "authorized_labels": 800, "complete_groups": 6, "seed": None,
        "randomness_scope": "deterministic_target_only_no_neural_training",
        "candidates": [candidate.record() for candidate in candidates],
        "baseline_name": "gaussian_fixed", "native_threads": 1,
        "runtime": {"python": sys.executable, "python_version": sys.version,
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "source_sha256": source, "source_manifest_sha256": semantic_sha256(source),
        "replay_scope": "independent_future_queue_after_C800", "independent_test_claimed": False,
        "producer_directory": None if producer_directory is None else str(Path(producer_directory).resolve()),
        "authorization_preprocessing": PreprocessingSpec().record(),
        "equal_capacity_comparison_claimed": False, "product_promotion": False}
    atomic_write_json(output / "spec.json", spec)
    started = perf_counter()
    with cpu_thread_budget(1):
        prepared = prepare_bnci_calibration_groups(mat_path, source_sha256=mat_sha256,
            authorized_labels=800, group_count=6)
        atomic_write_json(output / "input_ledger.json", prepared.ledger)
        if producer_directory is None:
            fitted = fit_nested_calibration_selection(prepared.groups, candidates, baseline_name="gaussian_fixed")
            producer = {"kind": "new_fit", "source_manifest_sha256": spec["source_manifest_sha256"]}
        else:
            original = Path(producer_directory)
            original_spec = json.loads((original / "spec.json").read_bytes())
            if (original_spec["mat"]["sha256"] != mat_sha256
                    or semantic_sha256(original_spec["candidates"]) != semantic_sha256(spec["candidates"])):
                raise ValueError("saved native producer differs from the fixed input/program declaration")
            # The original harness failed after saving its legitimate fit.
            # Reuse those exact bytes; do not repeat fits or relabel their source.
            source_changes = [name for name, digest in original_spec["source_sha256"].items()
                if name.startswith("src/") and source.get(name) != digest]
            if source_changes:
                raise ValueError(f"producer source differs from the frozen replay core: {source_changes}")
            payload = json.loads((original / "package/calibration_selection.json").read_bytes())
            fitted = FittedCalibrationSelection.load(original / "package",
                expected_selection_sha256=semantic_sha256(payload["record"]))
            producer = {"kind": "preserved_original_fit", "spec_sha256": semantic_sha256(original_spec),
                "source_manifest_sha256": original_spec["source_manifest_sha256"],
                "fit_attempts": 0, "selection_sha256": semantic_sha256(payload["record"])}
        digest = semantic_sha256(fitted.record)
        package = fitted.save(output / "package")
        restored = FittedCalibrationSelection.load(package, expected_selection_sha256=digest)
        future = prepare_bnci_independent_prefixes(mat_path, source_sha256=mat_sha256,
            budgets=(800,), profiles=(), authorization_preprocessing=PreprocessingSpec())
        atomic_write_json(output / "future_ledger.json", future.ledger)
        replays = []
        for view in future.decisions:
            before, after = fitted.score(view).record(), restored.score(view).record()
            if before != after:
                raise AssertionError("native nested final package replay changed")
            replays.append({"decision_id": view.decision_id,
                "physical_support_sha256": view.physical_support_digest,
                "raw_scope": list(view.raw_scope), "available_sample": view.available_sample,
                "probability_available": before["log_probability"] is not None,
                "evidence_sha256": semantic_sha256(before), "exact_replay": True})
    changed = [name for name, digest in source.items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"executed source changed during native acceptance: {changed}")
    report = {"schema": spec["schema"], "evidence_scope": spec["evidence_scope"],
        "spec_sha256": semantic_sha256(spec), "input_ledger_sha256": semantic_sha256(prepared.ledger),
        "producer": producer, "future_ledger_sha256": semantic_sha256(future.ledger),
        "selection_sha256": digest, "outer_requested_count": len(fitted.record["outer_rows"]),
        "outer_statuses": [row["status"] for row in fitted.record["outer_rows"]],
        "selected_candidate": fitted.record["selected_candidate"]["name"],
        "original_fit_counters": {key: fitted.record[key] for key in
            ("actual_fixed_fit_states", "fit_calls", "exact_subset_cache_hits")},
        "new_fits_in_this_invocation": None if producer_directory is None else 0,
        "replays": replays, "wall_seconds": perf_counter() - started,
        "source_bytes_unchanged": True, "independent_test_claimed": False,
        "equal_capacity_comparison_claimed": False, "product_promotion": False}
    atomic_write_json(output / "report.json", report)
    print(f"native nested: {len(replays)} exact package replays; report={output / 'report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--mat-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-directory", type=Path)
    arguments = parser.parse_args()
    run(arguments.mat, arguments.mat_sha256, arguments.output, producer_directory=arguments.producer_directory)
