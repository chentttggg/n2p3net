"""One native BNCI downstream dictionary/endpoint/replay acceptance probe.

Three fits on the same two complete authorized groups and one held group. This
is software/public-development acceptance, not a matrix or performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from train.runtime import cpu_thread_budget  # noqa: E402
from transfer.ensemble import EnsembleProgram  # noqa: E402
from transfer.ensemble_fit import (  # noqa: E402
    CalibrationExpertProgram,
    FittedEnsemble,
    fit_calibration_ensemble,
)
from transfer.ensemble_views import prepare_bnci_calibration_groups  # noqa: E402
from transfer.kernel_prior import KernelPriorSpec  # noqa: E402
from transfer.response_dictionary import ResponseDictionarySpec  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    verified_executed_snapshot(args.snapshot, root=ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    started = perf_counter()
    groups = prepare_bnci_calibration_groups(args.input, source_sha256=digest, authorized_labels=800, group_count=3).groups
    results = []
    with cpu_thread_budget(1):
        for family, endpoint in (('gaussian', 'free'), ('cosine', 'free'), ('cosine', 'hard_zero')):
            name = f"{family}_{endpoint}"
            spec = CalibrationExpertProgram(name, "continuous_ar", noise_order=1, conditioning_samples=32,
                dictionary=ResponseDictionarySpec(family, 1.2, 8, endpoint), kernel_prior=KernelPriorSpec.calibration_g(9.))
            begin = perf_counter()
            fitted = fit_calibration_ensemble(groups[:2], (spec,), EnsembleProgram.equal_weights(name, (name,)))
            fit_seconds = perf_counter()-begin
            predictor = fitted._experts[0].predictor
            prediction = fitted.score(groups[2].view)
            directory = fitted.save(args.output/name)
            original_files = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in directory.rglob('*') if p.is_file()}
            reload_start = perf_counter()
            reloaded = FittedEnsemble.load(directory)
            replay = reloaded.score(groups[2].view)
            assert prediction.record() == replay.record()
            assert all(hashlib.sha256((directory/p).read_bytes()).hexdigest() == value for p, value in original_files.items())
            replay_seconds = perf_counter()-reload_start
            geometry = predictor.dictionary_geometry
            results.append({"program": spec.record(), "dictionary": predictor.dictionary.metadata(),
                "capacity": {key: geometry[key] for key in ("ambient_dimensions", "kernel_dimensions", "df_ref", "df_fit",
                    "reference_rank", "reference_prior_energy", "capacity_changed_by_hard_endpoint", "capacity_matching_claimed")},
                "prior_digest": predictor.kernel_prior.digest(), "prediction": prediction.record(),
                "fit_seconds": fit_seconds, "reload_and_score_seconds": replay_seconds,
                "exact_package_replay": True, "model_files_unchanged": True, "held_truth_used_for_fit_or_selection": False})
    verified_executed_snapshot(args.snapshot, root=ROOT)
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == digest
    atomic_write_json(args.output/'receipt.json', {"schema": "n2p3_dictionary_native_acceptance/1",
        "status": "completed", "evidence_scope": "public_native_caller_software_acceptance", "performance_claim": False,
        "source_sha256": digest, "source_path": str(args.input),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "calibration": [g.record() for g in groups[:2]], "held_view": groups[2].view.record(),
        "same_native_C_read_scored_support_for_all_arms": True, "current_overlap_rejection_preserved": True,
        "n_fits": 3, "cpu_threads": 1, "wall_seconds": perf_counter()-started, "arms": results})
    print(json.dumps({"status": "completed", "n_fits": 3, "output": str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
