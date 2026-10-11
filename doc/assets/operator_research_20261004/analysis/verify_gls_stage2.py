"""Separate stage-2 audit: preserve stage-1 evidence and test fixed joint updates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
RESEARCH = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(RESEARCH), str(RESEARCH/"continuous_mechanism")]
from continuous_model import OperatorGLS  # noqa: E402
from verify_gls import fixture, legacy_bridge  # noqa: E402

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402


def main():
    plan = json.loads((RESEARCH/"calibration_group_plan_stage2.json").read_bytes())
    old_path = ROOT/"experiments/runs/operator_research_20261004/calibration_groups_v2/executed_source/doc/assets/operator_research_20261004/continuous_model.py"
    spec = importlib.util.spec_from_file_location("old_stage1_operator_gls", old_path)
    old_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_module
    spec.loader.exec_module(old_module)
    records, basis, held, onsets, membership = fixture()
    settings = dict(basis=basis, sfreq=32., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"), ridge=1.)
    options = {key: settings[key] for key in ("sfreq", "channels", "candidate_ids")}
    identities = {record.record_id for record in records}
    source = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in
              ["doc/assets/operator_research_20261004/continuous_model.py", "doc/assets/operator_research_20261004/run_calibration_study.py",
               "src/transfer/temporal_noise.py", "src/data/finite_filter_operator.py"]}
    sos = butter(2, [.3, 10.], btype="bandpass", fs=32., output="sos")
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), 29)
    rows = []
    with threadpool_limits(limits=2), tempfile.TemporaryDirectory() as directory:
        reference_bridge = legacy_bridge()
        for index, arm in enumerate(plan["arms"]):
            arm_basis = basis.copy()
            if arm.get("basis_boundary") == "smooth_zero_endpoints":
                window = np.sin(np.linspace(0, np.pi, len(basis)))**2
                window[[0, -1]] = 0.
                arm_basis = basis*window[:, None]
                arm_basis *= np.linalg.norm(basis, axis=0)/np.linalg.norm(arm_basis, axis=0)
                np.testing.assert_array_equal(arm_basis[[0, -1]], np.zeros((2, basis.shape[1])))
                np.testing.assert_allclose(np.linalg.norm(arm_basis, axis=0), np.linalg.norm(basis, axis=0), rtol=2e-15)
            parameters = dict(order=arm["order"], noise_method=arm["noise_method"],
                operator=operator if arm["filter"] else None, filter_design=arm["filter_design"])
            model = OperatorGLS.fit(records, **{**settings, "basis": arm_basis}, **parameters,
                feasible_iterations=arm.get("feasible_iterations", 1), channel_weighted_noise=arm.get("channel_weighted_noise", False),
                spatial_shrinkage=arm.get("spatial_shrinkage", "pooled_lw"))
            expected_passes = arm.get("feasible_iterations", 1)
            assert model.provenance["noise_fit_passes"] == expected_passes
            iterations = model.provenance["feasible_GLS_iterations"]
            assert len(iterations) == expected_passes
            for iteration in iterations:
                metric = iteration["temporal_fit_channel_metric_sha256"]
                assert (metric is not None) == arm.get("channel_weighted_noise", False)
                assert iteration["residual_diagnostics"]["cross_record_lag_pairs"] == 0
            assert {record["record_id"] for record in model.provenance["records"]} == identities
            if arm["noise_method"] == "burg":
                assert {record["record_id"] for record in model.noise.provenance["records"]} == identities
            score = model.score(held, onsets, membership, record_id="held", **options)
            path = Path(directory)/f"{index}.npz"
            model.save(path)
            loaded = OperatorGLS.load(path)
            replay = loaded.score(held, onsets, membership, record_id="held", **options)
            np.testing.assert_array_equal(score.log_evidence, replay.log_evidence)
            before = model.record()
            # Changed held signal affects only this read-only prediction.
            model.score(held+1e-3*np.arange(len(held))[:, None]**2, onsets, membership, record_id="held-change", **options)
            assert model.record() == before
            row = {"arm": arm["name"], "passes": expected_passes, "weighted": arm.get("channel_weighted_noise", False),
                   "all_training_ids_exclude_held": True, "score_does_not_update_model": True, "exact_saved_replay": True}
            if index < 5:
                old = old_module.OperatorGLS.fit(records, **settings, **parameters)
                old_result = old.score(held, onsets, membership, record_id="held", **options)
                np.testing.assert_array_equal(old_result.log_evidence, score.log_evidence)
                for field in ("mean", "row_covariance", "channel_covariance"):
                    np.testing.assert_array_equal(getattr(old, field), getattr(model, field))
                row["stage1_default_exact_numeric_bridge"] = True
            rows.append(row)
    current = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source}
    assert current == source, "source changed during stage2 verification"
    result = {"schema": "n2p3_operator_stage2_gls_verification/1", "source_sha256": source,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stage1_reference_source_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
              "old_continuous_reference_bridge": reference_bridge, "arms": rows, "n_arms": len(rows),
              "no_real_data_used": True, "evidence_scope": "software_correctness"}
    destination = RESEARCH/"stage2_gls_verification.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"arms_verified": len(rows), "old_default_exact": 5, "path": str(destination)}))


if __name__ == "__main__":
    main()
