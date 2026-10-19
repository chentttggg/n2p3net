"""Stage-3 priors, stationary innovation equivalence and fixed-arm replay."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import butter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
RESEARCH = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(RESEARCH), str(RESEARCH/"continuous_mechanism"), str(RESEARCH/"analysis_stage2")]
from continuous_model import OperatorGLS  # noqa: E402
from probe_conditioning import prior_scale_gauge  # noqa: E402
from verify_gls import fixture, legacy_bridge  # noqa: E402

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from transfer.temporal_noise import TemporalNoise  # noqa: E402


def main():
    paths = ["doc/assets/operator_research_20261004/continuous_model.py", "doc/assets/operator_research_20261004/run_calibration_study.py",
             "doc/assets/operator_research_20261004/selection_erp_model.py", "src/transfer/temporal_noise.py"]
    sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    plan = json.loads((RESEARCH/"calibration_group_plan_stage3.json").read_bytes())
    records, basis, held, onsets, member = fixture()
    settings = dict(basis=basis, sfreq=32., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"), ridge=1.)
    predict = {key: settings[key] for key in ("sfreq", "channels", "candidate_ids")}
    op = FiniteFilterOperator(tuple(map(tuple, butter(2, [.3, 10.], btype="bandpass", fs=32., output="sos"))), 29)
    rows = []
    with threadpool_limits(limits=2), tempfile.TemporaryDirectory() as directory:
        bridge = legacy_bridge()
        marginal_a = OperatorGLS.fit(records, **settings, order=2, kernel_prior_scale="marginal")
        marginal_b = OperatorGLS.fit(records, **settings, order=2, kernel_prior_scale="marginal", startup_mode="stationary_innovations")
        score_a = marginal_a.score(held, onsets, member, record_id="held", **predict)
        score_b = marginal_b.score(held, onsets, member, record_id="held", **predict)
        lattice_error = float(np.max(np.abs(score_a.log_evidence-score_b.log_evidence)))
        assert lattice_error < 1e-9
        fixed = TemporalNoise.from_parameters([.7], innovation_variance=1., reference_id="prior_scale_contract")
        with patch.object(TemporalNoise, "fit", return_value=fixed):
            scaled = OperatorGLS.fit(records, **settings, order=1, kernel_prior_scale="marginal")
            explicit = OperatorGLS.fit(records, **{**settings, "ridge": 1/fixed.marginal_variance_per_unit_innovation},
                                       order=1, kernel_prior_scale="innovation")
        np.testing.assert_array_equal(scaled.mean, explicit.mean)
        np.testing.assert_array_equal(scaled.row_covariance, explicit.row_covariance)
        np.testing.assert_array_equal(scaled.channel_covariance, explicit.channel_covariance)
        for arm in plan["arms"]:
            if arm.get("family"):
                continue
            arm_basis = basis.copy()
            if arm.get("basis_boundary") == "smooth_zero_endpoints":
                window = np.sin(np.linspace(0, np.pi, len(basis)))**2
                window[[0, -1]] = 0
                arm_basis *= window[:, None]
                arm_basis *= np.linalg.norm(basis, axis=0)/np.linalg.norm(arm_basis, axis=0)
            model = OperatorGLS.fit(records, **{**settings, "basis": arm_basis}, order=arm["order"], noise_method=arm["noise_method"],
                operator=op if arm["filter"] else None, filter_design=arm["filter_design"],
                feasible_iterations=arm.get("feasible_iterations", 1), channel_weighted_noise=arm.get("channel_weighted_noise", False),
                kernel_prior_scale=arm.get("kernel_prior_scale", "innovation"), startup_mode=arm.get("startup_mode", "stationary"))
            for step in model.provenance["feasible_GLS_iterations"]:
                expected = 1/step["marginal_variance_per_unit_innovation"] if arm.get("kernel_prior_scale") == "marginal" else 1.
                assert step["effective_ridge"] == expected
            score = model.score(held, onsets, member, record_id="held", **predict)
            path = Path(directory)/(arm["name"]+".npz")
            model.save(path)
            loaded = OperatorGLS.load(path)
            repeated = loaded.score(held, onsets, member, record_id="held", **predict)
            np.testing.assert_array_equal(score.log_evidence, repeated.log_evidence)
            assert all(record["record_id"].startswith("C-") for record in model.provenance["records"])
            rows.append({"arm": arm["name"], "family": "continuous_gls", "exact_numeric_replay": True,
                         "effective_ridge_each_pass_checked": True, "same_training_only_record_ids": True,
                         "startup_mode": model.noise.startup_mode, "all_prefix_rows_retained": model.noise.startup_mode != "conditional_prefix"})
        # Reuse the already executed independent owner audit only after source identity checks.
        owner_bytes = (RESEARCH/"selection_erp_verification.json").read_bytes()
        owner = json.loads(owner_bytes)
        assert owner["source_sha256"] == sources["doc/assets/operator_research_20261004/selection_erp_model.py"]
        assert owner["verification_sha256"] == hashlib.sha256((RESEARCH/"verify_selection_erp_model.py").read_bytes()).hexdigest()
        for arm in plan["arms"]:
            if not arm.get("family"):
                continue
            evidence = next(item for item in owner["models"] if item["family"] == arm["family"] and item["baseline"] == arm["baseline_mode"])
            assert evidence["owner_logit_max_error"] == 0 and evidence["numeric_reload_exact"]
            rows.append({"arm": arm["name"], "family": arm["family"], "exact_numeric_replay": True,
                         "validation_origin": "source-matched independently executed owner audit", "owner_logit_max_error": 0.})
        gauge = prior_scale_gauge()
    assert sources == {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    result = {"schema": "n2p3_stage3_fixed_route_verification/1", "source_sha256": sources,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "legacy_bridge": bridge, "stationary_versus_lattice_marginal_prior_max_log_evidence_error": lattice_error,
              "marginal_effective_ridge_matches_explicit_innovation_ridge_exactly": True,
              "physical_covariance_gauge_probe": gauge, "arms": rows, "n_arms": len(rows),
              "erp_owner_evidence_sha256": hashlib.sha256(owner_bytes).hexdigest(),
              "no_real_data_used": True, "evidence_scope": "software_and_math_verification"}
    (RESEARCH/"stage3_gls_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print(json.dumps({"verified_arms": len(rows), "lattice_bridge_error": lattice_error,
                      "effective_ridge_equivalence": "exact", "owner_audit_reused_with_source_hash": 4}))


if __name__ == "__main__":
    main()
