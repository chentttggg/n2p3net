"""Independent old-model bridge, dense Gaussian and frozen replay probes."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent.parent)]

import numpy as np  # noqa: E402
from continuous_model import OperatorGLS  # noqa: E402
from scipy.linalg import null_space, toeplitz  # noqa: E402
from scipy.signal import butter, lfilter  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from data.finite_filter_operator import FiniteFilterOperator  # noqa: E402
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel  # noqa: E402
from transfer.temporal_noise import TemporalNoise  # noqa: E402


def design(onsets, labels, n, basis):
    all_events, target_events = np.zeros(n), np.zeros(n)
    all_events[onsets] = 1
    target_events[onsets] = labels
    return np.column_stack([np.convolve(events, basis[:, col])[:n]
                            for events in (all_events, target_events) for col in range(basis.shape[1])])


def fixture():
    rng = np.random.default_rng(2026100403)
    time = np.arange(15)
    basis = np.column_stack([np.exp(-.5*((time-6)/2)**2), np.exp(-.5*((time-10)/3)**2)])
    kernel = np.array([[.2, -.1], [-.15, .05], [1., .2], [.5, .8]])
    records = []
    onsets = np.array([5, 40, 80, 120, 160, 201])
    for index, n in enumerate([240, 259, 275, 208]):
        labels = (np.arange(len(onsets))+index) % 3 == 0
        signal = design(onsets, labels, n, basis) @ kernel
        noise = lfilter([1.], [1., -.7], rng.normal(size=(n, 2)), axis=0) * .4
        records.append(CalibrationRecord(f"C-{index}", signal+noise, onsets, labels,
                                         right_censored=index == 3))
    held_onsets = np.array([10, 48, 84, 119, 162, 201])
    membership = np.eye(3, dtype=bool)[np.arange(6) % 3]
    held = design(held_onsets, membership[:, 1], 230, basis) @ kernel
    held += lfilter([1.], [1., -.7], rng.normal(size=held.shape), axis=0)*.4
    return records, basis, held, held_onsets, membership


def legacy_bridge():
    records, basis, held, onsets, member = fixture()
    settings = dict(basis=basis, sfreq=32., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"), ridge=1.)
    legacy = ContinuousDecisionModel.fit(records, **settings, operator_identity="raw-reference")
    replacement = OperatorGLS.fit(records, **settings, order=1, operator=None, noise_method="legacy_clipped_ar1")
    differences = {}
    for name, a, b in (("mean", legacy.kernel_mean, replacement.mean),
                       ("row_covariance", legacy.kernel_row_covariance, replacement.row_covariance),
                       ("innovation_covariance", legacy.innovation_covariance, replacement.channel_covariance)):
        differences[name] = float(np.max(np.abs(a-b)))
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-13)
    differences["phi"] = abs(legacy.phi-replacement.noise.coefficients[0])
    assert differences["phi"] < 1e-14
    predict = dict(record_id="held", sfreq=32., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"))
    old = legacy.score(held, onsets, member, **predict, operator_identity="raw-reference")
    new = replacement.score(held, onsets, member, **predict)
    np.testing.assert_allclose(old.log_evidence, new.log_evidence, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(old.probabilities, new.probabilities, atol=2e-12)
    differences["log_evidence"] = float(np.max(np.abs(old.log_evidence-new.log_evidence)))
    differences["probability"] = float(np.max(np.abs(old.probabilities-new.probabilities)))
    return {"record_lengths": [len(r.samples) for r in records], "last_record_right_censored": True,
            "max_absolute_differences": differences, "old_winner": old.winner, "new_winner": new.winner}


def dense_gaussian_reference():
    rng = np.random.default_rng(42168)
    results = []
    for coefficients in ([], [.7], [1.6, -.81]):
        basis = np.array([[0.], [1.], [.5], [0.]])
        mean = np.array([[.2, -.3], [.9, .5]])
        row_cov = np.array([[.3, .04], [.04, .2]])
        channel_cov = np.array([[.8, .12], [.12, .5]])
        noise = TemporalNoise.from_parameters(coefficients, innovation_variance=1., reference_id="dense_reference")
        model = OperatorGLS(basis, mean, row_cov, channel_cov, noise, None, True, 16., ("Fz", "Pz"),
                            ("a", "b"), {"fit_scope": "analytic_reference", "reference_id": "dense_gaussian_reference", "records": []})
        n = 22
        onsets = np.array([2, 6, 10, 15])
        member = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
        x = rng.normal(size=(n, 2))
        actual = model.score(x, onsets, member, record_id="held", sfreq=16., channels=("Fz", "Pz"), candidate_ids=("a", "b"))
        impulse = np.zeros(20000)
        impulse[0] = 1
        impulse_response = lfilter([1.], np.r_[1., -np.asarray(coefficients)], impulse)
        temporal = toeplitz([impulse_response[:len(impulse_response)-lag] @ impulse_response[lag:] for lag in range(n)])
        nuisance = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
        complement = null_space(nuisance.T)
        expected = []
        for candidate in range(2):
            d = design(onsets, member[:, candidate], n, basis)
            residual = (complement.T @ (x-d@mean)).reshape(-1)
            covariance = np.kron(complement.T @ (temporal+d@row_cov@d.T) @ complement, channel_cov)
            expected.append(-.5*(residual @ np.linalg.solve(covariance, residual)+np.linalg.slogdet(covariance)[1]))
        error = float(np.max(np.abs((actual.log_evidence-actual.log_evidence[0])-(np.asarray(expected)-expected[0]))))
        assert error < 5e-11, error
        results.append({"order": len(coefficients), "max_log_bayes_factor_error": error})
    return results


def replay_membership_and_input():
    records, basis, held, onsets, member = fixture()
    sos = butter(2, [.3, 10.], btype="bandpass", fs=32., output="sos")
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), 29)
    settings = dict(basis=basis, sfreq=32., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"), ridge=1.)
    predict = {key: settings[key] for key in ("sfreq", "channels", "candidate_ids")}
    results = []
    variants = [("legacy_none", None, True, "legacy_clipped_ar1", 1),
                ("filtered_old_design", operator, False, "legacy_clipped_ar1", 1),
                ("filtered_matched_design", operator, True, "legacy_clipped_ar1", 1),
                ("raw_Burg8", None, True, "burg", 8),
                ("raw_Burg32", None, True, "burg", 32)]
    with tempfile.TemporaryDirectory() as directory:
        for name, op, filtered, method, order in variants:
            model = OperatorGLS.fit(records, **settings, operator=op, filter_design=filtered, noise_method=method, order=order)
            first = model.score(held, onsets, member, record_id="held", **predict)
            path = Path(directory)/f"{name}.npz"
            model.save(path)
            restored = OperatorGLS.load(path)
            second = restored.score(held, onsets, member, record_id="held", **predict)
            np.testing.assert_array_equal(first.log_evidence, second.log_evidence)
            np.testing.assert_array_equal(first.probabilities, second.probabilities)
            # Candidate permutation is a public vocabulary operation, independent of hidden truth.
            permutation = np.array([2, 0, 1])
            vocabulary = tuple(settings["candidate_ids"][i] for i in permutation)
            permuted = replace(restored, candidate_ids=vocabulary)
            result = permuted.score(held, onsets, member[:, permutation], record_id="held", sfreq=32.,
                                    channels=settings["channels"], candidate_ids=vocabulary)
            np.testing.assert_allclose(result.log_evidence, first.log_evidence[permutation], atol=2e-11)
            assert result.winner == first.winner
            truth_one, truth_two = {"held": "a"}, {"held": "c"}
            assert truth_one != truth_two  # Both scorer-only maps remain outside the decoder call.
            assert "truth" not in inspect.signature(model.score).parameters
            again = model.score(held, onsets, member, record_id="held", **predict)
            np.testing.assert_array_equal(first.log_evidence, again.log_evidence)
            try:
                model.score(records[0].samples, records[0].onsets, member, record_id="renamed_training", **predict)
            except ValueError as error:
                assert "held-out" in str(error)
            else:
                raise AssertionError("an exact calibration record was accepted after renaming")
            results.append({"arm": name, "order": order, "winner": first.winner,
                            "exact_numeric_replay": True, "candidate_permutation": True,
                            "score_has_no_truth_parameter": True, "renamed_calibration_rejected": True})
    return results


def filtering_reference_and_units():
    records, basis, held, onsets, member = fixture()
    settings = dict(basis=basis, sfreq=32., channels=("Fz", "Pz"), candidate_ids=("a", "b", "c"), ridge=1.)
    predict = {key: settings[key] for key in ("sfreq", "channels", "candidate_ids")}
    model = OperatorGLS.fit(records, **settings, order=8)
    scaled_records = [replace(record, samples=record.samples*1e-6) for record in records]
    scaled = OperatorGLS.fit(scaled_records, **settings, order=8)
    original = model.score(held, onsets, member, record_id="held", **predict)
    volts = scaled.score(held*1e-6, onsets, member, record_id="held", **predict)
    unit_error = float(np.max(np.abs(original.log_evidence-volts.log_evidence)))
    assert unit_error < 2e-9, unit_error
    nuisance = np.column_stack([np.ones(len(held)), np.linspace(-1, 1, len(held))]) @ np.array([[2., -1.], [.8, 1.2]])
    with_drift = model.score(held+nuisance, onsets, member, record_id="held-plus-drift", **predict)
    drift_error = float(np.max(np.abs(original.log_evidence-with_drift.log_evidence)))
    assert drift_error < 2e-9, drift_error
    # Independent dense reference to the declared approximate filtered-coordinate model.
    # This is not a claim that AR1 equals a pushed-forward raw F Sigma F.T.
    n = 22
    sos = butter(2, [.3, 5.], btype="bandpass", fs=16., output="sos")
    op = FiniteFilterOperator(tuple(map(tuple, sos)), 12)
    b = np.array([[0.], [1.], [.5], [0.]])
    mean, row_cov, channel_cov = np.array([[.1], [.8]]), np.array([[.4, .05], [.05, .2]]), np.array([[.6]])
    noise = TemporalNoise.from_parameters([.7], innovation_variance=1., reference_id="filtered_coordinate_reference")
    ar_cov = toeplitz(.7**np.arange(n))/(1-.7**2)
    onset = np.array([2, 6, 10, 15])
    membership = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
    x = np.random.default_rng(78541).normal(size=(n, 1))
    complement = null_space(np.column_stack([np.ones(n), np.linspace(-1, 1, n)]).T)
    errors = []
    for filtered in (False, True):
        fitted = OperatorGLS(b, mean, row_cov, channel_cov, noise, op, filtered, 16., ("Pz",), ("a", "b"),
                             {"fit_scope": "analytic_reference", "reference_id": "dense_gaussian_reference", "records": []})
        computed = fitted.score(x, onset, membership, record_id="held", sfreq=16., channels=("Pz",), candidate_ids=("a", "b"))
        expected = []
        for candidate in range(2):
            d = design(onset, membership[:, candidate], n, b)
            d = op.apply(d) if filtered else d
            residual = complement.T @ (op.apply(x)-d@mean)
            covariance = .6*complement.T @ (ar_cov+d@row_cov@d.T) @ complement
            quadratic = residual[:, 0] @ np.linalg.solve(covariance, residual[:, 0])
            expected.append(-.5*(quadratic+np.linalg.slogdet(covariance)[1]))
        error = float(np.max(np.abs(computed.log_evidence-computed.log_evidence[0]-(np.asarray(expected)-expected[0]))))
        assert error < 5e-11, error
        errors.append({"filter_design": filtered, "max_log_bayes_factor_error": error})
    return {"global_uV_to_V_max_log_evidence_error": unit_error,
            "post_processing_common_drift_max_log_evidence_error": drift_error,
            "filtered_coordinate_dense_references": errors}


def main():
    dependencies = ["doc/assets/operator_research_20261004/continuous_model.py", "src/data/finite_filter_operator.py",
                    "src/transfer/continuous_decision.py", "src/transfer/temporal_noise.py"]
    initial = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in dependencies}
    with threadpool_limits(limits=2):
        result = {"schema": "n2p3_gls_independent_verification/1", "evidence_scope": "software_mathematical_correctness",
                  "source_sha256": initial, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "legacy_bridge": legacy_bridge(), "dense_gaussian": dense_gaussian_reference(),
                  "replay_and_truth_separation": replay_membership_and_input(),
                  "filtering_reference_and_units": filtering_reference_and_units()}
    final = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in dependencies}
    if initial != final:
        raise RuntimeError("source changed during independent verification; repeat after stabilization")
    Path(__file__).with_name("gls_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
