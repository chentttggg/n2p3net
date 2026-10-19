import numpy as np
import pytest
from scipy.linalg import null_space

from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionModel


def test_calibration_freeze_reload_and_unknown_target_prediction(tmp_path):
    rng = np.random.default_rng(44)
    basis = np.exp(-0.5 * ((np.arange(20) - 9) / 3) ** 2)[:, None]
    onsets = np.arange(30, 231, 25)
    records = []
    for record in range(4):
        target = np.arange(len(onsets)) % 3 == record % 3
        samples = rng.normal(0, 0.2, size=(280, 1))
        for onset, label in zip(onsets, target, strict=True):
            samples[onset:onset + 20] += basis * (0.5 + 3 * label)
        records.append(CalibrationRecord(f"cal-{record}", samples, onsets, target))
    model = ContinuousDecisionModel.fit(
        records, sfreq=50., channels=("Cz",), candidate_ids=("a", "b", "c"),
        operator_identity="raw-test-operator", basis=basis, ridge=1.,
    )
    membership = np.eye(3, dtype=bool)[np.arange(len(onsets)) % 3]
    samples = rng.normal(0, 0.2, size=(280, 1))
    for onset, member in zip(onsets, membership[:, 2], strict=True):
        samples[onset:onset + 20] += basis * (0.5 + 3 * member)
    settings = dict(operator_identity="raw-test-operator", sfreq=50., channels=("Cz",),
                    candidate_ids=("a", "b", "c"), record_id="held-out")
    result = model.score(samples, onsets, membership, **settings)
    assert result.winner == "c"
    assert result.probabilities[2] > 0.99
    assert result.n_unique_samples == len(samples)
    path = tmp_path / "continuous.npz"
    model.save(path)
    loaded = ContinuousDecisionModel.load(path)
    repeated = loaded.score(samples, onsets, membership, **settings)
    np.testing.assert_array_equal(repeated.log_evidence, result.log_evidence)
    assert not loaded.kernel_mean.flags.writeable
    assert not loaded.kernel_row_covariance.flags.writeable


@pytest.mark.parametrize("phi", [0., 0.7])
def test_predictive_covariance_matches_independent_dense_gaussian(phi):
    # Two candidate designs with unequal event spacing exercise candidate logdet.
    basis = np.array([[0.], [1.], [0.5]])
    mean = np.array([[0.2], [0.8]])
    Vh = np.array([[0.5, 0.1], [0.1, 0.3]])
    model = ContinuousDecisionModel(basis, mean, Vh, np.array([[0.4]]), phi, 10.,
                                    ("Cz",), ("a", "b"), "finite", {"fit_scope": "analytic_reference", "reference_id": "independent_dense_gaussian", "records": []})
    onset = np.array([2, 5, 7, 10])
    membership = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    X = np.linspace(-0.1, 0.1, 15)[:, None] + np.sin(np.arange(15))[:, None]
    result = model.score(X, onset, membership, operator_identity="finite", sfreq=10.,
                         channels=("Cz",), candidate_ids=("a", "b"), record_id="test")
    # Independent dense density in the orthogonal complement of intercept/slope.
    complement = null_space(np.column_stack([np.ones(15), np.linspace(-1, 1, 15)]).T)
    expected = []
    for c in range(2):
        impulse = np.zeros(15)
        impulse[onset] = 1
        targets = np.zeros(15)
        targets[onset] = membership[:, c]
        D = np.column_stack([np.convolve(impulse, basis[:, 0])[:15],
                             np.convolve(targets, basis[:, 0])[:15]])
        residual = complement.T @ (X - D @ mean)
        ar_covariance = phi ** np.abs(np.arange(15)[:, None] - np.arange(15)[None, :]) / (1 - phi**2)
        covariance = complement.T @ (0.4 * (ar_covariance + D @ Vh @ D.T)) @ complement
        expected.append(-0.5 * (float((residual.T @ np.linalg.solve(covariance, residual)).item())
                                + np.linalg.slogdet(covariance)[1]))
    np.testing.assert_allclose(result.log_evidence - result.log_evidence[0],
                               np.array(expected) - expected[0], atol=1e-12)
    with pytest.raises(ValueError, match="channel order"):
        model.score(X, onset, membership, operator_identity="finite", sfreq=10.,
                    channels=("Pz",), candidate_ids=("a", "b"), record_id="test")
    with pytest.raises(ValueError, match="vocabulary/order"):
        model.score(X, onset, membership, operator_identity="finite", sfreq=10.,
                    channels=("Cz",), candidate_ids=("b", "a"), record_id="test")
    with pytest.raises(ValueError, match="missing candidate"):
        # Three-candidate data requires evidence for every frozen candidate.
        model3 = ContinuousDecisionModel(basis, mean, Vh, np.array([[0.4]]), 0., 10.,
                                         ("Cz",), ("a", "b", "c"), "finite", {"fit_scope": "analytic_reference", "reference_id": "independent_dense_gaussian", "records": []})
        model3.score(X, onset, np.column_stack([membership, np.zeros(4)]),
                     operator_identity="finite", sfreq=10., channels=("Cz",),
                     candidate_ids=("a", "b", "c"), record_id="test")
    with pytest.raises(ValueError, match="ordered"):
        model.score(X, np.array([5, 2, 7, 10], dtype=np.uint64), membership,
                    operator_identity="finite", sfreq=10., channels=("Cz",),
                    candidate_ids=("a", "b"), record_id="test")


def test_serialized_noise_hyperparameters_are_bound_to_artifact(tmp_path):
    import json

    model = ContinuousDecisionModel(np.ones((3, 1)), np.zeros((2, 1)), np.eye(2),
                                    np.eye(1), 0.2, 10., ("Cz",), ("a", "b"), "x", {"fit_scope": "analytic_reference", "reference_id": "serialization_fixture", "records": []})
    path = tmp_path / "original.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(values["metadata"]))
    metadata["phi"] = 0.8
    values["metadata"] = np.asarray(json.dumps(metadata))
    altered = tmp_path / "altered.npz"
    np.savez_compressed(altered, **values)
    with pytest.raises(ValueError, match="digest"):
        ContinuousDecisionModel.load(altered)


def test_continuous_fit_ledger_cannot_be_erased_or_resigned(tmp_path):
    rng = np.random.default_rng(5)
    record = CalibrationRecord("cal", rng.normal(size=(40, 1)), np.array([5, 15, 25]), np.array([0, 1, 0]))
    model = ContinuousDecisionModel.fit([record], sfreq=10., channels=("Cz",),
        candidate_ids=("a", "b"), operator_identity="raw", basis=np.ones((3, 1)))
    with pytest.raises((TypeError, AttributeError)):
        model.provenance["records"].clear()
    with pytest.raises(TypeError, match="frozen"):
        model.provenance["records"][0]["record_id"] = "other"
    object.__setattr__(model, "provenance", {**model.provenance, "records": []})
    with pytest.raises(ValueError, match="integrity"):
        model.score(record.samples, record.onsets, np.eye(2)[record.target_mask],
                    operator_identity="raw", sfreq=10., channels=("Cz",), candidate_ids=("a", "b"), record_id="cal")
    with pytest.raises(ValueError, match="integrity"):
        model.save(tmp_path / "changed.npz")


def test_explicit_right_censored_calibration_clips_response_without_reading_future(tmp_path):
    basis = np.array([[0.], [1.], [2.], [1.], [0.]])
    onsets = np.array([3, 13, 23, 33, 43, 53])
    labels = np.arange(6) % 2
    # The next unlabelled flash occurs at sample 56: its signal is unavailable.
    samples = np.random.default_rng(51).normal(0, .1, (56, 1))
    for onset, label in zip(onsets, labels, strict=True):
        retained = min(len(basis), len(samples) - onset)
        samples[onset:onset + retained] += basis[:retained] * (1 + label)
    settings = dict(sfreq=10., channels=("Cz",), candidate_ids=("a", "b"),
                    operator_identity="calibration-cut-at-next-flash", basis=basis)
    with pytest.raises(ValueError, match="complete response tails"):
        ContinuousDecisionModel.fit([CalibrationRecord("cal", samples, onsets, labels)], **settings)
    model = ContinuousDecisionModel.fit([
        CalibrationRecord("cal", samples, onsets, labels, right_censored=True)], **settings)
    ledger = model.provenance["records"][0]
    assert ledger["right_censored"] is True
    assert ledger["n_samples"] == 56 and ledger["n_events"] == 6
    assert ledger["right_censored_events"] == 1
    path = tmp_path / "censored.npz"
    model.save(path)
    restored = ContinuousDecisionModel.load(path)
    np.testing.assert_array_equal(restored.kernel_mean, model.kernel_mean)
    # Calibration censoring permission must not leak into held-out scoring.
    with pytest.raises(ValueError, match="complete response tails"):
        restored.score(samples.copy() + .001, onsets, np.eye(2)[labels],
                       operator_identity=settings["operator_identity"], sfreq=10., channels=("Cz",),
                       candidate_ids=("a", "b"), record_id="test")


def test_continuous_loading_rejects_a_resigned_empty_training_ledger(tmp_path):
    import json

    from transfer.noise_model import metadata_digest
    record = CalibrationRecord("cal", np.random.default_rng(5).normal(size=(40, 1)),
                               np.array([5, 15, 25]), np.array([0, 1, 0]))
    model = ContinuousDecisionModel.fit([record], sfreq=10., channels=("Cz",),
        candidate_ids=("a", "b"), operator_identity="raw", basis=np.ones((3, 1)))
    path = tmp_path / "original.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    metadata = json.loads(payload["metadata"].item())
    metadata["provenance"]["records"] = []
    metadata.pop("artifact_sha256")
    metadata["artifact_sha256"] = metadata_digest(metadata)
    payload["metadata"] = np.asarray(json.dumps(metadata))
    changed = tmp_path / "changed.npz"
    np.savez(changed, **payload)
    with pytest.raises(ValueError, match="training ledger"):
        ContinuousDecisionModel.load(changed)
