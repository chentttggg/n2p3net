"""Independent public-owner, grid, LLR and numeric-replay checks for ERP anchors."""
# ruff: noqa: E402 -- explicit repository path and native-thread caps precede imports.
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np
from scipy.special import logsumexp
from selection_erp_model import SelectionERP, selection_epochs
from threadpoolctl import threadpool_limits

from baselines.structured_erp import StructuredERP
from data.epochs import PreprocessingSpec
from data.finite_filter_operator import FiniteFilterOperator
from data.preprocess import design_continuous_sos
from transfer.binary import fit_shrinkage_feature_model, target_shrinkage_features
from transfer.continuous_decision import CalibrationRecord
from transfer.noise_model import metadata_digest

HERE = Path(__file__).resolve().parent


def rejected(action, message):
    try:
        action()
    except (ValueError, TypeError):
        return
    raise AssertionError(message)


def main():
    rng = np.random.default_rng(20261005)
    channels = ("Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8")
    candidates = tuple(f"candidate-{i}" for i in range(6))
    onsets = 51 + 64 * np.arange(120, dtype=np.int64)
    n_samples = int(onsets[-1] + 307)
    topography = np.array([.4, .6, 1., .4, .8, .7, .4, .4])
    t = np.arange(307) / 256
    erp = 8e-6 * np.exp(-.5 * ((t - .35) / .10)**2) - 2e-6 * np.exp(-.5 * ((t - .17) / .05)**2)

    def samples(labels):
        x = rng.normal(scale=4e-6, size=(n_samples, 8))
        x += np.sin(np.arange(n_samples)[:, None] / 600) * rng.normal(scale=10e-6, size=(1, 8))
        for onset, label in zip(onsets, labels, strict=True):
            x[onset:onset + 307] += (erp[:, None] * topography) * (1. if label else .1)
        return x

    records = []
    for index in range(5):
        labels = np.arange(120) % 6 == index
        records.append(CalibrationRecord(f"synthetic-calibration-{index}", samples(labels), onsets.copy(), labels))
    test = samples(np.arange(120) % 6 == 1)
    membership = np.eye(6, dtype=bool)[np.arange(120) % 6]
    membership[np.arange(0, 120, 5), 0] = True  # unequal exposure challenges prior removal
    sos = design_continuous_sos(256., l_freq=.1, h_freq=30., order=4,
                                notch_freqs_hz=(50.,), notch_quality_factor=30.)
    padding = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    operator = FiniteFilterOperator(tuple(map(tuple, sos)), padding)
    rejected(lambda: PreprocessingSpec(sfreq=128., n_times=179, tmin_ms=-51 / 256 * 1000,
        tmax_ms=306 / 256 * 1000).validate(), "legacy closed tmax silently accepted inconsistent output count")
    # Independently spell out the exact source support and official FFT transform.
    processed = operator.apply(test)
    physical = np.stack([processed[o - 51:o + 307].T for o in onsets])
    assert physical.shape == (120, 8, 358)
    resampled = mne.filter.resample(physical, 128., 256., method="fft", npad="auto",
                                  window="auto", pad="edge", verbose=False)
    assert resampled.shape == (120, 8, 179)
    independent_none = resampled[:, :, :178]
    independent_baseline = independent_none - independent_none[:, :, :26].mean(axis=2, keepdims=True)
    np.testing.assert_allclose(selection_epochs(test, onsets, operator=operator, baseline_mode="none", n_channels=8),
                               independent_none, atol=0, rtol=0)
    np.testing.assert_allclose(selection_epochs(test, onsets, operator=operator, baseline_mode="mean_only", n_channels=8),
                               independent_baseline, atol=0, rtol=0)
    rejected(lambda: selection_epochs(np.vstack([test, np.zeros((1, 8))]), onsets,
        operator=operator, baseline_mode="none", n_channels=8), "forbidden extra endpoint sample accepted")
    rejected(lambda: selection_epochs(test[:-1], onsets, operator=operator,
        baseline_mode="none", n_channels=8), "last authorized sample missing but accepted")

    results = []
    for family, baseline in (("waveform_lda", "mean_only"), ("waveform_lda", "none"),
                             ("toeplitz_lda", "none"), ("xdawn_erp", "mean_only")):
        model = SelectionERP.fit(records, family=family, baseline_mode=baseline, sfreq=256.,
            channels=channels, candidate_ids=candidates, operator=operator)
        X = np.concatenate([selection_epochs(r.samples, r.onsets, operator=operator,
                            baseline_mode=baseline, n_channels=8) for r in records])
        y = np.concatenate([r.target_mask for r in records]).astype(int)
        test_epochs = independent_baseline if baseline == "mean_only" else independent_none
        if family == "waveform_lda":
            anchor = fit_shrinkage_feature_model(target_shrinkage_features(X, sfreq=128.), y)
            owner_logits = anchor.decision_function(target_shrinkage_features(test_epochs, sfreq=128.))
        else:
            anchor = StructuredERP(family, baseline_mode=baseline, bin_width=4 if family == "toeplitz_lda" else 1,
                                   nfilter=4).fit(X, y)
            owner_logits = anchor.predict_logit(test_epochs)
        replay_logits = model._logit(test_epochs)
        np.testing.assert_allclose(replay_logits, owner_logits, rtol=2e-10, atol=1e-10)
        scored = model.score(test, onsets, membership, record_id="synthetic-held", sfreq=256.,
                             channels=channels, candidate_ids=candidates)
        llr = owner_logits - np.log(y.mean() / (1 - y.mean()))
        np.testing.assert_allclose(scored.log_evidence, membership.T @ llr, rtol=2e-10, atol=1e-8)
        joint = (membership * -np.logaddexp(0., -llr)[:, None]
                 + (~membership) * -np.logaddexp(0., llr)[:, None]).sum(axis=0)
        np.testing.assert_allclose(scored.probabilities, np.exp(joint - logsumexp(joint)), atol=1e-12)
        assert not np.allclose(membership.T @ llr, membership.T @ owner_logits)
        assert scored.n_unique_samples == n_samples
        with tempfile.TemporaryDirectory(prefix="selection-erp-") as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = SelectionERP.load(path)
            replay = loaded.score(test, onsets, membership, record_id="synthetic-held", sfreq=256.,
                                  channels=channels, candidate_ids=candidates)
            np.testing.assert_array_equal(scored.log_evidence, replay.log_evidence)
            assert model.record() == loaded.record()
            with np.load(path, allow_pickle=False) as archive:
                contents = {key: archive[key].copy() for key in archive.files}
            metadata = json.loads(str(contents["metadata"]))
            metadata["provenance"]["records"] = []
            metadata.pop("artifact_sha256")
            metadata["artifact_sha256"] = metadata_digest(metadata)
            contents["metadata"] = np.asarray(json.dumps(metadata))
            tampered = Path(directory) / "resigned-empty-ledger.npz"
            np.savez_compressed(tampered, **contents)
            rejected(lambda tampered=tampered: SelectionERP.load(tampered), "re-signed empty training ledger accepted")
        rejected(lambda model=model: model.score(records[0].samples, onsets, membership, record_id="renamed-training",
            sfreq=256., channels=channels, candidate_ids=candidates), "renamed training bytes accepted as held data")
        rejected(lambda model=model: model.score(test, onsets, membership.astype(float), record_id="synthetic-held",
            sfreq=256., channels=channels, candidate_ids=candidates), "non-boolean membership accepted")
        rejected(lambda model=model: model.score(test, onsets, membership, record_id="synthetic-held",
            sfreq=256., channels=channels[::-1], candidate_ids=candidates), "channel permutation accepted")
        rejected(lambda model=model: model.provenance["records"][0].__setitem__("record_id", "changed"), "nested provenance mutable")
        rejected(lambda model=model: dict(model.state)["coefficient"].setflags(write=True), "numeric state mutable")
        results.append({"family": family, "baseline": baseline, "owner_logit_max_error": float(np.max(np.abs(owner_logits - replay_logits))),
                        "numeric_reload_exact": True, "label_count": sum(model.provenance["class_counts"]),
                        "state_sha256": model.record()["state_sha256"], "grid": model.provenance["epoch_geometry"]})
        print(f"PASS {family}/{baseline}", flush=True)

    rejected(lambda: SelectionERP.fit(records, family="waveform_lda", baseline_mode="mean_only", sfreq=256.,
        channels=channels, candidate_ids=candidates, operator=operator, expected_label_count=599), "wrong budget accepted")
    reduced = [CalibrationRecord(r.record_id, np.column_stack([r.samples[:, :3], r.samples[:, :3], r.samples[:, :2]]),
                                 r.onsets, r.target_mask) for r in records]
    rejected(lambda: SelectionERP.fit(reduced, family="xdawn_erp", baseline_mode="mean_only", sfreq=256.,
        channels=channels, candidate_ids=candidates, operator=operator), "xDAWN4 accepted empirical rank3")
    report = {"scope": "software_correctness_synthetic_only", "models": results,
        "grid_endpoint_extra_sample_rejected": True, "grid_missing_last_sample_rejected": True,
        "both_waveform_bridges_use_original_owner": True, "unequal_exposure_prior_removal_verified": True,
        "observed_xdawn_rank_gate_verified": True, "tamper_replay_membership_identity_rejections_verified": True,
        "source_sha256": hashlib.sha256((HERE / "selection_erp_model.py").read_bytes()).hexdigest(),
        "verification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / "selection_erp_verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
