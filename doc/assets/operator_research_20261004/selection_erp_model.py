"""Mature ERP anchors on exactly the continuous selection's authorized raw grid.

Training delegates to the existing waveform or StructuredERP owner. Every
selection is filtered independently; event epochs consume exactly samples
[-51,+306] at 256 Hz. Shared FFT yields 179 points, then the truthful closed
source endpoint contract retains the first 178 output points.
Candidate scores sum binary LLRs under an explicitly approximate independence
model. There is no held-data probability calibration or target truth input.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import baselines.features as feature_owner
import baselines.riemann as riemann_owner
import baselines.structured_erp as structured_owner
import data.epochs as epoch_owner
import data.finite_filter_operator as filter_owner
import data.preprocess as preprocessing_owner
import transfer.binary as waveform_owner
from baselines.structured_erp import StructuredERP
from data.epochs import PreprocessingSpec
from data.finite_filter_operator import FiniteFilterOperator
from data.preprocess import transform_epoch_samples
from transfer.binary import fit_shrinkage_feature_model, target_shrinkage_features
from transfer.continuous_decision import CalibrationRecord, ContinuousDecisionResult
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
)

SCHEMA = "n2p3_selection_erp_research/1"
FAMILIES = frozenset({"waveform_lda", "toeplitz_lda", "xdawn_erp"})
SOURCE_RATE = 256.0
OUTPUT_RATE = 128.0
START = -51
STOP = 307  # exclusive; sample +307 is never requested or consumed
RAW_SAMPLES = STOP - START
RESAMPLED_SAMPLES = 179
OUTPUT_SAMPLES = 178
_SOURCE_SHA256 = {
    owner.__name__: hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest()
    for owner in (feature_owner, riemann_owner, structured_owner, epoch_owner,
                  filter_owner, preprocessing_owner, waveform_owner)
}
_SOURCE_SHA256["selection_erp_model"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _preprocessing(baseline_mode):
    # Keep the actual closed source endpoint. The shared contract derives
    # floor(357 * 128 / 256)=178 output samples after its 179-point FFT.
    result = PreprocessingSpec(name="selection_erp_physical_epoch/1", sfreq=OUTPUT_RATE,
        l_freq=None, h_freq=None, tmin_ms=START / SOURCE_RATE * 1000,
        tmax_ms=(STOP - 1) / SOURCE_RATE * 1000, n_times=OUTPUT_SAMPLES,
        baseline_mode=baseline_mode, filter_phase="zero",
        causal_iir_initial_state="not_applicable", notch_freqs_hz=())
    result.validate()
    return result


def _geometry(baseline_mode):
    record = {"schema": "n2p3_selection_epoch_grid/1", "source_sample_rate_hz": SOURCE_RATE,
        "source_start_inclusive": START, "source_stop_exclusive": STOP,
        "source_last_inclusive": STOP - 1, "source_samples": RAW_SAMPLES,
        "source_start_microseconds": START / SOURCE_RATE * 1e6,
        "source_last_microseconds": (STOP - 1) / SOURCE_RATE * 1e6,
        "output_sample_rate_hz": OUTPUT_RATE, "output_samples": OUTPUT_SAMPLES,
        "fft_resampled_samples": RESAMPLED_SAMPLES, "first_T_discarded_output_samples": 1,
        "output_start_microseconds": START / SOURCE_RATE * 1e6,
        "output_last_microseconds": (START / SOURCE_RATE + (OUTPUT_SAMPLES - 1) / OUTPUT_RATE) * 1e6,
        "baseline_samples": 26 if baseline_mode == "mean_only" else 0,
        "baseline_interval": "output physical time <0, half-open",
        "signal_unit": "V", "resampling": "shared MNE epoch FFT, auto padding/window, edge pad",
        "physical_preprocessing_spec": asdict(_preprocessing(baseline_mode)),
        "source_epoch_builder_used": False,
        "filter_scope": "each independent complete selection, then epoch slices"}
    return json.loads(json.dumps(record, allow_nan=False))


def _validate_declaration(family, baseline_mode, sfreq, channels, candidate_ids, operator):
    if family not in FAMILIES:
        raise ValueError("selection ERP family is not a declared mature anchor")
    if baseline_mode not in {"mean_only", "none"}:
        raise ValueError("selection ERP baseline must be mean_only or none")
    if family == "toeplitz_lda" and baseline_mode != "none":
        raise ValueError("standard Toeplitz anchor requires no baseline")
    if family == "xdawn_erp" and baseline_mode != "mean_only":
        raise ValueError("declared ERP covariance anchor requires mean-only baseline")
    if isinstance(sfreq, bool) or sfreq != SOURCE_RATE:
        raise ValueError("selection raw grid is fixed at 256 Hz")
    require_identifiers(channels, "physical channel order")
    require_identifiers(candidate_ids, "complete candidate vocabulary")
    if len(candidate_ids) < 2 or not isinstance(operator, FiniteFilterOperator):
        raise ValueError("complete candidate vocabulary and explicit finite operator required")


def _input(samples, onsets, n_channels):
    x = np.asarray(samples, dtype=np.float64)
    onset = np.asarray(onsets)
    if (x.ndim != 2 or x.shape[1] != n_channels or len(x) < RAW_SAMPLES
            or not np.isfinite(x).all()):
        raise ValueError("selection samples must be finite, complete, samples by physical channels")
    if (onset.ndim != 1 or len(onset) < 1 or onset.dtype.kind not in "iu"
            or np.any(onset[1:] <= onset[:-1])):
        raise ValueError("selection event onsets must be distinct ordered integer samples")
    if int(onset[0]) != -START or int(onset[-1]) + STOP != len(x):
        raise ValueError("selection raw support must be exactly first_onset-51 through last_onset+307 exclusive")
    return x, onset.astype(np.int64, copy=False)


def selection_epochs(samples, onsets, *, operator, baseline_mode, n_channels):
    """Numerical seam shared by the model and independent owner-parity probes."""
    x, onset = _input(samples, onsets, n_channels)
    if not isinstance(operator, FiniteFilterOperator):
        raise ValueError("explicit frozen finite filter required")
    filtered = operator.apply(x)
    epochs = np.stack([filtered[int(t) + START:int(t) + STOP].T for t in onset])
    if epochs.shape != (len(onset), n_channels, RAW_SAMPLES):
        raise ValueError("physical epoch support is incomplete")
    transformed = transform_epoch_samples(epochs, SOURCE_RATE, _preprocessing(baseline_mode),
                                           require_linear=True)
    if transformed.shape != (len(onset), n_channels, OUTPUT_SAMPLES):
        raise ValueError("shared transform does not realize the declared 358-to-179-to-178 grid")
    return transformed


@dataclass(frozen=True)
class SelectionERP:
    family: str
    baseline_mode: str
    sfreq: float
    channels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    operator: FiniteFilterOperator
    prior_log_odds: float
    owner_metadata: dict
    state: tuple[tuple[str, np.ndarray], ...]
    provenance: dict

    def __post_init__(self):
        _validate_declaration(self.family, self.baseline_mode, self.sfreq,
                              self.channels, self.candidate_ids, self.operator)
        if not np.isfinite(self.prior_log_odds):
            raise ValueError("fitted binary prior must be finite")
        states = dict(self.state)
        if len(states) != len(self.state) or not states or any(not np.isfinite(v).all() for v in states.values()):
            raise ValueError("numeric model state requires distinct names and finite arrays")
        object.__setattr__(self, "state", tuple(
            (key, frozen_array(states[key]).reshape(np.asarray(states[key]).shape)) for key in sorted(states)))
        owner = json.loads(json.dumps(self.owner_metadata, allow_nan=False))
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if (provenance.get("fit_scope") != "target_calibration" or not provenance.get("records")
                or provenance.get("source_sha256") != _SOURCE_SHA256
                or provenance.get("epoch_geometry") != _geometry(self.baseline_mode)):
            raise ValueError("selection ERP needs complete current-source fit and physical-grid provenance")
        records = provenance["records"]
        require_identifiers([r["record_id"] for r in records], "fitted physical record IDs")
        counts = provenance.get("class_counts", [])
        if (len(counts) != 2 or any(type(n) is not int or n < 2 for n in counts)
                or sum(counts) != sum(r["events"] for r in records)
                or sum(counts) != provenance.get("requested_label_count")
                or not np.isclose(self.prior_log_odds, np.log(counts[1] / counts[0]), rtol=0, atol=1e-14)):
            raise ValueError("fitted class prior and physical label ledger disagree")
        if self.family == "waveform_lda":
            dimension = len(self.channels) * round(OUTPUT_SAMPLES * 20 / OUTPUT_RATE)
            shapes = {"feature_mean": (dimension,), "feature_scale": (dimension,),
                      "coefficient": (dimension,), "intercept": ()}
            if (set(states) != set(shapes) or any(states[k].shape != shape for k, shape in shapes.items())
                    or np.any(states["feature_scale"] <= 0)
                    or owner != {"owner": "transfer.binary.fit_shrinkage_feature_model",
                                  "features": "transfer.binary.target_shrinkage_features", "target_hz": 20.0,
                                  "prediction_dtype": owner.get("prediction_dtype")}
                    or owner.get("prediction_dtype") not in {"float32", "float64"}):
                raise ValueError("waveform anchor numeric state differs from original owner")
        else:
            expected_bin = 4 if self.family == "toeplitz_lda" else 1
            if (owner.get("family") != self.family or owner.get("baseline_mode") != self.baseline_mode
                    or owner.get("bin_width") != expected_bin or owner.get("input_shape") != [len(self.channels), OUTPUT_SAMPLES]
                    or owner.get("nfilter") != 4):
                raise ValueError("structured anchor state differs from frozen family/grid")
            StructuredERP.from_state(owner, states)
            if self.family == "xdawn_erp" and (owner["observed_rank"] < 4 or owner["effective_nfilter"] != 4):
                raise ValueError("fixed xDAWN4 requires at least four observed spatial dimensions")
        object.__setattr__(self, "owner_metadata", freeze_metadata(owner))
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_certificate", metadata_digest(self.record()))

    @classmethod
    def fit(cls, records, *, family, baseline_mode, sfreq, channels, candidate_ids,
            operator, expected_label_count=600):
        channels, candidate_ids = tuple(channels), tuple(candidate_ids)
        _validate_declaration(family, baseline_mode, sfreq, channels, candidate_ids, operator)
        records = tuple(records)
        if not records or any(not isinstance(r, CalibrationRecord) for r in records):
            raise ValueError("fit requires authorized complete CalibrationRecords")
        require_identifiers([r.record_id for r in records], "calibration physical records")
        if type(expected_label_count) is not int or expected_label_count < 4:
            raise ValueError("explicit positive authorized label budget required")
        epochs, labels, ledger = [], [], []
        for record in records:
            if record.right_censored is not False:
                raise ValueError("selection ERP anchor requires complete response support, not a censored prefix")
            x, onset = _input(record.samples, record.onsets, len(channels))
            y = np.asarray(record.target_mask)
            if y.dtype.kind not in "biu" or y.shape != onset.shape or not np.isin(y, [0, 1]).all():
                raise ValueError("aligned exact binary calibration labels required")
            transformed = selection_epochs(x, onset, operator=operator, baseline_mode=baseline_mode,
                                            n_channels=len(channels))
            epochs.append(transformed)
            labels.append(y.astype(np.int64))
            ledger.append({"record_id": record.record_id, "samples": len(x), "events": len(onset),
                "target_count": int(y.sum()), "input_sha256": array_digest(x, onset, y),
                "signal_sha256": array_digest(x, onset), "epoch_sha256": array_digest(transformed),
                "row_onsets": onset.tolist(), "row_start_inclusive": (onset + START).tolist(),
                "row_stop_exclusive": (onset + STOP).tolist(), "right_censored": False})
        X, y = np.concatenate(epochs), np.concatenate(labels)
        if len(y) != expected_label_count or set(y.tolist()) != {0, 1} or min(np.bincount(y)) < 2:
            raise ValueError("actual complete rows must equal the authorized budget with at least two per class")
        if family == "waveform_lda":
            features = target_shrinkage_features(X, sfreq=OUTPUT_RATE)
            fitted = fit_shrinkage_feature_model(features, y)
            scaler, classifier = fitted.steps[0][1], fitted.steps[1][1]
            if not np.array_equal(classifier.classes_, [0, 1]):
                raise ValueError("original waveform owner must produce non-target/target ordered classes")
            owner = {"owner": "transfer.binary.fit_shrinkage_feature_model",
                     "features": "transfer.binary.target_shrinkage_features", "target_hz": 20.0,
                     "prediction_dtype": str(classifier.coef_.dtype)}
            state = {"feature_mean": scaler.mean_, "feature_scale": scaler.scale_,
                     "coefficient": classifier.coef_[0], "intercept": classifier.intercept_[0]}
        else:
            if family == "xdawn_erp":
                observed = X.transpose(1, 0, 2).reshape(len(channels), -1)
                observed_rank = np.linalg.matrix_rank(observed - observed.mean(axis=1, keepdims=True))
                if observed_rank < 4:
                    raise ValueError("fixed xDAWN4 requires actual observed rank >=4 before shrinkage")
            fitted = StructuredERP(family, baseline_mode=baseline_mode,
                                   bin_width=4 if family == "toeplitz_lda" else 1, nfilter=4).fit(X, y)
            owner, state = fitted.export_state()
        counts = np.bincount(y, minlength=2).tolist()
        provenance = {"fit_scope": "target_calibration", "records": ledger,
            "requested_label_count": expected_label_count, "class_counts": counts,
            "epoch_geometry": _geometry(baseline_mode), "source_sha256": _SOURCE_SHA256,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "mne", "pyriemann")},
            "evidence_scope": "research_mature_ERP_anchor",
            "decision_aggregation": "sum_membership_binary_LLR_after_fitted_prior_log_odds_removal",
            "candidate_prior": "uniform", "probability_interpretation": "conditional_independence_approximation_for_overlapping_epochs",
            "held_probability_calibration": False, "independent_epochs_claimed": False}
        return cls(family, baseline_mode, sfreq, channels, candidate_ids, operator,
                   float(np.log(counts[1] / counts[0])), owner, tuple(state.items()), provenance)

    def assert_integrity(self):
        if metadata_digest(self.record()) != self._certificate:
            raise ValueError("frozen selection ERP state or provenance changed")

    def _logit(self, epochs):
        states = dict(self.state)
        if self.family == "waveform_lda":
            feature = target_shrinkage_features(epochs, sfreq=OUTPUT_RATE)
            # Match original StandardScaler's in-place floating dtype rounding,
            # then original sklearn LDA's two-dimensional affine operation.
            feature = feature.copy()
            feature -= states["feature_mean"].astype(feature.dtype, copy=False)
            feature /= states["feature_scale"].astype(feature.dtype, copy=False)
            dtype = np.dtype(self.owner_metadata["prediction_dtype"])
            coefficient = states["coefficient"].astype(dtype, copy=False)[None]
            intercept = states["intercept"].astype(dtype, copy=False).reshape(1)
            return (feature @ coefficient.T + intercept).ravel()
        return StructuredERP.from_state(dict(self.owner_metadata), states).predict_logit(epochs)

    def score(self, samples, onsets, membership, *, record_id, sfreq, channels, candidate_ids):
        self.assert_integrity()
        if sfreq != self.sfreq or tuple(channels) != self.channels or tuple(candidate_ids) != self.candidate_ids:
            raise ValueError("frozen sample rate, physical channels or vocabulary order differs")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("prediction requires a distinct physical record identity")
        x, onset = _input(samples, onsets, len(self.channels))
        if (record_id in {r["record_id"] for r in self.provenance["records"]}
                or array_digest(x, onset) in {r["signal_sha256"] for r in self.provenance["records"]}):
            raise ValueError("training selection cannot be rescored as held-out evidence")
        member = np.asarray(membership)
        if (member.shape != (len(onset), len(self.candidate_ids)) or member.dtype != np.bool_
                or np.any(member.sum(axis=0) == 0) or np.any(member.sum(axis=1) == 0)
                or np.any(member.sum(axis=1) == len(self.candidate_ids))):
            raise ValueError("complete exact boolean public membership required")
        epochs = selection_epochs(x, onset, operator=self.operator, baseline_mode=self.baseline_mode,
                                   n_channels=len(self.channels))
        llr = np.asarray(self._logit(epochs), dtype=np.float64) - self.prior_log_odds
        evidence = member.T @ llr
        if not np.isfinite(evidence).all():
            raise ValueError("nonfinite binary LLR or candidate evidence")
        best = int(np.argmax(evidence))
        tied = np.count_nonzero(np.isclose(evidence, evidence[best], rtol=0, atol=1e-10)) > 1
        return ContinuousDecisionResult(self.candidate_ids, frozen_array(evidence),
            frozen_array(np.exp(evidence - logsumexp(evidence))), None if tied else self.candidate_ids[best],
            "tied_evidence" if tied else "ok", len(x), array_digest(x, onset, member))

    def record(self):
        states = dict(self.state)
        return {"schema": SCHEMA, "family": self.family, "baseline_mode": self.baseline_mode,
            "sfreq": self.sfreq, "channels": self.channels, "candidate_ids": self.candidate_ids,
            "operator": self.operator.record(), "prior_log_odds": self.prior_log_odds,
            "owner_metadata": self.owner_metadata, "provenance": self.provenance,
            "state_keys": sorted(states), "state_sha256": array_digest(*(states[k] for k in sorted(states)))}

    def save(self, path):
        self.assert_integrity()
        metadata = self.record()
        metadata["artifact_sha256"] = metadata_digest(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{f"state__{k}": v for k, v in self.state},
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta.pop("artifact_sha256", None) != metadata_digest(meta) or meta.get("schema") != SCHEMA:
                raise ValueError("selection ERP artifact metadata changed or schema unsupported")
            keys = meta["state_keys"]
            if keys != sorted(set(keys)) or set(archive.files) != {"metadata", *(f"state__{k}" for k in keys)}:
                raise ValueError("selection ERP numeric archive keys differ")
            state = tuple((k, archive[f"state__{k}"]) for k in keys)
        if array_digest(*(v for _, v in state)) != meta["state_sha256"]:
            raise ValueError("selection ERP numeric state digest mismatch")
        op = meta["operator"]
        operator = FiniteFilterOperator(tuple(map(tuple, op["sos"])), op["padding_limit"])
        if op != operator.record():
            raise ValueError("selection ERP finite operator declaration differs")
        return cls(meta["family"], meta["baseline_mode"], meta["sfreq"], tuple(meta["channels"]),
            tuple(meta["candidate_ids"]), operator, meta["prior_log_odds"], meta["owner_metadata"], state,
            meta["provenance"])
