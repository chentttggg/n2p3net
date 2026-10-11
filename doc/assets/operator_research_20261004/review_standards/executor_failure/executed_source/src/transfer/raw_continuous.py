"""Attested BNCI MAT records to legal continuous calibration and decision views.

This is a distinct 256 Hz, native-reference research route. Epoch baseline and
resampling are never applied. The existing physical prefix split controls raw
authorization; the continuous calibration view stops before the first unknown
flash and uses explicitly right-censored response kernels for the last labels.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np

from data.bnci2014_008_candidate import (
    BNCI2014_008_CHANNELS,
    BNCI2014_008_SOURCE_REFERENCE,
    BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
    read_bnci2014_008_candidate_record,
)
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.preprocess import (
    design_continuous_sos,
    filter_bidirectional_segment,
    prefix_raw_scopes,
    source_epoch_support,
)
from research.contracts import semantic_sha256
from transfer.binary import BinaryTimeSplit, resolve_calibration_splits
from transfer.candidate_decision import candidate_event_key, row_column_target
from transfer.continuous_decision import CalibrationRecord
from transfer.noise_model import array_digest, frozen_array
from transfer.outcomes import CandidateCoverage

SFREQ = BNCI2014_008_SOURCE_SAMPLE_RATE_HZ
CANDIDATES = tuple(row_column_target(row, column) for row in range(6) for column in range(6))


def _fixed_array(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ContinuousInputPlan:
    """Predeclared numerical route, distinct from the epoch classifier contract."""

    response_seconds: float = 1.2
    n_basis: int = 8
    ridge: float = 1.0
    pre_roll_seconds: float = 0.2
    right_censored_calibration: bool = True
    minimum_test_trials: int = 200
    test_size_policy: str = "limited_data_80_percent"

    def __post_init__(self):
        if (not np.isfinite(self.response_seconds) or not 0 < self.response_seconds <= 1.2
                or not np.isfinite(self.pre_roll_seconds) or not 0 <= self.pre_roll_seconds <= .2
                or type(self.n_basis) is not int or not 1 <= self.n_basis <= round(SFREQ * self.response_seconds)
                or not np.isfinite(self.ridge) or self.ridge <= 0):
            raise ValueError("invalid continuous response, basis, ridge or preroll plan")
        if self.right_censored_calibration is not True:
            raise ValueError("continuous prefix fitting requires explicit right-censored calibration")
        if type(self.minimum_test_trials) is not int or self.minimum_test_trials < 1:
            raise ValueError("minimum test trials must be positive")
        if self.test_size_policy not in {"limited_data_80_percent", "strict_minimum"}:
            raise ValueError("unknown test size policy")

    def record(self):
        spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
        return {"schema": "n2p3_continuous_input_plan/1", **asdict(self),
                "sample_rate_hz": SFREQ, "channels": list(BNCI2014_008_CHANNELS),
                "reference": BNCI2014_008_SOURCE_REFERENCE, "signal_unit": "V",
                "baseline": "none", "resampling": "none", "filter_phase": "zero",
                "filter_edge_policy": spec.zero_phase_edge_policy,
                "l_freq": spec.l_freq, "h_freq": spec.h_freq, "filter_order": spec.filter_order,
                "notch_freqs_hz": list(spec.notch_freqs_hz), "notch_quality_factor": spec.notch_quality_factor,
                "authorization_epoch_support_ms": [spec.tmin_ms, spec.tmax_ms],
                "decision_model": "finite_causal_ERP_kernel_after_scope_filtering_approximation",
                "independent_filtered_decisions_claimed": False,
                "filter_state_uncertainty_integrated": False}


@dataclass(frozen=True)
class ContinuousDecisionView:
    """Prediction-facing view: public schedule and samples, with no target truth."""

    decision_id: str
    samples: np.ndarray | None
    onsets: np.ndarray
    membership: np.ndarray
    coverage: CandidateCoverage
    failure_reason: str | None
    start_sample: int
    stop_sample: int
    onset_start_sample: int
    onset_end_sample: int
    available_sample: int


@dataclass(frozen=True)
class ContinuousSubjectViews:
    subject: str
    calibration: dict[int, CalibrationRecord]
    decisions: tuple[ContinuousDecisionView, ...]
    scoring_truth: dict[str, str]
    ledger: dict


def _membership(flash_codes):
    """Only physical row/column flash codes enter the 36 hypothesis designs."""
    flash = np.asarray(flash_codes)
    if flash.ndim != 1 or flash.dtype.kind not in "iu" or np.any((flash < 0) | (flash >= 12)):
        raise ValueError("BNCI membership requires physical flash codes 0..11")
    return (flash[:, None] == np.repeat(np.arange(6), 6)[None, :]) | (
        flash[:, None] == 6 + np.tile(np.arange(6), 6)[None, :])


def _filter_scope(record, start, stop, sos, padlen):
    values = np.asarray(record.eeg_uv[start:stop], dtype=np.float64).T * 1e-6
    if values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("continuous scope has insufficient finite raw observations")
    result = filter_bidirectional_segment(sos, values, padding_limit=padlen)
    data = frozen_array(result.data.T)
    return data, {"start_sample": start, "stop_sample": stop,
                  "raw_values_sha256": array_digest(values), "filtered_values_sha256": array_digest(data),
                  "padding_samples": result.padding_samples,
                  "effective_filter_states_sha256": array_digest(result.left_state, result.right_state)}


def _calibration_view(record, budget, authorized_stop, plan, sos, padlen):
    # A K prefix ends inside a selection. Reading its full last response tail
    # would include the next unauthorized flash. Censor samples, not K labels.
    next_onset = int(record.flash_sample[budget])
    stop = min(int(authorized_stop), next_onset)
    samples, receipt = _filter_scope(record, 0, stop, sos, padlen)
    onsets = np.array(record.flash_sample[:budget], dtype=np.int64, copy=True)
    labels = np.array(record.raw_is_target[:budget], dtype=bool, copy=True)
    width = round(plan.response_seconds * SFREQ)
    censored = onsets + width > stop
    fitted = CalibrationRecord(f"{record.subject_id}:C{budget}:{record.source_sha256}", samples,
                               onsets, labels, right_censored=True)
    return fitted, {**receipt, "authorized_stop_sample": int(authorized_stop),
                    "excluded_next_flash_onset": next_onset, "last_included_sample": stop - 1,
                    "authorized_label_count": budget, "retained_label_count": len(labels),
                    "label_sha256": array_digest(labels), "right_censored_calibration": True,
                    "right_censored_event_indices": np.flatnonzero(censored).tolist(),
                    "observed_response_samples": np.minimum(width, stop - onsets).tolist(),
                    "input_sha256": array_digest(samples, onsets, labels),
                    "record_id": fitted.record_id}


def _prepare_record(record, budgets, plan):
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    events = np.column_stack([record.flash_sample, np.zeros(len(record.flash_sample), dtype=int),
                              np.ones(len(record.flash_sample), dtype=int)])
    # Geometry only: planning never reads EEG or target labels.
    class Geometry:
        first_samp = 0
        n_times = len(record.eeg_uv)
        info = {"sfreq": SFREQ}

    starts, stops = source_epoch_support(events, SFREQ, spec)
    rows = np.arange(len(events))

    def scopes(k, maximum):
        return prefix_raw_scopes(Geometry(), events, spec, calibration_events=k,
                                 maximum_calibration_events=maximum)

    def builder(k, maximum):
        if maximum == 0:
            return BinaryTimeSplit(rows[:0], rows.copy(), rows[:0])
        _, test_scope = scopes(k, maximum)
        later = rows >= maximum
        valid = later & (starts >= test_scope.start_sample) & (stops <= test_scope.stop_sample)
        return BinaryTimeSplit(rows[:k], rows[valid], rows[later & ~valid])

    splits, adjustments = resolve_calibration_splits(tuple(budgets), builder,
        minimum_test_trials=plan.minimum_test_trials, available_trials=len(events),
        test_size_policy=plan.test_size_policy)
    maximum = max(splits)
    if maximum == 0:
        raise ValueError("continuous classifier requires positive authorized calibration labels")
    _, test_scope = scopes(0, maximum)
    sos = design_continuous_sos(SFREQ, l_freq=spec.l_freq, h_freq=spec.h_freq,
        order=spec.filter_order, notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor)
    padlen = int(mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)["padlen"])
    test, test_receipt = _filter_scope(record, test_scope.start_sample, test_scope.stop_sample, sos, padlen)
    width = round(plan.response_seconds * SFREQ)
    preroll = round(plan.pre_roll_seconds * SFREQ)
    selections = np.unique(record.selection_index)
    selection_rows = {int(selection): np.flatnonzero(record.selection_index == selection) for selection in selections}
    gaps = [int(record.flash_sample[selection_rows[int(right)][0]] - record.flash_sample[selection_rows[int(left)][-1]])
            for left, right in zip(selections[:-1], selections[1:], strict=True)]
    calibration, receipts = {}, {}
    for budget in splits:
        if budget:
            authorized, _ = scopes(budget, maximum)
            calibration[budget], receipts[str(budget)] = _calibration_view(
                record, budget, authorized.stop_sample, plan, sos, padlen)
    expected = {candidate_event_key(code): 10 for code in range(12)}
    scored_rows = set(splits[maximum].test_rows.tolist())
    decisions, truths, sample_scopes = [], {}, []
    for selection in selections:
        indices = selection_rows[int(selection)]
        if not np.any(indices >= maximum):
            continue
        decision_id = str(record.selection_id[indices[0]])
        # Truth is exported exclusively to the scorer, never to a prediction view.
        truths[decision_id] = row_column_target(int(record.target_row[indices[0]]), int(record.target_col[indices[0]]))
        observed = np.asarray([row for row in indices if int(row) in scored_rows], dtype=int)
        codes = record.candidate_id[observed]
        coverage = CandidateCoverage.from_mappings(expected, {
            candidate_event_key(code): int(np.sum(codes == code)) for code in range(12)})
        first, last = int(record.flash_sample[indices[0]]), int(record.flash_sample[indices[-1]])
        start, stop = first - preroll, last + width
        reason = None
        if not coverage.complete:
            reason = "partial_decision_after_fixed_Kmax_and_original_epoch_support_purge"
        elif start < test_scope.start_sample or stop > test_scope.stop_sample:
            reason = "decision_response_outside_authorized_test_scope"
        elif (int(selection) > 0 and first - int(record.flash_sample[selection_rows[int(selection) - 1][-1]]) < width + preroll):
            reason = "previous_unknown_decision_overlaps_finite_response_support"
        elif (int(selection) < int(selections[-1]) and int(record.flash_sample[selection_rows[int(selection) + 1][0]]) < stop):
            reason = "next_unknown_decision_overlaps_finite_response_support"
        samples = None
        if reason is None:
            samples = frozen_array(test[start - test_scope.start_sample:stop - test_scope.start_sample])
            if sample_scopes and start < sample_scopes[-1][1]:
                raise ValueError("continuous decision scopes reuse raw sample positions")
            sample_scopes.append([start, stop])
        decisions.append(ContinuousDecisionView(decision_id, samples,
            _fixed_array(record.flash_sample[observed] - start, np.int64),
            _fixed_array(_membership(codes), bool), coverage, reason, start, stop, first, last,
            test_scope.stop_sample))
    return ContinuousSubjectViews(record.subject_id, calibration, tuple(decisions), truths, {
        "schema": "n2p3_bnci_continuous_subject_scope/1", "source_sha256": record.source_sha256,
        "source_relative_path": record.source_relative_path, "continuous_input_plan": plan.record(),
        "operator_identity": semantic_sha256(plan.record()), "sos_sha256": array_digest(sos),
        "requested_Ks": list(budgets), "actual_Ks": sorted(calibration),
        "adjustments": {str(k): reason for k, reason in adjustments.items()},
        "maximum_calibration_events": maximum, "calibration": receipts,
        "test": {**test_receipt, "scope": test_scope.record(),
                 "eligible_event_indices": sorted(scored_rows), "purged_event_indices": splits[maximum].purged_rows.tolist(),
                 "test_budget": splits[maximum].test_budget.record()},
        "selection_gaps_samples": gaps, "minimum_gap_seconds": min(gaps) / SFREQ,
        "requested_decisions": len(decisions), "scorable_decisions": sum(d.samples is not None for d in decisions),
        "decision_sample_scopes": sample_scopes,
        "raw_samples_used_once_per_decision_likelihood": True,
        "different_decisions_use_disjoint_sample_positions": True,
        "scope_filtering_makes_decisions_statistically_independent": False,
        "offline_filter_availability_sample": test_scope.stop_sample})


def prepare_bnci_continuous(path, *, source_sha256, budgets, plan=None):
    """Verify original bytes once and decode those bytes, then enforce split roles."""
    plan = ContinuousInputPlan() if plan is None else plan
    if not isinstance(plan, ContinuousInputPlan):
        raise TypeError("continuous input requires an explicit ContinuousInputPlan")
    path = Path(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source_sha256:
        raise ValueError("BNCI MAT differs from the frozen source byte declaration")
    record = read_bnci2014_008_candidate_record(io.BytesIO(payload), source_relative_path=path.name,
                                               source_sha256=source_sha256)
    return _prepare_record(record, budgets, plan)
