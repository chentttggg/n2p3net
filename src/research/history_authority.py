"""Original decoder/prefix authority for opt-in frozen-C history prediction.

Numerical histories cannot authorize themselves. CalibrationPrefixCompiler is
the sole public factory: it derives event prefixes from its verified original
MAT and replays the current view and actual C authority before issuing a scope.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from research.contracts import semantic_sha256
from research.latent_response_reference import ResponseState, _admit_law
from research.response_history_identification import NeighborEventPrefix, PublicSelection
from transfer.noise_model import array_digest, freeze_metadata


class HistoryUnavailableError(ValueError):
    def __init__(self, reason, *, failure_kind="not_available", evidence=None):
        self.reason, self.failure_kind = reason, failure_kind
        self.evidence = None if evidence is None else freeze_metadata(evidence)
        super().__init__(reason)

    def record(self):
        record = {"kind": "authorized_history_unavailable", "reason": self.reason,
                  "failure_kind": self.failure_kind}
        if self.evidence is not None:
            record["evidence"] = self.evidence
        return record


@dataclass(frozen=True, init=False)
class FrozenHistoryLaw:
    """A predeclared finite response law and independent uniform protocol q law.

    It extends the frozen C posterior; C is not refitted with latent states.
    An explicit correlated joint protocol table is retained and marginalized
    exactly. Learned q laws require a separate actual source/C producer rather
    than silently relabelling an external table as learned evidence.
    """

    states: tuple[ResponseState, ...]
    record: dict
    plan_bytes: bytes
    joint_mass: np.ndarray | None

    @classmethod
    def from_protocol_plan(cls, payload, *, model, prefix):
        from transfer.ensemble_views import calibration_prefix_identity

        if not isinstance(payload, bytes):
            raise TypeError("history law requires the actual frozen protocol plan bytes")
        plan = json.loads(payload)
        required = {"schema", "prior_kind", "freeze_role", "model_digest", "dictionary_digest",
                    "calibration_prefix_digest", "response_states"}
        explicit = plan.get("prior_kind") == "explicit_joint_selection_targets_and_response_states"
        if explicit:
            required |= {"selection_ids", "candidate_vocabularies", "joint_mass"}
        if (set(plan) != required or plan["schema"] != "n2p3_frozen_history_protocol_law/1"
                or plan["prior_kind"] not in {"independent_uniform_selection_targets",
                                               "explicit_joint_selection_targets_and_response_states"}
                or plan["freeze_role"] != "predeclared_protocol"
                or plan["model_digest"] != model.digest()
                or model.dictionary is None or plan["dictionary_digest"] != model.dictionary.digest()
                or plan["calibration_prefix_digest"] != calibration_prefix_identity(prefix)):
            raise ValueError("history protocol law differs from its frozen source/C/model authority")
        states = _admit_law(ResponseState(**state) for state in plan["response_states"])
        mass = None
        if explicit:
            from transfer.noise_model import frozen_array, require_identifiers

            names = tuple(plan["selection_ids"])
            require_identifiers(names, "complete joint history selection identities")
            vocabularies = plan["candidate_vocabularies"]
            if len(vocabularies) != len(names):
                raise ValueError("joint history target axes differ from declared selections")
            for vocabulary in vocabularies:
                require_identifiers(vocabulary, "complete joint history target vocabulary")
            mass = np.asarray(plan["joint_mass"], dtype=float)
            shape = tuple(len(v) for v in vocabularies)+(len(states),)*len(names)
            if (mass.shape != shape or not np.isfinite(mass).all() or np.any(mass < 0)
                    or not np.isclose(mass.sum(), 1., rtol=0, atol=64*np.finfo(float).eps)):
                raise ValueError("explicit prior must preserve complete target/response state Cartesian mass")
            amplitudes = np.array([[state.common_amplitude, state.target_amplitude] for state in states])
            for axis in range(len(names), len(shape)):
                marginal = mass.sum(axis=tuple(a for a in range(mass.ndim) if a != axis))
                # The global table normalization is admitted above. Multiple
                # axis reductions can have a different binary64 sum; the
                # physical mean anchor uses this marginal's normalized mass.
                if not np.allclose((marginal/marginal.sum())@amplitudes, [1., 1.],
                                   rtol=0, atol=64*np.finfo(float).eps):
                    raise ValueError("each explicit selection-state marginal must preserve the declared amplitude mean-one anchor")
            mass = frozen_array(mass)
        result = object.__new__(cls)
        object.__setattr__(result, "states", states)
        object.__setattr__(result, "plan_bytes", payload)
        object.__setattr__(result, "joint_mass", mass)
        object.__setattr__(result, "record", freeze_metadata({**plan,
            "plan_file_sha256": hashlib.sha256(payload).hexdigest(),
            "posterior_program": "frozen_original_C_posterior_predictive_extension_not_C_latent_refit"}))
        return result

    @property
    def digest(self):
        return semantic_sha256(self.record)

    def support(self, dictionary, onset):
        """Exact union over every state, never a caller-selected nominal L."""
        ranges = self.state_supports(dictionary, onset)
        return min(start for start, _ in ranges), max(stop for _, stop in ranges)

    def state_supports(self, dictionary, onset):
        """Every exact half-open interval; gaps in the envelope are not data."""
        rate = Fraction.from_float(float(dictionary.sample_rate_hz))
        ranges = []
        for state in self.states:
            start = Fraction(int(onset))+rate*Fraction.from_float(float(state.latency_seconds))
            stop = start+Fraction.from_float(float(state.width_multiplier))*len(dictionary.matrix)
            ranges.append((-(-start.numerator//start.denominator), -(-stop.numerator//stop.denominator)))
        return tuple(ranges)

    def condition_on_c_labels(self, selections, target_factors):
        """Exact full joint mass conditioned on original-owner C binary labels.

        State probabilities in response_states are the independent-law defaults.
        An explicit joint table owns its per-selection state marginals, including
        an exactly nominal C axis and nonnominal future axes.
        """
        from transfer.noise_model import frozen_array

        if self.joint_mass is None:
            return None
        if (tuple(self.record["selection_ids"]) != tuple(s.selection_id for s in selections)
                or tuple(tuple(v) for v in self.record["candidate_vocabularies"])
                != tuple(s.candidate_ids for s in selections)):
            raise ValueError("explicit joint prior axes differ from original cutoff history")
        mass = np.array(self.joint_mass, copy=True)
        for axis, selection in enumerate(selections):
            possible = np.asarray(target_factors[selection.selection_id]) > 0
            shape = [1]*mass.ndim
            shape[axis] = len(possible)
            mass *= possible.reshape(shape)
        total = mass.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("frozen joint history prior assigns zero mass to authorized C labels")
        return frozen_array(mass/total)


@dataclass(frozen=True, init=False)
class AuthorizedResponseHistory:
    selections: tuple
    current_selection_id: str
    view: object
    law: FrozenHistoryLaw
    binding: dict
    conditioned_joint_mass: np.ndarray | None

    @property
    def digest(self):
        return semantic_sha256(self.binding)

    def assert_model(self, model):
        if (model.digest() != self.binding["model_digest"]
                or model.noise.identity_digest != self.binding["noise_digest"]
                or model.dictionary.digest() != self.binding["dictionary_digest"]
                or self.law.digest != self.binding["law_digest"]):
            raise ValueError("authorized history model/noise/dictionary/law bytes changed")


def _issue_history_scope(compiler, view, *, prefix, model, law):
    """Called only by the original CalibrationPrefixCompiler ingress."""
    from data.bnci2014_008_candidate import BNCI2014_008_GRID_SHAPE
    from data.candidate_task import MembershipDecision
    from data.preprocess import prefix_raw_scopes
    from transfer.ensemble_views import (
        PredictionDecisionView,
        calibration_prefix_identity,
    )

    if not isinstance(view, PredictionDecisionView) or not isinstance(law, FrozenHistoryLaw):
        raise TypeError("history requires original current view and frozen protocol law")
    prefix = compiler.assert_original_prefix(prefix)
    if (prefix.raw is None or model.noise is None or model.dictionary is None
            or model._posterior_precision_root is None):
        raise HistoryUnavailableError("actual_frozen_C_posterior_noise_dictionary_required")
    if (law.record["model_digest"] != model.digest()
            or law.record["calibration_prefix_digest"] != calibration_prefix_identity(prefix)):
        raise ValueError("history law is not bound to this actual frozen C producer")
    if not view.decision.schedule_complete or np.any(view.decision.evidence_indices < 0):
        raise HistoryUnavailableError("original_current_complete_observation_gate_failed", failure_kind="incomplete")
    target, raw = compiler._arguments["target"], compiler._record
    requested_view = view
    overlap_reasons = {"previous_unknown_decision_overlaps_finite_response_support",
                       "next_unknown_decision_overlaps_finite_response_support"}
    if view.samples is None:
        if view.unavailable_reason not in overlap_reasons:
            raise HistoryUnavailableError(view.unavailable_reason or "original_current_observation_unavailable")
        # The baseline remains unavailable. The opt-in source owner supplies
        # exactly the same requested interval for the new history programme;
        # coverage/held/read/actual-clock gates below still independently apply.
        view = replace(view, samples=np.asarray(raw.eeg_uv[slice(*view.raw_scope)], dtype=np.float64)*1e-6,
                       unavailable_reason=None)
    physical = view.physical_record()
    for key in ("participant_key", "source_sha256", "source_relative_path", "source_dataset_id",
                "source_subject_id", "sample_rate_hz", "reference", "signal_unit"):
        if physical[key] != target[key]:
            raise ValueError("current history view differs from original participant/raw/reference authority")
    if (view.channel_names != tuple(target["channel_names"])
            or view.channel_types != tuple(target["channel_types"])
            or model.sfreq != view.sample_rate_hz or model.channels != view.channel_names
            or model.operator_identity != "native_unfiltered_volts/1"):
        raise HistoryUnavailableError("history_operator_projection_or_propagated_filter_not_authorized")
    if (model.candidate_ids != view.decision.candidate_vocabulary
            or view.decision.candidate_vocabulary != tuple(target["candidate_vocabulary"])):
        raise ValueError("history current vocabulary/order differs from original fitted candidate authority")
    current_rows = np.flatnonzero(raw.selection_id == view.decision_id)
    if not len(current_rows):
        raise ValueError("current physical selection is absent from original actual event ledger")
    current = MembershipDecision.row_column(raw.candidate_id[current_rows],
        n_rows=BNCI2014_008_GRID_SHAPE[0], n_columns=BNCI2014_008_GRID_SHAPE[1],
        event_ids=tuple(f"{target['source_dataset_id']}:{raw.subject_id}:{int(row)}" for row in current_rows),
        repetition_indices=raw.repetition_index[current_rows], repetitions=view.decision.repetitions)
    if (current.record() != view.decision.record()
            or not np.array_equal(view.onsets+view.raw_scope[0], raw.flash_sample[current_rows])
            or not np.array_equal(view.samples, np.asarray(raw.eeg_uv[slice(*view.raw_scope)], dtype=np.float64)*1e-6)):
        raise ValueError("current samples/events/membership differ from actual original-source replay")
    _, test_scope = prefix_raw_scopes(compiler._raw, compiler._events,
        compiler._arguments["authorization_preprocessing"], calibration_events=0,
        maximum_calibration_events=compiler._arguments["maximum"])
    start, stop = view.raw_scope
    if not test_scope.start_sample <= start < stop <= test_scope.stop_sample or view.available_sample != stop:
        raise ValueError("history current read/cutoff exceeds original authorized held scope")
    if start < prefix.target_contract["raw_scope"][1]:
        raise ValueError("history read overlaps actual C fit support")
    fit = model.provenance.get("ensemble_fit_binding")
    if (fit is None or semantic_sha256(fit.get("target_contract")) != semantic_sha256(prefix.target_contract)
            or not fit.get("training_receipts")):
        raise ValueError("history model lacks its actual parent C fit authority")
    expected_signal = array_digest(prefix.raw.samples, prefix.raw.onsets)
    if (len(model.provenance["records"]) != 1
            or model.provenance["records"][0]["record_id"] != prefix.raw.record_id
            or model.provenance["records"][0]["signal_sha256"] != expected_signal):
        raise ValueError("frozen response posterior did not consume this exact authorized C signal")
    native = model.provenance["records"][0]
    expected_native = {"n_samples": len(prefix.raw.samples), "n_events": prefix.budget,
        "n_targets": int(prefix.raw.target_mask.sum()),
        "input_sha256": array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask),
        "raw_input_sha256": array_digest(prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask),
        "right_censored": prefix.raw.right_censored, "read_sample_support": [0, len(prefix.raw.samples)],
        "input_dtypes": {"samples": prefix.raw.samples.dtype.str, "onsets": prefix.raw.onsets.dtype.str,
                         "target_mask": prefix.raw.target_mask.dtype.str}}
    if (model.provenance["fit_scope"] != "target_calibration"
            or any(semantic_sha256(native.get(key)) != semantic_sha256(value)
                   for key, value in expected_native.items())):
        raise ValueError("actual native fitter samples/labels/dose/dtypes differ from original C authority")
    receipts = fit["training_receipts"]
    if (len(receipts) != 1 or receipts[0]["raw_input_digest"] != array_digest(
            prefix.raw.samples, prefix.raw.onsets, prefix.raw.target_mask)
            or semantic_sha256(receipts[0]["label_use"]) != semantic_sha256(prefix.label_use)
            or tuple(receipts[0]["raw_scope"]) != tuple(prefix.target_contract["raw_scope"])):
        raise ValueError("parent fit receipts differ from actual C labels/read authority")
    if model.noise.startup_mode not in {"stationary", "stationary_innovations"}:
        raise HistoryUnavailableError("conditional_noise_has_no_joint_history_startup_law")
    if law.support(model.dictionary, 0)[1] > start:
        raise HistoryUnavailableError("outside_record_event_history_can_affect_observed_union_support")
    occurred = np.flatnonzero(raw.flash_sample < view.available_sample)
    event_ledger, selections, prior_factors, zero_axes, c_selections = [], [], {}, [], []
    for name in dict.fromkeys(raw.selection_id[occurred]):
        indices = occurred[raw.selection_id[occurred] == name]
        member = MembershipDecision.row_column(raw.candidate_id[indices],
            n_rows=6, n_columns=6,
            event_ids=tuple(f"{target['source_dataset_id']}:{raw.subject_id}:{int(row)}" for row in indices),
            repetition_indices=raw.repetition_index[indices], repetitions=view.decision.repetitions)
        onsets = raw.flash_sample[indices]
        selection_type = PublicSelection if name == view.decision_id else NeighborEventPrefix
        selections.append(selection_type(str(name), member.candidate_vocabulary, onsets, member.membership))
        # Only authorized C binary labels constrain protocol target states.
        # Decoder held target fields are never read here.
        known = indices < prefix.budget
        possible = np.ones(len(member.candidate_vocabulary), dtype=bool)
        if known.any():
            c_selections.append(str(name))
            possible = np.all(member.membership[known] == prefix.raw.target_mask[indices[known], None], axis=0)
        if not possible.any():
            raise ValueError("authorized C labels contradict original public history membership")
        prior_factors[str(name)] = (possible.astype(float)/possible.sum()).tolist()
        union_visible = any(max(lo, start) < min(hi, stop) for o in onsets
                            for lo, hi in law.state_supports(model.dictionary, int(o)))
        if not union_visible and name != view.decision_id:
            zero_axes.append(str(name))
        for row, onset in zip(indices, onsets, strict=True):
            lo, hi = law.support(model.dictionary, int(onset))
            event_ledger.append({"event_id": f"{target['source_dataset_id']}:{raw.subject_id}:{int(row)}",
                "selection_id": str(name), "actual_onset_sample": int(onset),
                "union_response_support": [lo, hi], "retained_support": [max(lo, start), min(hi, stop)]
                if max(lo, start) < min(hi, stop) else None, "right_censored": hi > stop})
    if any(s.selection_id == view.decision_id and len(s.onsets) != len(current_rows) for s in selections):
        raise HistoryUnavailableError("current_schedule_not_fully_occurred_at_actual_cutoff")
    binding = {"schema": "n2p3_authorized_response_history/1", "current": view.record(),
        "original_requested_current": requested_view.record(),
        "baseline_gate": requested_view.unavailable_reason,
        "opt_in_overlap_recovery": requested_view.samples is None,
        "original_source_sha256": raw.source_sha256, "source_owner": "CalibrationPrefixCompiler",
        "source_owner_implementation": compiler.history_owner_source,
        "prefix_identity": calibration_prefix_identity(prefix), "fit_binding_sha256": semantic_sha256(fit),
        "model_digest": model.digest(), "noise_digest": model.noise.identity_digest,
        "dictionary_digest": model.dictionary.digest(), "law_digest": law.digest,
        "history_selections": [{"selection_id": s.selection_id, "candidate_ids": list(s.candidate_ids),
            "onsets": s.onsets.tolist(), "membership": s.membership.tolist(),
            "role": "current_complete" if isinstance(s, PublicSelection) else "neighbour_actual_event_prefix"}
            for s in selections], "full_actual_event_ledger": event_ledger,
        "target_prior_factors": prior_factors,
        "prior_factorization": "predeclared_independent_uniform_targets_conditioned_only_on_authorized_C_labels",
        "response_state_factors": [state.probability for state in law.states],
        "C_fit_selection_ids": c_selections,
        "exact_zero_design_axis_marginalization": {"selection_ids": zero_axes,
            "proof": "all_original_events_all_frozen_response_states_support_disjoint_from_read_and_prefix",
            "mass_operation": "sum_complete_independent_target_and_response_state_mass_equals_one",
            "full_vocabulary_and_state_law_retained": True},
        "clock_kind": "original_decoder_actual_native_integer_sample",
        "operator_identity": model.operator_identity, "read_support": [start, stop],
        "conditioning_support": [start, start+model.conditioning_samples],
        "score_support": [start+model.conditioning_samples, stop], "available_sample": stop,
        "current_response_right_censored": any(law.support(model.dictionary, int(o))[1] > stop
                                               for o in raw.flash_sample[current_rows]),
        "posterior_program": "frozen_C_posterior_predictive_extension", "truth_consumed": False}
    if law.joint_mass is not None:
        names = tuple(s.selection_id for s in selections)
        if (tuple(law.record["selection_ids"]) != names
                or tuple(tuple(v) for v in law.record["candidate_vocabularies"])
                != tuple(s.candidate_ids for s in selections)):
            raise ValueError("explicit complete joint prior axes differ from original cutoff history")
        binding["prior_factorization"] = "explicit_frozen_joint_mass_exact_C_conditioning_and_axis_marginalization"
        binding["exact_zero_design_axis_marginalization"]["mass_operation"] = (
            "condition_complete_joint_mass_on_authorized_C_then_sum_the_actual_zero_image_axes")
    conditioned_mass = law.condition_on_c_labels(selections, prior_factors)
    binding["C_label_conditioned_joint_mass_sha256"] = (
        None if conditioned_mass is None else array_digest(conditioned_mass))
    binding["C_selection_state_admission"] = []
    binding["exact_zero_design_axis_marginalization"]["proof"] = (
        "complete_original_ledger_union_support_or_actual_C_conditioned_positive_state_full_prefix_zero_design")
    from research.latent_response_reference import selection_design

    for axis, selection in enumerate(selections):
        if selection.selection_id not in c_selections:
            continue
        if conditioned_mass is None:
            # Support is an exact Boolean predicate. Multiplying a tiny
            # positive state mass by q=1/36 can underflow and must not erase
            # a physically active C state before the mixture sees it.
            local_mass = np.logical_and(np.asarray(prior_factors[selection.selection_id])[:, None] > 0,
                np.asarray([state.probability for state in law.states])[None, :] > 0)
        else:
            retained_axes = (axis, axis+len(selections))
            local_mass = conditioned_mass.sum(axis=tuple(a for a in range(conditioned_mass.ndim) if a not in retained_axes))
        positive = np.argwhere(local_mass > 0)
        positive_states = sorted(set(int(state) for _, state in positive))
        nominal = all((law.states[state].common_amplitude, law.states[state].target_amplitude,
                       law.states[state].latency_seconds, law.states[state].width_multiplier) == (1., 1., 0., 1.)
                      for state in positive_states)
        if selection.selection_id in zero_axes:
            binding["C_selection_state_admission"].append({"selection_id": selection.selection_id,
                "program": "complete_union_support_structural_zero", "positive_state_indices": positive_states})
            continue
        # A theta is owned by the whole selection, including its continuation
        # after a partial C fit. Inspect the complete, C-conditioned positive
        # q/theta support; global defaults cannot override an explicit C clamp.
        witness = None
        for candidate, state_index in positive:
            design, execution = selection_design(model.dictionary, selection.onsets,
                selection.membership[:, candidate], origin_sample=start, observed_samples=stop-start,
                state=law.states[state_index], support="translated_physical")
            if np.any(design != 0):
                witness = {"response_state_index": int(state_index), "candidate_index": int(candidate),
                    "actual_design_sha256": array_digest(design), "execution": execution}
                break
        if witness is None:
            zero_axes.append(selection.selection_id)
            binding["exact_zero_design_axis_marginalization"]["selection_ids"] = list(zero_axes)
            binding["C_selection_state_admission"].append({"selection_id": selection.selection_id,
                "program": "all_C_conditioned_positive_q_state_designs_exactly_zero",
                "positive_state_indices": positive_states})
        elif nominal:
            binding["C_selection_state_admission"].append({"selection_id": selection.selection_id,
                "program": "all_C_conditioned_positive_states_are_original_nominal",
                "positive_state_indices": positive_states, "actual_nonzero_image": witness})
        else:
            raise HistoryUnavailableError("C_history_latent_state_not_conditioned_by_original_frozen_fit",
                evidence={"original_authority": binding, "C_fit_selection_id": selection.selection_id,
                    "positive_state_indices": positive_states, **witness,
                    "old_C_posterior": "beta_given_C_nominal_selection_state",
                    "new_C_latent_producer_required": True})
    binding["exact_zero_design_axis_marginalization"]["selection_ids"] = zero_axes
    result = object.__new__(AuthorizedResponseHistory)
    for name, value in {"selections": tuple(selections), "current_selection_id": view.decision_id,
                        "view": view, "law": law, "binding": freeze_metadata(binding),
                        "conditioned_joint_mass": conditioned_mass}.items():
        object.__setattr__(result, name, value)
    return result
