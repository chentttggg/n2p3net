"""Complete adaptive-input perturbation audits on declared native EEG support.

This owner composes the actual segment-local harmonic cleaner and finite SOS
operator; it never estimates another filter or a covariance/noise model. Every
challenge/draw reexecutes the complete program, including fitted frequencies,
robust iterations and application gates. Empirical output moments describe the
explicit perturbation distribution conditional on one supplied base record.
They are neither F Sigma F.T nor a Gaussian candidate likelihood or an efficacy
claim. A caller must separately bind physical source/time/C authorization.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from data import finite_filter_operator as finite_owner
from data import harmonic as harmonic_owner
from data.finite_filter_operator import FiniteFilterOperator
from data.harmonic import HarmonicConfig, clean_harmonic_segments
from research.boundary_observability import NativeBoundaryGrid
from research.contracts import semantic_sha256
from transfer.noise_model import array_digest, finite_real_array, freeze_metadata, frozen_array

if TYPE_CHECKING:
    from transfer.ensemble_views import PredictionDecisionView

_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_OWNER_SOURCE = freeze_metadata({
    "data.harmonic": hashlib.sha256(Path(harmonic_owner.__file__).read_bytes()).hexdigest(),
    "data.finite_filter_operator": hashlib.sha256(Path(finite_owner.__file__).read_bytes()).hexdigest(),
})


def _scope(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(item) is not int for item in value) or not 0 <= value[0] < value[1]):
        raise ValueError("input segments require absolute integer half-open sample bounds")
    return tuple(value)


def _identity(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicitly identified")
    return value


@dataclass(frozen=True)
class ContaminationInput:
    """An immutable native record with explicit observation and diagnostic role.

The canonical caller supplies an actual view/grid and authorized segments. An
identity string, role or SHA cannot issue physical read/fit permission. EOG and
missing channels are deliberately excluded from this EEG-only temporal audit.
"""

    grid: NativeBoundaryGrid
    samples: np.ndarray = field(repr=False, compare=False)
    segments: tuple[tuple[int, int], ...]
    available_at_sample: int
    role: str
    input_level: str
    observation_identity: str

    def __post_init__(self):
        if not isinstance(self.grid, NativeBoundaryGrid):
            raise TypeError("input audit requires the actual native physical grid")
        if any(name.upper() in {"EOG", "HEOG", "VEOG", "REF", "GND"}
               for name in self.grid.channel_names):
            raise ValueError("temporal input audit only consumes explicitly measured EEG channels")
        if (self.role not in {"source_training", "target_calibration", "test_offline_adaptive", "synthetic_math"}
                or self.input_level not in {"distributed_processed_eeg", "physical_adc_eeg", "synthetic_math"}
                or ((self.role == "synthetic_math") != (self.input_level == "synthetic_math"))):
            raise ValueError("declare the actual signal level and permitted input-audit role")
        _identity(self.observation_identity, "actual observation")
        values = self.grid.values(self.samples, self.grid.context_scope, "observed native EEG")
        scopes = tuple(_scope(value) for value in self.segments)
        if (not scopes or scopes[0][0] != self.grid.context_scope[0]
                or scopes[-1][1] != self.grid.context_scope[1]
                or any(left[1] != right[0] for left, right in zip(scopes[:-1], scopes[1:], strict=True))
                or type(self.available_at_sample) is not int
                or self.available_at_sample < self.grid.context_scope[1]):
            raise ValueError("actual segments must partition the observed grid and respect availability")
        object.__setattr__(self, "samples", frozen_array(values))
        object.__setattr__(self, "segments", scopes)

    def record(self):
        return {"schema": "n2p3_contamination_input/1", "native_grid": self.grid.record(),
                "segments": [list(value) for value in self.segments],
                "available_at_sample": self.available_at_sample, "role": self.role,
                "input_level": self.input_level, "observation_identity": self.observation_identity,
                "samples_sha256": array_digest(self.samples), "authority_issued_by_audit": False,
                "upstream_processing_inverted_or_reconstructed": False}

    @classmethod
    def from_prediction_view(cls, view: PredictionDecisionView, *, role, input_level, segments=None):
        """Consume the existing unlabeled native view without importing model code.

        The caller supplies an already admitted PredictionDecisionView. This
        adapter keeps its real source/availability/channel identities, and never
        accepts CalibrationSelection or fetches a hidden target label. Physical
        authorization remains with that canonical producer.
        """
        physical = view.physical_record()
        if (physical.get("schema") != "n2p3_prediction_decision_physical_support/1"
                or view.samples is None or tuple(view.channel_types) != ("EEG",) * len(view.channel_names)):
            raise ValueError("audit needs an available canonical unlabeled native EEG observation")
        grid = NativeBoundaryGrid(tuple(view.raw_scope), view.sample_rate_hz, tuple(view.channel_names),
            view.reference, view.source_sha256,
            f"{view.source_dataset_id}:{view.source_subject_id}:{view.source_relative_path}", view.signal_unit)
        return cls(grid, view.samples, (tuple(view.raw_scope),) if segments is None else segments,
                   view.available_sample, role, input_level, view.raw_input_digest)


@dataclass(frozen=True)
class ContaminationProgram:
    """One declared offline complete T; actual owners retain all numerical work.

Execution order is harmonic segment-local estimation, then each segment's
finite SOS. Either stage may be absent, giving a predeclared control. No raw
reference, channel mapping, resampling, EOG regression or gap fill is implied.
"""

    harmonic: HarmonicConfig | None = None
    finite_operator: FiniteFilterOperator | None = None

    def __post_init__(self):
        if self.harmonic is not None and not isinstance(self.harmonic, HarmonicConfig):
            raise TypeError("harmonic program requires the existing validated config type")
        if self.finite_operator is not None and not isinstance(self.finite_operator, FiniteFilterOperator):
            raise TypeError("finite stage requires the actual existing finite SOS owner")

    def record(self):
        return {"schema": "n2p3_contamination_program/1",
                "stages": [name for name, value in (("segment_local_adaptive_harmonic", self.harmonic),
                                                    ("segment_local_finite_SOS", self.finite_operator)) if value is not None],
                "harmonic": None if self.harmonic is None else asdict(self.harmonic),
                "finite_operator": None if self.finite_operator is None else self.finite_operator.record(),
                "state_role": "full_segment_refit_at_each_actual_input" if self.harmonic is not None else "fixed_deterministic_map",
                "output_stage": "same_native_samples_channels_reference_volts",
                "default_or_policy_selected_here": False}

    def _execute(self, observation, values):
        # This private path consumes the once-validated native shape. Independent
        # executions differ only in explicit additive challenges/draws.
        origin = observation.grid.context_scope[0]
        local_segments = tuple((left - origin, right - origin) for left, right in observation.segments)
        windows = ()
        output = values
        if self.harmonic is not None:
            cleaned = clean_harmonic_segments(values.T, observation.grid.sample_rate_hz,
                                              local_segments, self.harmonic)
            output = cleaned.cleaned.T
            windows = tuple({**row, "physical_channel": observation.grid.channel_names[row["channel"]],
                             "absolute_read_scope": [row["start"] + origin, row["stop"] + origin],
                             "available_at_sample": row["stop"] + origin} for row in cleaned.windows)
        if self.finite_operator is not None:
            result = np.empty_like(output)
            for left, right in local_segments:
                result[left:right] = self.finite_operator.apply(output[left:right])
            output = result
        if not np.isfinite(output).all():
            raise ValueError("actual input program produced nonfinite output")
        return output, windows

    def apply(self, observation):
        """Execute only the explicitly supplied observed record; no label or oracle."""
        if not isinstance(observation, ContaminationInput):
            raise TypeError("input processing requires its owned native observation")
        output, windows = self._execute(observation, observation.samples)
        return frozen_array(output), freeze_metadata(self._execution_record(observation, output, windows))

    def _execution_record(self, observation, output, windows):
        return {"program": self.record(), "input": observation.record(),
                "output_sha256": array_digest(output), "harmonic_windows": windows,
                "complete_read_scope": list(observation.grid.context_scope),
                "earliest_whole_output_sample": observation.grid.context_scope[1],
                "input_available_at_sample": observation.available_at_sample,
                "output_available_at_sample": observation.available_at_sample,
                "finite_segment_read_scopes": [list(value) for value in observation.segments]
                    if self.finite_operator is not None else [],
                "harmonic_receipt_is_frozen_linear_operator": False,
                "frequency_and_weights_are_physical_parameters_identified": False}


def _transition(base, changed):
    if len(base) != len(changed) or any(
            tuple(a[key] for key in ("segment", "channel", "start", "stop"))
            != tuple(b[key] for key in ("segment", "channel", "start", "stop"))
            for a, b in zip(base, changed, strict=True)):
        raise ValueError("pipeline challenge changed its actual harmonic observation windows")
    shifts = [float(b["frequency_hz"] - a["frequency_hz"]) for a, b in zip(base, changed, strict=True)]
    switches = [{"physical_channel": b["physical_channel"], "absolute_read_scope": b["absolute_read_scope"],
                 "before_applied": a["applied"], "after_applied": b["applied"]}
                for a, b in zip(base, changed, strict=True) if a["applied"] != b["applied"]]
    return {"requested_windows": len(base), "application_switches": switches,
            "n_application_switches": len(switches), "frequency_shifts_hz": shifts,
            "maximum_absolute_frequency_shift_hz": max(map(abs, shifts), default=None),
            "robust_reweighting_changed_windows": sum(
                a["downweighted_fraction"] != b["downweighted_fraction"]
                for a, b in zip(base, changed, strict=True)),
            "receipt_cannot_reconstruct_last_executed_weight_matrix": True}


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _increment_metrics(increment, reference, rate):
    error = increment - reference
    ref_norm, out_norm = float(np.linalg.norm(reference)), float(np.linalg.norm(increment))
    peak, ref_peak = np.argmax(np.abs(increment), axis=0), np.argmax(np.abs(reference), axis=0)
    valid = np.any(increment != 0, axis=0) & np.any(reference != 0, axis=0)
    return {"reference_norm_volts": ref_norm, "increment_norm_volts": out_norm,
            "relative_error": _ratio(float(np.linalg.norm(error)), ref_norm),
            "relative_error_undefined_reason": "zero_reference_increment" if ref_norm == 0 else None,
            "energy_fraction_remaining": _ratio(out_norm**2, ref_norm**2),
            "maximum_absolute_error_volts": float(np.max(np.abs(error))),
            "signed_peak_volts_per_channel": increment[peak, np.arange(increment.shape[1])].tolist(),
            "peak_latency_error_seconds_per_channel": [float((x - y) / rate) if ok else None
                for x, y, ok in zip(peak, ref_peak, valid, strict=True)],
            "area_error_volt_seconds_per_channel": (error.sum(axis=0) / rate).tolist()}


@dataclass(frozen=True)
class ContaminationAudit:
    """One empirical audit with compact output covariance factor, not a model.

Rows of ``output_covariance_factor`` are centered output increments divided by
sqrt(B-1), in sample-then-channel columns. Thus F.T F is the sample covariance.
The mean increment is explicit; covariance is not a gain or density. Original
perturbation arrays remain caller-owned artifacts identified in the report.
"""

    report: dict
    mean_output_increment: np.ndarray = field(repr=False, compare=False)
    output_covariance_factor: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "report", freeze_metadata(self.report))
        object.__setattr__(self, "mean_output_increment", frozen_array(self.mean_output_increment))
        object.__setattr__(self, "output_covariance_factor", frozen_array(self.output_covariance_factor))

    def save(self, directory):
        """Write a new diagnostic artifact; never overwrite source/models/results."""
        import json

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "moments.npz").open("xb") as stream:
            np.savez_compressed(stream, mean_output_increment=self.mean_output_increment,
                                output_covariance_factor=self.output_covariance_factor)
        with (destination / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(self.report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return destination


def audit_input_contamination(observation, program, challenges: Mapping, *, challenge_design_identity,
                              perturbations=None, perturbation_design_identity=None,
                              byte_budget=64 * 1024 * 1024):
    """Refit actual T for independent increments and an explicit draw distribution.

Challenge entries are {kind: 'ERP'|'line'|'glitch'|'drift'|'reference', values: h}.
They are additive declarations, never inferred clean truth or hidden target
labels. Each h uses the same actual native grid. Reference contamination here
is an explicit shared-voltage challenge, not a new reference or EOG estimate.

Noise draws are caller-supplied additive voltages of shape (B,N,C), B>=2. Report
both complete-output second moments and baseline-finite reference moments. All
state/gate changes remain in the denominator. There is no winner selection,
linear covariance substitution, delta-method approximation or posterior fit.
"""
    if not isinstance(observation, ContaminationInput) or not isinstance(program, ContaminationProgram):
        raise TypeError("audit requires owned actual observation and declared processing program")
    _identity(challenge_design_identity, "independent challenge design")
    if not isinstance(challenges, Mapping) or not challenges:
        raise ValueError("named independent input challenges are required")
    if type(byte_budget) is not int or byte_budget < 1:
        raise ValueError("explicit audit array byte budget must be positive")
    raw, shape = observation.samples, observation.samples.shape
    draws = None if perturbations is None else finite_real_array(perturbations, name="explicit additive noise draws")
    if draws is not None:
        _identity(perturbation_design_identity, "explicit perturbation distribution")
        if draws.ndim != 3 or draws.shape[1:] != shape or len(draws) < 2:
            raise ValueError("empirical moments require B>=2 draws on the exact native input grid")
    elif perturbation_design_identity is not None:
        raise ValueError("a perturbation distribution cannot be claimed without actual draws")
    required = raw.nbytes * (12 + 4 * (0 if draws is None else len(draws)))
    if byte_budget < required:
        raise ValueError("complete pipeline audit exceeds its explicit array byte budget")
    # Normalize every external challenge before executing any expensive stage.
    prepared = {}
    for identity, entry in challenges.items():
        _identity(identity, "challenge")
        if (not isinstance(entry, Mapping) or set(entry) != {"kind", "values"}
                or entry["kind"] not in {"ERP", "line", "glitch", "drift", "reference"}):
            raise ValueError("each challenge must declare its component kind and actual native increment")
        values = observation.grid.values(entry["values"], observation.grid.context_scope, "independent input increment")
        prepared[identity] = (entry["kind"], values)
    baseline, base_windows = program._execute(observation, raw)
    finite_reference = ContaminationProgram(finite_operator=program.finite_operator)
    zero_output, _ = program._execute(observation, np.zeros_like(raw))
    entries = []
    for identity, (kind, h) in prepared.items():
        output, windows = program._execute(observation, raw + h)
        increment = output - baseline
        reference, _ = finite_reference._execute(observation, h)
        isolated, isolated_windows = program._execute(observation, h)
        standalone_increment = isolated - zero_output
        entries.append({"identity": identity, "kind": kind, "input_increment_sha256": array_digest(h),
                        "complete_increment_sha256": array_digest(increment),
                        "complete_increment_metrics": _increment_metrics(increment, reference, observation.grid.sample_rate_hz),
                        "isolated_pipeline_increment_metrics": _increment_metrics(standalone_increment, reference, observation.grid.sample_rate_hz),
                        "refit_vs_isolated_increment_relative_norm": _ratio(
                            float(np.linalg.norm(increment - standalone_increment)), float(np.linalg.norm(reference))),
                        "base_to_injected_state": _transition(base_windows, windows),
                        "injected_windows": windows, "isolated_windows": isolated_windows,
                        "contamination_truth_inferred_from_observed_base": False})
    mean = np.zeros_like(raw)
    factor = np.empty((0, raw.size))
    noise_report = {"status": "not_requested", "requested_draws": 0, "completed_draws": 0,
                   "moments_unavailable_reason": "no_explicit_perturbation_distribution"}
    if draws is not None:
        increments, references, transitions, actual_windows = [], [], [], []
        for noise in draws:
            output, windows = program._execute(observation, raw + noise)
            increments.append(output - baseline)
            reference, _ = finite_reference._execute(observation, noise)
            references.append(reference)
            transitions.append(_transition(base_windows, windows))
            actual_windows.append(windows)
        increments, references = np.asarray(increments), np.asarray(references)
        mean = increments.mean(axis=0)
        factor = (increments - mean).reshape(len(draws), -1) / np.sqrt(len(draws) - 1)
        input_second = float(np.sum(draws**2) / len(draws))
        reference_second = float(np.sum(references**2) / len(draws))
        output_second = float(np.sum(increments**2) / len(draws))
        noise_report = {"status": "completed", "requested_draws": len(draws), "completed_draws": len(draws),
                        "distribution_identity": perturbation_design_identity, "input_draws_sha256": array_digest(draws),
                        "estimator": "complete_program_refit_increments_conditional_on_supplied_base_and_draw_distribution",
                        "input_mean_increment_norm_volts": float(np.linalg.norm(draws.mean(axis=0))),
                        "output_mean_increment_norm_volts": float(np.linalg.norm(mean)),
                        "input_empirical_second_moment_volts_squared": input_second,
                        "finite_reference_empirical_second_moment_volts_squared": reference_second,
                        "output_empirical_second_moment_volts_squared": output_second,
                        "output_second_moment_relative_to_input": _ratio(output_second, input_second),
                        "output_second_moment_relative_to_finite_reference": _ratio(output_second, reference_second),
                        "relative_gain_undefined_reason": "zero_reference_draw_energy" if reference_second == 0 else None,
                        "sample_covariance_trace_volts_squared": float(np.sum(factor**2)),
                        "empirical_covariance_rank_upper_bound": len(draws) - 1,
                        "moments_numeric_sha256": array_digest(mean, factor),
                        "covariance_factor_layout": "draw_rows_by_sample_then_physical_channel",
                        "covariance_factor_rule": "centered_output_increments_divided_by_sqrt_B_minus_1",
                        "state_transitions": transitions, "draw_windows": actual_windows,
                        "zero_energy_draws": int(sum(not np.any(value) for value in draws)),
                        "fixed_F_covariance_or_Gaussian_likelihood_claimed": False,
                        "asymptotic_or_population_gain_claimed": False}
    report = {"schema": "n2p3_complete_input_contamination_audit/1", "status": "completed",
              "execution": program._execution_record(observation, baseline, base_windows),
              "program_sha256": semantic_sha256(program.record()),
              "implementation_sha256": _IMPLEMENTATION_SHA256,
              "operator_owner_source_sha256_at_import": _OWNER_SOURCE,
              "challenge_design_identity": challenge_design_identity,
              "challenge_independence_authorized_here": False,
              "requested_challenges": len(entries), "completed_challenges": len(entries),
              "challenges": entries, "empirical_output_moments": noise_report,
              "held_truth_used": False, "parameters_selected_from_audit": False,
              "diagnostic_failure_does_not_remove_requested_participant": True,
              "evidence_scope": "declared_input_mechanism_perturbation_diagnostic",
              "product_or_public_performance_claimed": False}
    return ContaminationAudit(report, mean, factor)
