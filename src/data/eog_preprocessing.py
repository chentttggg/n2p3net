"""An explicit, reusable EOG preprocessing policy with EEG-only model outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from data.channel import canonical_channel_name
from data.ocular import (
    EOG_FIT_DOMAINS,
    NATIVE_EOG_FIT_DOMAIN,
    PASSBAND_EOG_FIT_DOMAIN,
    PRESERVE_ACQUIRED_EOG,
    fit_eog_regression,
)
from research.contracts import semantic_sha256


@dataclass(frozen=True)
class EOGPreprocessingPolicy:
    name: str
    enabled: bool
    missing_eog: str
    reference_channel: str = "CZ"
    continuous_qc: bool = False
    fit_domain: str = NATIVE_EOG_FIT_DOMAIN
    boundary_action: str = "artifact_edges"

    def __post_init__(self):
        if not self.name or type(self.enabled) is not bool or type(self.continuous_qc) is not bool:
            raise ValueError("EOG preprocessing requires a name and explicit enabled flag")
        if self.missing_eog not in {"error", "record_unavailable"}:
            raise ValueError("missing EOG action must be explicit")
        if canonical_channel_name(self.reference_channel) != "CZ":
            raise ValueError("this source comparison declares the measured Cz reference")
        if self.fit_domain not in EOG_FIT_DOMAINS or self.boundary_action not in {"artifact_edges","all_edges"}:
            raise ValueError("unsupported ocular fit domain or boundary action")

    @classmethod
    def corrected(cls, *, continuous_qc=True):
        return cls("QC_eog_passband_scopes_v2", True, "record_unavailable", continuous_qc=continuous_qc,
                   fit_domain=PASSBAND_EOG_FIT_DOMAIN,
                   boundary_action="all_edges" if continuous_qc else "artifact_edges")

    def record(self):
        fields = asdict(self)
        fields.pop("continuous_qc")
        fields.pop("fit_domain")
        fields.pop("boundary_action")
        return {"schema": "n2p3_EOG_preprocessing_policy/1", **fields,
                "method": ("scope_centered_passband_OLS" if self.fit_domain == PASSBAND_EOG_FIT_DOMAIN
                           else "centered_calibration_OLS") if self.enabled else "basic",
                **({"EOG_fit_domain":self.fit_domain,
                    "EOG_fit_guard_rule":"min_30_seconds_or_longest_authorised_clean_scope_quarter"}
                   if self.enabled and self.fit_domain == PASSBAND_EOG_FIT_DOMAIN else {}),
                "EOG_model_feature": False, "correction_strength": 1.,
                "EOG_reference_policy": PRESERVE_ACQUIRED_EOG,
                **({"continuous_qc": {"difference_z": 12., "difference_floor_v": 1e-7,
                    "threshold_aggregation": "per_record_MAD_envelope",
                    "settle_seconds": .1, "flatline_min_duration_s": 5., "minimum_channel_fraction": .5,
                    "range_action": "diagnostic", "boundary_action": self.boundary_action,
                    "recovery_tolerance_v": 1e-6, "recovery_inflation": 2.,
                    "recovery_warmup_seconds": 30., "recovery_probe_step_seconds": 1.,
                    "recovery_warmup_rule": "cap_by_longest_authorized_segment_quarter",
                    "fit_role": "source_training_or_target_calibration_only", "labels_used": False}}
                   if self.continuous_qc else {})}

    def digest(self):
        return semantic_sha256(self.record())

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict):
            raise ValueError("EOG preprocessing policy must be a complete record")
        try:
            policy = cls(**{key: record[key] for key in
                ("name", "enabled", "missing_eog", "reference_channel")},
                continuous_qc="continuous_qc" in record,
                fit_domain=record.get("EOG_fit_domain", NATIVE_EOG_FIT_DOMAIN),
                boundary_action=record.get("continuous_qc",{}).get("boundary_action","artifact_edges"))
        except (KeyError, TypeError) as exc:
            raise ValueError("EOG preprocessing policy must be a complete record") from exc
        if policy.record() != record:
            raise ValueError("unsupported or inconsistent EOG preprocessing policy")
        return policy


def prepare_source_eog(raw, *, policy, eeg_names, source_reference, participant_key,
                       session_id, fit_mask, preprocessing=None):
    """Fit the shared source/C map, returning pointwise-corrected native raw.

    EOG is retained solely for a pointwise correction. No absent EOG is
    synthesised from scalp electrodes. Reference is established before ocular
    fitting, using the existing owner while preserving acquired EOG voltage.
    A matched passband is applied only to isolated calibration copies; final
    EEG filtering and epoching remain with the common preprocessing owner.
    """
    if not isinstance(policy, EOGPreprocessingPolicy):
        raise TypeError("source EOG preparation requires its explicit policy")
    eog = tuple(n for n, t in zip(raw.ch_names, raw.get_channel_types(), strict=True) if t == "eog")
    if not policy.enabled:
        return raw, source_reference, {"policy": policy.record(), "status": "disabled", "recorded_EOG": list(eog)}
    if not eog and policy.missing_eog == "error":
        raise ValueError("QC preprocessing requires a real recorded EOG channel; source has none")
    if not eog:
        return raw, source_reference, {"policy": policy.record(), "status": "unavailable_no_recorded_EOG",
            "recorded_EOG": [], "correction_applied": False, "classifier_EOG_features": 0}
    names = [n for n in raw.ch_names if n in set(eeg_names) | set(eog)]
    output = raw.copy().pick(names).load_data()
    anchors = [n for n in eeg_names if canonical_channel_name(n) == "CZ"]
    if len(anchors) != 1:
        raise ValueError("source EOG preparation requires one measured Cz anchor")
    mask = np.asarray(fit_mask)
    if mask.dtype != bool or mask.shape != (raw.n_times,):
        raise ValueError("ocular fit requires an aligned, explicit authorised sample mask")
    model = fit_eog_regression(output, participant_key=participant_key,
        source_reference=source_reference, reference_channel="CZ", fit_sample_mask=mask,
        eog_reference_policy=PRESERVE_ACQUIRED_EOG,
        fit_domain=policy.fit_domain, preprocessing=preprocessing,
        fit_provenance={"calibration_session_ids": [session_id], "fit_role": "source_training_preprocessing_only",
                        "labels_used": False, "preprocessing_policy_digest": policy.digest()})
    corrected = model.apply(output, source_reference=source_reference)
    for channel in eog:
        if not np.array_equal(corrected.get_data(picks=[channel]), output.get_data(picks=[channel])):
            raise RuntimeError("ocular preprocessing changed the acquired EOG reference")
    return corrected, "measured_channel:CZ", {"policy": policy.record(), "status": "applied",
        "recorded_EOG": list(eog), "correction_applied": True, "fitted_model": model.record(),
        "model_digest": semantic_sha256(model.record()), "classifier_EOG_features": 0}
