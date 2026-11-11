"""Public-record QC using the same native detector and recovery as BrainSync."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from data.channel import canonical_channel_name
from data.continuous_qc import ContinuousQCSpec, fit_continuous_qc, signal_dependency_matrix
from data.epochs import materialize_dataset_lineage, select_scheduled_events
from data.lineage import DataLineage
from data.ocular import eog_regression_from_record
from data.preprocess import continuous_filter_segments
from data.qc_features import compute_epoch_qc_features
from data.recovery import fit_epoch_state_recovery
from research.contracts import semantic_sha256


def continuous_parameters(policy, sample_rate_hz):
    """Lower the same physical-time policy at the actual acquisition rate."""
    settings = policy.record().get("continuous_qc")
    if settings is None:
        raise ValueError("native QC requires an explicitly enabled continuous policy")
    rate = float(sample_rate_hz)
    spec = ContinuousQCSpec(difference_z=settings["difference_z"],
        difference_floor_v=settings["difference_floor_v"], settle_samples=max(1, round(settings["settle_seconds"] * rate)),
        channel_local=True, reference_channel="Cz", include_eog=True,
        minimum_channel_fraction=settings["minimum_channel_fraction"],
        flatline_min_duration_s=settings["flatline_min_duration_s"],
        range_action=settings["range_action"], boundary_action=settings["boundary_action"])
    recovery = dict(tolerance_v=settings["recovery_tolerance_v"], inflation=settings["recovery_inflation"],
        warmup_samples=round(settings["recovery_warmup_seconds"] * rate),
        probe_step_samples=max(1, round(settings["recovery_probe_step_seconds"] * rate)))
    return spec, recovery


def scope_recovery_parameters(raw, mask, preprocessing, parameters, *, rule):
    """Preserve probes on short authorised runs without reading another record."""
    if rule != "cap_by_longest_authorized_segment_quarter":
        raise ValueError("unknown recovery warmup scope policy")
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (raw.n_times,):
        raise ValueError("recovery warmup needs an aligned authorized sample mask")
    longest = 0
    for start, stop in continuous_filter_segments(raw):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[start:stop], False]))
        longest = max(longest, max((right-left for left, right in edges.reshape(-1, 2)), default=0))
    rate = float(raw.info["sfreq"])
    width = round(preprocessing.tmax_ms / 1000 * rate) - round(preprocessing.tmin_ms / 1000 * rate) + 1
    margin = int(longest) - width - parameters["probe_step_samples"]
    if margin < 4:
        raise ValueError("authorized calibration is too short for bidirectional recovery probes")
    return {**parameters, "warmup_samples": min(parameters["warmup_samples"], margin // 4)}


def calibration_group_envelope(models, *, fit_provenance, epoch_recovery=None, recovery=None):
    """Freeze a band containing every C record's robust native difference band.

    This handles measured scale variation between recordings. Adding a quiet
    record cannot erase an already authorised record's robust acceptance band.
    """
    models = tuple(models)
    if not models:
        raise ValueError("grouped continuous QC needs declared calibration records")
    first = models[0]
    if any((model.channel_names, model.sample_rate_hz, model.spec) !=
           (first.channel_names, first.sample_rate_hz, first.spec) for model in models):
        raise ValueError("grouped QC requires the same native channel roles, rate and policy")
    centers = np.asarray([model.difference_center_v for model in models])
    center = np.median(centers, axis=0)
    scales = np.asarray([model.difference_scale_v for model in models])
    scale = np.max(scales + np.abs(centers-center) / first.spec.difference_z, axis=0)
    ranges = None if first.range_threshold_v is None else tuple(np.max(
        np.asarray([model.range_threshold_v for model in models]), axis=0))
    provenance = {**fit_provenance, "threshold_aggregation": "per_record_MAD_envelope", "labels_used": False,
        "threshold_groups": [{"model_digest": semantic_sha256(model.record()),
            "fit_provenance": dict(model.fit_provenance), "difference_center_v": list(model.difference_center_v),
            "difference_scale_v": list(model.difference_scale_v)} for model in models]}
    return replace(first, difference_center_v=tuple(center), difference_scale_v=tuple(scale),
        fit_provenance=provenance, range_threshold_v=ranges, epoch_recovery=epoch_recovery, recovery=recovery)


def fit_source_continuous(raw, *, policy, preprocessing, fit_mask, participant_key, session_id, source_reference):
    if preprocessing.filter_phase != "zero":
        raise ValueError("this matched QC program requires its declared offline bidirectional preprocessing")
    spec, parameters = continuous_parameters(policy, raw.info["sfreq"])
    requested_warmup = parameters["warmup_samples"]
    rule = policy.record()["continuous_qc"]["recovery_warmup_rule"]
    parameters = scope_recovery_parameters(raw, fit_mask, preprocessing, parameters, rule=rule)
    names = tuple(n for n, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
                  if kind in {"eeg", "eog"} and n not in raw.info["bads"])
    provenance = {"participant_key": participant_key, "source_reference": source_reference,
        "calibration_session_ids": [session_id], "calibration_end_timestamp_s": raw.n_times / float(raw.info["sfreq"]),
        "timestamp_basis": "source_record_relative_seconds", "role": "source_training_preprocessing_only",
        "preprocessing_policy_digest": policy.digest(), "labels_used": False}
    provenance.update(requested_warmup_samples=requested_warmup, recovery_warmup_rule=rule)
    provenance["threshold_aggregation"] = policy.record()["continuous_qc"]["threshold_aggregation"]
    recovery = fit_epoch_state_recovery(raw, preprocessing=preprocessing, channels=names,
        fit_sample_mask=fit_mask, fit_provenance=provenance, **parameters)
    native = fit_continuous_qc(raw, spec=spec, fit_sample_mask=fit_mask,
        epoch_recovery=recovery, fit_provenance=provenance)
    fitted = calibration_group_envelope([native], fit_provenance=provenance, epoch_recovery=recovery)
    detected = fitted.detect(raw)
    return fitted, detected


def apply_source_continuous(raw, *, model, detected, ocular_record, eeg_names):
    """Reference before filtering and propagate every native artifact dependency."""
    ocular = (eog_regression_from_record(ocular_record["fitted_model"])
              if ocular_record["status"] == "applied" else None)
    output = raw.copy().load_data()
    if ocular is None:
        anchor = [n for n in raw.ch_names if canonical_channel_name(n) == "CZ"]
        if len(anchor) != 1:
            raise ValueError("source QC requires one measured Cz reference")
        output.set_eeg_reference(anchor, projection=False, verbose=False)
        output.set_channel_types({anchor[0]: "misc"}, on_unit_change="ignore")
    channels = tuple(n for n, kind in zip(output.ch_names, output.get_channel_types(), strict=True)
                     if kind == "eeg" and n in eeg_names)
    mixing = signal_dependency_matrix(model.channel_names, channels, reference_channel="Cz", ocular_model=ocular)
    boundaries = [start for start, _ in continuous_filter_segments(raw) if start]
    projected = model.project_result(detected, mixing, channels, boundaries=boundaries)
    annotated = projected.annotate(output)
    return annotated, projected, model.epoch_recovery.project(mixing, channels)


def apply_source_epoch_masks(dataset, *, raw, result, recovery, model):
    """Retain the scheduled ledger while removing only unsupported epoch evidence."""
    timeline = dataset.event_timeline
    samples = np.asarray(timeline.onset_samples) - int(raw.first_samp)
    channels = tuple(dataset.channel_names)
    # Dataset labels are canonical; recovery keeps the actual acquisition names.
    native_names = {canonical_channel_name(n): n for n in result.channel_names}
    ordered_names = tuple(native_names[n] for n in channels)
    quality = result.epoch_channel_mask(samples, channels=ordered_names,
        tmin_s=dataset.preprocessing.tmin_ms / 1000, tmax_s=dataset.preprocessing.tmax_ms / 1000)
    boundary, boundary_record = recovery.assess_epochs(raw, samples, boundary_action=model.spec.boundary_action)
    positions = [recovery.channel_names.index(n) for n in ordered_names]
    quality &= boundary[:, positions]
    events = np.flatnonzero(timeline.evidence_indices >= 0)
    events = events[np.argsort(timeline.evidence_indices[events])]
    masks = quality[events]
    keep = masks.sum(axis=1) >= int(np.ceil(dataset.n_channels * model.spec.minimum_channel_fraction))
    if not keep.any():
        raise ValueError("source QC left no supported EEG epochs; preserve failure instead of silently dropping the record")
    event_keep = np.zeros(timeline.n_events, dtype=bool)
    event_keep[events[keep]] = True
    selected = select_scheduled_events(dataset, event_keep,
        reason="continuous_quality_insufficient_observed_channels", preserve_schedule=True)
    rejected = (timeline.evidence_indices >= 0) & (selected.event_timeline.evidence_indices < 0)
    statuses = np.array(selected.event_timeline.statuses, dtype=object)
    statuses[rejected] = "artifact_rejected"
    available = np.array(selected.event_timeline.evidence_available_times_s, copy=True)
    available[np.isfinite(available)] = np.maximum(available[np.isfinite(available)],
        (int(raw.first_samp) + raw.n_times) / float(raw.info["sfreq"]))
    selected = replace(selected, event_timeline=replace(selected.event_timeline, statuses=statuses.astype(str),
        evidence_available_times_s=available, timing_source=timeline.timing_source + ";continuous_qc_complete_recording"))
    masks = masks[keep]
    values = np.where(masks[:, :, None], selected.X, 0.)
    provenance = {**selected.provenance, "continuous_artifact_processing": {"status": "applied",
        "model": model.record(), "model_digest": semantic_sha256(model.record())},
        "continuous_quality_ledger": result.record(), "bidirectional_boundary_recovery": boundary_record}
    output = replace(selected, X=values, channel_mask=masks.any(axis=0), trial_channel_mask=masks,
        qc_features=compute_epoch_qc_features(values, channel_mask=masks.any(axis=0), trial_channel_mask=masks),
        provenance=provenance, verified_cache_attestation=None,
        lineage=DataLineage.derive([materialize_dataset_lineage(selected)], operation="source_continuous_channel_QC",
            parameters={"model_digest": semantic_sha256(model.record()), "labels_used": False}))
    output.validate(require_labels=True)
    return output
