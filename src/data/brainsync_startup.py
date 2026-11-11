"""Owner-authorised BrainSync startup cuts on the original sample axis."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from importlib.resources import files
from math import ceil
from pathlib import Path

import numpy as np

POLICY = "first_flash_4s_round_up_to_complete_epoch/1"
ANNOTATION = "BAD_acq_skip_owner_startup"


def plan_startup_cut(onsets, sample_rate_hz, n_samples):
    """Round the first four stimulus seconds up to a full retained baseline."""
    samples = np.asarray(onsets)
    rate = float(sample_rate_hz)
    if (samples.ndim != 1 or not len(samples) or not np.issubdtype(samples.dtype, np.integer)
            or np.any(np.diff(samples) <= 0) or samples[0] < 0
            or not np.isfinite(rate) or rate <= 0 or n_samples <= samples[-1]):
        raise ValueError("startup cut requires ordered native stimulus samples and a valid recording")
    baseline = round(-.2 * rate)
    response = round(1.2 * rate) + 1
    nominal = int(samples[0]) + ceil(4 * Fraction(str(rate)))
    supported = np.flatnonzero((samples + baseline >= nominal) & (samples + response <= n_samples))
    if not len(supported):
        raise ValueError("startup cut leaves no complete retained trial")
    first = int(supported[0])
    return {"policy": POLICY, "sample_rate_hz": rate, "seconds_after_first_flash": 4.,
            "baseline_ms": -200., "response_end_ms": 1200.,
            "first_stimulus_sample": int(samples[0]), "nominal_stop_sample": nominal,
            "stop_sample": int(samples[first]) + baseline,
            "first_retained_event_index": first, "first_retained_onset_sample": int(samples[first]),
            "dropped_event_indices": list(range(first)), "original_sample_count": int(n_samples),
            "sample_interval": "half_open", "order": "exclude_original_raw_before_any_fit_or_filter"}


def startup_cuts():
    record = json.loads(files("data").joinpath("brainsync_startup_cuts.json").read_text(encoding="utf-8"))
    if record.get("schema") != "brainsync_startup_cuts/1":
        raise ValueError("unsupported startup cut registry")
    rows = record["recordings"]
    result = {r["raw_sha256"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("startup cut raw identities must be unique")
    return result


def startup_cut_for(bids):
    """Bind the cut to actual raw bytes and the actual original event ledger."""
    registry = startup_cuts()
    if not registry:
        return None
    binding = bids.content_record()["files"].get("recording")
    if binding is None:
        with bids.raw_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        digest = binding["sha256"]
    row = registry.get(digest)
    if row is None:
        return None
    plan = plan_startup_cut(np.array([s.sample for s in bids.stimuli]), bids.sample_rate_hz,
                            row["original_sample_count"])
    if any(row.get(k) != v for k, v in plan.items()):
        raise ValueError("startup cut differs from its original stimulus/sample authority")
    if round(bids.duration_seconds * bids.sample_rate_hz) != row["original_sample_count"]:
        raise ValueError("startup cut original raw duration changed")
    return dict(row)


def apply_startup_cut(raw, cut, fit_mask=None):
    """Exclude raw support without changing original clocks or joining samples.

    Shared filtering and QC use acquisition-skip segments. Keeping the original
    axis also retains all requested events and the existing pause boundaries.
    The separately exported EDF is the exact physical suffix of this same cut.
    """
    if cut is None:
        return
    stop = cut["stop_sample"]
    if (cut["policy"] != POLICY or type(stop) is not int or not 0 < stop < raw.n_times
            or raw.n_times != cut["original_sample_count"]
            or float(raw.info["sfreq"]) != cut["sample_rate_hz"]):
        raise ValueError("startup cut does not match the original raw sampling geometry")
    if not any(str(d) == ANNOTATION for d in raw.annotations.description):
        raw.annotations.append(raw.first_time, stop / raw.info["sfreq"], ANNOTATION)
    if fit_mask is not None:
        if fit_mask.dtype != bool or fit_mask.shape != (raw.n_times,):
            raise ValueError("startup fit mask must use the original native sample axis")
        fit_mask[:stop] = False


def startup_fit_provenance(cuts):
    return {"startup_cut_policy": POLICY} if any(c is not None for c in cuts) else {}


def require_startup_fit(provenance, cut):
    expected = None if cut is None else POLICY
    if provenance.get("startup_cut_policy") != expected:
        raise ValueError("startup raw support changed; refit QC/EOG on the cut calibration recordings")
    registry = startup_cuts()
    sources = provenance.get("sources", ())
    if expected is not None and (not isinstance(sources, (list, tuple)) or not sources):
        raise ValueError("startup fit sources are missing; refit QC/EOG on every cut source")
    if any(s.get("raw_startup_cut") != registry.get(s.get("raw_sha256")) for s in sources):
        raise ValueError("calibration source startup support changed; refit QC/EOG on every cut source")


def write_cut_edf(source, output, stop_sample):
    """Export retained original digital samples with their original calibration."""
    import edfio

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source:
        raise FileExistsError(output)
    original = edfio.read_edf(source)
    rates = {s.sampling_frequency for s in original.signals}
    counts = {len(s.digital) for s in original.signals}
    if len(rates) != 1 or len(counts) != 1:
        raise ValueError("BrainSync native cut requires one actual channel sampling axis")
    rate, count = rates.pop(), counts.pop()
    if type(stop_sample) is not int or not 0 < stop_sample < count:
        raise ValueError("cut must leave a nonempty native suffix")
    expected = [s.digital[stop_sample:].copy() for s in original.signals]
    calibrations = [(s.label, s.physical_range, s.digital_range, s.physical_dimension)
                    for s in original.signals]
    # One native sample per EDF record represents any exact sample boundary.
    original.update_data_record_duration(1. / rate)
    original.slice_between_seconds(stop_sample / rate, count / rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        original.write(stream)
    saved = edfio.read_edf(output)
    for index, signal in enumerate(saved.signals):
        if (not np.array_equal(signal.digital, expected[index])
                or (signal.label, signal.physical_range, signal.digital_range, signal.physical_dimension)
                != calibrations[index]):
            raise RuntimeError("cut EDF differs from the original digital suffix or calibration")
    return {"digital_suffix_equal": True, "original_samples": count,
            "removed_samples": stop_sample, "retained_samples": count - stop_sample,
            "sample_rate_hz": rate, "filters_applied": [], "resampling_applied": False,
            "original_axis_offset_samples": stop_sample}
