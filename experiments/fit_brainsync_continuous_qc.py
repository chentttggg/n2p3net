"""Fit frozen raw QC from explicitly enumerated clean calibration intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.brainsync import load_brainsync_session  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.brainsync_startup import (  # noqa: E402
    apply_startup_cut,
    startup_cut_for,
    startup_fit_provenance,
)
from data.continuous_qc import (  # noqa: E402
    ContinuousQCSpec,
    FilterStateRecovery,
    fit_continuous_qc,
    save_continuous_qc,
)
from data.dataset import read_raw  # noqa: E402
from data.epochs import materialize_dataset_identity, parse_preprocessing_spec  # noqa: E402
from data.recovery import fit_epoch_state_recovery  # noqa: E402


def fit_brainsync_continuous_manifest(path: str | Path):
    with ExitStack() as input_leases:
        path = Path(path).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "brainsync_continuous_qc_calibration/1":
            raise ValueError("explicit continuous QC calibration manifest is required")
        if not isinstance(manifest.get('preprocessing'), dict):
            raise ValueError('continuous QC requires an explicit complete preprocessing declaration')
        preprocessing = parse_preprocessing_spec(manifest['preprocessing'])
        if not isinstance(manifest.get('sessions'), list) or not manifest['sessions']:
            raise ValueError('continuous QC requires a nonempty declared calibration session list')
        for item in manifest['sessions']:
            if (not isinstance(item, dict) or not isinstance(item.get('session_dir'), str) or not item['session_dir'].strip()
                    or not isinstance(item.get('clean_intervals_samples'), list) or not item['clean_intervals_samples']
                    or not isinstance(item.get('clean_interval_evidence'), str) or not item['clean_interval_evidence'].strip()):
                raise ValueError('each continuous QC session requires a path, explicit clean intervals and selection evidence')
            for interval in item['clean_intervals_samples']:
                if (not isinstance(interval, list) or len(interval) != 2
                        or any(type(value) is not int for value in interval) or not 0 <= interval[0] < interval[1]):
                    raise ValueError('clean calibration intervals must be half-open native sample bounds')
        specification = dict(manifest["spec"])
        if preprocessing.filter_phase == "zero":
            specification.setdefault("range_action", "diagnostic")
            specification.setdefault("boundary_action", "artifact_edges")
            if manifest.get("epoch_recovery") is None:
                raise ValueError("zero-phase QC fitting requires explicit epoch_recovery calibration parameters")
        spec = ContinuousQCSpec(**specification)
        recovery = manifest.get('recovery')
        recovery = None if recovery is None else FilterStateRecovery(**recovery)
        epoch_parameters = manifest.get('epoch_recovery')
        if epoch_parameters is not None:
            expected = {'tolerance_v', 'inflation', 'warmup_samples', 'probe_step_samples'}
            if not isinstance(epoch_parameters, dict) or set(epoch_parameters) != expected:
                raise ValueError('epoch_recovery requires exactly tolerance_v, inflation, warmup_samples and probe_step_samples')
        raws, masks, sessions, participants, references, records = [], [], [], set(), set(), []
        startup_cuts = []
        for item in manifest["sessions"]:
            root = (path.parent / item["session_dir"]).resolve()
            session = validate_brainsync_bids_session(root)
            input_leases.callback(session.bids.close)
            dataset = load_brainsync_session(root, preprocessing=preprocessing)
            participants.add(materialize_dataset_identity(dataset).record_for(session.subject_id).authority_key("source_or_global"))
            references.add(session.bids.source_reference)
            raw = read_raw(session.bids.raw_path, preload=True)
            raw.set_channel_types(dict(zip(raw.ch_names, [kind.lower() for kind in session.bids.channel_types], strict=True)))
            raw.info["bads"] = [name for name, status in zip(raw.ch_names, session.bids.channel_statuses, strict=True) if status == "bad"]
            mask = np.zeros(raw.n_times, dtype=bool)
            for start, stop in item["clean_intervals_samples"]:
                if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= raw.n_times:
                    raise ValueError("clean calibration intervals must be half-open native sample bounds")
                mask[start:stop] = True
            startup_cut = startup_cut_for(session.bids)
            apply_startup_cut(raw, startup_cut, mask)
            startup_cuts.append(startup_cut)
            if not mask.any() or not item.get("clean_interval_evidence"):
                raise ValueError("clean interval selection requires samples and evidence")
            clock = session.manifest["recording"].get("clock")
            if clock is not None:
                acquired = clock["acquired_samples"]
                mask[acquired:] = False
                if acquired < raw.n_times:
                    raw.annotations.append(acquired / raw.info["sfreq"],
                        (raw.n_times-acquired) / raw.info["sfreq"], "BAD_acq_skip_edf_padding")
            if session.bids.acquisition_boundaries:
                raw.annotations.append(session.bids.acquisition_boundaries,
                    [0.] * len(session.bids.acquisition_boundaries), ["EDGE boundary"] * len(session.bids.acquisition_boundaries))
            for interval in session.bids.excluded_intervals:
                start = int(np.floor(interval.onset_seconds * raw.info["sfreq"]))
                stop = int(np.ceil((interval.onset_seconds + interval.duration_seconds) * raw.info["sfreq"]))
                mask[max(0, start):min(raw.n_times, stop)] = False
            raws.append(raw)
            masks.append(mask)
            sessions.append(session)
            records.append({"session_id": session.session_id,
                            "raw_sha256": hashlib.sha256(session.bids.raw_path.read_bytes()).hexdigest(),
                            "clean_intervals_samples": item["clean_intervals_samples"],
                            "raw_startup_cut": startup_cut,
                            "clean_interval_evidence": item["clean_interval_evidence"]})
        if (len(participants) != 1 or len(references) != 1
                or len({session.session_id for session in sessions}) != len(sessions)):
            raise ValueError("continuous QC requires unique sessions from one participant and reference")
        provenance = {"participant_key": next(iter(participants)),
                            **startup_fit_provenance(startup_cuts),
                            "source_reference": next(iter(references)), "sources": records,
                            "calibration_session_ids": [session.session_id for session in sessions],
                            "calibration_end_timestamp_s": max(session.ended_timestamp_s for session in sessions),
                            "role": "explicit_clean_calibration_intervals",
                            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "preprocessing": preprocessing.record()}
        group_models = None
        if "threshold_aggregation" in manifest:
            if manifest["threshold_aggregation"] != "per_record_MAD_envelope":
                raise ValueError("unknown declared continuous threshold aggregation")
            group_models = [fit_continuous_qc(group_raw, spec=spec, fit_sample_mask=group_mask,
                fit_provenance={**provenance, "calibration_session_ids": [group_session.session_id],
                    "role": "single_authorized_calibration_record_thresholds"})
                for group_raw, group_mask, group_session in zip(raws, masks, sessions, strict=True)]
        raw = mne.concatenate_raws(raws, verbose=False)
        mask = np.concatenate(masks)
        epoch_recovery = None
        if manifest.get("epoch_recovery") is not None:
            if "recovery_warmup_rule" in manifest:
                from data.source_qc import scope_recovery_parameters

                epoch_parameters = scope_recovery_parameters(raw, mask, preprocessing, epoch_parameters,
                    rule=manifest["recovery_warmup_rule"])
                provenance["recovery_warmup_rule"] = manifest["recovery_warmup_rule"]
            channels = tuple(raw.ch_names[index] for index in mne.pick_types(raw.info,
                eeg=True, eog=spec.include_eog, exclude="bads"))
            epoch_recovery = fit_epoch_state_recovery(raw, preprocessing=preprocessing,
                channels=channels, fit_sample_mask=mask, fit_provenance=provenance,
                **epoch_parameters)
        if group_models is not None:
            from data.source_qc import calibration_group_envelope

            return calibration_group_envelope(group_models, fit_provenance=provenance,
                recovery=recovery, epoch_recovery=epoch_recovery)
        return fit_continuous_qc(raw, spec=spec, fit_sample_mask=mask,
            recovery=recovery,
            epoch_recovery=epoch_recovery, fit_provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = fit_brainsync_continuous_manifest(args.calibration_manifest)
    print(save_continuous_qc(args.output, model))


if __name__ == "__main__":
    main()
