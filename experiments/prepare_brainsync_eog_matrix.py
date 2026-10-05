"""Prepare K-limited raw calibration and three EOG strengths for the fixed P001 matrix."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from data.brainsync import load_brainsync_sessions  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.domain import adapt_observed_montage  # noqa: E402
from data.epochs import (  # noqa: E402
    save_epoch_dataset,
    select_epoch_channels,
    select_scheduled_events,
)
from data.ocular import PRESERVE_ACQUIRED_EOG, save_eog_regression  # noqa: E402
from data.source_records import SIX, SOURCE_EOG_PREPROCESSING, file_sha256  # noqa: E402
from experiments.fit_brainsync_continuous_qc import fit_brainsync_continuous_manifest  # noqa: E402
from experiments.fit_brainsync_eog import fit_brainsync_eog_sessions  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.binary import binary_recording_split, resolve_calibration_splits  # noqa: E402


def observed_event_ids(dataset):
    mask = dataset.trial_channel_mask
    if mask is None:
        mask = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
    complete_rows = mask.all(axis=1)
    evidence = dataset.event_timeline.evidence_indices
    available = evidence >= 0
    complete = np.zeros(len(evidence), bool)
    complete[available] = complete_rows[evidence[available]]
    return set(dataset.event_timeline.event_ids[complete].astype(str))


def prepare(inputs, output, calibration_ids, test_ids):
    roots = [inputs / s for s in (*calibration_ids, *test_ids)]
    sessions = [validate_brainsync_bids_session(p) for p in roots]
    if tuple(s.session_id for s in sessions) != (*calibration_ids, *test_ids):
        raise ValueError("explicit matrix identities differ from session declarations")
    if len({s.subject_id for s in sessions}) != 1:
        raise ValueError("all personal matrix sessions must identify the confirmed same participant")
    native = load_brainsync_sessions(roots, preprocessing=SOURCE_EOG_PREPROCESSING)
    timeline = native.event_timeline
    available = timeline.onset_times_s + SOURCE_EOG_PREPROCESSING.tmax_ms / 1000
    def builder(k, maximum):
        return binary_recording_split(timeline.onset_times_s, available, timeline.session_ids,
            recording_order=(*calibration_ids, *test_ids), test_recordings=test_ids,
            calibration_trials=k, maximum_calibration_trials=maximum)
    splits, adjustments = resolve_calibration_splits((0, 200, 800, 2000), builder,
        minimum_test_trials=1, available_trials=int(np.isin(timeline.session_ids, calibration_ids).sum()),
        separate_test_recordings=True)
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(schema="n2p3_brainsync_eog_matrix_inputs/1", calibration_sessions=list(calibration_ids),
        test_sessions=list(test_ids), adjustments=adjustments, budgets={},
        budget_basis="scheduled acquisition prefix; QC does not extend raw authorization to replace rejected rows",
        preprocessing=SOURCE_EOG_PREPROCESSING.record(), participant=sessions[0].subject_id)
    no_target_fit = adapt_observed_montage(native, SIX, reference_channel="Cz")
    zero_test = select_scheduled_events(no_target_fit, np.isin(timeline.session_ids, test_ids), reason="K0 independent test; no target-fitted transformations")
    zero_path = save_epoch_dataset(output / "K0_test.npz", zero_test)
    plan["zero_test"] = dict(path=str(zero_path), sha256=file_sha256(zero_path))
    for budget, split in splits.items():
        if not budget:
            continue
        folder = output / f"K{budget}"
        folder.mkdir(exist_ok=True)
        allowed_events = set(timeline.event_ids[split.calibration_rows].astype(str))
        scoped_sessions = [s for s in sessions if s.session_id in set(timeline.session_ids[split.calibration_rows])]
        fit_intervals = {}
        for session in scoped_sessions:
            selected = split.calibration_rows[timeline.session_ids[split.calibration_rows] == session.session_id]
            rate = float(session.bids.sample_rate_hz)
            stop = int(timeline.onset_samples[selected].max() + round(SOURCE_EOG_PREPROCESSING.tmax_ms / 1000 * rate) + 1)
            fit_intervals[session.session_id] = [[0, stop]]
        # Threshold and state-envelope fitting see only this acquisition prefix. Explicit BIDS gaps remain excluded.
        qc_manifest = dict(schema="brainsync_continuous_qc_calibration/1",
            preprocessing=SOURCE_EOG_PREPROCESSING.record(),
            spec=dict(difference_z=12, difference_floor_v=1e-7, settle_samples=25,
                channel_local=True, reference_channel="Cz", include_eog=True, minimum_channel_fraction=.5,
                range_window_samples=351, range_quantile=.995, range_inflation=2., range_floor_v=50e-6,
                artifact_pre_guard_samples=350, flatline_min_duration_s=5.),
            epoch_recovery=dict(tolerance_v=1e-6, inflation=2., warmup_samples=7500, probe_step_samples=250),
            sessions=[dict(session_dir=str(s.root), clean_intervals_samples=fit_intervals[s.session_id],
                clean_interval_evidence="prespecified K-authorized calibration prefix; robust median/MAD and empirical state envelope; BIDS exclusions retained")
                for s in scoped_sessions])
        qc_path = folder / "continuous_manifest.json"
        atomic_write_json(qc_path, qc_manifest)
        qc = fit_brainsync_continuous_manifest(qc_path)
        atomic_write_json(folder / "continuous.json", {"model": qc.record(), "digest": semantic_sha256(qc.record())})
        ocular = fit_brainsync_eog_sessions([s.root for s in scoped_sessions], reference_channel="Cz",
            continuous_qc_model=qc, eog_reference_policy=PRESERVE_ACQUIRED_EOG,
            fit_intervals_by_session=fit_intervals, preprocessing=SOURCE_EOG_PREPROCESSING)
        save_eog_regression(folder / "ocular_beta.json", ocular)
        variants, supports = {}, {}
        for alpha in (0., .5, 1.):
            model = None if alpha == 0 else replace(ocular, correction_strength=alpha)
            data = load_brainsync_sessions([s.root for s in scoped_sessions] + [inputs / s for s in test_ids],
                ocular_model=model, continuous_qc_model=qc, preprocessing=SOURCE_EOG_PREPROCESSING)
            if data.provenance["source_reference"] != "measured_channel:CZ":
                raise ValueError("continuous preprocessing did not establish the declared measured Cz reference")
            data = select_epoch_channels(data, SIX)
            train = select_scheduled_events(data, np.isin(data.event_timeline.event_ids, list(allowed_events)),
                reason=f"authorized K={budget} calibration events; raw EOG/QC fit bounded by recorded native sample intervals")
            test = select_scheduled_events(data, np.isin(data.event_timeline.session_ids, test_ids),
                reason="locked later development recordings; no fitting")
            train.provenance["calibration_raw_authorization"] = dict(
                scheduled_event_ids=sorted(allowed_events), fit_intervals_by_session=fit_intervals,
                continuous_digest=semantic_sha256(qc.record()), beta_digest=semantic_sha256(ocular.record()))
            variants[alpha] = (train, test)
            supports[alpha] = (observed_event_ids(train), observed_event_ids(test))
        common_train = set.intersection(*(pair[0] for pair in supports.values()))
        common_test = set.intersection(*(pair[1] for pair in supports.values()))
        record = dict(acquisition_K=budget, authorized_event_ids=sorted(allowed_events),
                      fit_intervals_by_session=fit_intervals, common_train_events=sorted(common_train),
                      common_test_events=sorted(common_test), variants={})
        for alpha, (train, test) in variants.items():
            for view in ("pipeline", "common"):
                selected_train = train if view == "pipeline" else select_scheduled_events(train,
                    np.isin(train.event_timeline.event_ids, list(common_train)),
                    reason="prespecified alpha-complete calibration support; no test information", preserve_schedule=True)
                # Keep the complete request ledger at test; record common scoring support separately.
                key = f"alpha{alpha:g}_{view}"
                train_path = save_epoch_dataset(folder / f"{key}_train.npz", selected_train)
                test_path = save_epoch_dataset(folder / f"{key}_test.npz", test)
                record["variants"][key] = dict(alpha=alpha, view=view, actual_training_rows=selected_train.n_epochs,
                    train=dict(path=str(train_path), sha256=file_sha256(train_path)),
                    test=dict(path=str(test_path), sha256=file_sha256(test_path)),
                    scoring_event_ids=None if view == "pipeline" else sorted(common_test))
        plan["budgets"][str(budget)] = record
        atomic_write_json(output / "prepared.json", plan)
    plan["status"] = "completed"
    atomic_write_json(output / "prepared.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    args = parser.parse_args()
    prepare(args.inputs.resolve(), args.output.resolve(), tuple(args.calibration), tuple(args.test))


if __name__ == "__main__":
    main()
