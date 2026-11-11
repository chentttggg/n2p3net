"""Isolate frozen EOG subtraction while holding QC masks/filter segments fixed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data import brainsync  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.brainsync_startup import apply_startup_cut, startup_cut_for  # noqa: E402
from data.continuous_qc import load_continuous_qc  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402
from data.ocular import load_eog_regression  # noqa: E402
from data.source_records import SIX, file_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    plan = json.loads((run / "target_startup_cut_nominal_v1/prepared.json").read_text(encoding="utf-8"))
    quality = json.loads((run / "confirmed_EOG_inputs/quality_manifest.json").read_text(encoding="utf-8"))
    roots = {r["directory_id"]: r["input_root"] for r in quality["sessions"]}
    cell = plan["cells"]["QC_K1"]
    ocular_path = run / "target_startup_cut_v1/QC_K1/ocular.json"
    qc_path = run / "target_startup_cut_v1/QC_K1/continuous.json"
    ocular, qc = load_eog_regression(ocular_path), load_continuous_qc(qc_path)
    scope = brainsync.IndependentRecordingScope(tuple(cell["calibration_sessions"]), tuple(cell["test_sessions"]))
    original_builder = brainsync.build_subject
    calibration = validate_brainsync_bids_session(roots[cell["calibration_sessions"][0]])
    try:
        raw = brainsync.read_raw(calibration.bids.raw_path, preload=True)
        raw.set_channel_types(dict(zip(raw.ch_names, (k.lower() for k in calibration.bids.channel_types), strict=True)))
        raw.info["bads"] = [n for n,s in zip(raw.ch_names, calibration.bids.channel_statuses, strict=True) if s=="bad"]
        mask = np.ones(raw.n_times, dtype=bool)
        apply_startup_cut(raw, startup_cut_for(calibration.bids), mask)
        for interval in calibration.bids.excluded_intervals:
            mask[round(interval.onset_seconds*raw.info["sfreq"]):round(interval.end_seconds*raw.info["sfreq"])] = False
        if calibration.bids.acquisition_boundaries:
            raw.annotations.append(calibration.bids.acquisition_boundaries,
                [0.]*len(calibration.bids.acquisition_boundaries), ["EDGE boundary"]*len(calibration.bids.acquisition_boundaries))
        mask &= qc.detect(raw).valid_sample_mask
        import hashlib

        assert hashlib.sha256(mask.tobytes()).hexdigest() == ocular.fit_provenance["fit_sample_mask_sha256"]
        e = raw.get_data(picks=list(ocular.eog_channels))[0,mask]
        y = raw.get_data(picks=list(ocular.eeg_channels))[:,mask]-raw.get_data(picks=["Cz"])[0,mask]
        e = e-e.mean()
        y = y-y.mean(axis=1,keepdims=True)
        beta = y@e/(e@e)
        np.testing.assert_allclose(beta,np.asarray(ocular.coefficients)[:,0],rtol=1e-10,atol=1e-10)
        time = np.flatnonzero(mask)/raw.info["sfreq"]
        time = time-time.mean()
        u = time/np.linalg.norm(time)
        covariance = y@e
        trend_covariance = (y@u)*(e@u)
        covariance_diagnosis = {"fit_mask_reproduced":True,"raw_OLS_beta_reproduced":True,
            "EOG_variance_fraction_in_linear_time_trend":float((e@u)**2/(e@e)),
            "EEG_EOG_covariance_signed_fraction_in_linear_time_trend":dict(zip(ocular.eeg_channels,
                (trend_covariance/covariance).tolist(),strict=True)),
            "meaning":"orthogonal decomposition on original C fit-mask samples; signed covariance contribution can exceed one if residual opposes trend"}
        raw.close()
    finally:
        calibration.bids.close()
    rows = []
    for entry in cell["rows"]:
        if entry["role"] != "fixed_future_test":
            continue
        captured = {}

        def build_pair(raw, *positional, _captured=captured, **keywords):
            actual = original_builder(raw, *positional, **keywords)
            no_subtraction = raw.copy().load_data()
            artifact = raw.get_data(picks=list(ocular.eog_channels))
            artifact -= np.asarray(ocular.calibration_eog_mean_v)[:, None]
            correction = ocular.effective_coefficients @ artifact
            for index, name in enumerate(ocular.eeg_channels):
                no_subtraction._data[no_subtraction.ch_names.index(name)] += correction[index]
            control = original_builder(no_subtraction, *positional, **keywords)
            _captured["actual"], _captured["control"] = actual, control
            return actual

        brainsync.build_subject = build_pair
        try:
            actual = brainsync.load_brainsync_session(roots[entry["session_id"]], channels=SIX,
                preprocessing=qc.epoch_recovery.preprocessing, ocular_model=ocular,
                continuous_qc_model=qc, recording_scope=scope)
        finally:
            brainsync.build_subject = original_builder
        saved = load_epoch_dataset(entry["path"], validation="attested")
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("fixed-test cache differs from its frozen prepared manifest")
        np.testing.assert_array_equal(actual.X, saved.X)
        np.testing.assert_array_equal(actual.trial_channel_mask, saved.trial_channel_mask)
        control = captured["control"]
        full_ids = actual.event_timeline.event_ids
        ids = np.flatnonzero(actual.event_timeline.evidence_indices >= 0)
        ids = ids[np.argsort(actual.event_timeline.evidence_indices[ids])]
        lookup = {str(event): int(evidence) for event, evidence in
            zip(control.event_timeline.event_ids, control.event_timeline.evidence_indices, strict=True) if evidence >= 0}
        indices = [lookup[str(full_ids[i])] for i in ids]
        before = control.data[indices]
        mask = actual.trial_channel_mask
        mask = np.broadcast_to(actual.channel_mask, actual.X.shape[:2]) if mask is None else mask
        p1 = np.max(np.where(mask, np.ptp(actual.X, axis=2)*1e6, -np.inf), axis=1)
        p0 = np.max(np.where(mask, np.ptp(before, axis=2)*1e6, -np.inf), axis=1)
        maximum = int(np.argmax(p1))
        channel = int(np.argmax(np.where(mask[maximum], np.ptp(actual.X[maximum],axis=1), -np.inf)))
        fig, ax = plt.subplots(figsize=(10,3), layout="constrained")
        time = actual.preprocessing.tmin_ms+np.arange(actual.X.shape[2])*1000/actual.preprocessing.sfreq
        ax.plot(time, before[maximum,channel]*1e6, color="#0072B2", label="same QC support, EOG subtraction undone")
        ax.plot(time, actual.X[maximum,channel]*1e6, color="#D55E00", label="frozen C1 EOG subtraction")
        onset = float(actual.event_timeline.onset_times_s[ids[maximum]])
        ax.set(xlabel="Time relative to flash (ms)",ylabel="Voltage (uV)",
               title=f'{entry["session_id"]}: {actual.channel_names[channel]}, onset={onset:.3f}s')
        ax.legend(fontsize=8)
        image = entry["session_id"]+"_EOG_control.png"
        fig.savefig(args.output/image,dpi=150)
        plt.close(fig)
        row = {"session_id":entry["session_id"], "epochs":actual.n_epochs,
            "input_cache_sha256":entry["sha256"],
            "replay_bitwise_equal":True,"same_QC_mask_and_filter_segments":True,
            "full_QC_max_PTP_uv":float(p1.max()),"without_EOG_subtraction_max_PTP_uv":float(p0.max()),
            "full_QC_above_1000":int(np.count_nonzero(p1>1000)),
            "without_EOG_subtraction_above_1000":int(np.count_nonzero(p0>1000)),
            "worst_onset_s":onset,"worst_channel":actual.channel_names[channel],"image":image}
        rows.append(row)
        print(row,flush=True)
    atomic_write_json(args.output/"results.json", {"schema":"brainsync_frozen_EOG_subtraction_isolation/1",
        "status":"passed","analysis_source_sha256":file_sha256(__file__),"new_fits":0,
        "ocular_sha256":file_sha256(ocular_path),"continuous_QC_sha256":file_sha256(qc_path),
        "coefficients_by_EEG":dict(zip(ocular.eeg_channels,[float(r[0]) for r in ocular.coefficients],strict=True)),
        "calibration_covariance_decomposition":covariance_diagnosis,
        "control":"undo only the saved pointwise EOG subtraction after QC annotations; same acquired reference, masks, segments, filters, epoch indices and baseline",
        "records":rows})


if __name__ == "__main__":
    main()
