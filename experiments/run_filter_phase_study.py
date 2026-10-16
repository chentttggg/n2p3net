"""Paired completed-recording filter study; development evidence, with saved predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import sosfreqz  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from data.brainsync import load_brainsync_session  # noqa: E402
from data.brainsync_contract import validate_brainsync_bids_session  # noqa: E402
from data.contract import (  # noqa: E402
    BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
    BRAIN_SYNC_CAUSAL_DATA_CONTRACT,
)
from data.epochs import preprocessing_spec_from_contract  # noqa: E402
from data.preprocess import design_continuous_sos, filter_continuous, preprocess  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def profiles():
    forward = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    return {
        "forward": forward,
        "zero_legacy": replace(forward, name="brainsync_zero_legacy_control",
                               filter_phase="zero", causal_iir_initial_state="not_applicable"),
        "zero_reflect": preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT),
    }


def filter_options(spec):
    return dict(l_freq=spec.l_freq, h_freq=spec.h_freq, phase=spec.filter_phase,
                causal_iir_initial_state=spec.causal_iir_initial_state,
                notch_freqs_hz=spec.notch_freqs_hz, notch_quality_factor=spec.notch_quality_factor,
                zero_phase_edge_policy=spec.zero_phase_edge_policy)


def raw_signal(values):
    return mne.io.RawArray(np.array(values, dtype=float, copy=True)[None],
                           mne.create_info(["Cz"], 250, "eeg"), verbose=False)


def signal_study(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sos = design_continuous_sos(250, l_freq=.1, h_freq=30, notch_freqs_hz=(50,))
    f = np.geomspace(.03, 80, 1200)
    _, response = sosfreqz(sos, worN=f, fs=250)
    params = mne.filter.construct_iir_filter(dict(sos=sos), verbose=False)
    time = np.arange(30000) / 250 - 60
    erp = 8e-6 * np.exp(-.5 * ((time - .35) / .08)**2) - 3e-6 * np.exp(-.5 * ((time - .15) / .05)**2)
    event = np.array([[15000, 0, 1]])
    unfiltered = preprocess(raw_signal(erp), event, l_freq=None, h_freq=None).data[0, 0]
    metrics, waveforms, future = {}, {}, {}
    for name, spec in profiles().items():
        options = filter_options(spec)
        full = filter_continuous(raw_signal(erp), **options).get_data()[0]
        result = preprocess(raw_signal(erp), event, **{k: v for k, v in options.items() if k != "phase"},
                            filter_phase=spec.filter_phase).data[0, 0]
        waveforms[name] = result
        t = np.arange(len(result)) / 128 - .2
        peak = (t >= .2) & (t <= .6)
        # Same known ERP near each boundary; long-record output is a sensitivity reference.
        endpoint_errors = []
        for onset in (1., 5., 10., 18.):
            start = 15000 - round(onset * 250)
            short = filter_continuous(raw_signal(erp[start:start + 5000]), **options).get_data()[0]
            a = round((onset - .2) * 250)
            b = round((onset + 1.2) * 250)
            diff = short[a:b] - full[start + a:start + b]
            diff -= diff[:50].mean()
            endpoint_errors.append(float(np.sqrt(np.mean(diff**2)) * 1e6))
        step = np.zeros(30000)
        step[15000:] = 1e-3
        filtered_step = filter_continuous(raw_signal(step), **options).get_data()[0]
        future[name] = filtered_step
        metrics[name] = dict(peak_ms=float(t[peak][np.argmax(result[peak])] * 1000),
                             erp_rmse_uv=float(np.sqrt(np.mean((result-unfiltered)**2)) * 1e6),
                             boundary_onsets_s=[1, 5, 10, 18], boundary_rmse_uv=endpoint_errors,
                             future_step_preceding_1s_max_uv=float(np.max(np.abs(filtered_step[14750:15000])) * 1e6))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    axes[0].semilogx(f, 20*np.log10(np.maximum(np.abs(response), 1e-10)), label="Forward")
    axes[0].semilogx(f, 40*np.log10(np.maximum(np.abs(response), 1e-10)), label="Forward-backward")
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", ylim=(-65, 2), title="Same SOS, different magnitude")
    axes[1].plot(t*1000, unfiltered*1e6, color="black", linestyle="--", label="Known ERP")
    for name in waveforms:
        axes[1].plot(t*1000, waveforms[name]*1e6, label=name)
        part = (time >= -3) & (time <= 3)
        axes[2].plot(time[part], future[name][part]*1e6, label=name)
    axes[1].set(xlabel="Time after event (ms)", ylabel="Amplitude (uV)", title="Production epoch + baseline")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Time from 1 mV step (s)", ylabel="Filtered amplitude (uV)", title="Backward spread before a future step")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "filter_phase.png", dpi=180)
    plt.close(fig)
    return dict(sample_rate_hz=250, scipy_notch_pad_samples=33,
                impulse_response_pad_samples=int(params["padlen"]),
                conditions=metrics, boundary_reference="same-mode known long synthetic recording",
                waveform_reference="unfiltered production resample and baseline")


def features(dataset, reference):
    x = dataset.X.astype(float, copy=True)
    names = [s.upper() for s in dataset.channel_names]
    if reference == "cz":
        cz = names.index("CZ")
        picks = [i for i in range(len(names)) if i != cz]
        x = x[:, picks] - x[:, cz:cz+1]
    times = dataset.preprocessing.tmin_ms + np.arange(x.shape[2]) * 1000 / dataset.preprocessing.sfreq
    edges = np.arange(-200, 1201, 100)
    return np.stack([x[:, :, (times >= a) & (times < b)].mean(-1)
                     for a, b in zip(edges[:-1], edges[1:], strict=True)], -1).reshape(len(x), -1) * 1e6


def candidate_metrics(dataset, scores):
    timeline = dataset.event_timeline
    groups = np.unique(timeline.group_ids)
    ledger = []
    for group in groups:
        rows = np.flatnonzero((timeline.group_ids == group) & (timeline.repetition_indices < 5))
        complete = True
        sums = []
        for digit in map(str, range(1, 10)):
            selected = rows[timeline.candidate_ids[rows] == digit]
            indices = timeline.evidence_indices[selected]
            if (len(selected) != 5 or np.any(indices < 0)
                    or sorted(timeline.repetition_indices[selected].tolist()) != list(range(5))
                    or not np.isfinite(scores[indices]).all()):
                complete = False
                break
            sums.append(float(scores[indices].sum()))
        truth = np.unique(timeline.target_candidate_ids[rows])
        if len(truth) != 1 or truth[0] not in list(map(str, range(1, 10))):
            raise ValueError("decision requires one explicit target in the nine-candidate vocabulary")
        credit = 0.
        if complete:
            winners = np.flatnonzero(np.isclose(sums, np.max(sums), rtol=0, atol=1e-12))
            credit = float((int(truth[0])-1 in winners) / len(winners))
        ledger.append(dict(group=str(group), complete=complete, credit=credit, candidate_scores=sums if complete else None))
    eligible = sum(row["complete"] for row in ledger)
    hits = sum(row["credit"] for row in ledger)
    return dict(requested=len(groups), eligible=eligible, hits=hits,
                hit_r5_all=hits/len(groups), hit_r5_eligible=hits/eligible if eligible else None, ledger=ledger,
                availability="completed_recording; retrospective first-five-cycle score, not early stopping")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sessions = [validate_brainsync_bids_session(p) for p in args.sessions]
    if len(sessions) < 3 or len({s.session_id for s in sessions}) != len(sessions):
        raise ValueError("at least three distinct complete recordings are required")
    if len({s.subject_id for s in sessions}) != 1:
        raise ValueError("this within-person study requires one declared participant")
    sources = []
    for root, session in zip(args.sessions, sessions, strict=True):
        manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
        raw_path = root / manifest["recording"]["path"]
        sources.append(dict(session=session.session_id, root=str(root.resolve()), raw=str(raw_path.resolve()),
                            sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest()))
    protocol = dict(schema="n2p3_filter_phase_study/1", role="within_person_retrospective_development",
                    sources=sources, profiles={k:v.record() for k,v in profiles().items()},
                    split="leave_one_complete_recording_out", references=["native", "cz"],
                    classifier="LDA lsqr auto shrinkage, fixed equal priors; 100 ms window means in uV",
                    continuous_qc="not_applied; common raw-supported rows, includes observed contamination",
                    candidate_estimand="one declared target decision per session; first five cycles scored only after full recording processing",
                    ocular="not_applied", expected_fits=len(sessions)*6,
                    software=dict(numpy=np.__version__, mne=mne.__version__),
                    source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), *sorted((ROOT/"src").rglob("*.py"))]})
    atomic_write_json(output / "protocol.json", protocol)
    with threadpool_limits(limits=1):
        signal = signal_study(output)
        datasets = {k:[load_brainsync_session(root, preprocessing=spec) for root in args.sessions]
                    for k,spec in profiles().items()}
        base = datasets["forward"]
        for arm in datasets.values():
            for a,b in zip(base,arm,strict=True):
                np.testing.assert_array_equal(a.event_timeline.event_ids, b.event_timeline.event_ids)
                np.testing.assert_array_equal(a.event_timeline.evidence_indices, b.event_timeline.evidence_indices)
                np.testing.assert_array_equal(a.y, b.y)
        results = []
        for reference in protocol["references"]:
            for name, arm in datasets.items():
                f = [features(d,reference) for d in arm]
                for heldout, test in enumerate(arm):
                    training = [i for i in range(len(arm)) if i != heldout]
                    x = np.concatenate([f[i] for i in training])
                    y = np.concatenate([arm[i].y for i in training])
                    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5,.5]).fit(x,y)
                    scores = model.decision_function(f[heldout])
                    np.testing.assert_allclose(scores, (f[heldout]@model.coef_.T+model.intercept_).ravel(), atol=1e-10)
                    artifact = f"{reference}_{name}_{heldout}.npz"
                    np.savez_compressed(output/artifact, scores=scores, y=test.y, coef=model.coef_,
                                        intercept=model.intercept_, test_features=f[heldout],
                                        training_sessions=np.array([sessions[i].session_id for i in training]),
                                        event_ids=test.event_timeline.event_ids,
                                        evidence_indices=test.event_timeline.evidence_indices)
                    result = dict(reference=reference, profile=name, heldout=sessions[heldout].session_id,
                                  training_epochs=len(y), test_epochs=len(scores), scheduled=test.event_timeline.n_events,
                                  auc=float(roc_auc_score(test.y,scores)), r5=candidate_metrics(test,scores),
                                  artifact=artifact, sha256=hashlib.sha256((output/artifact).read_bytes()).hexdigest())
                    results.append(result)
                    print(json.dumps({k:v for k,v in result.items() if k not in {"r5","sha256"}}), flush=True)
        for source in sources:
            assert hashlib.sha256(Path(source["raw"]).read_bytes()).hexdigest() == source["sha256"]
        atomic_write_json(output / "results.json", dict(protocol=protocol, signal=signal, results=results,
                                                        completed_fits=len(results), raw_hashes_unchanged=True))


if __name__ == "__main__":
    main()
