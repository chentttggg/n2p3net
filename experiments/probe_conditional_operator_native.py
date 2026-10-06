"""One explicit BNCI native-sample observed-continuation development probe.

The MAT contains distributed processed EEG, not ADC acquisition. This probe
applies no additional filtering, resampling or FFT. It verifies real downstream
conditional scoring, not identification of unobserved boundary filter states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data.bnci2014_008_candidate import (  # noqa: E402
    BNCI2014_008_CHANNELS,
    BNCI2014_008_SOURCE_REFERENCE,
    BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
    read_bnci2014_008_candidate_record,
)
from experiments.run_operator_decision_study import main as operator_main  # noqa: E402
from research.evaluation import verified_executed_snapshot  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.noise_model import array_digest  # noqa: E402
from transfer.operator_lda import OperatorAwareLDA  # noqa: E402


def _feature_maps():
    channels, samples = len(BNCI2014_008_CHANNELS), 192
    conditioner = np.zeros((channels, samples*channels))
    response = np.zeros((2*channels, samples*channels))
    for channel in range(channels):
        conditioner[channel, np.arange(24)*channels+channel] = 1/24
        for block, start in enumerate((96, 144)):
            response[block*channels+channel, np.arange(start, start+48)*channels+channel] = 1/48
    return conditioner, response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    verified_executed_snapshot(args.snapshot, root=ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"schema": "n2p3_conditional_native_probe/1", "status": "running",
              "evidence_scope": "public_processed_dataset_development_software_acceptance",
              "scope": "one_subject_observed_continuation_not_unobserved_filter_state_identification",
              "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest()}
    atomic_write_json(args.output/"execution.json", status)
    try:
        _run(args, status)
        verified_executed_snapshot(args.snapshot, root=ROOT)
    except BaseException as error:
        atomic_write_json(args.output/"execution.json", {**status, "status": "failed",
                          "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    atomic_write_json(args.output/"execution.json", {**status, "status": "completed"})
    return 0


def _run(args, status):
    payload = args.input.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    import io
    record = read_bnci2014_008_candidate_record(io.BytesIO(payload), source_relative_path=args.input.name,
        source_sha256=digest, source_provenance={"stage": "public_processed_dataset_development", "physical_adc_raw": False})
    c_map, y_map = _feature_maps()
    np.savez_compressed(args.output/"native_feature_maps.npz", conditioning_map=c_map, response_map=y_map)
    support = {"conditioning_sample_support": [0, 24], "response_sample_support": [96, 192], "available_samples": 192}
    roles = {"calibration": (0, 1), "held": (2,)}
    arrays, ledgers = {}, {}
    for role, selections in roles.items():
        raw, response, labels, ledger = [], [], [], []
        first = int(record.flash_sample[np.flatnonzero(record.selection_index == min(selections))[0]])-32
        cutoff = int(record.flash_sample[np.flatnonzero(record.selection_index == max(selections)+1)[0]])-32
        for index in np.flatnonzero(np.isin(record.selection_index, selections)):
            onset = int(record.flash_sample[index])
            begin, end = onset-32, onset+160
            if begin < first or end > cutoff:
                raise ValueError("declared native draw exceeds its complete selection authorization")
            window = record.eeg_uv[begin:end]*1e-6
            flat = window.reshape(-1)  # sample then authoritative physical channel
            x, y = c_map @ flat, y_map @ flat
            np.testing.assert_allclose(x, window[:24].mean(axis=0), rtol=1e-14, atol=1e-20)
            np.testing.assert_allclose(y, np.concatenate((window[96:144].mean(axis=0), window[144:192].mean(axis=0))), rtol=1e-14, atol=1e-20)
            raw.append(x)
            response.append(y)
            labels.append(int(record.row_code[index] == record.target_row[index] or record.col_code[index] == record.target_col[index]))
            ledger.append({"draw_id": f"{record.selection_id[index]}:flash{int(index):04d}",
                           "selection_id": str(record.selection_id[index]), "onset_sample": onset,
                           "full_read_sample_support": [begin, end], "conditioning_sample_support": [begin, begin+24],
                           "scored_sample_support": [begin+96, end], "available_at_sample": end,
                           "native_window_sha256": array_digest(window), "feature_sha256": array_digest(x, y),
                           "authorized_role_sample_support": [first, cutoff]})
        arrays[role] = (np.asarray(raw), np.asarray(response), np.asarray(labels))
        ledgers[role] = ledger
    assert not set(r["selection_id"] for r in ledgers["calibration"]) & set(r["selection_id"] for r in ledgers["held"])
    x, y, labels = arrays["calibration"]
    class_x = np.stack([x[labels == q].mean(axis=0) for q in (0, 1)])
    class_y = np.stack([y[labels == q].mean(axis=0) for q in (0, 1)])
    np.savez_compressed(args.output/"noise_draws.npz", raw_residual_draws=x-class_x[labels], state_draws=y-class_y[labels])
    noise_plan = {"input": "noise_draws.npz", "fit_scope": "target_calibration", "scope_id": f"{record.subject_id}:selections00-01",
                  "draw_ids": [r["draw_id"] for r in ledgers["calibration"]]}
    atomic_write_json(args.output/"noise_plan.json", noise_plan)
    operator_main(["noise-fit", "--plan", str(args.output/"noise_plan.json"), "--output", str(args.output/"noise")])
    np.savez_compressed(args.output/"operators.npz", A=np.zeros((y.shape[1], x.shape[1])), B=np.eye(y.shape[1]))
    np.savez_compressed(args.output/"calibration.npz", features=y, labels=labels, conditioning_features=x)
    np.savez_compressed(args.output/"calibration_marginal.npz", features=y, labels=labels)
    test_x, test_y, truth = arrays["held"]
    np.savez_compressed(args.output/"held.npz", features=test_y, observed_raw=test_x)
    np.savez_compressed(args.output/"held_marginal.npz", features=test_y)
    np.savez_compressed(args.output/"held_truth_scorer_only.npz", truth=truth)
    results = {}
    for arm in ("conditional", "marginal"):
        identity = f"{record.subject_id}:native256-right-ear:prefix24-response2x48:{arm}"
        plan = {"input": "calibration.npz" if arm == "conditional" else "calibration_marginal.npz", "noise_model": "noise/noise.npz",
                "operators": "operators.npz", "calibration_ids": noise_plan["draw_ids"], "scope_id": noise_plan["scope_id"],
                "input_identity": identity, "ridge_relative": 1e-3}
        metadata = {"input_identity": identity}
        if arm == "conditional":
            plan["observation_support"] = support
            metadata["observation_support"] = support
        atomic_write_json(args.output/f"{arm}_plan.json", plan)
        atomic_write_json(args.output/f"{arm}_metadata.json", metadata)
        operator_main(["operator-lda-fit", "--plan", str(args.output/f"{arm}_plan.json"), "--output", str(args.output/f"{arm}_fit")])
        model_path = args.output/f"{arm}_fit/model.npz"
        before = model_path.read_bytes()
        operator_main(["operator-lda-predict", "--input", str(args.output/("held.npz" if arm == "conditional" else "held_marginal.npz")),
                       "--model", str(model_path), "--metadata", str(args.output/f"{arm}_metadata.json"), "--output", str(args.output/f"{arm}_prediction")])
        prediction = json.loads((args.output/f"{arm}_prediction/prediction.json").read_text(encoding="utf-8"))
        logits = np.asarray(prediction["logits"])
        reloaded = OperatorAwareLDA.load(model_path)
        replay = reloaded.predict_logit(test_y, observed_raw=test_x if arm == "conditional" else None, input_identity=identity)
        np.testing.assert_array_equal(logits, replay)
        assert model_path.read_bytes() == before
        with np.load(args.output/"held_truth_scorer_only.npz", allow_pickle=False) as scorer:
            held_labels = scorer["truth"]
        probability = 1/(1+np.exp(-logits))
        results[arm] = {"n_scored": len(logits), "n_targets": int(held_labels.sum()), "binary_auc": float(roc_auc_score(held_labels, logits)),
                        "mean_nll_fixed_equal_prior": float(np.mean(np.logaddexp(0, logits)-held_labels*logits)),
                        "mean_brier": float(np.mean((probability-held_labels)**2)), "exact_reload_replay": True,
                        "model_unchanged_at_predict": True, "no_raw_likelihood_multiplier": True}
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == digest
    atomic_write_json(args.output/"receipt.json", {**status, "status": "completed", "source_file": str(args.input),
        "source_sha256": digest, "subject": record.subject_id, "sfreq": BNCI2014_008_SOURCE_SAMPLE_RATE_HZ,
        "channels": list(BNCI2014_008_CHANNELS), "reference": BNCI2014_008_SOURCE_REFERENCE,
        "additional_filter_or_fft": False, "map_layout": "sample_then_authoritative_physical_channel",
        "native_maps_sha256": hashlib.sha256((args.output/"native_feature_maps.npz").read_bytes()).hexdigest(),
        "conditioning_read_onset_relative_samples": [-32, -8], "response_read_onset_relative_samples": [64, 160],
        "state_coordinate_role": "observed_response_residual_not_unobserved_filter_state", "draw_ledgers": ledgers,
        "same_calibration_residual_draws_for_both_arms": True, "results": results,
        "performance_claim": "none_one_subject_binary_ERP_caller_acceptance"})
    print(json.dumps(results))


if __name__ == "__main__":
    raise SystemExit(main())
