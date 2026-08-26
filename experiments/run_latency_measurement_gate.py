"""Two-fold synthetic gate for the v12 LatencyMeasurement object.

This runner is the Phase-1 synthetic evidence gate. It uses the project's
existing synthetic P3b generator, fits the measurement object on
optimization subjects, calibrates posterior sharpness on known-shift trials,
and reports the pre-registered thresholds. It never loads development folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_latency_identifiability import (  # noqa: E402
    SFREQ,
    _colored_noise,
    _synthetic_trial,
    _time_ms,
    make_synthetic_training_data,
)

from measurement.latency_measurement import LatencyMeasurement  # noqa: E402

_THRESHOLDS = {
    "bias_ms_max": 5.0,
    "rmse_ms_max": 10.0,
    "slope_lo": 0.9,
    "slope_hi": 1.1,
    "coverage_lo": 0.85,
    "coverage_hi": 0.95,
    "tau0_bias_ms_max": 5.0,
    "init_sensitivity_ms_max": 2.0,
    "amplitude_confound_tau_bias_ms_max": 5.0,
    "amplitude_ratio_error_max": 0.10,
}


def _run_fold(seed: int) -> dict[str, object]:
    data = make_synthetic_training_data(
        n_subjects=6,
        n_target_per_subject=30,
        n_nontarget_per_subject=30,
        base_latency_ms=460.0,
        train_jitter_ms=40.0,
        noise_std=0.5,
        seed=seed,
    )
    estimator = LatencyMeasurement(
        anchor_tau0_ms=460.0,
        sfreq=SFREQ,
        time_ms=_time_ms(),
        grid_radius_ms=60.0,
        grid_step_ms=0.5,
        woody_iterations=3,
    ).fit(data.X, data.y, data.groups)

    target = np.flatnonzero(data.y == 1)
    target_subjects = np.unique(data.groups[target])
    calibration_subjects = target_subjects[: len(target_subjects) // 2]
    test_subjects = target_subjects[len(target_subjects) // 2 :]
    calibration = target[np.isin(data.groups[target], calibration_subjects)]
    test = target[np.isin(data.groups[target], test_subjects)]
    estimator.calibrate_posterior_scale(
        data.X[calibration],
        data.true_p3b_latency_ms[calibration],
        target_coverage=0.90,
        scale_lo=0.5,
        scale_hi=12.0,
        n_candidates=81,
    )
    posterior = estimator.predict(data.X[test])
    truth = data.true_p3b_latency_ms[test]
    errors = posterior.mean_ms - truth
    tau0 = estimator.population_tau0(posterior, prior_mean_ms=460.0, prior_sd_ms=30.0)

    # Amplitude/latency confound probes: same noise realization, +40 ms shift,
    # amplitudes 1.0 vs 0.9. A latency estimator that tracks morphology must
    # keep tau bias <5 ms and recover the amplitude ratio within 10%.
    amp_rng = np.random.default_rng(seed + 100_000)
    amp_high: list[np.ndarray] = []
    amp_low: list[np.ndarray] = []
    for _ in range(80):
        noise = _colored_noise(amp_rng, 0.5)
        gain = float(amp_rng.uniform(0.85, 1.15))
        amp_high.append(_synthetic_trial(noise, target=True, p3b_latency_ms=500.0, subject_gain=gain))
        amp_low.append(
            _synthetic_trial(noise, target=True, p3b_latency_ms=500.0, subject_gain=0.9 * gain)
        )
    high = estimator.predict(np.stack(amp_high))
    low = estimator.predict(np.stack(amp_low))
    amplitude_ratios = low.amplitude_mean / np.maximum(np.abs(high.amplitude_mean), 1e-9)
    amplitude_ratio_error = float(np.mean(np.abs(amplitude_ratios - 0.9)))
    amplitude_confound_tau_bias_ms = float(np.abs(np.mean(low.mean_ms - 500.0)))
    return {
        "bias_ms": float(np.mean(errors)),
        "rmse_ms": float(np.sqrt(np.mean(errors**2))),
        "slope": float(np.polyfit(truth, posterior.mean_ms, 1)[0]),
        "coverage": float(np.mean((truth >= posterior.lower_ms) & (truth <= posterior.upper_ms))),
        "tau0_bias_ms": abs(tau0["tau0_ms"] - 460.0),
        "init_sensitivity_ms": tau0["init_sensitivity_ms"],
        "posterior_scale": estimator.posterior_scale,
        "n_test_trials": int(test.size),
        "amplitude_confound_tau_bias_ms": amplitude_confound_tau_bias_ms,
        "amplitude_ratio_error": amplitude_ratio_error,
    }


def _passes(metrics: dict[str, object]) -> bool:
    return bool(
        abs(float(metrics["bias_ms"])) < _THRESHOLDS["bias_ms_max"]
        and float(metrics["rmse_ms"]) < _THRESHOLDS["rmse_ms_max"]
        and _THRESHOLDS["slope_lo"] <= float(metrics["slope"]) <= _THRESHOLDS["slope_hi"]
        and _THRESHOLDS["coverage_lo"] <= float(metrics["coverage"]) <= _THRESHOLDS["coverage_hi"]
        and float(metrics["tau0_bias_ms"]) < _THRESHOLDS["tau0_bias_ms_max"]
        and float(metrics["init_sensitivity_ms"]) < _THRESHOLDS["init_sensitivity_ms_max"]
        and float(metrics["amplitude_confound_tau_bias_ms"])
        < _THRESHOLDS["amplitude_confound_tau_bias_ms_max"]
        and float(metrics["amplitude_ratio_error"]) < _THRESHOLDS["amplitude_ratio_error_max"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    folds = [_run_fold(args.seed + fold) for fold in range(2)]
    report = {
        "schema": "n2p3net_latency_measurement_gate/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "thresholds": _THRESHOLDS,
        "passed": all(_passes(metrics) for metrics in folds),
        "folds": folds,
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[latency-gate] report: {path}", flush=True)
    print(f"[latency-gate] passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
