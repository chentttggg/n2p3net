"""Known low-frequency drift and ERP fidelity through the production operator."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import sosfreqz  # noqa: E402

from data.brainsync import BRAIN_SYNC_PREPROCESSING  # noqa: E402
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT  # noqa: E402
from data.epochs import preprocessing_spec_from_contract  # noqa: E402
from data.preprocess import design_continuous_sos, filter_continuous, preprocess  # noqa: E402


def raw(values):
    return mne.io.RawArray(np.array(values, copy=True)[None], mne.create_info(["Pz"], 250, "eeg"), verbose=False)


def options(spec):
    return dict(l_freq=spec.l_freq, h_freq=spec.h_freq, notch_freqs_hz=spec.notch_freqs_hz,
                notch_quality_factor=spec.notch_quality_factor, causal_iir_initial_state=spec.causal_iir_initial_state,
                zero_phase_edge_policy=spec.zero_phase_edge_policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    forward = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
    profiles = {"forward": forward, "bidirectional": BRAIN_SYNC_PREPROCESSING}
    time = np.arange(120000)/250
    interior = (time >= 120) & (time < 360)
    drift_rows = []
    for frequency in (.03, .05, .08):
        drift = 1e-3*np.sin(2*np.pi*frequency*time)
        basis = np.column_stack((np.sin(2*np.pi*frequency*time[interior]),
                                 np.cos(2*np.pi*frequency*time[interior]), np.ones(interior.sum())))
        row = {"frequency_hz": frequency, "input_amplitude_uv": 1000.}
        for name, spec in profiles.items():
            values = filter_continuous(raw(drift), phase=spec.filter_phase, **options(spec)).get_data()[0]
            coeff = np.linalg.lstsq(basis, values[interior], rcond=None)[0]
            row[name+"_amplitude_uv"] = float(np.hypot(*coeff[:2])*1e6)
            sos = design_continuous_sos(250, l_freq=spec.l_freq, h_freq=spec.h_freq,
                                        notch_freqs_hz=spec.notch_freqs_hz)
            gain = np.abs(sosfreqz(sos, worN=[frequency], fs=250)[1][0])
            row[name+"_theoretical_amplitude_uv"] = float(1000*gain**(2 if spec.filter_phase=="zero" else 1))
        drift_rows.append(row)
    epoch_time = np.arange(60000)/250-120
    erp_rows, example = [], {}
    event = np.array([[30000, 0, 1]])
    for latency in (.3, .5, .8):
        for width in (.06, .12, .22):
            erp = 8e-6*np.exp(-.5*((epoch_time-latency)/width)**2)
            drift = 1e-3*np.sin(2*np.pi*.05*epoch_time+np.pi/3)
            truth = preprocess(raw(erp), event, l_freq=None, h_freq=None).data[0,0]
            row = dict(latency_s=latency, width_s=width, drift_frequency_hz=.05, drift_amplitude_uv=1000.)
            for name,spec in profiles.items():
                clean = preprocess(raw(erp), event, filter_phase=spec.filter_phase, **options(spec)).data[0,0]
                noisy = preprocess(raw(erp+drift), event, filter_phase=spec.filter_phase, **options(spec)).data[0,0]
                row[name] = dict(clean_erp_rmse_uv=float(np.sqrt(np.mean((clean-truth)**2))*1e6),
                                 erp_with_drift_rmse_uv=float(np.sqrt(np.mean((noisy-truth)**2))*1e6))
                if latency==.5 and width==.12:
                    example[name] = noisy
                    example["truth"] = truth
            erp_rows.append(row)
    fig, axes = plt.subplots(1,2,figsize=(10.8,4),layout="constrained")
    x = np.arange(len(drift_rows))
    for offset,name,label in ((-.18,"forward","Forward"),(.18,"bidirectional","Forward-backward")):
        axes[0].bar(x+offset,[r[name+"_amplitude_uv"] for r in drift_rows],.36,label=label)
    axes[0].set(xticks=x,xticklabels=[str(r["frequency_hz"]) for r in drift_rows],
                xlabel="Drift frequency (Hz)",ylabel="Residual amplitude (uV)",yscale="log",
                title="1,000 uV sinusoidal drift; 0.1 Hz high-pass")
    t = np.arange(179)/128*1000-200
    for name,label in (("truth","Known ERP"),("forward","Forward + drift"),("bidirectional","Forward-backward + drift")):
        axes[1].plot(t,example[name]*1e6,label=label)
    axes[1].set(xlabel="Time after event (ms)",ylabel="Amplitude (uV)",title="Actual epoch resampling and baseline")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(args.output/"drift_and_erp.png",dpi=180)
    plt.close(fig)
    record = dict(role="known_signal_operator_validation", source_rate_hz=250,
        profiles={k:v.record() for k,v in profiles.items()}, drift=drift_rows, erp=erp_rows,
        source_files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                      ("src/data/preprocess.py","src/data/contract.py","experiments/audit_bidirectional_drift.py")})
    (args.output/"results.json").write_text(json.dumps(record,indent=2)+'\n',encoding="utf-8")
    print(json.dumps(drift_rows,indent=2))


if __name__ == "__main__":
    main()
