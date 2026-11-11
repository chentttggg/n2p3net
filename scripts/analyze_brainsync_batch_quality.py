"""Read-only, single-person BrainSync batch diagnostics; no model or QC fitting.

Run with the repository venv. Each output directory is a new immutable analysis.
Threshold counts are sensitivity diagnostics and never remove EEG or epochs.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import psutil
from scipy.signal import welch
from threadpoolctl import threadpool_limits

from data.brainsync import load_brainsync_inference_session
from data.brainsync_contract import validate_brainsync_bids_session
from research.recording_quality import create_recording_quality_report, file_sha256

REPO = Path(__file__).resolve().parents[1]
THRESHOLDS_UV = (100, 200, 500)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def quantiles(values):
    values = np.asarray(values)
    return {name: float(np.quantile(values, q)) for name, q in (("p50", .5), ("p95", .95), ("p99", .99))} if values.size else {}


def original_files(root):
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def diagnostic_raw(root, validated, report, directory):
    """Native volts; 2 s windows never cross actual acquisition boundaries."""
    raw = mne.io.read_raw_edf(validated.bids.raw_path, preload=True, verbose="ERROR")
    try:
        clock = validated.manifest["recording"].get("clock") or {}
        count = int(clock.get("acquired_samples", raw.n_times))
        fs = float(raw.info["sfreq"])
        values = raw.get_data(stop=count) * 1e6
        cuts = sorted({0, count, *(round(t * fs) for t in validated.bids.acquisition_boundaries)})
        segments = list(zip(cuts[:-1], cuts[1:], strict=True))
        channels = []
        psd_sum, weight = None, 0
        centers, window_ptp = [], []
        for a, b in segments:
            size = round(2 * fs)
            usable = (b - a) // size * size
            if usable:
                blocks = values[:, a:a + usable].reshape(len(values), -1, size)
                window_ptp.append(np.ptp(blocks, axis=-1).T)
                centers.extend((a + np.arange(blocks.shape[1]) * size + size / 2) / fs)
            if b - a >= size:
                frequencies, power = welch(values[:, a:b], fs=fs, nperseg=size, axis=-1)
                windows = 1 + (b - a - size) // (size // 2)
                psd_sum = power * windows if psd_sum is None else psd_sum + power * windows
                weight += windows
        powers = psd_sum / weight if weight else None
        ptp = np.concatenate(window_ptp) if window_ptp else np.empty((0, len(values)))
        for i, name in enumerate(raw.ch_names):
            slopes, jumps = [], []
            for a, b in segments:
                jumps.extend(np.abs(np.diff(values[i, a:b])))
                size = round(fs)
                usable = (b - a) // size * size
                medians = np.median(values[i, a:a + usable].reshape(-1, size), axis=1)
                if len(medians) > 30:
                    slopes.extend(np.abs(medians[30:] - medians[:-30]) / 30)
            row = {"name": name, "type": validated.bids.channel_types[i],
                   "raw_2s_ptp_uv": quantiles(ptp[:, i]), "raw_step_uv": quantiles(jumps),
                   "abs_30s_median_slope_uv_per_s": quantiles(slopes)}
            if powers is not None:
                df = frequencies[1] - frequencies[0]
                for label, lo, hi in (("band_1_30_rms_uv", 1, 30), ("line_49_51_rms_uv", 49, 51), ("band_30_80_rms_uv", 30, 80)):
                    row[label] = float(np.sqrt(powers[i, (frequencies >= lo) & (frequencies < hi)].sum() * df))
            channels.append(row)
        np.savez_compressed(directory / "raw_window_metrics.npz", times_s=centers, ptp_uv=ptp,
                            channel_names=np.asarray(raw.ch_names))
        return channels
    finally:
        raw.close()


def extra_diagnostics(root, report, directory, dataset):
    validated = validate_brainsync_bids_session(root)
    try:
        raw_channels = diagnostic_raw(root, validated, report, directory)
        timeline = dataset.event_timeline
        markers = validated.bids.stimuli
        assert len(markers) == len(timeline.evidence_indices)
        assert np.array_equal(np.asarray([m.sample for m in markers]), timeline.onset_samples)
        available = np.asarray(timeline.evidence_indices) >= 0
        indices = np.asarray(timeline.evidence_indices)[available]
        assert set(indices.tolist()) == set(range(dataset.n_epochs))
        labels = np.empty(dataset.n_epochs, dtype=bool)
        blocks = np.empty(dataset.n_epochs, dtype=int)
        for marker, index in zip(markers, timeline.evidence_indices, strict=True):
            if index >= 0:
                labels[index] = int(marker.candidate_id) == validated.thought_digit
                blocks[index] = marker.block_id
        scheduled_labels = np.asarray([int(m.candidate_id) == validated.thought_digit for m in markers])
        ptp = np.ptp(dataset.X, axis=-1) * 1e6
        channel_metrics = [{"name": name, "epoch_ptp_uv": quantiles(ptp[:, i]),
                            "epoch_count_above_ptp_uv": {str(t): int((ptp[:, i] > t).sum()) for t in THRESHOLDS_UV}}
                           for i, name in enumerate(dataset.channel_names)]
        onset_differences = []
        for left, right in zip(markers[:-1], markers[1:], strict=True):
            if left.block_id == right.block_id:
                onset_differences.append(right.onset_seconds - left.onset_seconds)
        counts = Counter()
        with validated.bids.events_path.open(encoding="utf-8", newline="") as stream:
            event_rows = list(csv.DictReader(stream, delimiter="\t"))
        counts.update(r["trial_type"] for r in event_rows)
        per_block_candidates = {str(block): dict(Counter(m.candidate_id for m in markers if m.block_id == block))
                                for block in dict.fromkeys(m.block_id for m in markers)}
        time_ms = dataset.preprocessing.tmin_ms + np.arange(dataset.n_times) / dataset.preprocessing.sfreq * 1000
        evoked_target = dataset.X[labels].mean(axis=0) * 1e6
        evoked_other = dataset.X[~labels].mean(axis=0) * 1e6
        difference = evoked_target - evoked_other
        odd = blocks % 2 == 1
        halves = []
        for mask in (odd, ~odd):
            halves.append((dataset.X[mask & labels].mean(axis=0) - dataset.X[mask & ~labels].mean(axis=0)) * 1e6)
        np.savez_compressed(directory / "erp_descriptive.npz", time_ms=time_ms,
                            channel_names=np.asarray(dataset.channel_names), target_uv=evoked_target,
                            nontarget_uv=evoked_other, difference_uv=difference,
                            odd_blocks_difference_uv=halves[0], even_blocks_difference_uv=halves[1])
        figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), layout="constrained", sharex=True)
        for i, name in enumerate(dataset.channel_names):
            axes[0].plot(time_ms, difference[i], label=name, linewidth=1)
        axes[0].set(ylabel="Target - nontarget (uV)", title=f"{root.name}: all available epochs, no fitted QC")
        axes[0].legend(ncol=7, fontsize=8)
        for name in ("CZ", "PZ"):
            i = dataset.channel_names.index(name)
            axes[1].plot(time_ms, halves[0][i], label=f"{name} odd blocks", linewidth=1.2)
            axes[1].plot(time_ms, halves[1][i], label=f"{name} even blocks", linestyle="--", linewidth=1.2)
        for axis in axes:
            axis.axhline(0, color=".4", linewidth=.6)
            axis.axvline(0, color=".4", linewidth=.6)
            axis.grid(alpha=.15)
        axes[1].legend(ncol=2, fontsize=9)
        axes[1].set(xlabel="Nominal model epoch time (ms)", ylabel="Target - nontarget (uV)")
        figure.savefig(directory / "erp_descriptive.png", dpi=140, facecolor="white")
        plt.close(figure)
        summary_window = (time_ms >= 250) & (time_ms < 600)
        erp = [{"name": name, "mean_difference_250_600_uv": float(difference[i, summary_window].mean()),
                "odd_blocks_mean_difference_uv": float(halves[0][i, summary_window].mean()),
                "even_blocks_mean_difference_uv": float(halves[1][i, summary_window].mean())}
               for i, name in enumerate(dataset.channel_names)]
        validated.bids.assert_content_unchanged()
        return {"raw_channels": raw_channels, "epoch_channels": channel_metrics,
                "event_types": dict(counts), "candidate_counts_per_block": per_block_candidates,
                "balanced_5_each_per_block": all(row == {str(k): 5 for k in range(1, 10)} for row in per_block_candidates.values()),
                "within_block_soa_s": {**quantiles(onset_differences), "min": min(onset_differences), "max": max(onset_differences)},
                "target_counts": {"scheduled": int(scheduled_labels.sum()), "preprocessed": int(labels.sum()),
                                  "nontarget_scheduled": int((~scheduled_labels).sum()), "nontarget_preprocessed": int((~labels).sum())},
                "epoch_any_channel_above_ptp_uv": {str(t): int((ptp > t).any(axis=1).sum()) for t in THRESHOLDS_UV},
                "descriptive_erp": erp, "erp_target_labels_used": True,
                "erp_note": "Descriptive, all available epochs; no artifact removal or inference of efficacy. One fixed target per session; odd/even blocks are repeated measurements."}
    finally:
        validated.bids.close()


def overview(rows, directory):
    good = [r for r in rows if r.get("extra")]
    labels = [r["directory_id"].removeprefix("P001_").removesuffix("_01") for r in good]
    names = [c["name"] for c in good[0]["extra"]["epoch_channels"]]
    matrix = np.asarray([[c["epoch_ptp_uv"]["p50"] for c in r["extra"]["epoch_channels"]] for r in good])
    figure, axes = plt.subplots(1, 2, figsize=(14, max(6, len(good) * .4)), layout="constrained")
    image = axes[0].imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_xticks(range(len(names)), names)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_title("Median epoch peak-to-peak (uV), shared preprocessing")
    for i in range(len(good)):
        for j in range(len(names)):
            axes[0].text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8,
                         color="white" if image.norm(matrix[i, j]) < .5 else "black")
    figure.colorbar(image, ax=axes[0], label="uV")
    y = np.arange(len(good))
    scheduled = np.asarray([r["report"]["counts"]["scheduled_trials"] for r in good])
    available = np.asarray([r["report"]["counts"]["preprocessed_epochs"] for r in good])
    axes[1].barh(y, scheduled, color="#dddddd", label="Scheduled")
    axes[1].barh(y, available, color="#0072b2", label="Basic preprocessing available")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 510), xlabel="Flashes (not artifact-free certifications)", title="All requested flashes retained in denominator")
    for i, (n, total) in enumerate(zip(available, scheduled, strict=True)):
        axes[1].text(total + 5, i, f"{n}/{total}", va="center", fontsize=8)
    axes[1].legend(loc="lower left", fontsize=8)
    figure.savefig(directory / "overview.png", dpi=155, facecolor="white")
    plt.close(figure)
    columns = ["group", "directory_id", "status", "target", "scheduled", "available", "complete_blocks", "epoch_any_ptp_gt_100uv", "epoch_any_ptp_gt_200uv", "epoch_any_ptp_gt_500uv", "report_error"]
    with (directory / "sessions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            counts = r["report"].get("counts", {})
            extra = r.get("extra", {})
            writer.writerow({"group": r["group"], "directory_id": r["directory_id"], "status": r["status"],
                             "target": r["target"], "scheduled": counts.get("scheduled_trials"),
                             "available": counts.get("preprocessed_epochs"),
                             "complete_blocks": sum(b["preprocessed_epochs"] == 45 for b in r["report"].get("blocks", [])),
                             **{f"epoch_any_ptp_gt_{t}uv": extra.get("epoch_any_channel_above_ptp_uv", {}).get(str(t)) for t in THRESHOLDS_UV},
                             "report_error": r["report"].get("error", {}).get("message", "")})
    parts = ["<!doctype html><meta charset='utf-8'><title>BrainSync 同一个体数据质量分析</title>",
             "<style>body{max-width:1250px;margin:30px auto;padding:0 20px;font:15px/1.7 system-ui}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}img{max-width:100%}</style>",
             "<h1>BrainSync 同一个体数据质量分析</h1>",
             "<p>按用户确认映射为同一个体；原始参与者与会话编号保留。基础预处理可用数量与伪迹质量分别报告。未拟合分类器、QC或EOG模型。</p>",
             "<p>统一处理：原生七导、左耳参考、50 Hz/Q30陷波、0.1–30 Hz双向滤波、128 Hz、[-200,1200) ms、刺激前均值基线。ERP仅作带标签描述，幅度门限统计不删数据。</p>",
             "<img src='overview.png' alt='逐会话逐导epoch幅度以及完整排程的基础输入覆盖'><table><tr><th>组别 / 会话</th><th>状态</th><th>预处理输入 / 排程</th><th>任一导PTP超过100/200/500 μV的epoch数</th><th>图表与机器结果</th></tr>"]
    for r in rows:
        counts = r["report"].get("counts", {})
        flag = r.get("extra", {}).get("epoch_any_channel_above_ptp_uv", {})
        sid = html.escape(r["directory_id"])
        base = f"sessions/{r['group']}/{sid}"
        details = f"<a href='{base}/report.html'>无标签基础质量报告</a>"
        if r.get("extra"):
            details += f" · <a href='{base}/erp_descriptive.png'>描述性ERP</a> · <a href='{base}/extra.json'>详细数值</a>"
        parts.append(f"<tr><td>{r['group']} / {sid}</td><td>{html.escape(r['status'])}</td><td>{counts.get('preprocessed_epochs', '—')} / {counts.get('scheduled_trials', '—')}</td><td>{'/'.join(str(flag.get(str(t), '—')) for t in THRESHOLDS_UV)}</td><td>{details}</td></tr>")
    parts.append("</table><p>原始信号、频谱、幅度分布、全部闪烁覆盖见逐会话报告；原始2秒窗口指标和ERP数组可复算。输入完整不等于无伪迹，描述性ERP不等于九选一准确率。</p>")
    (directory / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), *(REPO / "src/data").glob("*.py"), REPO / "src/research/recording_quality.py", REPO / "src/transfer/quality.py"]
    manifest = {"schema": "brainsync_batch_quality/1", "created_utc": datetime.now(UTC).isoformat(),
                "role": "single_person_descriptive_quality", "participant_linkage_authority": "User explicitly confirmed all added and previous samples are the same person on 2026-10-30; original IDs unchanged.",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "git_status_before": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
                "source_sha256": {p.relative_to(REPO).as_posix(): file_sha256(p) for p in source_files},
                "python": sys.version, "interpreter": sys.executable, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "mne", "matplotlib")},
                "logical_cpus": os.cpu_count(), "available_ram_bytes_at_start": psutil.virtual_memory().available,
                "native_thread_limit": 2, "fit_performed": False, "classifier_evaluated": False,
                "quality_target_labels_used": False, "secondary_descriptive_erp_target_labels_used": True,
                "diagnostic_ptp_thresholds_uv": THRESHOLDS_UV,
                "methods": {"raw_windows": "Nonoverlapping complete 2 s native windows per acquisition segment; partial windows omitted only from window statistics.",
                            "psd": "2 s Welch with 50 percent overlap, per actual acquisition segment; window-count weighted mean; constant detrend; band RMS=sqrt(sum PSD bins * df).",
                            "drift": "Absolute difference of native 1 s medians 30 s apart divided by 30; within acquisition segments only.",
                            "erp": "All basic-preprocessing epochs; target minus nontarget; odd/even block comparison descriptive; 250<=t<600 ms summary.",
                            "replicates": "One human; sessions repeated measurements; fixed-target blocks are not independent target choices."},
                "sessions": []}
    inputs = [("historical4", REPO / "experiments/runs/brainsync_p001_20260907/inputs"),
              ("historical2", REPO / "experiments/runs/brainsync_quality_20260909/inputs"),
              ("Brainsync1", REPO / "mne_data/Brainsync1"), ("Brainsync2", REPO / "mne_data/Brainsync2")]
    rows = []
    with threadpool_limits(limits=2):
        for group, parent in inputs:
            for session_path in sorted(parent.glob("*/session.json")):
                root = session_path.parent
                print(f"START {group}/{root.name}", flush=True)
                before = original_files(root)
                m = json.loads(session_path.read_text(encoding="utf-8-sig"))
                cached = {}

                def loader(path, cached=cached):
                    cached["dataset"] = load_brainsync_inference_session(path)
                    return cached["dataset"]

                output = directory / "sessions" / group / root.name
                report = create_recording_quality_report(root, output, loader=loader)
                row = {"group": group, "directory_id": root.name, "original_manifest_session_id": m["session_id"],
                       "subject_id_as_saved": m["experiment"]["subject_id"], "person_analysis_id": "same_person_01",
                       "input_root": str(root), "status": m["status"],
                       "target": (m.get("target_label") or {}).get("thought_digit"),
                       "recording_clock": (m.get("recording") or {}).get("clock"),
                       "impedance": (m.get("quality") or {}).get("impedance"),
                       "original_quality_job": json.loads((root / "analysis/quality_report.json").read_text(encoding="utf-8")) if (root / "analysis/quality_report.json").exists() else None,
                       "input_sha256": before, "report": report}
                if report["status"] in {"completed", "attention"}:
                    row["extra"] = extra_diagnostics(root, report, output, cached["dataset"])
                    write_json(output / "extra.json", row["extra"])
                assert before == original_files(root), f"Original input changed: {root}"
                row["original_files_unchanged"] = True
                rows.append(row)
                write_json(output / "session_analysis.json", row)
                print(f"DONE {root.name}: {report['status']} {report.get('counts', {})}", flush=True)
    overview(rows, directory)
    manifest["sessions"] = rows
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    manifest["source_unchanged"] = all(file_sha256(REPO / p) == sha for p, sha in manifest["source_sha256"].items())
    assert manifest["source_unchanged"]
    manifest["input_files_unchanged"] = all(r["original_files_unchanged"] for r in rows)
    manifest["outputs"] = {p.relative_to(directory).as_posix(): file_sha256(p) for p in directory.rglob("*") if p.is_file()}
    write_json(directory / "manifest.json", manifest)
    print(f"COMPLETE {directory}: {len(rows)} sessions", flush=True)


if __name__ == "__main__":
    main()
