"""Read-only, target-free recording reports using production preprocessing and QC."""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.signal import welch

from data.brainsync import BrainSyncNoUsableEvidenceError, load_brainsync_inference_session
from data.brainsync_contract import validate_brainsync_bids_recording
from data.channel import canonical_channel_name
from data.dataset import read_raw
from data.qc_features import compute_epoch_qc_features
from research.execution import atomic_write_json
from transfer.quality import transform_epoch_quality

REPORT_SCHEMA = "n2p3_recording_quality_report/1"
PLOT_COLORS = ("#147d70", "#3879b6", "#8c63ac", "#ae642e", "#3c8e9b", "#74852e", "#bc5370", "#787f86")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_plot(figure, directory, filename):
    figure.savefig(directory / filename, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return filename


def _raw_diagnostics(validated, directory):
    raw = read_raw(validated.bids.raw_path, preload=False)
    try:
        rate = float(raw.info["sfreq"])
        if tuple(map(canonical_channel_name, raw.ch_names)) != tuple(map(canonical_channel_name, validated.bids.channel_names)):
            raise ValueError("raw channel order differs from the BIDS recording contract")
        if not np.isclose(rate, validated.bids.sample_rate_hz, rtol=0, atol=1e-6):
            raise ValueError("raw sample rate differs from the BIDS recording contract")
        clock = validated.manifest["recording"].get("clock") or {}
        acquired = int(clock.get("acquired_samples", raw.n_times))
        if not 2 <= acquired <= raw.n_times:
            raise ValueError("recording does not contain two acquired samples")
        values = raw.get_data(start=0, stop=acquired)
        if not np.isfinite(values).all():
            raise ValueError("raw recording contains non-finite samples")
        cuts = sorted({0, acquired, *(round(t * rate) for t in validated.bids.acquisition_boundaries)})
        segments = [(a, b) for a, b in zip(cuts[:-1], cuts[1:], strict=True) if 0 <= a < b <= acquired]
        summary = {
            "sample_rate_hz": rate, "acquired_samples": acquired,
            "file_samples": int(raw.n_times), "edf_padding_samples": int(raw.n_times) - acquired,
            "duration_s": acquired / rate, "reference": validated.bids.source_reference,
            "segment_count": len(segments), "channels": [],
        }
        for index, name in enumerate(raw.ch_names):
            signal = values[index]
            longest = 1
            for a, b in segments:
                run_edges = np.r_[0, np.flatnonzero(np.diff(signal[a:b]) != 0) + 1, b - a]
                longest = max(longest, int(np.diff(run_edges).max()))
            summary["channels"].append({
                "name": name, "type": validated.bids.channel_types[index],
                "status": validated.bids.channel_statuses[index],
                "ptp_uv": float(np.ptp(signal) * 1e6), "std_uv": float(signal.std() * 1e6),
                "median_uv": float(np.median(signal) * 1e6),
                "longest_exact_constant_s": longest / rate,
                "entire_recording_constant": bool(np.ptp(signal) == 0),
            })
        figure, axes = plt.subplots(len(values), 1, figsize=(12, 1.05 * len(values) + 0.6), sharex=True,
                                    squeeze=False, layout="constrained")
        for index, axis in enumerate(axes[:, 0]):
            for a, b in segments:
                # Min/max envelopes retain brief transients when compressing a long recording.
                width = max(1, int(np.ceil((b - a) / 1800)))
                starts = np.arange(a, b, width)
                lo = np.minimum.reduceat(values[index, a:b], starts - a) * 1e6
                hi = np.maximum.reduceat(values[index, a:b], starts - a) * 1e6
                axis.fill_between(starts / rate, lo, hi, color=PLOT_COLORS[index % len(PLOT_COLORS)], alpha=.4)
                axis.plot(starts / rate, (lo + hi) / 2, color=PLOT_COLORS[index % len(PLOT_COLORS)], linewidth=.6)
            axis.set_ylabel(f"{raw.ch_names[index]}\nuV", fontsize=9)
            axis.grid(alpha=.16)
        axes[-1, 0].set_xlabel("Stored recording time (s)")
        raw_plot = _save_plot(figure, directory, "raw_signal.png")
        figure, axis = plt.subplots(figsize=(12, 3.8), layout="constrained")
        nperseg = max(2, round(2 * rate))
        accumulated, weight = None, 0
        for a, b in segments:
            if b - a < nperseg:
                continue
            frequencies, power = welch(values[:, a:b], fs=rate, nperseg=nperseg, axis=-1)
            windows = 1 + (b - a - nperseg) // (nperseg - nperseg // 2)
            accumulated = power * windows if accumulated is None else accumulated + power * windows
            weight += windows
        if weight:
            power = accumulated / weight * 1e12
            selected = (frequencies > 0) & (frequencies <= min(80, rate / 2))
            if (power[:, selected] > 0).any():
                for index, name in enumerate(raw.ch_names):
                    axis.semilogy(frequencies[selected], np.where(power[index, selected] > 0, power[index, selected], np.nan),
                                  label=name, color=PLOT_COLORS[index % len(PLOT_COLORS)], linewidth=1)
                axis.legend(ncol=4, fontsize=9)
            else:
                axis.text(.5, .5, "No nonzero spectral power", transform=axis.transAxes, ha="center")
        else:
            axis.text(.5, .5, "Segments shorter than the 2 s spectral window", transform=axis.transAxes, ha="center")
        axis.set(xlabel="Frequency (Hz)", ylabel="Raw PSD (uV squared / Hz)")
        axis.grid(alpha=.16)
        psd_plot = _save_plot(figure, directory, "raw_spectrum.png")
        summary["psd_window_s"] = nperseg / rate
        summary["psd_windows"] = weight
        return summary, [
            {"file": raw_plot, "title": "原始信号全程概览", "caption": "逐导显示真实幅度范围。包络保留短时峰值；手动暂停的采集段分别绘制，EDF补齐样本不计入。"},
            {"file": psd_plot, "title": "原始信号频谱", "caption": "Welch功率谱按真实采集段计算，EOG单独列出。频带能量变化是诊断信息，不能单独证明伪迹。"},
        ]
    finally:
        raw.close()


def _block_counts(block_ids, evidence_indices, retained):
    blocks = np.asarray(block_ids).astype(str)
    indices = np.asarray(evidence_indices, dtype=int)
    if blocks.shape != indices.shape:
        raise ValueError("block identities must cover the full requested event ledger")
    available = indices >= 0
    qc_events = np.zeros(len(indices), dtype=bool)
    if retained is not None:
        qc_events[available] = retained[indices[available]]
    return [{"block_id": block, "scheduled_trials": int((blocks == block).sum()),
             "preprocessed_epochs": int(((blocks == block) & available).sum()),
             "qc_retained_epochs": None if retained is None else int(((blocks == block) & qc_events).sum())}
            for block in dict.fromkeys(blocks.tolist())]


def _epoch_diagnostics(dataset, epoch_quality, directory, scheduled_blocks):
    features = compute_epoch_qc_features(dataset.X, channel_mask=dataset.channel_mask,
                                        trial_channel_mask=dataset.trial_channel_mask)
    observed = features.observed_mask
    ptp = np.ptp(dataset.X, axis=2) * 1e6
    constant = observed & (ptp == 0)
    timeline = dataset.event_timeline
    indices = np.asarray(timeline.evidence_indices, dtype=int)
    available = indices >= 0
    states = np.zeros((len(indices), dataset.n_channels), dtype=np.int8)
    states[available] = np.where(observed[indices[available]], 1, 0)
    states[available] = np.where(constant[indices[available]], 2, states[available])
    retained = None
    qc_record = {"status": "not_configured", "fit_performed": False}
    if epoch_quality is not None:
        transformed, rejected = transform_epoch_quality(dataset, epoch_quality)
        retained = ~rejected
        effective = transformed.trial_channel_mask & retained[:, None]
        states[available] = np.where(effective[indices[available]], 1, 2)
        diagnostics = getattr(transformed, "diagnostic_masks", {})
        if diagnostics:
            flagged = np.logical_or.reduce(list(diagnostics.values()))
            states[available] = np.where(flagged[indices[available]] & effective[indices[available]], 3, states[available])
        qc_record = {"status": "applied", "fit_performed": False, "method": epoch_quality.record()["method"],
                     "retained_epochs": int(retained.sum()), "rejected_epochs": int(rejected.sum()),
                     "fit_provenance": epoch_quality.fit_provenance, "summary": transformed.summary()}
    counts = {
        "scheduled_trials": len(indices), "preprocessed_epochs": dataset.n_epochs,
        "preprocessing_unavailable": int((~available).sum()),
        "qc_retained_epochs": None if retained is None else int(retained.sum()),
        "constant_channel_epochs": int(constant.sum()),
        "preprocessing_reasons": dict(Counter(np.asarray(timeline.status_details)[~available].astype(str))),
    }
    blocks = _block_counts(scheduled_blocks, indices, retained)
    figure, axis = plt.subplots(figsize=(12, 3.6), layout="constrained")
    values = [ptp[observed[:, index], index] for index in range(dataset.n_channels)]
    axis.boxplot([value if len(value) else [np.nan] for value in values], tick_labels=dataset.channel_names,
                 patch_artist=True, boxprops={"facecolor": "#d7ece6"}, medianprops={"color": "#147d70"})
    axis.set_ylabel("Epoch peak-to-peak (uV)")
    axis.grid(axis="y", alpha=.16)
    amplitude_plot = _save_plot(figure, directory, "epoch_amplitude.png")
    figure, axis = plt.subplots(figsize=(12, 3.7), layout="constrained")
    picture = axis.imshow(states.T, aspect="auto", interpolation="nearest", vmin=-.5, vmax=3.5,
                          cmap=ListedColormap(["#d9e1e4", "#218673", "#c95758", "#e9b153"]))
    axis.set_yticks(np.arange(dataset.n_channels), labels=dataset.channel_names)
    axis.set(xlabel="Scheduled flash index (all requested events)", ylabel="EEG channel")
    block_ids = np.asarray(scheduled_blocks).astype(str)
    for position in np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1:
        axis.axvline(position - .5, color="white", linewidth=.6, alpha=.8)
    bar = figure.colorbar(picture, ax=axis, ticks=range(4), pad=.02)
    bar.ax.set_yticklabels(["Unavailable", "Observed", "QC / constant", "Diagnostic only"])
    coverage_plot = _save_plot(figure, directory, "epoch_coverage.png")
    channels = [{"name": name, "observed_epochs": int(observed[:, index].sum()),
                 "median_ptp_uv": float(np.median(values[index])) if len(values[index]) else None,
                 "constant_epochs": int(constant[:, index].sum())}
                for index, name in enumerate(dataset.channel_names)]
    return counts, qc_record, channels, blocks, [
        {"file": amplitude_plot, "title": "预处理后逐导幅度分布", "caption": "使用共享epoch QC特征与实际观测mask。幅度大小及离群点仅供诊断，不据此自动删除动态EEG。"},
        {"file": coverage_plot, "title": "全部闪烁的通道覆盖", "caption": "灰色为预处理不可用或未观测；红色为冻结QC拒绝或精确恒值；黄色仅是相对校准的分布诊断。保留所有请求闪烁作为分母。"},
    ]


def _write_html(report, directory):
    def escape(value):
        return html.escape(str(value))
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>BrainSync 数据质量报告</title><style>body{margin:0;background:#f3f6f7;color:#24343c;font:15px/1.7 'Microsoft YaHei',sans-serif}main{max-width:1080px;margin:auto;padding:28px}h1{font-size:26px}h2{font-size:19px;margin-top:30px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:9px;border-bottom:1px solid #d4dfe2}.note{color:#5c6e77}section{border-top:1px solid #d4dfe2;margin-top:24px}p{overflow-wrap:anywhere}</style><main>",
             "<h1>BrainSync 数据质量报告</h1>", f"<p>{escape(report['session_id'])} · {escape(report['headline'])}</p>"]
    for note in report["notes"]:
        parts.append(f"<p class='note'>{escape(note)}</p>")
    for title, values in report["sections"]:
        parts.append(f"<section><h2>{escape(title)}</h2><table>")
        for key, value in values:
            parts.append(f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>")
        parts.append("</table></section>")
    for chart in report["plots"]:
        parts.append(f"<section><h2>{escape(chart['title'])}</h2><p class='note'>{escape(chart['caption'])}</p><img src='{escape(chart['file'])}' alt='{escape(chart['title'])}'></section>")
    parts.append("</main></html>")
    (directory / "report.html").write_text("\n".join(parts), encoding="utf-8")


def create_recording_quality_report(session_dir, output_dir, *, loader=load_brainsync_inference_session,
                                    epoch_quality=None, input_validator=None, progress=None, request=None):
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "report.json").exists():
        raise FileExistsError("quality reports are immutable; choose a new output directory")
    root = Path(session_dir).resolve()
    notify = progress or (lambda _value, _message: None)
    report = {"schema": REPORT_SCHEMA, "session_id": root.name, "session_dir": str(root),
              "created_utc": datetime.now(UTC).isoformat(), "status": "not_evaluable", "headline": "记录未能完成质量评估",
              "target_labels_used": False, "fit_performed": False, "classifier_evaluated": False,
              "request": request or {}, "notes": [], "sections": [], "plots": [],
              "software": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mne", "matplotlib")},
              "generator_sha256": file_sha256(Path(__file__))}
    try:
        notify(5, "检查录制文件与采集合同")
        validated = validate_brainsync_bids_recording(root)
        device = validated.manifest.get("device")
        device = device if isinstance(device, dict) else {}
        report.update(session_id=validated.session_id, subject_id=validated.subject_id,
                      simulation_status="declared_mock" if device.get("transport") == "mock" else "not_declared")
        report["input_sha256"] = {"raw": file_sha256(validated.bids.raw_path), "events": file_sha256(validated.bids.events_path)}
        notify(20, "生成原始信号和频谱图")
        raw_summary, plots = _raw_diagnostics(validated, directory)
        report.update(raw=raw_summary, plots=plots)
        report["sections"].append(("采集概况", [
            ("受试者", validated.subject_id), ("原始参考", raw_summary["reference"]),
            ("采样率", f"{raw_summary['sample_rate_hz']:g} Hz"), ("采集数据时长", f"{raw_summary['duration_s']:.2f} s"),
            ("已采集 / 文件样本", f"{raw_summary['acquired_samples']} / {raw_summary['file_samples']}"),
            ("采集段数", raw_summary["segment_count"]),
        ]))
        constant_channels = [item["name"] for item in raw_summary["channels"] if item["entire_recording_constant"]]
        if constant_channels:
            report["notes"].append("原始信号全程恒值通道：" + ", ".join(constant_channels))
        notify(45, "运行共享预处理与冻结质量检查")
        dataset = loader(root)
        if dataset.y is not None:
            raise ValueError("quality reports require the target-free recording loader")
        if input_validator is not None:
            input_validator(dataset)
        fitted = epoch_quality() if callable(epoch_quality) else epoch_quality
        counts, qc_record, channels, blocks, plots = _epoch_diagnostics(
            dataset, fitted, directory, [marker.block_id for marker in validated.bids.stimuli])
        report.update(counts=counts, epoch_quality=qc_record, epoch_channels=channels, blocks=blocks,
                      preprocessing=dataset.preprocessing.record(),
                      continuous_quality=dataset.provenance.get("continuous_quality_ledger"),
                      continuous_processing=dataset.provenance.get("continuous_artifact_processing", {}),
                      ocular_processing=dataset.provenance.get("ocular_artifact_processing", {}))
        report["plots"].extend(plots)
        diagnostic_counts = qc_record.get("summary", {}).get("diagnostic_channel_epochs", {})
        report["status"] = "attention" if constant_channels or counts["preprocessing_unavailable"] or counts["constant_channel_epochs"] or qc_record.get("rejected_epochs") or any(diagnostic_counts.values()) else "completed"
        report["headline"] = "预处理完成，存在待复核项" if report["status"] == "attention" else "预处理与基础检查完成"
        report["notes"].insert(0, f"本次采集数据 {raw_summary['duration_s']:.2f} 秒；{counts['scheduled_trials']} 次闪烁中，"
                                 f"{counts['preprocessed_epochs']} 个epoch完成预处理，{counts['preprocessing_unavailable']} 次因输入支持或质量约束不可用。")
        spec = dataset.preprocessing
        report["sections"].append(("预处理与数据使用量", [
            ("处理后参考", dataset.provenance["source_reference"]), ("处理后采样率", f"{spec.sfreq:g} Hz"),
            ("滤波", f"{spec.l_freq:g}-{spec.h_freq:g} Hz · " + {"forward": "因果单向", "zero": "零相位"}.get(spec.filter_phase, spec.filter_phase)),
            ("工频陷波", (", ".join(f"{f:g} Hz" for f in spec.notch_freqs_hz) + f" · Q={spec.notch_quality_factor:g}")
             if spec.notch_freqs_hz else "未启用（按本次输入合同）"),
            ("Epoch时间窗", f"[{spec.tmin_ms:g}, {spec.tmax_ms:g}) ms · {spec.n_times} 点"),
            ("基线", {"mean_only": "刺激前均值校正", "none": "未校正"}.get(spec.baseline_mode, spec.baseline_mode)), ("原排程闪烁", counts["scheduled_trials"]),
            ("预处理保留epoch", counts["preprocessed_epochs"]), ("预处理不可用", counts["preprocessing_unavailable"]),
            ("精确恒值导-epoch", counts["constant_channel_epochs"]),
            ("连续QC", "已应用冻结模型" if report["continuous_processing"].get("status") == "applied" else "未配置"),
            ("眼电处理", "已应用冻结模型" if report["ocular_processing"].get("status") == "applied" else "未应用回归"),
            ("冻结epoch QC", qc_record.get("method", "未配置")),
            ("冻结QC保留epoch", counts["qc_retained_epochs"] if counts["qc_retained_epochs"] is not None else "未评估"),
        ]))
        if counts["preprocessing_reasons"]:
            reason_labels = {"BAD_brainsync_rest": "与休息或排除区间重叠",
                             "continuous_quality_insufficient_observed_channels": "连续QC后有效EEG通道不足",
                             "BAD boundary": "跨越采集断点", "TOO_SHORT": "响应时间窗不完整", "NO_DATA": "缺少对应样本"}
            report["sections"].append(("预处理不可用原因", [(reason_labels.get(reason, reason), count)
                                                             for reason, count in counts["preprocessing_reasons"].items()] ))
        if dataset.provenance.get("continuous_artifact_processing", {}).get("status") != "applied":
            report["notes"].append("未配置冻结连续QC模型，未完成瞬态及滤波恢复区间判定；滤波后平滑不代表信号合格。")
        if fitted is None:
            report["notes"].append("未配置冻结epoch QC模型；本报告仅计算共享质量特征和恒值诊断，不根据本次数据拟合或调整门限。")
        elif qc_record["method"] == "fold_local_ptp_cv":
            report["notes"].append("本次按历史相对幅度QC模型重放；其分布超界不等于伪迹已得到独立证实。")
        if report["input_sha256"] != {"raw": file_sha256(validated.bids.raw_path), "events": file_sha256(validated.bids.events_path)}:
            raise ValueError("录制输入在报告生成期间发生变化，请重新生成报告")
    except BrainSyncNoUsableEvidenceError as error:
        report["continuous_quality"] = error.ledger
        report["counts"] = {"scheduled_trials": error.scheduled_trials, "preprocessed_epochs": 0,
                            "preprocessing_unavailable": error.scheduled_trials, "qc_retained_epochs": 0}
        report["headline"] = "连续QC后没有可用epoch"
        report["sections"].append(("可用性结论", [("原排程闪烁", error.scheduled_trials),
                                                   ("预处理保留epoch", 0), ("连续QC不可用", error.scheduled_trials)]))
        report["notes"].append("所有请求闪烁均保留在不可用分母中，本记录不能形成有效模型证据。")
    except Exception as error:
        report.update(status="not_evaluable", headline="记录未能完成质量评估")
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["notes"].append("未完成的处理阶段：" + str(error))
    report["notes"].append("报告不读取目标标签、不训练分类器；幅度分布和信号覆盖不是九选一准确率或成人效能证明。")
    if report.get("simulation_status") == "declared_mock":
        report["notes"].insert(0, "模拟采集数据，仅用于软件流程验证。")
    notify(90, "保存文字结论与图表")
    _write_html(report, directory)
    atomic_write_json(directory / "report.json", report)
    notify(100, "数据质量报告已生成")
    return report
