"""Measure observed EEG amplitudes after native startup cuts; no model scoring."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data.brainsync import load_brainsync_session  # noqa: E402
from data.domain import adapt_observed_montage  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402
from data.source_records import SIX, file_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402

THRESHOLDS = (200, 500, 1000)


def measurements(data):
    if data.preprocessing.signal_unit != "V":
        raise ValueError("amplitude analysis requires physical volt inputs")
    mask = (np.broadcast_to(data.channel_mask, data.X.shape[:2]) if data.trial_channel_mask is None
            else np.asarray(data.trial_channel_mask))
    if not np.isfinite(data.X[mask]).all() or not np.all(mask.any(axis=1)):
        raise ValueError("retained epochs require finite, actually observed EEG")
    ptp = np.ptp(data.X, axis=2).astype(float) * 1e6
    absolute = np.max(np.abs(data.X), axis=2).astype(float) * 1e6
    row_max = np.max(np.where(mask, ptp, -np.inf), axis=1)
    available = np.flatnonzero(data.event_timeline.evidence_indices >= 0)
    available = available[np.argsort(data.event_timeline.evidence_indices[available])]
    onset = data.event_timeline.onset_times_s[available]
    rows = []
    for index in range(data.n_epochs):
        channel = int(np.argmax(np.where(mask[index], ptp[index], -np.inf)))
        meta = data.metadata.iloc[index]
        rows.append({"event_id": str(data.event_timeline.event_ids[available[index]]),
            "scheduled_index": int(available[index]), "onset_s": float(onset[index]),
            "trial_id": str(meta["trial_id"]), "block_id": int(meta["block_id"]),
            "channel": data.channel_names[channel], "max_PTP_uv": float(row_max[index]),
            "max_absolute_uv": float(np.max(absolute[index, mask[index]])), "row": index})
    summary = {"requested_events": data.event_timeline.n_events, "retained_epochs": data.n_epochs,
        "observed_channel_fraction": float(mask.mean()),
        "median_epoch_max_PTP_uv": float(np.median(row_max)),
        "p95_epoch_max_PTP_uv": float(np.quantile(row_max, .95)),
        "maximum_PTP_uv": float(row_max.max()),
        "maximum_absolute_uv": float(absolute[mask].max()),
        "above_PTP_uv": {str(t): int(np.count_nonzero(row_max > t)) for t in THRESHOLDS},
        "by_channel": {name: {"observed_epochs": int(mask[:, c].sum()),
            "maximum_PTP_uv": float(ptp[mask[:, c], c].max()) if mask[:, c].any() else None,
            "above_PTP_uv": {str(t): int(np.count_nonzero(mask[:, c] & (ptp[:, c] > t))) for t in THRESHOLDS}}
            for c, name in enumerate(data.channel_names)},
        "top_epochs": sorted(rows, key=lambda r: r["max_PTP_uv"], reverse=True)[:12]}
    return summary, rows, mask, ptp, row_max, onset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut-manifest", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eog-audit", type=Path)
    args = parser.parse_args(argv)
    batch = json.loads(args.cut_manifest.read_text(encoding="utf-8"))
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    eog_audit = json.loads(args.eog_audit.read_text(encoding="utf-8")) if args.eog_audit else None
    if prepared["status"] != "prepared":
        raise ValueError("all eight fresh preprocessing cells must finish before amplitude comparison")
    args.output.mkdir(parents=True, exist_ok=False)
    cells = prepared["cells"]
    if eog_audit:
        expected = {r["session_id"]:r for r in cells["QC_K1"]["rows"] if r["role"]=="fixed_future_test"}
        records = eog_audit.get("records", [])
        if (eog_audit.get("status") != "passed" or len(records) != len(expected)
                or {r["session_id"] for r in records} != set(expected)
                or any(not r.get("replay_bitwise_equal") or not r.get("same_QC_mask_and_filter_segments")
                    or r.get("input_cache_sha256") != expected[r["session_id"]]["sha256"]
                    or r["epochs"] != expected[r["session_id"]]["n_epochs"] for r in records)):
            raise ValueError("EOG isolation audit must bind the same replayed fixed-test inputs and support")
    parents = {arm: {r["session_id"]: r for r in cells[arm+"_K10"]["rows"]} for arm in ("basic", "QC")}
    loaded, measured = {}, {}
    result = {"schema": "brainsync_post_startup_cut_amplitude/1", "status": "analyzing",
        "analysis_source_sha256": file_sha256(__file__),
        "EOG_subtraction_isolation": eog_audit,
        "units": "microvolt", "statistic": "within-epoch channel peak-to-peak after mean baseline; observed slots only",
        "thresholds_diagnostic_only_uv": list(THRESHOLDS), "labels_used_for_quality": False,
        "new_classifier_fits": 0, "cut_manifest_sha256": file_sha256(args.cut_manifest),
        "prepared_manifest_sha256": file_sha256(args.prepared), "sessions": {}, "fixed_test_by_dose": {}}
    csv_rows, plot_rows = [], []
    for source in batch["recordings"]:
        sid = source["session_id"]
        if source["recording_review"]:
            result["sessions"][sid] = {"status": "whole_record_excluded"}
            continue
        entry = {"status": "analyzed", "raw_startup_cut": source["stop_sample"], "methods": {}}
        for arm in ("basic", "QC"):
            parent = parents[arm].get(sid)
            if parent is None:
                if arm == "QC":
                    entry["QC_unavailable_reason"] = "outside current P001 C10/fixed-test matrix; original P003 identity retained"
                    continue
                data = adapt_observed_montage(load_brainsync_session(source["original_root"]), SIX, reference_channel="Cz")
            else:
                if file_sha256(parent["path"]) != parent["sha256"]:
                    raise ValueError("fresh preprocessing cache identity changed")
                data = load_epoch_dataset(parent["path"], validation="attested")
            assert data.provenance["raw_startup_cut"]["stop_sample"] == source["stop_sample"]
            values = measurements(data)
            loaded[sid, arm], measured[sid, arm] = data, values
            summary, rows, *_ = values
            entry["methods"][arm] = summary
            if parent is not None:
                entry["methods"][arm]["cache_sha256"] = parent["sha256"]
            for row in rows:
                csv_rows.append({"session": sid, "method": arm, **row})
        if "QC" in entry["methods"]:
            ma, mb = measured[sid, "basic"], measured[sid, "QC"]
            lookup = {r["event_id"]: i for i, r in enumerate(ma[1])}
            aligned = [(lookup[r["event_id"]], j) for j, r in enumerate(mb[1]) if r["event_id"] in lookup]
            left, right = np.asarray(aligned).T
            mask = ma[2][left] & mb[2][right]
            entry["common_observed_event_channels"] = {"count": int(mask.sum()),
                "basic_maximum_PTP_uv": float(ma[3][left][mask].max()),
                "QC_maximum_PTP_uv": float(mb[3][right][mask].max()),
                "basic_above_1000": int(np.count_nonzero(ma[3][left][mask] > 1000)),
                "QC_above_1000": int(np.count_nonzero(mb[3][right][mask] > 1000))}
        result["sessions"][sid] = entry
        plot_rows.append(sid)
        print("ANALYZED", sid, {k: (v["retained_epochs"], v["maximum_PTP_uv"], v["above_PTP_uv"])
            for k,v in entry["methods"].items()}, flush=True)
    for dose in (1, 3, 5, 10):
        for arm in ("basic", "QC"):
            rows = [r for r in cells[f"{arm}_K{dose}"]["rows"] if r["role"] == "fixed_future_test"]
            summaries = []
            for row in rows:
                if file_sha256(row["path"]) != row["sha256"]:
                    raise ValueError("fixed-test cache differs from its selected prepared row")
                summaries.append(measurements(load_epoch_dataset(row["path"], validation="attested"))[0])
            result["fixed_test_by_dose"][f"{arm}_K{dose}"] = {
                "retained_epochs": sum(r["retained_epochs"] for r in summaries),
                "requested_events": sum(r["requested_events"] for r in summaries),
                "maximum_PTP_uv": max(r["maximum_PTP_uv"] for r in summaries),
                "above_PTP_uv": {str(t):sum(r["above_PTP_uv"][str(t)] for r in summaries) for t in THRESHOLDS}}
    with (args.output / "epoch_amplitudes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots((len(plot_rows)+1)//2, 2, figsize=(14, 20), layout="constrained", sharey=True)
        for ax, sid in zip(axes.flat, plot_rows, strict=False):
            for arm, color, marker in (("basic", "#0072B2", "o"), ("QC", "#D55E00", "x")):
                if (sid, arm) not in measured:
                    continue
                vals = measured[sid, arm]
                assert np.all(vals[4] > 0), "log scale requires positive observed PTP"
                ax.scatter(vals[5], vals[4], s=7, color=color, marker=marker, label=arm, alpha=.7)
            for threshold, style in ((200, ":"), (500, "--"), (1000, "-.") ):
                ax.axhline(threshold, color="#666666", lw=.6, ls=style)
            ax.set(title=sid.replace("P001_", ""), yscale="log", ylabel="Epoch max channel PTP (uV), log10 axis",
                   xlabel="Original stored-sample time (s)")
            ax.legend(loc="upper right", fontsize=8)
        for ax in list(axes.flat)[len(plot_rows):]:
            ax.set_visible(False)
        fig.savefig(args.output / "amplitude_timeline.png", dpi=150)
        plt.close(fig)
    worst = sorted(((v[0]["maximum_PTP_uv"], sid) for (sid,arm),v in measured.items() if arm=="QC"), reverse=True)[:3]
    result["waveform_examples"] = []
    for rank, (_,sid) in enumerate(worst, 1):
        qc = loaded[sid,"QC"]
        top = measured[sid,"QC"][0]["top_epochs"][0]
        qrow, channel = top["row"], qc.channel_names.index(top["channel"])
        basic = loaded[sid,"basic"]
        brow = next(i for i,r in enumerate(measured[sid,"basic"][1]) if r["event_id"]==top["event_id"])
        fig, ax = plt.subplots(figsize=(10,3), layout="constrained")
        for data,index,arm,color in ((basic,brow,"basic","#0072B2"),(qc,qrow,"QC","#D55E00")):
            time = data.preprocessing.tmin_ms + np.arange(data.X.shape[2])*1000/data.preprocessing.sfreq
            ax.plot(time, data.X[index,channel]*1e6, color=color, label=arm, lw=1.)
        ax.set(xlabel="Time relative to flash (ms)", ylabel="Voltage (uV)",
               title=f'{sid}: {top["channel"]}, onset={top["onset_s"]:.3f}s, QC PTP={top["max_PTP_uv"]:.1f}uV')
        ax.legend()
        ax.axvline(0,color="gray",ls=":",lw=.5)
        image = f"worst_QC_{rank}.png"
        fig.savefig(args.output/image,dpi=150)
        plt.close(fig)
        result["waveform_examples"].append({"session":sid,**top,"image":image})
    result["status"] = "completed"
    atomic_write_json(args.output / "results.json", result)
    table = []
    for sid in plot_rows:
        methods = result["sessions"][sid]["methods"]
        vals = [sid]
        for arm in ("basic","QC"):
            m = methods.get(arm)
            vals.extend([f'{m["retained_epochs"]}/{m["requested_events"]}',f'{m["maximum_PTP_uv"]:.1f}',
                         str(m["above_PTP_uv"]["500"]),str(m["above_PTP_uv"]["1000"])] if m else ["未应用"]*4)
        table.append("<tr>"+"".join(f"<td>{html.escape(v)}</td>" for v in vals)+"</tr>")
    images = ''.join(f'<img src="{r["image"]}" alt="{html.escape(r["session"])}残留大幅波形">' for r in result["waveform_examples"])
    dose_rows = []
    for dose in (1,3,5,10):
        for arm in ("basic","QC"):
            m = result["fixed_test_by_dose"][f"{arm}_K{dose}"]
            values = [f"{arm} C{dose}",f'{m["retained_epochs"]}/{m["requested_events"]}',
                      f'{m["maximum_PTP_uv"]:.1f}',str(m["above_PTP_uv"]["500"]),str(m["above_PTP_uv"]["1000"])]
            dose_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in values)+'</tr>')
    dose_table = '<h2>同两份固定测试：校准量对幅值的影响</h2><table><tr><th>预处理</th><th>保留/请求</th><th>最大PTP μV</th><th>&gt;500 μV</th><th>&gt;1000 μV</th></tr>'+''.join(dose_rows)+'</table>'
    audit_html = ''
    if eog_audit:
        actual_high = sum(r['full_QC_above_1000'] for r in eog_audit['records'])
        control_high = sum(r['without_EOG_subtraction_above_1000'] for r in eog_audit['records'])
        assert actual_high == result['fixed_test_by_dose']['QC_K1']['above_PTP_uv']['1000']
        audit_html = f'<h2>C1冻结EOG回归的同支持对照</h2><p>同一QC mask、分段、滤波和基线，仅撤销冻结EOG扣除后，两份固定测试超过1000 μV的trial由{actual_high}个变为{control_high}个。未重新拟合参数。</p>'
        for r in eog_audit['records']:
            image_path = os.path.relpath(args.eog_audit.parent/r['image'],args.output).replace('\\','/')
            audit_html += f'<p>{r["session_id"]}：保留{r["epochs"]}；完整QC最大{r["full_QC_max_PTP_uv"]:.1f} μV，同支持撤销EOG后{r["without_EOG_subtraction_max_PTP_uv"]:.1f} μV。</p><img src="{html.escape(image_path)}" alt="冻结EOG扣除的同支持对照">'
    (args.output/"index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>裁剪后预处理幅值复核</title>'
        '<style>body{font:16px system-ui;margin:32px}td,th{padding:8px;border-bottom:1px solid #ccc}img{max-width:100%}</style>'
        '<h1>裁剪后预处理幅值复核</h1><p>全部已排除首刺激起4秒并向上取整。比较实际六导Cz参考输入，QC采用新拟合的C10 EOG/QC；'
        '只统计真实保留通道。200/500/1000 μV均为诊断门限，未据此再删数据。PTP是epoch峰峰值，并非ERP效应幅度。</p>'
        '<p>14:45仍整份排除。9月8日10:30原P003记录列出basic，保原身份，不套用P001的QC包。</p>'
        +dose_table+audit_html+'<h2>C10预处理：各录制全段幅值</h2><table><tr><th>录制</th><th>basic保留/请求</th><th>最大PTP μV</th><th>&gt;500 μV</th><th>&gt;1000 μV</th>'
        '<th>QC保留/请求</th><th>最大PTP μV</th><th>&gt;500 μV</th><th>&gt;1000 μV</th></tr>'
        +''.join(table)+'</table><p><a href="epoch_amplitudes.csv">逐trial数据</a> · <a href="results.json">完整统计与共同支持对照</a></p>'
        '<img src="amplitude_timeline.png" alt="各录制全段epoch幅值，纵轴为对数，虚线为200、500、1000微伏">'
        '<h2>QC后幅值最大的三个会话实例</h2>'+images+'</html>',encoding="utf-8")


if __name__ == "__main__":
    main()
