"""Matched-event waveform and availability figures for frozen recovery inputs."""

from __future__ import annotations

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
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.quality import apply_epoch_quality, load_epoch_quality  # noqa: E402


def dense(dataset):
    shape = (dataset.event_timeline.n_events, dataset.n_channels)
    values = np.full((*shape, dataset.n_times), np.nan)
    observed = np.zeros(shape, dtype=bool)
    evidence = np.asarray(dataset.event_timeline.evidence_indices)
    present = evidence >= 0
    mask = np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2]).copy()
    if dataset.trial_channel_mask is not None:
        mask &= dataset.trial_channel_mask
    values[present] = dataset.X[evidence[present]]
    observed[present] = mask[evidence[present]]
    return values, observed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    payload = json.loads(args.quality.read_text(encoding="utf-8"))
    assert payload["digest"] == semantic_sha256(payload["model"])
    quality = load_epoch_quality(args.quality)
    arms = {arm["name"]: arm for arm in plan["pipelines"]}
    sessions, counts, arrays = [], [], {}
    for index, session in enumerate(plan["evaluation_sessions"]):
        paths = [arms[name]["evaluation_caches"][index] for name in ("historical_no_qc", "local_recovery")]
        for path in paths:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == plan["input_sha256"][path]
        historical, continuous = [load_epoch_dataset(path, require_labels=True) for path in paths]
        final = apply_epoch_quality(continuous, quality)
        np.testing.assert_array_equal(historical.event_timeline.event_ids, final.event_timeline.event_ids)
        assert historical.channel_names == final.channel_names
        old, old_mask = dense(historical)
        _, intermediate = dense(continuous)
        new, mask = dense(final)
        common = old_mask & mask
        codes = np.asarray(final.event_timeline.candidate_ids, dtype=int)
        labels = codes == np.asarray(final.event_timeline.target_candidate_ids, dtype=int)
        status = np.zeros(mask.shape, dtype=np.uint8)
        status[old_mask] = 1
        status[intermediate] = 2
        status[mask] = 3
        rows = {"session_id": session, "scheduled_events": len(labels), "retained_epochs": final.n_epochs,
                "retained_channel_epochs": int(mask.sum()), "matched_channel_events": []}
        old_difference, new_difference = [], []
        for channel in range(final.n_channels):
            target = common[:, channel] & labels
            nontarget = common[:, channel] & ~labels
            rows["matched_channel_events"].append({"channel": final.channel_names[channel],
                "targets": int(target.sum()), "nontargets": int(nontarget.sum())})
            old_difference.append(old[target, channel].mean(axis=0) - old[nontarget, channel].mean(axis=0))
            new_difference.append(new[target, channel].mean(axis=0) - new[nontarget, channel].mean(axis=0))
        times = final.preprocessing.tmin_ms + 1000*np.arange(final.n_times)/final.preprocessing.sfreq
        sessions.append((session, final.channel_names, status, np.asarray(old_difference), np.asarray(new_difference)))
        counts.append(rows)
        arrays.update({session + "_status": status, session + "_common_event_mask": common,
                       session + "_historical_erp_difference_v": old_difference,
                       session + "_recovery_erp_difference_v": new_difference,
                       session + "_target_labels": labels,
                       session + "_event_ids": np.asarray(final.event_timeline.event_ids, dtype=str)})
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 5.5), layout="constrained")
        colors = ["#d7d7d7", "#c1443c", "#ddaf30", "#147d92"]
        cmap = ListedColormap(colors)
        for index, (session, names, status, _, _) in enumerate(sessions):
            ax = axes[index]
            ax.imshow(status.T, aspect="auto", interpolation="nearest", cmap=cmap,
                      norm=BoundaryNorm(np.arange(-.5, 4), cmap.N), extent=(.5, len(status)+.5, len(names)-.5, -.5))
            ax.set_yticks(np.arange(len(names)), names)
            ax.set_xticks(np.arange(0, 451, 45))
            ax.set_title(f"{session}: {counts[index]['retained_epochs']}/450 epochs; {counts[index]['retained_channel_epochs']}/2700 channel-epochs", loc="left")
            ax.set_xlabel("Original scheduled event index (block boundaries every 45 events)")
            for boundary in np.arange(45.5, 450, 45):
                ax.axvline(boundary, color="white", linewidth=.6)
        fig.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(colors,
            ("Unavailable in historical input", "Continuous-QC unavailable", "Epoch-QC unavailable", "Retained"), strict=True)],
            loc="outside lower center", ncol=4, frameon=False)
        fig.savefig(args.output / "event_channel_availability.png", dpi=180, facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(6, 2, figsize=(12, 13), layout="constrained")
        for column, (session, names, _, old, new) in enumerate(sessions):
            for channel, name in enumerate(names):
                ax = axes[channel, column]
                ax.plot(times, old[channel]*1e6, color="#a2531b", linestyle="--", linewidth=1.3, label="Historical, same events")
                ax.plot(times, new[channel]*1e6, color="#147d92", linewidth=1.3, label="Recovered, same events")
                ax.axhline(0, color="#777777", linewidth=.5)
                ax.axvline(0, color="#777777", linewidth=.6, linestyle=":")
                numbers = counts[column]["matched_channel_events"][channel]
                ax.set_title(f"{name}: target {numbers['targets']}, non-target {numbers['nontargets']}", loc="left")
                ax.set_xlim(-200, 1200)
                ax.set_ylabel("Target - non-target (uV)")
                if channel == 5:
                    ax.set_xlabel("Time from recorded stimulus (ms)")
                if channel == 0:
                    ax.text(.5, 1.22, session, transform=ax.transAxes, ha="center", fontsize=11)
        fig.suptitle("Descriptive ERP difference on identical retained channel-event pairs; no extra smoothing", fontsize=12)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
        fig.savefig(args.output / "matched_event_erp.png", dpi=160, facecolor="white")
        plt.close(fig)
    np.savez_compressed(args.output / "figure_data.npz", times_ms=times, **arrays)
    atomic_write_json(args.output / "figure_manifest.json", {"plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "quality_model_digest": payload["digest"], "rows": counts,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "estimand": "same retained event set per channel; descriptive target-minus-nontarget mean",
        "smoothing": "none beyond declared preprocessing", "uncertainty": "not estimated; one observed participant",
        "axes": "per-panel data limits; no clipping; numeric y ticks retained",
        "not_evidence_of": ["total physiological cleanliness", "independent adult P300 confirmation"]})
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
