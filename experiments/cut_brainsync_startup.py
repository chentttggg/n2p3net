"""Publish exact unprocessed BrainSync suffixes and content-bound startup cuts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from data.bids_eeg import validate_bids_eeg_recording  # noqa: E402
from data.brainsync_contract import BRAIN_SYNC_BIDS_CONTRACT  # noqa: E402
from data.brainsync_startup import POLICY, plan_startup_cut, write_cut_edf  # noqa: E402
from data.recording_review import recording_reviews  # noqa: E402
from data.source_records import file_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def publish(quality_manifest, output, registry):
    quality_manifest, output, registry = map(lambda p: Path(p).resolve(), (quality_manifest, output, registry))
    quality = json.loads(quality_manifest.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "brainsync_unprocessed_startup_cut_batch/1", "policy": POLICY,
        "status": "preparing", "authority": "2026-11-02 owner confirmed equipment fault and requested four seconds from first flash, rounded to complete trials",
        "quality_manifest": str(quality_manifest), "quality_manifest_sha256": file_sha256(quality_manifest),
        "filters_applied": [], "new_scientific_fits": 0, "recordings": [], "unavailable": []}
    original_registry = json.loads(registry.read_text(encoding="utf-8"))
    entries = {r["raw_sha256"]: r for r in original_registry["recordings"]}
    reviews = recording_reviews()
    for row in quality["sessions"]:
        if row["status"] != "completed":
            result["unavailable"].append({"session_id": row["directory_id"], "status": row["status"],
                "reason": "no admitted completed raw recording; original failed data preserved"})
            continue
        root = Path(row["input_root"])
        session = json.loads((root / "session.json").read_text(encoding="utf-8"))
        with validate_bids_eeg_recording(root, session["recording"]["bids"],
                                        contract=BRAIN_SYNC_BIDS_CONTRACT) as bids:
            binding = bids.content_record()
            sha = file_sha256(bids.raw_path)
            samples = np.array([e.sample for e in bids.stimuli])
            plan = plan_startup_cut(samples, bids.sample_rate_hz,
                                     round(bids.duration_seconds * bids.sample_rate_hz))
            entry = {**plan, "raw_sha256": sha, "session_id": row["directory_id"],
                "authority": result["authority"]}
            if sha in entries and entries[sha] != entry:
                raise ValueError("an existing startup cut authority differs from the requested cut")
            entries[sha] = entry
            folder = output / row["directory_id"]
            folder.mkdir()
            destination = folder / "raw_cut.edf"
            receipt = write_cut_edf(bids.raw_path, destination, plan["stop_sample"])
            first = plan["first_retained_event_index"]
            with (folder / "events.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["onset", "sample", "candidate_id", "trial_id", "block_id", "trial_index",
                                 "original_sample", "original_event_index"])
                for index, event in enumerate(bids.stimuli):
                    if index < first:
                        continue
                    sample = event.sample - plan["stop_sample"]
                    writer.writerow([sample / bids.sample_rate_hz, sample, event.candidate_id, event.trial_id,
                                     event.block_id, event.trial_index, event.sample, index])
            record = {**entry, **receipt, "original_root": str(root), "original_raw": str(bids.raw_path),
                "original_BIDS_content": binding, "raw_cut": str(destination),
                "raw_cut_sha256": file_sha256(destination), "events": str(folder / "events.tsv"),
                "events_sha256": file_sha256(folder / "events.tsv"),
                "original_scheduled_events": len(bids.stimuli), "remaining_scheduled_events": len(bids.stimuli)-first,
                "channel_names": list(bids.channel_names), "channel_types": list(bids.channel_types),
                "source_reference": bids.source_reference, "EOG_derivation": dict(bids.eog_derivation),
                "recording_review": reviews.get(sha),
                "original_clock": session["recording"].get("clock"),
                "acquisition_boundaries_original_samples": [round(t*bids.sample_rate_hz) for t in bids.acquisition_boundaries],
                "acquisition_boundaries_cut_samples": [round(t*bids.sample_rate_hz)-plan["stop_sample"]
                    for t in bids.acquisition_boundaries if round(t*bids.sample_rate_hz)>plan["stop_sample"]],
                "time_authority": "original sample axis plus declared integer offset; EDF clock header is not the stimulus timing authority",
                "consumer": "model input uses original BIDS schedule plus the same content-bound native support cut; source.json and events.tsv map the exported suffix"}
            bids.assert_content_unchanged()
            if file_sha256(bids.raw_path) != sha:
                raise ValueError("original EEG changed during startup-cut publication")
            atomic_write_json(folder / "source.json", record)
            result["recordings"].append(record)
            atomic_write_json(output / "manifest.json", result)
            print(row["directory_id"], "dropped_trials", first, "cut_sample", plan["stop_sample"], flush=True)
    if not result["recordings"]:
        raise ValueError("no complete raw recordings were cut")
    result.update(status="completed", completed_utc=datetime.now(UTC).isoformat(),
        original_sources_unchanged=True, registry=str(registry),
        implementation_sha256={p: file_sha256(ROOT / p) for p in
            ("src/data/brainsync_startup.py", "experiments/cut_brainsync_startup.py")})
    atomic_write_json(registry, {"schema": "brainsync_startup_cuts/1", "recordings": list(entries.values())})
    result["registry_sha256"] = file_sha256(registry)
    atomic_write_json(output / "manifest.json", result)
    rows = "".join("<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in
        (r["session_id"], len(r["dropped_event_indices"]), f'{r["stop_sample"]/r["sample_rate_hz"]:.3f}',
         r["remaining_scheduled_events"], "整份仍排除" if r["recording_review"] else "已裁剪"))+"</tr>"
        for r in result["recordings"])
    (output / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>BrainSync原始数据开头裁剪</title><style>body{font:16px system-ui;margin:40px;max-width:1100px}'
        'td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}</style>'
        '<h1>BrainSync原始数据开头裁剪</h1><p>从首个刺激起排除4秒，向上取整至首个保留trial的完整基线起点。'
        '裁剪EDF逐导数字采样与原始后缀完全相等，未滤波、未重采样。原始文件与排程身份保留。</p>'
        '<table><tr><th>录制</th><th>删除trial</th><th>原始轴裁剪至 / 秒</th><th>剩余闪烁</th><th>状态</th></tr>'
        +rows+'</table><p>各录制目录含 raw_cut.edf、events.tsv、source.json；manifest.json 保存批次来源和核验。</p></html>',
        encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args(argv)
    publish(args.quality_manifest, args.output, args.registry)


if __name__ == "__main__":
    main()
