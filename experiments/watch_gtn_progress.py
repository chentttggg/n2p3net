"""Human-readable live monitor for GTN progress.jsonl runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "experiments" / "runs"


def _latest_progress() -> Path:
    paths = list(RUNS_ROOT.rglob("progress.jsonl")) if RUNS_ROOT.exists() else []
    if not paths:
        raise FileNotFoundError(f"未找到 progress.jsonl：{RUNS_ROOT}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def _resolve_progress(run: str | None) -> Path:
    if not run:
        return _latest_progress()

    candidate = Path(run).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend(
            [
                ROOT / candidate,
                RUNS_ROOT / candidate / "progress.jsonl",
                RUNS_ROOT / run / "progress.jsonl",
            ]
        )
    for path in candidates:
        if path.is_dir():
            path = path / "progress.jsonl"
        if path.is_file() and path.name == "progress.jsonl":
            return path.resolve()
    raise FileNotFoundError(f"找不到指定 run 的 progress.jsonl：{run}")


def _read_progress(path: Path) -> tuple[dict, list[dict], bool]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                # The writer flushes complete lines, but tolerate a line being
                # observed while another process is appending it.
                continue
            if isinstance(row, dict):
                rows.append(row)
    manifest = next((row for row in rows if row.get("type") == "manifest"), {})
    folds = [row for row in rows if row.get("type") == "fold"]
    return manifest, folds, any(row.get("type") == "done" for row in rows)


def _read_latest_epochs(path: Path) -> list[dict]:
    """Read the newest epoch event for each fold without requiring fold completion."""

    epoch_dir = path.parent / "epochs"
    latest: dict[int, dict] = {}
    if not epoch_dir.exists():
        return []
    for epoch_path in epoch_dir.glob("fold_*.jsonl"):
        try:
            with epoch_path.open(encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or row.get("type") != "epoch":
                        continue
                    fold = int(row["fold"])
                    prior = latest.get(fold)
                    if prior is None or int(row.get("epoch", 0)) >= int(prior.get("epoch", 0)):
                        latest[fold] = row
        except (OSError, ValueError, TypeError):
            continue
    return [latest[fold] for fold in sorted(latest)]


def _elapsed_since(iso_value: str | None) -> float | None:
    if not iso_value:
        return None
    try:
        return max(0.0, time.time() - datetime.fromisoformat(iso_value.replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError):
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(round(seconds))
    if seconds >= 86400:
        return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def _format_metric(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _gpu_line() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    query = (
        "utilization.gpu,memory.used,memory.total,power.draw"
        ",temperature.gpu"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        if len(values) < 5:
            return None
        return (
            f"GPU: util {values[0]}% | memory {values[1]}/{values[2]} MiB | "
            f"power {values[3]} W | temp {values[4]} C"
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _render(path: Path, jobs: int, recent_count: int) -> None:
    manifest, folds, finished = _read_progress(path)
    latest_epochs = _read_latest_epochs(path)
    total = int(manifest.get("total_folds", len(folds) or 1))
    done = len(folds)
    started = manifest.get("started_utc")
    elapsed = _elapsed_since(started)
    fit_times = [float(row["fit_sec"]) for row in folds if row.get("fit_sec") is not None]
    recent_fit = fit_times[-recent_count:]
    average_fit = sum(recent_fit) / len(recent_fit) if recent_fit else None
    remaining = max(total - done, 0)
    eta = average_fit * remaining / max(jobs, 1) if average_fit is not None else None

    batch_size = manifest.get("trainer_kwargs", {}).get("batch_size", "?")
    run_name = manifest.get("run_name", path.parent.name)
    last_update = time.time() - path.stat().st_mtime
    status = "已完成" if finished else "运行中"

    print(f"GTN/N2P3Net 进度 | {status}")
    print(f"run: {run_name}")
    print(
        f"fold: {done}/{total} ({done / max(total, 1) * 100:.1f}%) | "
        f"batch: {batch_size} | 并发估计: {jobs}"
    )
    print(
        f"已运行: {_format_duration(elapsed)} | 最近写入: {_format_duration(last_update)} 前 | "
        f"预计剩余: {_format_duration(eta)}"
    )
    gpu = _gpu_line()
    if gpu:
        print(gpu)
    print()

    if latest_epochs:
        print("最近 epoch:")
        for row in latest_epochs[-recent_count:]:
            fold = int(row.get("fold", -1)) + 1
            epoch = row.get("epoch", "-")
            limit = row.get("epoch_limit", "?")
            print(
                f"  fold {fold:3d}/{total} | epoch={epoch}/{limit} | "
                f"val_loss={_format_metric(row.get('task_val_loss'))} | "
                f"val_AUC={_format_metric(row.get('task_val_auc'))}"
            )
        print()

    if not folds:
        print("尚无 fold 完成记录；当前 fold 的 epoch 输出已从 epochs/*.jsonl 读取。")
        return

    print("最近完成的 fold:")
    for row in folds[-recent_count:]:
        fold = int(row.get("fold", -1)) + 1
        subject = str(row.get("subject") or "-")
        epochs = row.get("epochs_ran", "-")
        fit_sec = row.get("fit_sec")
        task_val_aucs = row.get("task_val_aucs") or []
        latest_val_auc = task_val_aucs[-1] if task_val_aucs else None
        print(
            f"  fold {fold:3d}/{total} | subject={subject:<10} | "
            f"epochs={str(epochs):<3} | fit={_format_duration(fit_sec)} | "
            f"val_AUC={_format_metric(latest_val_auc)} | "
            f"bacc={_format_metric(row.get('fold_bacc'))} | "
            f"AUC={_format_metric(row.get('fold_auc'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="以人类可读格式监控 GTN progress.jsonl")
    parser.add_argument("--run", help="run 名称、run 目录或 progress.jsonl 路径；默认选择最近更新的 run")
    parser.add_argument("--interval", type=float, default=5.0, help="刷新间隔秒数，默认 5")
    parser.add_argument("--jobs", type=int, default=8, help="用于 ETA 的 fold 并发数，默认 8")
    parser.add_argument("--recent", type=int, default=8, help="显示最近多少个 fold，默认 8")
    parser.add_argument("--once", action="store_true", help="只打印一次，不持续刷新")
    parser.add_argument("--no-clear", action="store_true", help="刷新时不清屏，适合保存终端日志")
    args = parser.parse_args()
    if args.interval <= 0 or args.jobs <= 0 or args.recent <= 0:
        parser.error("--interval、--jobs、--recent 必须为正数")

    try:
        path = _resolve_progress(args.run)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    while True:
        if not args.no_clear:
            sys.stdout.write("\033[2J\033[H")
        print(f"监控文件: {path}")
        try:
            _render(path, args.jobs, args.recent)
        except (FileNotFoundError, OSError) as exc:
            print(f"读取失败：{exc}")
        sys.stdout.flush()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
