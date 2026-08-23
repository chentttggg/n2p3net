"""通用实验启动器：自动起 dashboard 服务器 → 跑实验 → 结束自动关服务器（数据保留）。

用法（任何实验命令前面加 run_with_dashboard.py 即可）：
    .venv/Scripts/python.exe experiments/run_with_dashboard.py -- \
        .venv/Scripts/python.exe experiments/run_bnci008_loso.py --models n2p3net8 --epochs 30

行为：
1. 找空闲端口（8812 起）启动 http.server（--directory experiments）
2. 后台线程每 5 秒把「最近更新的 runs/*/progress.jsonl」写入 experiments/active_run.txt
   —— dashboard.html 无 URL 参数时自动加载该 run（等效于自动改好地址参数）
3. 实验结束后（无论成败）自动关闭服务器；dashboard.html 与 progress.jsonl/record.json
   全部保留，随时可用任意静态服务器重开：
   .venv/Scripts/python.exe -m http.server 8812 --directory experiments
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPS = ROOT / "experiments"


def _free_port(start: int = 8812) -> int:
    for p in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


def _start_active_run_monitor(stop_event: threading.Event) -> threading.Thread:
    """每 5 秒把最近更新的 run 名写入 active_run.txt（dashboard 免参数定位）。"""

    def _monitor() -> None:
        while not stop_event.is_set():
            try:
                progs = sorted(
                    EXPS.glob("runs/*/progress.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if progs:
                    (EXPS / "active_run.txt").write_text(
                        progs[0].parent.name, encoding="utf-8"
                    )
            except OSError:
                pass
            stop_event.wait(5.0)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="实验命令（建议用 -- 分隔）")
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        ap.error("需要实验命令，例如：run_with_dashboard.py -- python experiments/xxx.py ...")

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", str(EXPS)]
    )
    stop_event = threading.Event()
    _start_active_run_monitor(stop_event)
    time.sleep(1.0)

    print(f"[dashboard] http://localhost:{port}/dashboard.html （自动跟踪最新 run）", flush=True)
    rc = 1
    try:
        rc = subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("[dashboard] 收到中断，实验与服务器将依次关闭", flush=True)
        rc = 130
    finally:
        stop_event.set()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        active = (EXPS / "active_run.txt").read_text(encoding="utf-8").strip()
        print(
            f"[dashboard] 服务器已关闭。数据已保留：experiments/runs/{active}/"
            f"（progress.jsonl + record.json，含逐 fold 训练/验证 LOSS）；"
            f"重开仪表盘：python -m http.server 8812 --directory experiments",
            flush=True,
        )
    sys.exit(rc)


if __name__ == "__main__":
    main()
