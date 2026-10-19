"""通用实验启动器：自动起 dashboard 服务器 → 跑实验 → 结束自动关服务器（数据保留）。

用法（任何实验命令前面加 run_with_dashboard.py 即可）：
    .venv/Scripts/python.exe experiments/run_with_dashboard.py -- \
        .venv/Scripts/python.exe experiments/run_eeg_loso.py \
            --dataset-cache experiments/cache/bnci008_v8.npz --epochs 30

行为：
1. 找空闲端口（8812 起）启动 dashboard server（--directory experiments）
2. 后台线程每 5 秒把「最近更新的 runs/*/progress.jsonl」写入 experiments/active_run.txt
   —— dashboard.html 无 URL 参数时自动加载该 run（等效于自动改好地址参数）
3. 实验结束后（无论成败）自动关闭服务器；dashboard.html 与 progress.jsonl/record.json
   全部保留，随时可用任意静态服务器重开：
   .venv/Scripts/python.exe experiments/dashboard_server.py --port 8812 \
       --directory experiments
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

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
    raise OSError(f"dashboard ports {start}..{start + 19} are all unavailable")


def _server_output(server, ready, port):
    expected = f"[dashboard] http://127.0.0.1:{port}/dashboard.html"
    for line in server.stdout:
        if line.strip() == expected:
            # Publish the URL only after this child's bind and HTTP readiness.
            ready.set()
        else:
            print(line, end="", flush=True)


def _wait_ready(server, ready, port, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"dashboard server exited during startup (exit {server.returncode})")
        if ready.is_set():
            try:
                request = Request(f"http://127.0.0.1:{port}/dashboard.html", method="HEAD")
                with urlopen(request, timeout=.2) as response:
                    if response.status == 200 and server.poll() is None:
                        return
            except (OSError, URLError):
                pass
        time.sleep(.05)
    raise RuntimeError("dashboard server did not become ready before the startup deadline")


def _start_active_run_monitor(stop_event: threading.Event) -> threading.Thread:
    """每 5 秒把最近更新的 run 名写入 active_run.txt（dashboard 免参数定位）。"""

    def _monitor() -> None:
        while not stop_event.is_set():
            try:
                # 含多模型 run 的嵌套子目录（runs/<run>/<spec>/progress.jsonl）
                progs = sorted(
                    list(EXPS.glob("runs/*/progress.jsonl"))
                    + list(EXPS.glob("runs/*/*/progress.jsonl")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if progs:
                    rel = progs[0].parent.relative_to(EXPS / "runs").as_posix()
                    (EXPS / "active_run.txt").write_text(rel, encoding="utf-8")
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

    try:
        port = _free_port()
    except OSError as error:
        ap.exit(1, f"[dashboard] {error}\n")
    server = subprocess.Popen(
        [
            sys.executable,
            str(EXPS / "dashboard_server.py"),
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(EXPS),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stop_event = threading.Event()
    ready = threading.Event()
    output = threading.Thread(target=_server_output, args=(server, ready, port), daemon=True)
    output.start()
    monitor = None
    rc = 1
    try:
        _wait_ready(server, ready, port)
        monitor = _start_active_run_monitor(stop_event)
        print(f"[dashboard] http://localhost:{port}/dashboard.html （自动跟踪最新 run）", flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
    except RuntimeError as error:
        print(f"[dashboard] {error}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("[dashboard] 收到中断，实验与服务器将依次关闭", flush=True)
        rc = 130
    finally:
        stop_event.set()
        if server.poll() is None:
            server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        output.join(timeout=1)
        server.stdout.close()
        if monitor is not None:
            monitor.join(timeout=1)
        active_path = EXPS / "active_run.txt"
        active = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else ""
        location = f"experiments/runs/{active}/" if active else "no run record was created"
        print(
            f"[dashboard] 服务器已关闭。数据已保留：{location}"
            f"（progress.jsonl + record.json，含逐 fold 训练/验证 LOSS）；"
            f"重开仪表盘：python experiments/dashboard_server.py --port 8812 --directory experiments",
            flush=True,
        )
    sys.exit(rc)


if __name__ == "__main__":
    main()
