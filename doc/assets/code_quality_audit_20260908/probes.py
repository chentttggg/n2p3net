"""Reproduce audit findings using synthetic data and owned temporary processes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(ROOT / ".mne_home"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.deep import DeepConfig  # noqa: E402
from baselines.n2p3net import N2P3NetBaseline  # noqa: E402
from research.source_selection import fit_source_trajectory  # noqa: E402
from tests.test_source_training_chain import _source_dataset  # noqa: E402


def windows_liveness():
    if os.name != "nt":
        return {"skipped": "Windows-specific behavior"}
    # Windows signal zero is CTRL_C_EVENT. Isolate the entire console, not just
    # the child process, so the probe cannot signal the audit runner's console.
    harness = """
import json, signal, subprocess, sys
from experiments.watch_resources import _pid_alive
signal.signal(signal.SIGINT, lambda *_: None)
victim = "import signal,time; signal.signal(signal.SIGINT, signal.default_int_handler); print('ready',flush=True); time.sleep(30)"
child = subprocess.Popen([sys.executable, '-c', victim], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
try:
    if child.stdout.readline().strip() != 'ready':
        raise RuntimeError('audit child did not become ready')
    before = child.poll()
    print(json.dumps({'phase': 'victim_ready', 'pid': child.pid}), flush=True)
    reported = _pid_alive(child.pid)
    try:
        code = child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        code = None
    print(json.dumps({'alive_before': before is None, 'reported_alive': reported,
                     'exited_after_probe': code is not None, 'exit_code': code}), flush=True)
finally:
    if child.poll() is None:
        child.terminate()
    child.wait(timeout=5)
"""
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", harness], cwd=ROOT, text=True, timeout=20,
            creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode & 0xFFFFFFFF != 0xC000013A or '"victim_ready"' not in error.output:
            raise
        return {"victim_ready_before_probe": True, "isolated_console_interrupted": True,
                "exit_code": error.returncode, "windows_status": "STATUS_CONTROL_C_EXIT",
                "signal_zero_equals_ctrl_c_event": True}
    return json.loads(output.strip().splitlines()[-1])


def source_reload():
    from transfer.checkpoint import (
        checkpoint_input_stats,
        load_n2p3_trunk_checkpoint,
        predict_n2p3_checkpoint,
    )

    dataset = _source_dataset()
    result = {}
    for standardize in (True, False):
        config = DeepConfig(fixed_updates=1, snapshot_updates=(1,), batch_size=4,
                            sampling="uniform_rows", compile_mode=None,
                            val_group_frac=None, standardize_input=standardize)
        model, payloads = fit_source_trajectory(
            dataset, np.arange(4), config=config, source_cache_sha256="a" * 64,
            source_snapshot_sha256="b" * 64, excluded_source_datasets=["target_library"],
            device=torch.device("cpu"),
        )
        trunk, payload = load_n2p3_trunk_checkpoint(
            payloads[1], dataset, target_subject="source::s3")
        before = model.predict_logit(dataset.X[4:])
        after = predict_n2p3_checkpoint(
            trunk, dataset.X[4:], input_stats=checkpoint_input_stats(payload, dataset.n_channels),
            device=torch.device("cpu"),
        )
        result[str(standardize)] = {
            "reload_max_abs_logit_difference": float(np.max(np.abs(before - after))),
            "reload_equal_at_1e_6": bool(np.allclose(before, after, rtol=1e-6, atol=1e-6)),
            "declared_standardize_input": payload["training_contract"]["optimizer"]["config"]["standardize_input"],
        }
    return result


def nonfinite_source_loss():
    from torch.nn import functional as F

    values = np.random.default_rng(4).normal(size=(12, 3, 180)).astype(np.float32)
    config = DeepConfig(fixed_updates=2, snapshot_updates=(1, 2), batch_size=4,
                        sampling="uniform_rows", compile_mode=None, val_group_frac=None)
    model = N2P3NetBaseline(3, 180, 128, config=config, device=torch.device("cpu"))
    original = F.cross_entropy

    def failed_loss(*args, **kwargs):
        return original(*args, **kwargs) * float("nan")

    with patch.object(F, "cross_entropy", side_effect=failed_loss):
        model.fit(values, np.tile([0, 1], 6))
    return {"fit_returned_as_fitted": model._fitted,
            "published_snapshot_updates": sorted(model.candidate_snapshots_),
            "all_state_finite": all(bool(torch.isfinite(x).all()) for x in model.model_.state_dict().values()),
            "all_losses_finite": all(np.isfinite(x["mean_loss"]) for x in model.last_history["update_losses"])}


def failed_adapter_refit():
    from tests.test_target_adapter_controls import _data, _trunk
    from transfer.recipes import TargetRecipe
    from transfer.subject_adapter import SubjectAdapter

    values, labels = _data()
    adapter = SubjectAdapter(_trunk(), config=TargetRecipe("audit", epochs=1, batch_size=4).adapter_config())
    adapter.fit(values, labels)
    previous_digest = adapter.last_history["fitted_state_sha256"]
    original = torch.optim.AdamW.step
    steps = 0

    def interrupted_step(optimizer, *args, **kwargs):
        nonlocal steps
        steps += 1
        if steps == 2:
            raise RuntimeError("audit-owned simulated interruption after one update")
        return original(optimizer, *args, **kwargs)

    failure = None
    with patch.object(torch.optim.AdamW, "step", interrupted_step):
        try:
            adapter.fit(values * 4, labels)
        except RuntimeError as error:
            failure = str(error)
    try:
        prediction_allowed = bool(np.isfinite(adapter.predict_logit(values)).all())
    except RuntimeError:
        prediction_allowed = False
    return {"fit_error": failure, "fitted_flag_after_error": adapter._fitted,
            "prediction_allowed_after_error": prediction_allowed,
            "history_keeps_previous_digest": adapter.last_history["fitted_state_sha256"] == previous_digest,
            "actual_state_matches_history": adapter.fitted_state_digest() == adapter.last_history["fitted_state_sha256"]}


def target_prediction_reuse(directory):
    from tests.test_brainsync import _write_current_checkpoint
    from tests.test_target_experiments import binary_dataset
    from transfer.binary import binary_time_split
    from transfer.target_experiments import build_target_tasks, execute_target_tasks

    dataset = binary_dataset()
    source = _write_current_checkpoint(directory / "source.pt", dataset, source_snapshot_sha256="a" * 64)
    payload = torch.load(source, weights_only=True)
    split = binary_time_split(np.arange(40) * 2., np.arange(40) * 2. + .8,
                              calibration_trials=20, maximum_calibration_trials=20, tmin_s=-.2)
    scopes = {"p1": (np.arange(40), {20: split})}
    context = dict(target_study="synthetic_p300", source_seed=1,
                   checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   target_cache_sha256="b" * 64, evaluation_snapshot={"archive_sha256": "c" * 64},
                   target_protocol={"kind": "single_recording_prefix"})
    original = build_target_tasks(dataset, scopes, stage="compact", batch_size=512)
    original["tasks"] = [task for task in original["tasks"] if task["arm"] == "compact_waveform_lda"]
    first = execute_target_tasks(dataset, payload, original, output=directory / "compact", context=context, device="cpu")
    prediction = directory / "compact/00000.npz"
    with np.load(prediction) as archive:
        altered = {name: archive[name].copy() for name in archive.files}
    altered["scores"] *= -1
    np.savez_compressed(prediction, **altered)
    later = build_target_tasks(dataset, scopes, stage="E1", batch_size=512)
    later["tasks"] = [task for task in later["tasks"] if task["arm"] == "e1_waveform_lda"]
    context["plan"] = {"reuse_target_records": [str(directory / "compact/record.json")]}
    second = execute_target_tasks(dataset, payload, later, output=directory / "e1", context=context, device="cpu")
    return {"original_auc": first["cells"][0]["auc"], "reused_auc": second["cells"][0]["auc"],
            "reuse_status": second["cells"][0]["status"],
            "fit_executed": second["cells"][0].get("prediction_reuse", {}).get("fit_executed"),
            "model_digest_unchanged": first["cells"][0]["fitted_state_sha256"] == second["cells"][0]["fitted_state_sha256"]}


def dashboard_process_matching():
    from experiments import dashboard_server as server

    fake_proc = SimpleNamespace(glob=lambda _pattern: [SimpleNamespace(name="12345")])
    with patch.object(server, "os", SimpleNamespace(name="posix")), patch.object(server, "Path", return_value=fake_proc), patch.object(
        server, "_command_line", return_value=["python", "watch_resources.py", "--run", "/tmp/audit-run"]
    ):
        found = server._matching_pids("watch_resources.py", "--run", {"/tmp/audit-run"})
    return {"matching_process_present": True, "matched_pids": found}


def dashboard_missing_numbers():
    script = """
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const scripts = [...html.matchAll(/<script(?:\\s[^>]*)?>([\\s\\S]*?)<\\/script>/gi)]
  .map(x => x[1]).filter(x => x.trim());
for (const script of scripts) new vm.Script(script);
const fn = html.match(/function finiteNumber\\(value\\)\\s*\\{[\\s\\S]*?\\n\\}/)[0];
console.log(JSON.stringify({inline_scripts_parsed: scripts.length,
  null_value: vm.runInNewContext(fn + ';finiteNumber(null)'),
  empty_value: vm.runInNewContext(fn + ';finiteNumber(\"\")'),
  false_value: vm.runInNewContext(fn + ';finiteNumber(false)')}));
"""
    output = subprocess.check_output(
        ["node", "-e", script, str(ROOT / "experiments/dashboard.html")], text=True,
    )
    return json.loads(output)


def main():
    torch.set_num_threads(1)
    result = {}
    with tempfile.TemporaryDirectory(prefix="n2p3-code-audit-") as directory:
        probes = {"windows_liveness": windows_liveness, "source_reload": source_reload,
                  "nonfinite_source_loss": nonfinite_source_loss, "failed_adapter_refit": failed_adapter_refit,
                  "target_prediction_reuse": lambda: target_prediction_reuse(Path(directory)),
                  "dashboard_process_matching": dashboard_process_matching,
                  "dashboard_missing_numbers": dashboard_missing_numbers}
        for name, probe in probes.items():
            try:
                result[name] = probe()
            except Exception as error:
                result[name] = {"probe_error": f"{type(error).__name__}: {error}"}
            print(name, json.dumps(result[name], allow_nan=False), flush=True)
    paths = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("experiments/*.py"), *ROOT.glob("tests/*.py"),
                    ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "experiments/dashboard.html",
                    ROOT / "open_dashboard.ps1", ROOT / "open_dashboard.cmd"])
    result["audited_files"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result["environment"] = {"python": sys.version, "torch": torch.__version__, "platform": sys.platform,
                             "cuda_available": torch.cuda.is_available()}
    Path(__file__).with_name("results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
