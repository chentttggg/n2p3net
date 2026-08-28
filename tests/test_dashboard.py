from __future__ import annotations

import json

import pytest

from experiments.dashboard_server import _list_runs, _run_dir


def test_dashboard_discovers_current_nested_run_and_done_state(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "dashboard-smoke" / "eegnet"
    run_dir.mkdir(parents=True)
    started = "2026-08-28T01:00:00+00:00"
    finished = "2026-08-28T01:01:00+00:00"
    manifest = {
        "type": "manifest",
        "run_name": "dashboard-smoke",
        "total_folds": 2,
        "started_utc": started,
    }
    (run_dir / "progress.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                manifest,
                {"type": "fold", "fold": 0, "n_folds_done": 1},
                {"type": "done", "ts": finished},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "record.json").write_text(
        json.dumps({"finished_utc": finished}), encoding="utf-8"
    )

    assert _run_dir(tmp_path / "runs", "dashboard-smoke/eegnet") == run_dir.resolve()
    runs = _list_runs(tmp_path / "runs")

    assert len(runs) == 1
    assert runs[0]["name"] == "dashboard-smoke/eegnet"
    assert runs[0]["done"] is True
    assert runs[0]["active"] is False


def test_dashboard_run_resolution_rejects_path_escape(tmp_path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "valid").mkdir(parents=True)

    with pytest.raises(ValueError):
        _run_dir(run_root, "../valid")
