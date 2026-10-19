"""Keep main-only E2 plan semantics when converging its current worker."""
from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.contracts import semantic_sha256
from research.execution import atomic_write_json
from tests import PROJECT_ROOT
from train.runtime import cpu_thread_budget
from transfer.result_reuse import seal_result_record

SOURCE = PROJECT_ROOT / "experiments/run_e2_prior_scale.py"


def _worker_from_source(path, globals_dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_subject")
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=[worker], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_run_subject"]


def _fixture(tmp_path, *, fail_seal=False):
    calls = []
    saved = {}
    result = SimpleNamespace(log_evidence=np.asarray([2.0, 0.0]), probabilities=np.asarray([0.8, 0.2]),
                             winner="a", input_sha256="1" * 64)

    class Model:
        noise = SimpleNamespace(coefficients=np.asarray([0.5]), marginal_variance_per_unit_innovation=4.0,
                                record=lambda: {"declared_noise": True})
        provenance = {"feasible_GLS_iterations": [{"effective_ridge": 9.25, "residual_diagnostics": {"retained": True}}]}

        @staticmethod
        def fit(records, **kwargs):
            calls.append(kwargs)
            return Model()

        def save(self, path):
            path.write_bytes(b"software-operator-fixture")
            saved[path] = self

        @staticmethod
        def load(path):
            return saved[path]

        def score(self, *args, **kwargs):
            return result

    groups = [(SimpleNamespace(record_id=str(i), samples=np.zeros((8, 1)), onsets=np.asarray([1])),
               np.ones((1, 2), dtype=bool), "a") for i in range(6)]
    identity = {"fixture": "0" * 64}
    operator = object()
    namespace = {"Path": Path, "np": np, "json": json, "time": time, "hashlib": hashlib, "gc": gc,
                 "atomic_write_json": atomic_write_json, "semantic_sha256": semantic_sha256,
                 "cpu_thread_budget": cpu_thread_budget, "logsumexp": logsumexp,
                 "seal_result_record": seal_result_record,
                 "BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT": BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT,
                 "preprocessing_spec_from_contract": preprocessing_spec_from_contract,
                 "BNCI2014_008_CHANNELS": BNCI2014_008_CHANNELS, "SFREQ": 1000.,
                 "ASSETS": PROJECT_ROOT / "doc/assets/operator_research_20261004"}
    namespace.update({"_executing_files": lambda: identity,
                      "_selection_views": lambda row, plan: (groups, {}), "OperatorGLS": Model,
                      "CANDIDATES": ("a", "b"), "FiniteFilterOperator": lambda *args: operator,
                      "smooth_erp_basis": lambda *args: np.ones((8, 1)),
                      "design_continuous_sos": lambda *args, **kwargs: np.asarray([[1., 0., 0., 1., 0., 0.]]),
                      "mne": SimpleNamespace(filter=SimpleNamespace(construct_iir_filter=lambda *args, **kwargs: {"padlen": 0}))})
    if fail_seal:
        def rejected_seal(*args, **kwargs):
            raise OSError("artifact unavailable")
        namespace["seal_result_record"] = rejected_seal
    worker = _worker_from_source(Path(os.environ.get("N2P3_E2_WORKER_SOURCE", SOURCE)), namespace)
    plan = {"response_seconds": 1.2, "basis_rank": 1, "cpu_threads_per_worker": 1,
            "evidence_scope": "software_correctness", "ridge_relative_grid": [0.1, 1.0],
            "arms": [{"name": "declared", "order": 1, "kernel_prior_scale": "marginal",
                      "filter": True, "noise_method": "legacy_clipped_ar1", "filter_design": False,
                      "spatial_shrinkage": "standardized_lw"}]}
    return worker, plan, identity, calls, operator


def test_e2_worker_honors_declared_operator_and_records_model_owned_ridge(tmp_path):
    worker, plan, identity, calls, operator = _fixture(tmp_path)
    directory = tmp_path / "worker"
    worker({"subject": "synthetic"}, plan, directory, identity)
    assert len(calls) == 12
    assert all(call["operator"] is operator and call["noise_method"] == "legacy_clipped_ar1"
               and not call["filter_design"] and call["spatial_shrinkage"] == "standardized_lw" for call in calls)
    records = json.loads((directory / "cells.json").read_bytes())
    assert len(records) == 6 and all(record["status"] == "completed" for record in records)
    assert all(record["effective_ridge"] == 9.25 for record in records)
    assert all(point["effective_ridge"] == 9.25 and point["operator_declared"] == "project_bidirectional"
               and point["noise_record"] == {"declared_noise": True}
               for record in records for point in record["ridge_grid_results"])
    prediction = json.loads((directory / "fold0_declared/primary_prediction.json").read_bytes())
    assert prediction["primary_ridge"] == 1.0


def test_e2_worker_preserves_missing_primary_and_seal_failure_ledgers(tmp_path):
    worker, plan, identity, _, _ = _fixture(tmp_path)
    missing = copy.deepcopy(plan)
    missing["ridge_relative_grid"] = [0.1]
    worker({"subject": "synthetic"}, missing, tmp_path / "missing", identity)
    records = json.loads((tmp_path / "missing/cells.json").read_bytes())
    assert len(records) == 6 and all(record["status"] == "failed" and "primary ridge 1.0" in record["error"] for record in records)
    worker, plan, identity, _, _ = _fixture(tmp_path, fail_seal=True)
    worker({"subject": "synthetic"}, plan, tmp_path / "seal", identity)
    records = json.loads((tmp_path / "seal/cells.json").read_bytes())
    assert len(records) == 6 and all(record["status"] == "failed" and "seal:" in record["error"] for record in records)
