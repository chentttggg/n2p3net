"""Explicit research CLI for segment line regression and continuous decisions.

All paths in a plan are relative to that plan. Outputs are new artifacts, never
in-place edits. This runner does not register a production preprocessing route.
Use --help on each command for the small NPZ/JSON interchange contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from data.harmonic import HarmonicConfig, clean_harmonic_segments  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402
from transfer.continuous_decision import (  # noqa: E402
    CalibrationRecord,
    ContinuousDecisionModel,
    smooth_erp_basis,
)
from transfer.noise_model import JointNoiseModel, metadata_digest  # noqa: E402
from transfer.operator_lda import ConditioningObservationSupport, OperatorAwareLDA  # noqa: E402


@contextmanager
def _input_snapshot(path):
    """Decode and identify one immutable read; large inputs spill to local disk."""
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024**2, mode="w+b") as snapshot:
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024**2), b""):
                digest.update(block)
                snapshot.write(block)
        snapshot.seek(0)
        yield snapshot, digest.hexdigest()


def _json(path):
    with _input_snapshot(path) as (snapshot, digest):
        return json.loads(snapshot.read().decode("utf-8-sig")), digest


@contextmanager
def _arrays(path):
    with _input_snapshot(path) as (snapshot, digest), np.load(snapshot, allow_pickle=False) as archive:
        yield archive, digest


def _model(model_type, path):
    with _input_snapshot(path) as (snapshot, digest):
        return model_type.load(snapshot), digest


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _plan_path(plan_file, value):
    path = Path(value)
    return path if path.is_absolute() else Path(plan_file).parent / path


def _source_provenance(command):
    return {"runner_source_sha256": _sha(__file__),
            "module_source_sha256": {
                name: _sha(ROOT / "src" / name) for name in
                ("data/harmonic.py", "transfer/continuous_decision.py", "transfer/noise_model.py",
                 "transfer/operator_lda.py")},
            "evidence_scope": "software_or_public_development_no_confirmed_performance",
            "command": command}


_IMPORTED_SOURCE = _source_provenance("import")

def _provenance(args):
    return args.execution_provenance


def _clean(args, output):
    plan, plan_sha = _json(args.plan)
    if plan.get("scope_role") not in {"source_training", "target_calibration", "test_offline_adaptive"}:
        raise ValueError("declare source_training, target_calibration or test_offline_adaptive scope_role")
    if not plan.get("scope_id"):
        raise ValueError("cleaning requires an explicit scope_id")
    path = _plan_path(args.plan, plan["input"])
    with _arrays(path) as (archive, input_sha):
        X = archive["X"]
    result = clean_harmonic_segments(X, plan["sfreq"], plan["segments"],
                                     HarmonicConfig(**plan.get("config", {})))
    np.savez_compressed(output / "cleaned.npz", X=result.cleaned)
    _write(output / "result.json", {**_provenance(args), **result.record(),
                                     "scope_id": plan["scope_id"], "scope_role": plan["scope_role"],
                                     "segments": plan["segments"], "sfreq": plan["sfreq"],
                                     "input_file_sha256": input_sha, "plan_sha256": plan_sha})


def _fit(args, output):
    plan, plan_sha = _json(args.plan)
    records, files = [], []
    for item in plan["records"]:
        path = _plan_path(args.plan, item["path"])
        with _arrays(path) as (archive, digest):
            records.append(CalibrationRecord(item["record_id"], archive["samples"],
                                              archive["onsets"], archive["target_mask"],
                                              right_censored=item.get("right_censored", False)))
        files.append({"record_id": item["record_id"], "path": str(path), "sha256": digest})
    basis_sha = None
    if "basis_path" in plan:
        with _input_snapshot(_plan_path(args.plan, plan["basis_path"])) as (snapshot, basis_sha):
            basis = np.load(snapshot, allow_pickle=False)
        if not isinstance(basis, np.ndarray):
            raise ValueError("basis_path must identify one numeric NPY array")
    else:
        basis = smooth_erp_basis(plan["sfreq"], plan.get("response_seconds", 0.8), plan.get("n_basis", 8))
    model = ContinuousDecisionModel.fit(
        records, sfreq=plan["sfreq"], channels=tuple(plan["channels"]),
        candidate_ids=tuple(plan["candidate_ids"]), operator_identity=plan["operator_identity"],
        basis=basis, ridge=plan.get("ridge", 1.), fit_scope=plan["fit_scope"],
    )
    path = output / "model.npz"
    model.save(path)
    reloaded, model_sha = _model(ContinuousDecisionModel, path)
    for field in ("basis", "kernel_mean", "kernel_row_covariance", "innovation_covariance"):
        np.testing.assert_array_equal(getattr(model, field), getattr(reloaded, field))
    _write(output / "fit.json", {**_provenance(args), "model_sha256": model_sha,
                                  "plan_sha256": plan_sha, "inputs": files, "basis_file_sha256": basis_sha,
                                  "fit": model.provenance, "reload_state_equal": True})


def _predict(args, output):
    metadata, metadata_sha = _json(args.metadata)
    model, model_sha = _model(ContinuousDecisionModel, args.model)
    with _arrays(args.input) as (archive, input_sha):
        if set(archive.files) != {"samples", "onsets", "membership"}:
            raise ValueError("prediction NPZ must contain only samples, onsets, membership; truth stays in scorer")
        result = model.score(archive["samples"], archive["onsets"], archive["membership"],
                             operator_identity=metadata["operator_identity"],
                             sfreq=metadata["sfreq"], channels=tuple(metadata["channels"]),
                             candidate_ids=tuple(metadata["candidate_ids"]), record_id=metadata["record_id"])
    _write(output / "prediction.json", {**_provenance(args), **result.record(),
                                         "model_sha256": model_sha, "input_file_sha256": input_sha,
                                         "metadata_sha256": metadata_sha})


def _noise_fit(args, output):
    plan, plan_sha = _json(args.plan)
    path = _plan_path(args.plan, plan["input"])
    with _arrays(path) as (archive, input_sha):
        model = JointNoiseModel.fit(archive["raw_residual_draws"], archive["state_draws"],
                                   fit_scope=plan["fit_scope"], scope_id=plan["scope_id"],
                                   draw_ids=plan["draw_ids"], shrinkage=plan.get("shrinkage"))
    model.save(output / "noise.npz")
    reloaded, model_sha = _model(JointNoiseModel, output / "noise.npz")
    np.testing.assert_array_equal(reloaded.mean, model.mean)
    np.testing.assert_array_equal(reloaded.covariance, model.covariance)
    _write(output / "fit.json", {**_provenance(args), "fit": model.provenance,
                                  "input_file_sha256": input_sha, "plan_sha256": plan_sha,
                                  "model_sha256": model_sha, "reload_state_equal": True})


def _noise_project(args, output):
    model, model_sha = _model(JointNoiseModel, args.model)
    with _arrays(args.input) as (archive, operator_sha):
        result = model.project(archive["A"], archive["B"])
    np.savez_compressed(output / "propagated.npz", mean=result.mean, covariance=result.covariance)
    _write(output / "result.json", {**_provenance(args), "model_sha256": model_sha,
                                     "operator_file_sha256": operator_sha,
                                     "output_dimension": len(result.mean),
                                     "min_eigenvalue": float(np.linalg.eigvalsh(result.covariance).min())})


def _noise_condition(args, output):
    model, model_sha = _model(JointNoiseModel, args.model)
    with _arrays(args.input) as (archive, input_sha):
        if set(archive.files) != {"raw_values", "raw_means", "state_means"}:
            raise ValueError("noise conditioning takes only observed raw and full candidate expectations")
        result = model.condition(archive["raw_values"], raw_means=archive["raw_means"], state_means=archive["state_means"])
    np.savez_compressed(output/"conditioned.npz", mean=result.mean, covariance=result.covariance)
    _write(output/"result.json", {**_provenance(args), "model_sha256": model_sha, "input_file_sha256": input_sha,
                                 "rank_ledger": result.rank_ledger, "extra_independent_evidence": False})


def _operator_lda_fit(args, output):
    plan, plan_sha = _json(args.plan)
    noise_path = _plan_path(args.plan, plan["noise_model"])
    noise, noise_sha = _model(JointNoiseModel, noise_path)
    data_path = _plan_path(args.plan, plan["input"])
    operator_path = _plan_path(args.plan, plan["operators"])
    support = ConditioningObservationSupport.from_record(plan["observation_support"]) if "observation_support" in plan else None
    with _arrays(data_path) as (data, input_sha), _arrays(operator_path) as (operators, operator_sha):
        expected = {"features", "labels"} | ({"conditioning_features"} if support is not None else set())
        if set(data.files) != expected:
            raise ValueError("operator calibration takes only declared features, labels and observed conditioning features")
        X = data["features"]
        raw = data["conditioning_features"] if support is not None else None
        model = OperatorAwareLDA.fit(X, data["labels"], noise=noise,
                                     raw_operator=operators["A"], state_operator=operators["B"],
                                     calibration_ids=plan["calibration_ids"], scope_id=plan["scope_id"],
                                     input_identity=plan["input_identity"], ridge_relative=plan.get("ridge_relative", 1e-3),
                                     conditioning_features=raw, observation_support=support)
    model_path = output / "model.npz"
    model.save(model_path)
    reloaded, model_sha = _model(OperatorAwareLDA, model_path)
    np.testing.assert_array_equal(model.predict_logit(X, observed_raw=raw, input_identity=plan["input_identity"]),
                                  reloaded.predict_logit(X, observed_raw=raw, input_identity=plan["input_identity"]))
    conditioning_report = ({} if support is None else {"conditioning_sample_support": list(support.conditioning_sample_support),
        "scored_sample_support": list(support.response_sample_support), "available_samples": support.available_samples,
        "rank_ledger": noise.conditioning_rank_ledger, "raw_likelihood_multiplied": False})
    _write(output / "fit.json", {**_provenance(args), "fit": model.provenance,
                                  "model_sha256": model_sha, "plan_sha256": plan_sha,
                                  "noise_artifact_sha256": noise_sha, "operator_file_sha256": operator_sha,
                                  "input_file_sha256": input_sha, "reload_score_equal": True,
                                  "identifiable_rank": int(np.linalg.matrix_rank(model.identifiable_projector)), **conditioning_report})


def _operator_lda_predict(args, output):
    metadata, metadata_sha = _json(args.metadata)
    model, model_sha = _model(OperatorAwareLDA, args.model)
    conditional = model.observation_support is not None
    if conditional and metadata_digest(metadata.get("observation_support")) != metadata_digest(model.observation_support.record()):
        raise ValueError("prediction conditioning/scored support differs from its frozen operator program")
    with _arrays(args.input) as (data, input_sha):
        if set(data.files) != {"features"} | ({"observed_raw"} if conditional else set()):
            raise ValueError("operator LDA prediction takes only declared features/observed raw, never labels or oracle states")
        logits = model.predict_logit(data["features"], observed_raw=data["observed_raw"] if conditional else None,
                                     input_identity=metadata["input_identity"])
    conditioning_report = ({} if not conditional else {"conditioning_sample_support": list(model.observation_support.conditioning_sample_support),
        "scored_sample_support": list(model.observation_support.response_sample_support),
        "available_samples": model.observation_support.available_samples, "raw_likelihood_multiplied": False})
    _write(output / "prediction.json", {**_provenance(args), "logits": logits.tolist(),
                                         "model_sha256": model_sha, "input_file_sha256": input_sha,
                                         "metadata_sha256": metadata_sha, **conditioning_report})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    specs = {
        "clean": ("Plan: input NPZ with X[channels,samples], sfreq, segments, scope_id, scope_role, config.", _clean),
        "fit": ("Plan: labeled records NPZ(samples,onsets,target_mask), sfreq, channels, candidate_ids, operator_identity, fit_scope, basis_path or n_basis/response_seconds, ridge.", _fit),
        "predict": ("Truth-free NPZ(samples,onsets,membership); metadata: sfreq, channels, candidate_ids in column order, operator_identity, record_id.", _predict),
        "noise-fit": ("Plan: input NPZ(raw_residual_draws,state_draws), fit_scope, scope_id, draw_ids, optional shrinkage.", _noise_fit),
        "noise-project": ("Frozen joint noise model and NPZ operators A, B; stack epochs to keep cross covariance.", _noise_project),
        "noise-condition": ("Frozen joint model; NPZ(raw_values,raw_means,state_means) with full candidate expectations; returns a|x,q, no extra evidence.", _noise_condition),
        "operator-lda-fit": ("Plan: noise_model, input NPZ(features,labels), operators NPZ(A,B), calibration_ids, scope_id, input_identity, ridge_relative. Conditional mode adds observation_support and NPZ conditioning_features.", _operator_lda_fit),
        "operator-lda-predict": ("Frozen operator LDA; NPZ(features), plus observed_raw for conditional mode; metadata input_identity and frozen observation_support.", _operator_lda_predict),
    }
    for name, (help_text, handler) in specs.items():
        child = subparsers.add_parser(name, help=help_text, description=help_text)
        child.add_argument("--output", type=Path, required=True)
        if name in {"clean", "fit", "noise-fit", "operator-lda-fit"}:
            child.add_argument("--plan", type=Path, required=True)
        else:
            child.add_argument("--input", type=Path, required=True)
            child.add_argument("--model", type=Path, required=True)
        if name in {"predict", "operator-lda-predict"}:
            child.add_argument("--metadata", type=Path, required=True)
        child.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    args.execution_provenance = _source_provenance(args.command)
    if {**args.execution_provenance, "command": "import"} != _IMPORTED_SOURCE:
        raise RuntimeError("operator implementation changed since module import")
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"schema": "n2p3_operator_research_execution/1", "status": "running",
              **args.execution_provenance}
    atomic_write_json(args.output / "execution.json", status)
    try:
        args.handler(args, args.output)
        if _source_provenance(args.command) != args.execution_provenance:
            raise RuntimeError("operator implementation changed during execution")
    except BaseException as error:
        atomic_write_json(args.output / "execution.json", {**status, "status": "failed",
                          "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    atomic_write_json(args.output / "execution.json", {**status, "status": "completed"})
    print(json.dumps({"status": "complete", "command": args.command, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
