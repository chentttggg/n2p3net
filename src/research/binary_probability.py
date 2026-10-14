"""Proper binary risk exports and fixed arithmetic posterior pools."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score

from research.comparison import ContrastPlan
from research.source_execution import current_input_admission
from transfer.ensemble import EnsembleProgram


def binary_probability_metrics(labels, posterior_logits):
    y, z = np.asarray(labels), np.asarray(posterior_logits, dtype=np.float64)
    if y.shape != z.shape or y.ndim != 1 or not len(y) or not np.isin(y, (0, 1)).all() or not np.isfinite(z).all():
        raise ValueError("binary risk requires every requested label and finite posterior logit")
    return {"nll": float(np.mean(np.logaddexp(0., z) - y * z)),
            "probability_auc": float(roc_auc_score(y, z)) if len(np.unique(y)) == 2 else None,
            "brier": float(np.mean((expit(z) - y) ** 2)), "probability_scored_rows": len(y)}


def arithmetic_binary_pool(posterior_logits, weights):
    z, w = np.asarray(posterior_logits, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if (z.ndim != 2 or z.shape[0] != len(w) or not np.isfinite(z).all()
            or np.any(w <= 0) or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("binary pool needs fixed normalized weights and complete declared posterior scores")
    positive = logsumexp(np.log(w)[:, None] - np.logaddexp(0., -z), axis=0)
    negative = logsumexp(np.log(w)[:, None] - np.logaddexp(0., z), axis=0)
    return positive - negative


def export_binary_pools(cells, declarations, output):
    programs = [EnsembleProgram(value["name"], tuple(value["expert_ids"]), tuple(value["weights"]))
                for value in declarations]
    if len({program.name for program in programs}) != len(programs):
        raise ValueError("binary probability pools require distinct names")
    keys = sorted({(cell["subject"], cell["K"], cell["source_seed"], cell["target_seed"]) for cell in cells})
    exported = []
    for subject, budget, source_seed, seed in keys:
        group = {cell["arm"]: cell for cell in cells if (cell["subject"], cell["K"], cell["source_seed"], cell["target_seed"])
                 == (subject, budget, source_seed, seed)}
        for program in programs:
            record = {"subject": subject, "K": budget, "source_seed": source_seed, "target_seed": seed,
                      "program": program.record(), "status": "unavailable",
                      "requested_Ks": sorted({request for name in program.expert_ids if name in group
                                               for request in group[name]["requested_Ks"]})}
            missing = [name for name in program.expert_ids if name not in group
                       or group[name].get("probability_status") != "completed"]
            if missing:
                record["unavailable_experts"] = missing
                exported.append(record)
                continue
            loaded, origins = [], []
            for name in program.expert_ids:
                cell = group[name]
                ContrastPlan(()).validate_pair(group[program.expert_ids[0]]["comparison_identity"], cell["comparison_identity"])
                path = output / cell["prediction_artifact"]["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != cell["prediction_artifact"]["sha256"]:
                    raise ValueError("binary pool prediction artifact differs from its actual scored bytes")
                with np.load(path, allow_pickle=False) as artifact:
                    loaded.append({key: artifact[key].copy() for key in ("y", "test_rows", "calibration_rows", "posterior_logits")})
                origins.append({"expert_id": name, "model_artifact": cell["model_artifact"],
                                "prediction_artifact": cell["prediction_artifact"], "probability_semantics": cell["probability_semantics"]})
            first = loaded[0]
            if any(not np.array_equal(first[key], value[key]) for value in loaded[1:]
                   for key in ("y", "test_rows", "calibration_rows")):
                raise ValueError("binary probability experts must share actual calibration/test rows and labels")
            logits = arithmetic_binary_pool(np.stack([value["posterior_logits"] for value in loaded]), program.weights)
            path = output / f"probability_pool_{len(exported):05d}.npz"
            arrays = {"y": first["y"], "test_rows": first["test_rows"], "calibration_rows": first["calibration_rows"],
                      "posterior_logits": logits, "probability": expit(logits)}
            # The epoch-cache writer uses this same NPY/ZIP/deflate growth
            # bound. Reserve before NumPy opens the final pool path.
            estimate = sum(value.nbytes for value in arrays.values()) * 1001 // 1000 + len(arrays) * 65536 + 2**20
            admission = current_input_admission()
            with admission.artifact_writer(path, estimate) if admission is not None else nullcontext():
                np.savez_compressed(path, **arrays)
            record.update(status="completed", expert_origins=origins, prediction_path=path.name,
                          prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          **binary_probability_metrics(first["y"], logits))
            exported.append(record)
    return exported
