"""Persist original joint flat numerical states for model-only predictive replay.

Compact augmented roots, original row counts/da volumes, all retained draws,
same-record joins and actual parameter bindings are saved. Loading verifies
external artifact identity and restores producer arrays without fitting or
recompiling QR. A numeric replay state grants no access to original EEG labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.joint_flat_response import FlatCompiledDecision, prepare_flat_prediction
from research.joint_response_sampling import JointResponseSamples, prepare_model
from research.joint_response_scale import (
    CompiledDecision,
    FixedHyperparameters,
    GlobalGaussianPrior,
)
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest

_ARRAY_FIELDS = {"augmented_root", "prefix_drift_rhs", "tail_root", "energy_error_columns", "tail_error_columns",
                 "tail_drift_root", "tail_drift_linear", "tail_data_norms"}
_LINK_FIELDS = {"prefix", "tail", "earlier_prefixes"}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")


def save_joint_flat_state(directory, samples, *, predictions=(), authority_binding=None, parameters=None):
    """Save a finite sampling program and optional original compiled future states."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    arrays = {"prior_mean": samples.model.prior.mean, "prior_root": samples.model.prior.root,
              "chart": samples.chart, "scales": samples.scales}
    nodes = []
    ids = {}

    def save_node(node):
        if node is None:
            return None
        if id(node) in ids:
            return ids[id(node)]
        if not isinstance(node, CompiledDecision):
            raise TypeError("only the original numerical decision factory state is serializable")
        index = len(nodes)
        ids[id(node)] = index
        nodes.append(None)
        scalar, numeric = {}, {}
        for name in node.__dataclass_fields__:
            if name in _LINK_FIELDS:
                continue
            value = getattr(node, name)
            if name in _ARRAY_FIELDS and value is not None:
                key = f"decision_{index}_{name}"
                arrays[key] = value
                numeric[name] = key
            else:
                scalar[name] = str(value) if name == "reference_nu" else value
        nodes[index] = {"kind": "flat" if isinstance(node, FlatCompiledDecision) else "proper",
            "scalars": scalar, "arrays": numeric, "prefix": save_node(node.prefix),
            "tail": save_node(node.tail),
            "earlier_prefixes": [save_node(v) for v in node.earlier_prefixes]
                if isinstance(node, FlatCompiledDecision) else []}
        return index

    block_ids = [save_node(record) for record in samples.model.blocks]
    prediction_ids = []
    for prediction in predictions:
        # Re-admit the actual original source/C numerical join before persisting.
        admitted = prepare_flat_prediction(samples.model, prediction.full)
        if admitted.prefix_already_in_model != prediction.prefix_already_in_model:
            raise ValueError("prediction prefix join changed before save")
        prediction_ids.append(save_node(prediction.full))
    parameter_record = None
    if parameters is not None:
        from research.joint_flat_authority import JointFlatParameters

        if (not isinstance(parameters, JointFlatParameters)
                or parameters.prior is not samples.model.prior
                or parameters.hyperparameters != samples.model.hyperparameters):
            raise ValueError("actual parameter owner must match the saved sampled prior/noise program")
        arrays["actual_basis"] = parameters.basis
        parameters.noise.save(target/"noise.json")
        parameter_record = {"binding": parameters.binding, "binding_sha256": parameters.digest,
            "noise_file_sha256": hashlib.sha256((target/"noise.json").read_bytes()).hexdigest()}
    state_file = target/"state.npz"
    with state_file.open("xb") as handle:
        np.savez(handle, **arrays)
    manifest = {"schema": "n2p3_joint_flat_sampling_artifact/2",
        "prior_program": samples.model.prior.program,
        "hyperparameters": {"degrees_of_freedom": str(samples.model.hyperparameters.degrees_of_freedom),
            "relative_noise_shape": samples.model.hyperparameters.relative_noise_shape,
            "program": samples.model.hyperparameters.program},
        "decisions": nodes, "model_block_ids": block_ids, "prediction_ids": prediction_ids,
        "sampling": {"burnin_per_chain": samples.burnin_per_chain,
            "draws_per_chain": samples.draws_per_chain, "chain_seeds": samples.chain_seeds,
            "elapsed_seconds": samples.elapsed_seconds,
            "program": ("Gaussian_limit_independent_proper_beta_draws" if samples.gaussian_limit
                        else "continuous_Gamma_Gibbs_collapsed_physical_da")},
        "actual_authority_binding": authority_binding,
        "parameter_owner": parameter_record,
        "array_records": {key: {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": array_digest(value)} for key, value in arrays.items()},
        "state_file_sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
        "producer_sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("joint_flat_artifact.py", "joint_flat_response.py", "joint_flat_authority.py",
                         "joint_response_scale.py", "joint_response_sampling.py", "channel_noise_mechanism.py")},
        "restore": "original_producer_compact_arrays_no_QR_noise_response_or_sampler_fit"}
    identity = metadata_digest(manifest)
    (target/"manifest.json").write_bytes(_json_bytes({"state": manifest, "artifact_sha256": identity}))
    return identity


def load_joint_flat_state(directory, *, artifact_sha256):
    """Model-only load requires the independently supplied original package SHA."""
    target = Path(directory)
    payload = json.loads((target/"manifest.json").read_bytes())
    manifest = payload["state"]
    if (set(payload) != {"state", "artifact_sha256"}
            or manifest["schema"] != "n2p3_joint_flat_sampling_artifact/2"
            or payload["artifact_sha256"] != artifact_sha256
            or metadata_digest(manifest) != artifact_sha256
            or hashlib.sha256((target/"state.npz").read_bytes()).hexdigest() != manifest["state_file_sha256"]):
        raise ValueError("joint flat artifact differs from original external producer identity")
    with np.load(target/"state.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["array_records"]):
            raise ValueError("saved numeric member set differs from original producer")
        arrays = {}
        for key, record in manifest["array_records"].items():
            value = archive[key]
            if (list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]
                    or value.dtype.kind != "f" or not np.isfinite(value).all()
                    or array_digest(value) != record["sha256"]):
                raise ValueError("saved numeric geometry/bytes differ from original producer")
            arrays[key] = frozen_array(value)
    prior = GlobalGaussianPrior(arrays["prior_mean"], arrays["prior_root"], manifest["prior_program"])
    hyper = dict(manifest["hyperparameters"])
    hyper["degrees_of_freedom"] = float(hyper["degrees_of_freedom"])
    hyper = FixedHyperparameters(**hyper)
    nodes = []
    for stored in manifest["decisions"]:
        cls = FlatCompiledDecision if stored["kind"] == "flat" else CompiledDecision
        if stored["kind"] not in {"flat", "proper"}:
            raise ValueError("unsupported stored decision likelihood measure")
        values = {**stored["scalars"], **{key: arrays[value] for key, value in stored["arrays"].items()}}
        if "reference_nu" in values:
            values["reference_nu"] = float(values["reference_nu"])
        expected = set(cls.__dataclass_fields__)-_LINK_FIELDS
        if set(values) != expected:
            raise ValueError("saved decision must preserve every actual factory field")
        node = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(node, key, freeze_metadata(value) if isinstance(value, dict) else value)
        object.__setattr__(node, "prefix", None)
        object.__setattr__(node, "tail", None)
        if isinstance(node, FlatCompiledDecision):
            if (node.observed_rows-node.nuisance_rank != node.effective_rows
                    or not 0 <= node.nuisance_rank <= node.observed_rows):
                raise ValueError("saved raw/rank/effective row accounting is inconsistent")
            object.__setattr__(node, "earlier_prefixes", ())
        if node.augmented_root.ndim != 2 or node.augmented_root.shape[1] != len(prior.mean)+1:
            raise ValueError("saved augmented root must use the original beta chart and residual row")
        nodes.append(node)
    for node, stored in zip(nodes, manifest["decisions"], strict=True):
        for key in ("prefix", "tail"):
            index = stored[key]
            object.__setattr__(node, key, None if index is None else nodes[index])
        if isinstance(node, FlatCompiledDecision):
            object.__setattr__(node, "earlier_prefixes", tuple(nodes[i] for i in stored["earlier_prefixes"]))
    model = prepare_model(prior, (nodes[i] for i in manifest["model_block_ids"]), hyper)
    sampling = {key: value for key, value in manifest["sampling"].items() if key != "program"}
    sampling["chain_seeds"] = tuple(sampling["chain_seeds"])
    samples = JointResponseSamples(model, arrays["chart"], arrays["scales"], **sampling)
    predictions = tuple(prepare_flat_prediction(model, nodes[i]) for i in manifest["prediction_ids"])
    parameters = None
    if manifest["parameter_owner"] is not None:
        from research.channel_noise_mechanism import FittedChannelNoise
        from research.joint_flat_authority import _parameters

        record = manifest["parameter_owner"]
        if (metadata_digest(record["binding"]) != record["binding_sha256"]
                or hashlib.sha256((target/"noise.json").read_bytes()).hexdigest() != record["noise_file_sha256"]):
            raise ValueError("parameter source/noise state differs from original saved producer")
        noise = FittedChannelNoise.load(target/"noise.json")
        if metadata_digest(noise.record()) != metadata_digest(record["binding"]["noise"]):
            raise ValueError("loaded numeric noise differs from actual parameter authority binding")
        parameters = _parameters(prior, hyper, arrays["actual_basis"], noise, record["binding"])
    return samples, predictions, freeze_metadata(manifest["actual_authority_binding"]), parameters
