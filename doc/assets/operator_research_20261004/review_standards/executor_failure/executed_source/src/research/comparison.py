"""Physical evidence, materialized inputs and explicit operator contrasts.

The same original trials may be processed by different declared operators in
a comparison. Result reuse instead requires exactly the same fit and score
materialization. This owner keeps these two relations separate.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from data.epochs import materialize_dataset_identity
from research.contracts import semantic_sha256

INPUT_IDENTITY_SCHEMA = "n2p3_evaluation_input_identity/2"
_AXES = {
    "baseline": {"baseline_mode", "trial_reference_window_ms", "trial_reference_center", "trial_reference_scale"},
    "filter": {"l_freq", "h_freq", "filter_order", "filter_method", "filter_phase", "causal_iir_initial_state",
               "zero_phase_edge_policy", "notch_freqs_hz", "notch_quality_factor"},
    "epoch_window": {"tmin_ms", "tmax_ms", "n_times"},
    "resampling": {"sfreq", "n_times", "resample_domain", "resample_method", "resample_npad",
                   "resample_window", "resample_pad"},
    "spatial": set(), "raw_support": set(), "operator_implementation": set(),
}


def _array_sha(value):
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("input identities cannot hash object arrays")
    digest = hashlib.sha256(semantic_sha256({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _indices(rows, n):
    raw = np.asarray(rows)
    if raw.ndim != 1 or (len(raw) and raw.dtype.kind not in "iu"):
        raise ValueError("evidence rows must be an integer vector")
    selected = raw.astype(np.int64)
    if np.any(selected < 0) or np.any(selected >= n) or len(np.unique(selected)) != len(selected):
        raise ValueError("evidence rows must be unique and within the dataset")
    return selected


def _raw_inputs(dataset):
    """Read known producer fields; never treat an implementation hash as raw."""
    records = {}

    def has_raw(value):
        if isinstance(value, Mapping):
            for key in ("consumed_inputs", "inputs"):
                items = value.get(key)
                if isinstance(items, list) and any(isinstance(item, Mapping)
                        and item.get("role") == "raw" and item.get("sha256") for item in items):
                    return True
            return any(has_raw(child) for child in value.values())
        return isinstance(value, (tuple, list)) and any(has_raw(child) for child in value)

    def visit(value):
        if isinstance(value, Mapping):
            inputs = value.get("consumed_inputs", ())
            if isinstance(inputs, list):
                for item in inputs:
                    if isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
                        binding = {"role": item["role"], "sha256": item["sha256"]}
                        records[semantic_sha256(binding)] = binding
            raw = value.get("raw_identity")
            if isinstance(raw, Mapping):
                for item in raw.get("inputs", ()):
                    if isinstance(item, Mapping) and "sha256" in item:
                        binding = {"role": item["role"], "sha256": item["sha256"]}
                        records[semantic_sha256(binding)] = binding
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(dataset.provenance)
    if dataset.lineage is not None:
        visit(dataset.lineage.payload())
    output = [records[key] for key in sorted(records)]
    for binding in output:
        digest = binding["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid original raw input hash")
    complete = has_raw(dataset.provenance)
    if dataset.lineage is not None:
        # Every source leaf must lead to a known raw-producing operation.
        # One attested child must not certify unrelated unbound parents.
        leaves, ancestors, covered = set(), {}, set()
        for entity in dataset.lineage.entities:
            digest = entity.digest()
            if entity.operation == "source_ingress":
                leaves.add(digest)
                ancestors[digest] = {digest}
            else:
                ancestors[digest] = set().union(*(ancestors[parent] for parent in entity.parent_entity_digests))
            if has_raw(entity.parameters):
                covered.update(ancestors[digest])
        complete = bool(leaves) and leaves <= covered
    return output, complete


def _scope_contexts(execution):
    contexts = []
    if isinstance(execution, Mapping):
        context = execution.get("source_context")
        if isinstance(context, Mapping):
            contexts.append({key: context.get(key) for key in (
                "sample_rate_hz", "first_samp", "n_times", "raw_scope", "filter_segments")})
        for child in execution.values():
            if isinstance(child, (Mapping, list)):
                contexts.extend(_scope_contexts(child))
    elif isinstance(execution, list):
        for child in execution:
            contexts.extend(_scope_contexts(child))
    return sorted(contexts, key=semantic_sha256)


def evaluation_input_identity(dataset, fit_rows, score_rows) -> dict:
    """Bind actual fit/score arrays and physical event identities, independently."""
    dataset.validate(require_labels=True)
    fit, score = (_indices(rows, dataset.n_epochs) for rows in (fit_rows, score_rows))
    if not len(score) or np.intersect1d(fit, score).size:
        raise ValueError("scored evidence must be nonempty and disjoint from fit evidence")
    table = materialize_dataset_identity(dataset)
    timeline = dataset.event_timeline
    event_rows = np.flatnonzero(timeline.evidence_indices >= 0)
    if not np.array_equal(np.sort(timeline.evidence_indices[event_rows]), np.arange(dataset.n_epochs)):
        raise ValueError("one physical event per materialized row is required")
    inverse = np.empty(dataset.n_epochs, dtype=np.int64)
    inverse[timeline.evidence_indices[event_rows]] = event_rows
    participants = {str(subject): table.record_for(subject).authority_key()
                    for subject in np.unique(dataset.subject_ids)}
    observed = (np.broadcast_to(dataset.channel_mask, dataset.X.shape[:2])
                if dataset.trial_channel_mask is None else dataset.trial_channel_mask)
    raw, raw_bound = _raw_inputs(dataset)
    spatial = {"channels": list(dataset.channel_names), "positions": np.asarray(dataset.channel_positions_m).tolist(),
               "reference": dataset.provenance.get("source_reference"),
               "domain_adapter": dataset.provenance.get("domain_adapter")}
    support = dataset.provenance.get("target_raw_record_scopes")
    if support is not None:
        support = [{key: row[key] for key in ("raw_identity", "raw_scope")} for row in support]
    else:
        support = _scope_contexts(dataset.provenance.get("preprocessing_execution"))
    operator = {"preprocessing": dataset.preprocessing.record(), "spatial": spatial,
                "raw_support": support,
                "execution": dataset.provenance.get("preprocessing_execution")}
    decision_semantics = {field: dataset.provenance.get(field) for field in (
        "event_contract", "candidate_task_contract", "candidate_vocabulary", "session_schema")}
    decision_semantics["timeline_complete"] = timeline.complete
    decision_semantics["requested_schedule_sha256"] = semantic_sha256({
        field: np.asarray(getattr(timeline, field)).tolist() for field in (
            "event_ids", "dataset_ids", "subject_ids", "session_ids", "run_ids", "group_ids",
            "selection_ids", "stimulus_ids", "candidate_ids", "target_candidate_ids", "repetition_indices",
            "onset_samples", "onset_times_s")})
    fit_context = {"timeline_sha256": timeline.fingerprint(),
                   "metadata_sha256": semantic_sha256(dataset.metadata.to_json(orient="table", index=False)),
                   "qc_features_sha256": (None if dataset.qc_features is None else semantic_sha256({
                       field: _array_sha(getattr(dataset.qc_features, field)) for field in (
                           "relative_ptp", "channel_std_v", "epoch_scale_v", "observed_mask")}))}
    scopes = {}
    for name, rows in (("fit", fit), ("score", score)):
        events = inverse[rows]
        physical = {field: np.asarray(getattr(timeline, field))[events].tolist() for field in (
            "event_ids", "dataset_ids", "session_ids", "run_ids", "onset_samples", "onset_times_s",
            "stimulus_ids", "group_ids", "selection_ids", "candidate_ids", "target_candidate_ids", "repetition_indices")}
        physical.update(participants=[participants[str(subject)] for subject in dataset.subject_ids[rows]],
                        labels=dataset.y[rows].tolist(), raw_inputs=raw)
        scopes[name] = {"evidence_sha256": semantic_sha256(physical), "n_rows": len(rows),
                        "values_sha256": _array_sha(dataset.X[rows]), "labels_sha256": _array_sha(dataset.y[rows]),
                        "observed_sha256": _array_sha(observed[rows]),
                        "rows_sha256": _array_sha(rows)}
    record = {"schema": INPUT_IDENTITY_SCHEMA, "fit": scopes["fit"], "score": scopes["score"],
              "raw_bytes_bound": raw_bound, "operator": operator,
              "decision_semantics": decision_semantics, "fit_context": fit_context}
    return {**record, "digest": semantic_sha256(record)}


def validate_input_identity(record):
    if not isinstance(record, Mapping) or record.get("schema") != INPUT_IDENTITY_SCHEMA:
        raise ValueError("comparison requires an explicit current input identity")
    if set(record) != {"schema", "fit", "score", "raw_bytes_bound", "operator",
                       "decision_semantics", "fit_context", "digest"}:
        raise ValueError("incomplete comparison input identity")
    if semantic_sha256({k: v for k, v in record.items() if k != "digest"}) != record.get("digest"):
        raise ValueError("comparison input identity digest mismatch")

    def require_hash(value):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid comparison identity hash")

    semantics, context = record["decision_semantics"], record["fit_context"]
    if not isinstance(semantics, Mapping) or set(semantics) != {
        "event_contract", "candidate_task_contract", "candidate_vocabulary", "session_schema",
        "timeline_complete", "requested_schedule_sha256",
    } or type(semantics["timeline_complete"]) is not bool:
        raise ValueError("incomplete decision semantics in comparison identity")
    require_hash(semantics["requested_schedule_sha256"])
    if not isinstance(context, Mapping) or set(context) != {
        "timeline_sha256", "metadata_sha256", "qc_features_sha256",
    }:
        raise ValueError("incomplete fit context in comparison identity")
    require_hash(context["timeline_sha256"])
    require_hash(context["metadata_sha256"])
    if context["qc_features_sha256"] is not None:
        require_hash(context["qc_features_sha256"])
    if type(record.get("raw_bytes_bound")) is not bool or not isinstance(record.get("operator"), Mapping):
        raise ValueError("comparison identity lacks operator and raw binding status")
    if set(record["operator"]) != {"preprocessing", "spatial", "raw_support", "execution"}:
        raise ValueError("invalid input operator declaration")
    if not isinstance(record["operator"]["preprocessing"], Mapping):
        raise ValueError("invalid preprocessing identity")
    for name in ("fit", "score"):
        if not isinstance(record.get(name), Mapping):
            raise ValueError("comparison identity lacks fit/score evidence")
        count = record[name].get("n_rows")
        if type(count) is not int or count < (1 if name == "score" else 0):
            raise ValueError("invalid fit/score row count")
        for field in ("evidence_sha256", "values_sha256", "labels_sha256", "observed_sha256", "rows_sha256"):
            require_hash(record[name].get(field))
    return record


@dataclass(frozen=True)
class ContrastPlan:
    """A declaration of which input operations may differ; model arms remain explicit."""

    allowed_input_axes: tuple[str, ...] = ()

    def __post_init__(self):
        axes = tuple(self.allowed_input_axes)
        if len(set(axes)) != len(axes) or set(axes) - set(_AXES):
            raise ValueError("contrast has unknown or duplicate allowed input axes")
        object.__setattr__(self, "allowed_input_axes", axes)

    def record(self):
        return {"schema": "n2p3_input_contrast_plan/1", "allowed_input_axes": list(self.allowed_input_axes)}

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"schema", "allowed_input_axes"} or record["schema"] != "n2p3_input_contrast_plan/1":
            raise ValueError("invalid input ContrastPlan")
        if not isinstance(record["allowed_input_axes"], list):
            raise ValueError("allowed input axes must be an explicit list")
        return cls(tuple(record["allowed_input_axes"]))

    def validate_pair(self, left, right):
        left, right = validate_input_identity(left), validate_input_identity(right)
        for scope in ("fit", "score"):
            if left[scope]["evidence_sha256"] != right[scope]["evidence_sha256"]:
                raise ValueError("matched physical fit/score evidence identity differs")
        if left["decision_semantics"] != right["decision_semantics"]:
            raise ValueError("matched physical decision semantics or requested schedule differs")
        if not self.allowed_input_axes:
            if left["digest"] != right["digest"]:
                raise ValueError("matched materialized input identity differs")
            return
        if not left["raw_bytes_bound"] or not right["raw_bytes_bound"]:
            raise ValueError("operator contrasts require original raw byte bindings")
        a, b = left["operator"], right["operator"]
        permitted = set().union(*(_AXES[axis] for axis in self.allowed_input_axes))
        changed = {key for key in set(a["preprocessing"]) | set(b["preprocessing"])
                   if a["preprocessing"].get(key) != b["preprocessing"].get(key)}
        if changed - permitted - {"name"}:
            raise ValueError(f"undeclared preprocessing contrast: {sorted(changed - permitted - {'name'})}")
        for field, axis in (("spatial", "spatial"), ("raw_support", "raw_support")):
            if a[field] != b[field] and axis not in self.allowed_input_axes:
                raise ValueError(f"undeclared {axis} contrast")
        if "spatial" not in self.allowed_input_axes:
            for scope in ("fit", "score"):
                if left[scope]["observed_sha256"] != right[scope]["observed_sha256"]:
                    raise ValueError("undeclared observed channel support change")
        def implementations(execution):
            if isinstance(execution, Mapping):
                result = ([execution["implementation"]] if "implementation" in execution else [])
                for child in execution.values():
                    result.extend(implementations(child))
                return result
            if isinstance(execution, list):
                return [item for child in execution for item in implementations(child)]
            return []
        if ("operator_implementation" not in self.allowed_input_axes
                and sorted(implementations(a["execution"]), key=semantic_sha256)
                != sorted(implementations(b["execution"]), key=semantic_sha256)):
            raise ValueError("undeclared operator_implementation contrast")
        if left["digest"] != right["digest"] and not changed - {"name"} and not any(
            a[key] != b[key] for key in ("spatial", "raw_support")
        ) and not ("operator_implementation" in self.allowed_input_axes and a["execution"] != b["execution"]):
            raise ValueError("changed materialized values without a declared changed operator")


def assert_matched_cells(cells, *, plan=None):
    """One pairing gate for both inline effects and full contrast statistics."""
    rows = list(cells)
    if len(rows) < 2:
        return
    contract = ContrastPlan() if plan is None else plan
    if not isinstance(contract, ContrastPlan):
        contract = ContrastPlan.from_record(contract)
    identities = [row.get("comparison_identity") for row in rows]
    for right in identities[1:]:
        contract.validate_pair(identities[0], right)
    if not contract.allowed_input_axes:
        # Preserve consistency with the emitted prediction-row/label bindings.
        fields = ("calibration_rows_sha256", "test_rows_sha256", "input_sha256", "test_labels_sha256")
        for field in fields:
            values = [row.get(field) for row in rows]
            if not all(isinstance(value, str) and len(value) == 64 for value in values) or len(set(values)) != 1:
                raise ValueError("matched same calibration and test input identity is required")


__all__ = ["ContrastPlan", "evaluation_input_identity", "validate_input_identity", "assert_matched_cells"]
