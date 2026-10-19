"""Opt-in persistence and replay of B03 strict-nested inner producers.

The recorder is attached to the *same* ``fit_nested_calibration_selection``
run.  It stores the immutable fitted objects and exact selection/temperature
score events; persistence never performs a second fit.  This remains bounded
calibration-development evidence and is not a product or efficacy gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, atomic_write_json
from transfer.ensemble_fit import FittedEnsemble, FrozenSourceCheckpoint
from transfer.noise_model import array_digest

SCHEMA = "n2p3_inner_model_replay/1"


def _oof_held_record_id(use):
    """Use the row's own held group while retaining ancestor scope fields."""
    if use["phase"] == "temperature_oof_inner":
        return use.get("temperature_held_record_id")
    if use["phase"] == "pool_probability_oof_inner":
        return use.get("pool_held_record_id")
    return use.get("temperature_held_record_id") or use.get("pool_held_record_id")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_identity(group):
    view = group.view
    if view.samples is None:
        raise ValueError("inner replay requires complete observed samples")
    return {
        "record_id": view.record_id,
        "decision_id": view.decision_id,
        "participant_key": view.participant_key,
        "source_subject_id": view.source_subject_id,
        "source_dataset_id": view.source_dataset_id,
        "source_sha256": view.source_sha256,
        "raw_scope": list(view.raw_scope),
        "raw_input_digest": view.raw_input_digest,
        "physical_support_digest": view.physical_support_digest,
        "view": view.record(),
        "group": group.record(),
        # Keep independent bytes digests.  A stale raw_input_digest must not
        # make a changed numerical sample appear to be the original input.
        "samples_sha256": array_digest(view.samples, view.onsets),
        "labels_sha256": array_digest(group.labels),
        "native_raw_fit_input_digest": array_digest(view.samples, view.onsets, group.labels),
    }


def _final_identity(model):
    from research.calibration_selection import _final_identity as identity

    return identity(model)


def _source_roles(candidates, sources):
    roles = {}
    for candidate in candidates:
        for spec in candidate.expert_programs:
            if spec.expert_id not in sources:
                roles.setdefault(spec.expert_id, {"digest": None, "role": None})
                continue
            role = spec.source_checkpoint_role
            roles[spec.expert_id] = {"digest": sources[spec.expert_id].digest, "role": role}
    return roles


def _normalise_sources(source_checkpoints):
    if source_checkpoints is None:
        return {}
    return {name: (value if isinstance(value, FrozenSourceCheckpoint)
                   else FrozenSourceCheckpoint.from_path(value))
            for name, value in source_checkpoints.items()}


def _recursive_close(left, right, *, atol, rtol, path="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"score record fields differ at {path}")
        for key in left:
            _recursive_close(left[key], right[key], atol=atol, rtol=rtol,
                             path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"score record length differs at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _recursive_close(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]")
        return
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                raise ValueError(f"score record differs at {path}")
        elif not np.isclose(float(left), float(right), atol=atol, rtol=rtol,
                            equal_nan=True):
            raise ValueError(f"score record differs at {path}")
        return
    if left != right:
        raise ValueError(f"score record differs at {path}")


class InnerReplayRecorder:
    """Run-local sink for actual fit/cache/OOF events.

    ``calibration_selection`` and ``ensemble_fit`` call these methods only
    when the optional recorder is supplied.  The sink holds model references,
    rather than re-fitting from receipts later.
    """

    def __init__(self):
        self.producers: dict[str, FittedEnsemble | ExpectedSubjectError] = {}
        self.uses: dict[str, list[dict[str, Any]]] = {}
        self.oof_rows: dict[str, list[dict[str, Any]]] = {}
        self.oof_score_failures: dict[str, list[dict[str, Any]]] = {}
        self.selection_scores: dict[str, list[dict[str, Any]]] = {}
        self.object_keys: dict[int, str] = {}
        self.context: dict[str, Any] = {"phase": "unclassified"}
        # Prefix callers use the same run-local sink as nested selection.  The
        # nested persistence contract intentionally does not consume these
        # fields; ``prefix_inner_replay`` owns their schema and receipt.
        self.prefix_programs: dict[str, Any] = {}
        self.prefix_context: dict[str, Any] = {}
        self.prefix_future_scores: list[dict[str, Any]] = []
        self.native_fit_inputs: dict[int, dict[str, Any]] = {}

    def record_native_fit_inputs(self, model, inputs):
        """Capture the original fit owner's actual data operands, never a wrapper."""
        self.native_fit_inputs[id(model)] = inputs

    def record_prefix_programs(self, programs, *, prefix, expert_programs, program,
                               source_checkpoints=None, context=None):
        """Capture actual prefix results from the original fit invocation.

        This stores object references only.  A later persistence call may
        serialize them, but it never reconstructs or re-fits a prefix model.
        """
        context = dict(self.context if context is None else context)
        self.prefix_context = context
        self.prefix_programs = dict(programs)
        self.prefix_fit_declaration = {
            "prefix": prefix,
            "expert_programs": tuple(expert_programs),
            "program": program,
            "source_checkpoints": {} if source_checkpoints is None else dict(source_checkpoints),
            "context": context,
        }
        from research.prefix_inner_replay import prefix_input_identity, prefix_native_fit_inputs

        self.prefix_fit_declaration["input_identity"] = prefix_input_identity(prefix)
        self.prefix_fit_declaration["native_fit_inputs"] = prefix_native_fit_inputs(prefix, expert_programs)

    def record_prefix_future_score(self, strategy, *, view, evidence):
        """Capture an already-scored label-free future view from this run."""
        from research.prefix_inner_replay import future_input_identity

        self.prefix_future_scores.append({"strategy": strategy,
            "input": future_input_identity(view),
            "expert_evidence": {name: item.record() for name, item in evidence.items()}})

    def set_context(self, **values):
        self.context = dict(values)

    def record_fit(self, fit_key, model, *, candidate, training, cache_hit=False, context=None):
        context = self.context if context is None else context
        canonical = (fit_key if isinstance(model, ExpectedSubjectError)
                     else self.object_keys.setdefault(id(model), fit_key))
        self.producers.setdefault(canonical, model)
        use = {
            "phase": context.get("phase", "unclassified"),
            "candidate_name": candidate.name,
            "training_record_ids": [group.view.record_id for group in training],
            "outer_held_record_id": context.get("outer_held_record_id"),
            "selection_held_record_id": context.get("selection_held_record_id"),
            "temperature_held_record_id": context.get("temperature_held_record_id"),
            "cache_hit": bool(cache_hit),
        }
        if context.get("pool_held_record_id") is not None:
            use["pool_held_record_id"] = context["pool_held_record_id"]
        rows = self.uses.setdefault(canonical, [])
        if use not in rows:
            rows.append(use)

    def record_direct_fit(self, fit_key, model, *, expert_programs, program, training, context=None, cache_hit=False):
        """Capture a fixed cache producer used by grouped-temperature OOF."""
        if fit_key is None:
            raise ValueError("recorder requires an immutable fixed-fit cache key")
        context = self.context if context is None else context
        candidate_name = context.get("candidate_name", program.name)
        canonical = (fit_key if isinstance(model, ExpectedSubjectError)
                     else self.object_keys.setdefault(id(model), fit_key))
        self.producers.setdefault(canonical, model)
        use = {
            "phase": context.get("phase", "temperature_oof_inner"),
            "candidate_name": candidate_name,
            "training_record_ids": [group.view.record_id for group in training],
            "outer_held_record_id": context.get("outer_held_record_id"),
            "selection_held_record_id": context.get("selection_held_record_id"),
            "temperature_held_record_id": context.get("temperature_held_record_id"),
            "cache_hit": bool(cache_hit),
            "program": program.record(),
            "expert_programs": [spec.record() for spec in expert_programs],
        }
        if context.get("pool_held_record_id") is not None:
            use["pool_held_record_id"] = context["pool_held_record_id"]
        rows = self.uses.setdefault(canonical, [])
        if use not in rows:
            rows.append(use)

    def record_oof_score(self, fit_key, *, held, training, scores, context=None):
        context = self.context if context is None else context
        canonical = self.object_keys.get(id(self.producers.get(fit_key)), fit_key)
        row = {
            "held_record_id": held.view.record_id,
            "training_record_ids": [group.view.record_id for group in training],
            "phase": context.get("phase", "temperature_oof_inner"),
            "evidence": {name: score.record() for name, score in scores.items()},
        }
        rows = self.oof_rows.setdefault(canonical, [])
        if not any(semantic_sha256(existing) == semantic_sha256(row) for existing in rows):
            rows.append(row)

    def record_oof_score_failure(self, fit_key, *, held, training, error, context=None):
        """Capture an actual raised score failure without inventing evidence."""
        context = self.context if context is None else context
        canonical = self.object_keys.get(id(self.producers.get(fit_key)), fit_key)
        self.oof_score_failures.setdefault(canonical, []).append({
            "held_record_id": held.view.record_id,
            "training_record_ids": [group.view.record_id for group in training],
            "phase": context.get("phase", "temperature_oof_inner"),
            "failure": error.record(subject=held.view.participant_key),
        })

    def record_selection_score(self, fit_key, *, held, evidence):
        canonical = self.object_keys.get(id(self.producers.get(fit_key)), fit_key)
        self.selection_scores.setdefault(canonical, []).append({
            "held_record_id": held.view.record_id,
            "evidence": evidence.record(),
        })

    def _producer(self, fit_key, calibration, candidates, source_roles, root, input_map):
        model = self.producers[fit_key]
        uses = self.uses.get(fit_key, [])
        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        candidate_name = uses[0]["candidate_name"]
        candidate = candidate_by_name[candidate_name]
        if isinstance(model, FittedEnsemble):
            training_ids = [row["record_id"] for row in model.fit_receipts]
        else:
            training_ids = list(uses[0]["training_record_ids"])
        roles = {spec.expert_id: source_roles.get(spec.expert_id, {"digest": None, "role": None})
                 for spec in candidate.expert_programs}
        package = None
        if isinstance(model, FittedEnsemble):
            package_path = root / "models" / fit_key
            model.save(package_path)
            package = str(package_path.relative_to(root)).replace("\\", "/")
        producer = {
            "fit_key": fit_key,
            "status": "completed" if isinstance(model, FittedEnsemble) else "unavailable",
            "candidate_name": candidate_name,
            "candidate": candidate.record(),
            "uses": uses,
            "training_record_ids": training_ids,
            "training_scope": {
                "group_records": [input_map[record_id]["group"] for record_id in training_ids],
                "actual_group_labels": sum(input_map[record_id]["group"]["authorized_label_count"]
                                            for record_id in training_ids),
            },
            "input_identities": [input_map[record_id] for record_id in training_ids],
            "source_checkpoint_roles": roles,
            "package": package,
            "package_sha256": None if package is None else _sha256(root / package / "ensemble.json"),
            "model_identity": None if not isinstance(model, FittedEnsemble) else _final_identity(model),
            "temperature_record": None if not isinstance(model, FittedEnsemble) else model.temperature_record,
            "oof_rows": self.oof_rows.get(fit_key, []),
            "selection_score_rows": self.selection_scores.get(fit_key, []),
            "failure": None if isinstance(model, FittedEnsemble) else model.record(
                subject=calibration[0].view.participant_key),
        }
        return producer

    def persist(self, selection, calibration, candidates, directory, *, source_checkpoints=None, device="cpu"):
        """Write captured producer objects and the strict selection receipt."""
        from research.calibration_selection import FittedCalibrationSelection

        if not isinstance(selection, FittedCalibrationSelection):
            raise TypeError("inner replay requires a FittedCalibrationSelection")
        if not self.producers:
            raise ValueError("inner replay recorder has no captured producers")
        groups = tuple(calibration)
        candidates = tuple(candidates)
        if semantic_sha256([candidate.record() for candidate in candidates]) != semantic_sha256(selection.record["candidates"]):
            raise ValueError("inner replay candidates differ from accepted declaration")
        sources = _normalise_sources(source_checkpoints)
        source_roles = _source_roles(candidates, sources)
        if {name: source.digest for name, source in sources.items()} != selection.record["source_checkpoint_sha256"]:
            raise ValueError("inner replay source checkpoint digest differs from accepted selection")
        input_map = {group.view.record_id: _group_identity(group) for group in groups}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "models").mkdir()
        producers = [self._producer(key, groups, candidates, source_roles, directory, input_map)
                     for key in self.producers]
        record = {
            "schema": SCHEMA,
            "evidence_scope": "calibration_development",
            "participant_key": selection.record["participant_key"],
            "candidate_roster": [candidate.record() for candidate in candidates],
            "baseline": selection.record["baseline"],
            "group_inputs": [input_map[group.view.record_id] for group in groups],
            "source_checkpoint_digests": source_roles,
            "producers": producers,
            "selection_decision": selection.record["final_selection"]["decision"],
            "selected_candidate": selection.record["selected_candidate"],
            "final_selection": selection.record["final_selection"],
            "final_refit_scope": selection.record["final_refit_scope"],
            "outer_requested": selection.record["outer_requested"],
            "outer_rows": selection.record["outer_rows"],
            "outer_metrics": selection.record["outer_metrics"],
            "outer_scores_used_for_final_selection": False,
            "inner_models_persisted": True,
            "requested_denominator": selection.record["outer_requested"],
            "selection_record_sha256": semantic_sha256(selection.record),
            "device": str(device),
        }
        atomic_write_json(directory / "inner_replay.json", {
            "record": record, "record_sha256": semantic_sha256(record),
        })
        return directory


def persist_inner_replay(selection, calibration, candidates, directory, *, recorder=None,
                         source_checkpoints=None, device="cpu"):
    """Persist a recorder captured during the original nested fit run."""
    if recorder is None:
        raise ValueError("inner persistence requires an opt-in recorder from the original fit run")
    if not isinstance(recorder, InnerReplayRecorder):
        raise TypeError("recorder must be an InnerReplayRecorder")
    return recorder.persist(selection, calibration, candidates, directory,
                            source_checkpoints=source_checkpoints, device=device)


def _load_manifest(directory):
    payload = json.loads((directory / "inner_replay.json").read_bytes())
    record = payload.get("record")
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("inner replay manifest has an unsupported schema")
    if payload.get("record_sha256") != semantic_sha256(record):
        raise ValueError("inner replay manifest digest changed")
    if not record.get("inner_models_persisted"):
        raise ValueError("inner replay manifest is not a persisted producer package")
    if record.get("outer_scores_used_for_final_selection") is not False:
        raise ValueError("inner replay final selection is allowed no outer metric input")
    return record


def _load_packages(directory, record):
    loaded = {}
    for producer in record["producers"]:
        key = producer["fit_key"]
        if producer["status"] != "completed":
            if producer["package"] is not None or producer["failure"] is None:
                raise ValueError("unavailable producer has an invalid package receipt")
            continue
        package = directory / producer["package"]
        if package.name != key or not (package / "ensemble.json").is_file():
            raise ValueError("inner producer package path differs from its fit key")
        if producer["package_sha256"] != _sha256(package / "ensemble.json"):
            raise ValueError("inner producer package digest changed")
        model = FittedEnsemble.load(package)
        if semantic_sha256(_final_identity(model)) != semantic_sha256(producer["model_identity"]):
            raise ValueError("inner numerical producer identity changed")
        loaded[key] = model
    return loaded


def _audit_boundaries(record, loaded):
    violations = []
    for producer in record["producers"]:
        training_ids = set(producer["training_record_ids"])
        for use in producer["uses"]:
            forbidden = {use.get("outer_held_record_id"), use.get("selection_held_record_id"),
                         use.get("temperature_held_record_id"), use.get("pool_held_record_id")} - {None}
            overlap = sorted(forbidden & training_ids)
            if overlap:
                violations.append({"fit_key": producer["fit_key"], "overlap": overlap})
        model = loaded.get(producer["fit_key"])
        if model is not None and set(row["record_id"] for row in model.fit_receipts) != training_ids:
            violations.append({"fit_key": producer["fit_key"], "reason": "training_receipts_scope"})
        for row in producer.get("oof_rows", []):
            held_and_training = set(row["training_record_ids"]) | {row["held_record_id"]}
            for use in producer["uses"]:
                if (tuple(use["training_record_ids"]) != tuple(row["training_record_ids"])
                        or _oof_held_record_id(use) != row["held_record_id"]
                        or use["phase"] != row["phase"]):
                    continue
                forbidden = {use.get("outer_held_record_id"), use.get("selection_held_record_id"),
                             use.get("pool_held_record_id"), use.get("temperature_held_record_id")} - {None, row["held_record_id"]}
                if forbidden & held_and_training:
                    violations.append({"fit_key": producer["fit_key"], "reason": "temperature_oof_scope"})
    return violations


def replay_inner_replay(directory, *, calibration=None, candidates=None,
                        source_checkpoints=None, device=None, atol=0.0, rtol=0.0):
    """Reload and score captured producers without fitting any new model."""
    directory = Path(directory)
    record = _load_manifest(directory)
    loaded = _load_packages(directory, record)
    violations = _audit_boundaries(record, loaded)
    if violations:
        raise ValueError(f"inner information boundary violation: {violations[0]}")
    if calibration is None:
        return {"status": "validated_package", "producer_count": len(record["producers"]),
                "boundary_violations": [], "exact_score_checks": 0}
    groups = tuple(calibration)
    actual_inputs = [_group_identity(group) for group in groups]
    if semantic_sha256(actual_inputs) != semantic_sha256(record["group_inputs"]):
        raise ValueError("inner replay input identity, labels, samples or scope changed")
    if candidates is None:
        from research.calibration_selection import CalibrationProgramCandidate

        candidates = tuple(CalibrationProgramCandidate.from_record(value)
                           for value in record["candidate_roster"])
    if semantic_sha256([candidate.record() for candidate in candidates]) != semantic_sha256(record["candidate_roster"]):
        raise ValueError("inner replay candidate declaration changed")
    sources = _normalise_sources(source_checkpoints)
    if _source_roles(candidates, sources) != record["source_checkpoint_digests"]:
        raise ValueError("inner replay source checkpoint role or digest changed")
    input_by_id = {group.view.record_id: group for group in groups}
    score_checks = 0
    for producer in record["producers"]:
        model = loaded.get(producer["fit_key"])
        if model is None:
            continue
        for use in producer["uses"]:
            held_id = (_oof_held_record_id(use) or use.get("selection_held_record_id")
                       or use.get("outer_held_record_id"))
            if held_id is None:
                continue
            # A loaded producer can score its held view without touching outer
            # truth; this checks exact/declared-tolerance numerical replay.
            score = {name: item.record() for name, item in model.score_experts(input_by_id[held_id].view).items()}
            expected = next((row["evidence"] for row in producer.get("oof_rows", [])
                             if row["held_record_id"] == held_id
                             and tuple(row["training_record_ids"]) == tuple(use["training_record_ids"])), None)
            if expected is not None:
                _recursive_close(score, expected, atol=atol, rtol=rtol)
            score_checks += 1
        for row in producer.get("selection_score_rows", []):
            held_id = row["held_record_id"]
            score = model.score(input_by_id[held_id].view).record()
            _recursive_close(score, row["evidence"], atol=atol, rtol=rtol)
            score_checks += 1
    return {"status": "replayed", "producer_count": len(record["producers"]),
            "boundary_violations": [], "exact_score_checks": score_checks}
