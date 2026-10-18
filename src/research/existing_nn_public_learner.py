"""Reuse the admitted leave-H NN and mature C-only target-learning owners.

This caller adds no neural training algorithm. It joins the original record
compiler, source admission, epoch selection/refit, frozen predictor and prefix
replay. Future truth is absent from every fitting and scoring argument.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from data.candidate_input import resolve_candidate_dataset_adapter
from data.study import dataset_study_membership
from models.candidate_evidence import CandidateEvidence
from research.contracts import semantic_sha256
from research.execution import ExpectedSubjectError, UnidentifiableFitError, atomic_write_json
from research.frozen_metadata import plain_metadata
from research.nested_selection_replay import InnerReplayRecorder
from research.prefix_inner_replay import (
    persist_prefix_inner_replay,
    prefix_input_identity,
    replay_prefix_inner_replay,
)
from research.source_catalog import SourceSeedCatalog
from transfer.calibration_candidates import (
    CalibrationCandidateFitCache,
    waveform_lda_recipe,
)
from transfer.calibration_split import dataset_calibration_inner_split
from transfer.calibration_state import FrozenTrialPredictor
from transfer.checkpoint import ScopedSourceCheckpointPreparation
from transfer.ensemble import EnsembleProgram, EnsembleScorer
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    FrozenSourceCheckpoint,
    _calibration_expert_fit_digest,
    _fit_expert,
    _FittedExpert,
)
from transfer.ensemble_views import RecordCalibrationPrefixCompiler, SelectionEpochProfile
from transfer.epoch_selection import (
    EpochSelectionPolicy,
    epoch_selection_recipes,
    fit_with_epoch_selection,
)
from transfer.recipes import erp_target_recipe

LEAVE_H_SOURCE_SHA256 = "134a5bc1beef8a64870bae7f2ca3a29da33ddad689ccc90b5419170f9c467236"
LEAVE_H_SOURCE_SEED = 20261014
LEAVE_H_CHANNELS = ("FZ", "P3", "PZ", "P4", "PO8", "OZ")
METHOD = "n2p3_leave_H_11study_Cz6_seed20261014"


def admit_existing_leave_h_source(path):
    """Keep the actual whole-library exclusion and source selection/refit law."""
    path = Path(path)
    catalog = SourceSeedCatalog([{
        "outer_target_study": "Huebner2017", "source_seed": LEAVE_H_SOURCE_SEED,
        "path": path.name, "sha256": LEAVE_H_SOURCE_SHA256,
        "procedure": {"kind": "selection_refit"},
    }], base_directory=path.parent, evidence_scope="public_development")
    return FrozenSourceCheckpoint(catalog.content("Huebner2017", LEAVE_H_SOURCE_SEED)), catalog.record()[0]


def leave_h_profile(*, native_sample_rate_hz, native_reference):
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract

    return SelectionEpochProfile.from_preprocessing(
        replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none"),
        source_sample_rate_hz=native_sample_rate_hz, channel_names=LEAVE_H_CHANNELS,
        reference=native_reference, reference_channel="CZ", allow_missing_channels=True)


def existing_nn_public_compiler(declaration, *, budgets=(200, 800)):
    """Resolve original input once; raw reads stay with its registered issuer."""
    adapter = resolve_candidate_dataset_adapter(declaration)
    first = adapter.recordings[0]
    profile = leave_h_profile(native_sample_rate_hz=first.sample_rate_hz, native_reference=first.reference)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=budgets, profiles=(profile,),
        authorization_preprocessing=profile.preprocessing, response_seconds=1.2, pre_roll_seconds=.2)
    return compiler, profile


@dataclass(frozen=True)
class ExistingNNTargetPolicy:
    target_seed: int = 20261014
    epochs: int = 60
    batch_size: int = 512
    epoch_selection: EpochSelectionPolicy = EpochSelectionPolicy(mode="patience")

    def recipes(self):
        recipes = epoch_selection_recipes(batch_size=self.batch_size, seed=self.target_seed, epochs=self.epochs)
        # The existing fixed-mass calibration risk supports the canonical LLR
        # probability transform independently of how the final batch is sized.
        return tuple(replace(recipes[name], trial_loss_reduction="calibration_balanced_fixed_mass")
                     for name in ("full", "scratch"))

    def record(self):
        return {"target_seed": self.target_seed, "epochs": self.epochs, "batch_size": self.batch_size,
            "epoch_selection": self.epoch_selection.record(), "recipes": [r.record() for r in self.recipes()]}


class _CapturedFits(CalibrationCandidateFitCache):
    """Capture actual mature fitter returns, without changing its optimization."""

    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.rows = []

    def fit(self, dataset, recipe, training_mask, **options):
        fitted = super().fit(dataset, recipe, training_mask, **options)
        validation = options.get("validation_mask")
        row = {"phase": options["phase"], "recipe": fitted.recipe.record(),
            "training_rows": np.flatnonzero(training_mask).tolist(),
            "validation_rows": [] if validation is None else np.flatnonzero(validation).tolist(),
            "history": fitted.fit_history}
        if options["phase"] == "inner":
            # Keep the selected inner state, not another copy of the final model.
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"inner_{len(self.rows):02d}.pt"
            predictor = FrozenTrialPredictor.from_fitted(fitted)
            predictor.save(path)
            row.update(model=path.name, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                predictor_digest=predictor.digest(),
                validation_raw_logits=predictor.predict_raw_logit(dataset.X[validation],
                    dataset.trial_channel_mask[validation]).tolist())
        self.rows.append(row)
        return fitted


@dataclass
class ExistingNNPublicFit:
    prefix: object
    compiler: RecordCalibrationPrefixCompiler
    programs: dict
    source_checkpoints: dict
    recorder: InnerReplayRecorder
    context: dict
    selection: dict


def _neural_expert(prefix, profile, recipe, source, prepared, subject, policy, capture, *, device, recording_order):
    data = prefix.trial_datasets.get(profile.digest)
    selected_recipe, selection, predictor, reason = recipe, None, None, None
    kind = "not_available"
    if data is None or data.n_epochs < 3 or set(np.unique(data.y)) != {0, 1}:
        reason = "insufficient_authorized_trial_classes_or_rows"
    else:
        try:
            fitted, selection = fit_with_epoch_selection(data, recipe, np.ones(data.n_epochs, dtype=bool),
                policy=policy.epoch_selection, source_trunk=prepared.trunk_for_target(subject),
                source_input_stats=prepared.input_statistics, device=device,
                recording_order=recording_order, fit_cache=capture)
            selected_recipe = fitted.recipe
            predictor = FrozenTrialPredictor.from_fitted(fitted)
        except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
            reason, kind = f"{type(error).__name__}: {error}", "fit_failure"
    name = METHOD + "_full_fine" if recipe.initialization == "source" else "n2p3_matched_Cz6_scratch"
    spec = CalibrationExpertProgram(name, "trial", recipe=selected_recipe, profile=profile)
    source_digest = source.digest if recipe.initialization == "source" and predictor is not None else None
    fit_digest = (_calibration_expert_fit_digest(spec, prefix.record(), source.digest) if predictor is not None else
        semantic_sha256({"program": spec.record(), "prefix": prefix.record(),
            "fit_failure" if kind == "fit_failure" else "unavailable": reason}))
    if predictor is not None:
        predictor = predictor.with_fit_binding({"program": spec.record(), "training_receipts": prefix.fit_receipts(),
            "model_source_digest": source_digest, "fit_digest": fit_digest, "logit_offset": 0.,
            "target_contract": prefix.target_contract})
    return _FittedExpert(spec, predictor, None if predictor is None else predictor.digest(),
        fit_digest, source_digest, 0., reason, kind), selection


def fit_existing_nn_public_prefix(compiler, prefix, profile, source, *, output,
                                  policy, device="cpu", source_preparation=None,
                                  context=None):
    """Fit an authorized positive K with the original source and matched scratch.

    Complete-record zero-phase validation uses the adapter's declared acquisition
    order. A one-record C retains the mature owner's explicit fixed-budget
    fallback. It never splits already filtered C into pseudo-independent data.
    """
    prefix = compiler.assert_original_prefix(prefix)
    if source.digest != LEAVE_H_SOURCE_SHA256:
        raise ValueError("existing leave-H learner requires its original admitted source bytes")
    if prefix.budget <= 0:
        raise ValueError("this target-learning slice requires a positive authorized C budget")
    data = prefix.trial_datasets.get(profile.digest)
    if data is None:
        raise ValueError("authorized C has no epochs in the declared original NN input profile")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    subject = data.identity_table.local_subject_ids[0]
    preparation = source_preparation or ScopedSourceCheckpointPreparation()
    prepared = preparation.prepare(source.content, data, subject)
    prepared.assert_library_exclusion(target_membership=dataset_study_membership(data))
    scoped_records = tuple(record for record in compiler.sources
        if record.record_id in prefix.target_contract["record_scopes"])
    order = {"run_order": tuple(dict.fromkeys(record.run_id for record in scoped_records)),
        "session_order": tuple(dict.fromkeys(record.session_id for record in scoped_records))}
    capture = _CapturedFits(output / "epoch_selection")
    experts, selections = [], {}
    for recipe in policy.recipes():
        expert, selection = _neural_expert(prefix, profile, recipe, source, prepared, subject,
            policy, capture, device=device, recording_order=order)
        experts.append(expert)
        selections[expert.program.expert_id] = selection
    # The mature classical fitter selects the actually observed fixed channel
    # union. Its frozen waveform_channels records exclude the unobserved PO8.
    classics = (("waveform_shrinkage_lda_Cz6", waveform_lda_recipe(batch_size=policy.batch_size, seed=policy.target_seed)),
        ("xdawn_erp_covariance_Cz6", erp_target_recipe("xdawn_erp", batch_size=policy.batch_size, seed=policy.target_seed)))
    for name, recipe in classics:
        spec = CalibrationExpertProgram(name, "trial", recipe=recipe, profile=profile)
        if data.n_epochs < 3 or set(np.unique(data.y)) != {0, 1}:
            reason, kind = "insufficient_authorized_trial_classes_or_rows", "not_available"
        else:
            try:
                experts.append(_fit_expert(spec, data, prefix.calibration_records, prefix.target_contract,
                    None, prefix.fit_receipts(), prefix.record(), device="cpu", target_contract=prefix.target_contract))
                continue
            except (ExpectedSubjectError, UnidentifiableFitError, np.linalg.LinAlgError) as error:
                reason, kind = f"{type(error).__name__}: {error}", "fit_failure"
        digest = semantic_sha256({"program": spec.record(), "prefix": prefix.record(),
            "fit_failure" if kind == "fit_failure" else "unavailable": reason})
        experts.append(_FittedExpert(spec, None, None, digest, None, 0., reason, kind))
    specs = tuple(expert.program for expert in experts)
    roster = EnsembleProgram.equal_weights("existing_nn_target_roster", tuple(s.expert_id for s in specs))
    fitted = FittedEnsemble(EnsembleScorer(roster), experts, prefix.fit_receipts(), target_contract=prefix.target_contract)
    sources = {expert.program.expert_id: source for expert in experts if expert.program.recipe.initialization in {"source", "scratch"}}
    selection_record = {"schema": "n2p3_existing_nn_public_epoch_selection/1", "method": METHOD,
        "policy": policy.record(), "source_checkpoint_sha256": source.digest,
        "prefix_input": prefix_input_identity(prefix), "selections": selections, "captured_fits": capture.rows,
        "source_preparation": preparation.record(), "source_fit_calls": 0}
    fit_context = {**(context or {}), "source_seed": LEAVE_H_SOURCE_SEED, "target_seed": policy.target_seed,
        "requested_Ks": [prefix.budget], "phase": "existing_nn_public_target_reuse",
        "epoch_selection_sha256": semantic_sha256(selection_record)}
    recorder = InnerReplayRecorder()
    recorder.record_prefix_programs({"fixed": fitted}, prefix=prefix, expert_programs=specs,
        program=roster, source_checkpoints=sources, context=fit_context)
    atomic_write_json(output / "epoch_selection.json", selection_record)
    capture.clear()
    return ExistingNNPublicFit(prefix, compiler, {"fixed": fitted}, sources, recorder, fit_context, selection_record)


def score_persist_replay_existing_nn(fit, future_views, *, output, device="cpu"):
    """Publish original typed evidence and replay its entire requested future."""
    views = tuple(future_views)
    if ([view.record_id for view in views] != list(fit.context["requested_decision_ids"])
            or len({view.record_id for view in views}) != len(views)):
        raise ValueError("future views differ from the complete original requested selection roster")
    rows = []
    for view in views:
        evidence = FittedEnsemble.score_programs(fit.programs, view)["fixed"]
        if any(not isinstance(value, CandidateEvidence) for value in evidence.values()):
            raise TypeError("original trial scoring did not return typed candidate evidence")
        fit.recorder.record_prefix_future_score("fixed", view=view, evidence=evidence)
        rows.append({"record_id": view.record_id, "decision_id": view.decision_id,
            "physical_support_digest": view.physical_support_digest,
            "evidence": {name: value.record() for name, value in evidence.items()}})
    directory = Path(output)
    package = directory / "package"
    persist_prefix_inner_replay(fit.prefix, fit.programs, package, recorder=fit.recorder,
        source_checkpoints=fit.source_checkpoints, context=fit.context, device=device)
    replay = replay_prefix_inner_replay(package, prefix=fit.prefix, future_views=views,
        source_checkpoints=fit.source_checkpoints, context=fit.context, device=device,
        expected_record_sha256=fit.recorder.prefix_receipt_sha256, prefix_compiler=fit.compiler)
    if len(replay["scores"]) != len(views) or replay["exact_score_checks"] != len(views):
        raise ValueError("saved parent replay did not consume the full original request roster")
    selection_replay = replay_existing_nn_epoch_selection(directory, compiler=fit.compiler, prefix=fit.prefix)
    result = {"schema": "n2p3_existing_nn_public_prediction/1", "method": METHOD,
        "actual_K": fit.prefix.budget, "requested_decisions": len(views), "rows": rows,
        "prefix_replay_receipt_sha256": fit.recorder.prefix_receipt_sha256,
        "parent_replay": plain_metadata(replay), "epoch_selection_sha256": semantic_sha256(fit.selection),
        "epoch_selection_replay": selection_replay}
    atomic_write_json(directory / "predictions.json", result)
    return result


def replay_existing_nn_epoch_selection(directory, *, compiler, prefix):
    """Check saved inner state/chronology and final refit without another fit."""
    from train.early_stopping import PatienceStopper

    directory = Path(directory)
    prefix = compiler.assert_original_prefix(prefix)
    record = json.loads((directory / "epoch_selection.json").read_bytes())
    receipt = json.loads((directory / "package/prefix_inner_replay.json").read_bytes())["record"]
    if (semantic_sha256(record) != receipt["fit_context"]["epoch_selection_sha256"]
            or semantic_sha256(record["prefix_input"]) != semantic_sha256(prefix_input_identity(prefix))):
        raise ValueError("epoch selection differs from its actual original C and prefix producer")
    loaded = FittedEnsemble.load_programs(directory / "package")["fixed"]
    rows = []
    for expert in loaded._experts:
        selection = record["selections"].get(expert.program.expert_id)
        if selection is None:
            continue
        data = prefix.trial_datasets[expert.program.profile.digest]
        allowed = np.ones(data.n_epochs, dtype=bool)
        scoped = tuple(r for r in compiler.sources if r.record_id in prefix.target_contract["record_scopes"])
        order = {"run_order": tuple(dict.fromkeys(r.run_id for r in scoped)),
            "session_order": tuple(dict.fromkeys(r.session_id for r in scoped))}
        split = dataset_calibration_inner_split(data, allowed, recording_order=order)
        captures = [row for row in record["captured_fits"]
            if row["recipe"]["initialization"] == expert.program.recipe.initialization]
        full = [row for row in captures if row["phase"] == "full_prefix"]
        if (len(full) != 1 or full[0]["training_rows"] != np.flatnonzero(allowed).tolist()
                or semantic_sha256(full[0]["history"]) != semantic_sha256(selection["refit_history"])
                or selection["refit_recipe"] != expert.program.recipe.record()
                or selection["selected_epochs"] != expert.program.recipe.epochs):
            raise ValueError("selected neural model differs from its actual all-C refit")
        actual_probability = {key: expert.predictor.record["class_calibration"]["fit_contract"][key]
            for key in full[0]["history"]["probability_fit"]}
        if semantic_sha256(actual_probability) != semantic_sha256(full[0]["history"]["probability_fit"]):
            raise ValueError("saved full-C numerical model differs from the mature refit receipt")
        if selection["policy"]["mode"] != "fixed" and selection["inner_split"] != split.record():
            raise ValueError("epoch selection changed original C recording chronology")
        if selection["selection_evidence"]:
            inner = [row for row in captures if row["phase"] == "inner"]
            if (len(inner) != 1 or inner[0]["training_rows"] != np.flatnonzero(split.training_mask).tolist()
                    or inner[0]["validation_rows"] != np.flatnonzero(split.validation_mask).tolist()
                    or semantic_sha256(inner[0]["history"]) != semantic_sha256(selection["inner_history"])
                    or inner[0]["history"]["initial_state_sha256"] != full[0]["history"]["initial_state_sha256"]):
                raise ValueError("inner state or independent full-C initialization changed")
            path = directory / "epoch_selection" / inner[0]["model"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != inner[0]["model_sha256"]:
                raise ValueError("selected inner numerical state changed")
            model = FrozenTrialPredictor.load(path)
            logits = model.predict_raw_logit(data.X[split.validation_mask], data.trial_channel_mask[split.validation_mask])
            if model.digest() != inner[0]["predictor_digest"] or not np.array_equal(logits, inner[0]["validation_raw_logits"]):
                raise ValueError("selected inner state did not replay its actual validation observations")
            stopper = PatienceStopper(selection["policy"]["patience"], selection["policy"]["min_delta"])
            for loss in selection["inner_history"]["val_losses"]:
                stopper.observe(loss)
            if stopper.best_check != selection["selected_epochs"]:
                raise ValueError("selected epoch differs from the original mature stopping rule")
        elif len(captures) != 1 or selection["selected_epochs"] != selection["requested_recipe"]["epochs"]:
            raise ValueError("fixed-budget fallback changed the declared target-learning budget")
        rows.append({"expert_id": expert.program.expert_id, "selected_epochs": selection["selected_epochs"],
            "selection_evidence": selection["selection_evidence"], "all_C_refit_replayed": True,
            "inner_numeric_state_replayed": selection["selection_evidence"]})
    return {"status": "replayed", "source_fits": 0, "target_fits": 0, "neural_learners": rows}
