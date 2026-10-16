"""Original continuous source evidence for the opt-in joint-history expert.

The NN SourceSeedCatalog keeps its own training contract. This catalogue admits
an existing continuous NPZ and joint codec; neither registration nor resolution
fits a source or a target. Successful source stages and aggregate exits retain
their separate, original evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from research.contracts import semantic_sha256
from research.frozen_metadata import freeze_metadata, plain_metadata

ENTRY_SCHEMA = "n2p3_joint_source_catalog_entry/2"
RECEIPT_SCHEMA = "n2p3_joint_source_resolution/1"
SOURCE_ROLE = "original_source_response_and_noise_parameters"
_SCOPES = {"software_correctness", "public_development", "public_calibration_development"}
_EXCLUSIONS = {"target_participant", "whole_target_dataset"}
_HUEBNER_PUBLIC_ROSTER = {
    "path": "D:/BCI/artifacts/huebner-original-inputs/full-original-13/roster.json",
    "bytes": 83686,
    "sha256": "ce370b4152a8e6f9fba9cad6efb357db3503fde97d89c219962836ce35ba08fd",
}
_ADMISSION_OWNER = object()


def _artifact(path, content=None):
    path = Path(path).resolve()
    content = path.read_bytes() if content is None else content
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def _read_artifact(receipt, base, *, prefix=False):
    path = Path(base) / receipt["path"]
    size = receipt["bytes"]
    if type(size) is not int or size < 1:
        raise ValueError("joint source artifact requires its actual positive byte count")
    with path.open("rb") as stream:
        content = stream.read(size if prefix else size + 1)
    if len(content) != size or hashlib.sha256(content).hexdigest() != receipt["sha256"]:
        raise ValueError("joint source artifact bytes differ from the original receipt")
    return content


def _source_summary(source, model):
    """Relate the codec's declared cohort to its actual embedded NPZ producer."""
    binding = source.parameters.binding
    if (model.provenance["fit_scope"] != "source_training"
            or model.digest() != binding["source_model_digest"]
            or semantic_sha256(model._metadata()) != semantic_sha256(binding["source_model"])):
        raise ValueError("joint source metadata differs from its actual continuous producer")
    if any(binding[name] != 0 for name in ("noise_parameter_fits_after_original_source_model",
            "target_parameter_fits", "target_labels_read")):
        raise ValueError("joint frozen source must retain zero post-source parameter fits and target labels")
    prefixes = binding["source_prefixes"]
    actual_views = binding.get("actual_source_record_views")
    fitted = model.provenance["records"]
    if actual_views is None:
        if len(prefixes) != len(fitted):
            raise ValueError("joint source lacks the original one-record prefix roster")
        views = [{**row["target_contract"], "record_id": record["record_id"],
            "raw_scope": row["target_contract"]["raw_scope"]}
            for row, record in zip(prefixes, fitted, strict=True)]
    else:
        views = actual_views
    if [row["record_id"] for row in views] != [row["record_id"] for row in fitted]:
        raise ValueError("joint source record order differs from its actual source fit")
    records = []
    for view, record in zip(views, fitted, strict=True):
        support = list(view["raw_scope"])
        if support != [0, record["n_samples"]]:
            raise ValueError("joint source native support differs from the actual fitted recording")
        records.append({"record_id": record["record_id"], "source_dataset_id": view["source_dataset_id"],
            "participant_key": view["participant_key"], "source_sha256": view["source_sha256"],
            "read_sample_support": support, "input_sha256": record["input_sha256"],
            "n_events": record["n_events"], "n_targets": record["n_targets"]})
    if (len({row["record_id"] for row in records}) != len(records)
            or len({row["source_sha256"] for row in records}) != len(records)):
        raise ValueError("joint source repeats an original acquisition record")
    return freeze_metadata({"parameter_authority_sha256": source.parameters.digest,
        "source_model_digest": model.digest(), "geometry": source.parameters.noise.geometry.record(),
        "source_records": records, "source_prefixes": prefixes,
        "beta_prior_sha256": binding["beta_prior_sha256"], "basis_sha256": binding["basis_sha256"],
        "excluded_target_participant": binding["excluded_target_participant"],
        "source_fit_scope": model.provenance["fit_scope"],
        "noise_parameter_fits_after_original_source_model": binding["noise_parameter_fits_after_original_source_model"],
        "target_parameter_fits": binding["target_parameter_fits"], "target_labels_read": binding["target_labels_read"]})


def _assert_exclusion(summary, exclusion):
    scope = exclusion["scope"]
    participant, study = exclusion["target_participant_key"], exclusion["target_study"]
    if scope not in _EXCLUSIONS or not participant or not study:
        raise ValueError("joint source requires its explicit actual target and exclusion scope")
    if participant != summary["excluded_target_participant"]:
        raise ValueError("joint source belongs to another original excluded target participant")
    records = summary["source_records"]
    if any(row["participant_key"] == participant for row in records):
        raise ValueError("joint source contains its excluded target participant")
    if scope == "whole_target_dataset" and any(row["source_dataset_id"] == study for row in records):
        raise ValueError("joint whole-study source contains the actual target dataset")


def _read_fit_evidence(evidence, base, model_receipt, codec_receipt, source_evidence_scope):
    """A completed source stage can belong to an aggregate actor that failed."""
    raw = _read_artifact(evidence["events"], base, prefix=True)
    rows = [json.loads(line) for line in raw.splitlines()]
    start, end = evidence["started_event_index"], evidence["completed_event_index"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end < len(rows)
            or end != len(rows) - 1):
        raise ValueError("joint source events require an ordered original fit-start/save prefix")
    started, completed = rows[start], rows[end]
    scopes = {row["source_evidence_scope"] for row in rows if "source_evidence_scope" in row}
    if any(row.get("source_fit_scope") == "explicit_participant_excluded_software_programme" for row in rows):
        scopes.add("software_correctness")
    if scopes != {source_evidence_scope}:
        raise ValueError("joint source evidence scope differs from the original executed source programme")
    if (started["phase"] != "source_fit_started" or completed["phase"] != "source_saved"
            or started["pid"] != completed["pid"] or started["time"] > completed["time"]
            or started["actual_public_source_fit_attempts"] != 1
            or completed["actual_successful_public_source_fits"] != 1
            or completed["noise_refits_after_original_source"] != 0):
        raise ValueError("joint source requires the actual completed original source fit event")
    for name, receipt in (("source_model", model_receipt), ("frozen_parameters", codec_receipt)):
        if any(completed[name][key] != receipt[key] for key in ("sha256", "bytes")):
            raise ValueError("joint source completed event belongs to different source artifacts")
    terminal = json.loads(_read_artifact(evidence["aggregate_terminal"], base))
    if (terminal["observed_original_fitter_pid"] != completed["pid"]
            or terminal["source_successful_fit_count"] != 1
            or terminal["source_model_sha256"] != model_receipt["sha256"]
            or terminal["frozen_source_sha256"] != codec_receipt["sha256"]
            or terminal["source_parameter_refits_after_original_source"] != 0
            or terminal["source_unchanged"] is not True or not terminal["source_after"]):
        raise ValueError("joint source stage differs from its original aggregate execution evidence")
    return freeze_metadata({"started": started, "completed": completed,
        "aggregate_terminal": terminal, "evidence": evidence,
        "source_fit_count": 1, "export_source_fit_count": 0, "actual_source_seed": None,
        "source_evidence_scope": source_evidence_scope})


def _verify_original_cohort(source, model, source_cohort, target_participant):
    """Reuse original acquisition issuers; no response/noise parameter fitting."""
    from research.joint_flat_authority import _original_record_source_cohort, _source_cohort
    from transfer.ensemble_views import CalibrationPrefixCompiler, RecordCalibrationPrefixCompiler
    from transfer.noise_model import array_digest

    source_cohort = tuple(source_cohort)
    if not source_cohort:
        raise ValueError("joint source export requires its actual original source cohort")
    compilers = {type(compiler) for compiler, _ in source_cohort}
    if compilers == {RecordCalibrationPrefixCompiler}:
        cohort, records, views, geometry = _original_record_source_cohort(source_cohort, target_participant)
    elif compilers == {CalibrationPrefixCompiler}:
        cohort, geometry = _source_cohort(source_cohort, target_participant)
        records = tuple(prefix.raw for _, prefix in cohort)
        views = tuple(prefix.target_contract for _, prefix in cohort)
    else:
        raise TypeError("joint source export requires a supported original acquisition issuer")
    if geometry != source.parameters.noise.geometry:
        raise ValueError("joint source original cohort changed its actual native physical geometry")
    fitted = model.provenance["records"]
    if [row["record_id"] for row in fitted] != [raw.record_id for raw in records]:
        raise ValueError("joint source original cohort differs from the actual source fit record order")
    summary = _source_summary(source, model)
    for raw, actual, saved in zip(records, views, summary["source_records"], strict=True):
        if (array_digest(raw.samples, raw.onsets, raw.target_mask) != saved["input_sha256"]
                or len(raw.samples) != saved["read_sample_support"][1]
                or len(raw.onsets) != saved["n_events"] or int(raw.target_mask.sum()) != saved["n_targets"]
                or any(actual[key] != saved[key] for key in
                    ("participant_key", "source_dataset_id", "source_sha256"))):
            raise ValueError("joint source original samples, labels, dose or identities differ from its source fit")
    if len(cohort) != len(summary["source_prefixes"]):
        raise ValueError("joint source original prefix roster differs from its parameter authority")
    return summary


def _numeric_source_identity(source):
    """Target export changes provenance, while all fitted numerical laws stay."""
    from transfer.noise_model import array_digest

    parameters = source.parameters
    noise = parameters.noise
    return {"original_model_sha256": hashlib.sha256(source.original_model_content).hexdigest(),
        "prior_mean_and_root_sha256": array_digest(parameters.prior.mean, parameters.prior.root),
        "basis_sha256": array_digest(parameters.basis), "geometry": noise.geometry.record(),
        "temporal_law": [item.record() for item in noise.temporal],
        "innovation_covariance_sha256": array_digest(noise.innovation_covariance),
        "startup_root_sha256": array_digest(noise.compiled_prefix_root),
        "conditioning_samples": noise.conditioning_samples,
        "degrees_of_freedom": str(parameters.hyperparameters.degrees_of_freedom),
        "relative_noise_shape": parameters.hyperparameters.relative_noise_shape}


def _entry_identity(entry):
    return semantic_sha256({key: entry[key] for key in ("schema", "source_id", "actual_source_seed",
        "source_evidence_scope", "exclusion", "codec", "original_model", "source_summary", "fit_stage", "export")})


def _export_receipt(source, *, fit_stage, exclusion, kind):
    return {"schema": "n2p3_joint_readonly_source_export/1", "kind": kind,
        "fit_stage_sha256": semantic_sha256(fit_stage),
        "original_model_sha256": fit_stage["original_model"]["sha256"],
        "original_fit_codec_sha256": fit_stage["codec"]["sha256"],
        "export_codec_sha256": source.digest, "parameter_authority_sha256": source.parameters.digest,
        "numeric_source": _numeric_source_identity(source), "exclusion": exclusion,
        "exporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_refits": 0, "target_fits": 0}


def issue_joint_source_entry(source_id, *, codec_path, original_model_path, source_cohort,
        target_contract, exclusion_scope, event_log_path, started_event_index, completed_event_index,
        aggregate_terminal_path, source_evidence_scope, _source_preparation=None):
    """Read-only export from real source arrays and original fit-stage evidence.

    The event receipt ends at the original source_saved line; later aggregate
    progress can append without changing that stage. Terminal bytes are kept
    verbatim, including a failed monitor or unknown scientific exit.
    """
    from transfer.continuous_decision import ContinuousDecisionModel

    codec, original = _artifact(codec_path), _artifact(original_model_path)
    codec_bytes, original_bytes = Path(codec["path"]).read_bytes(), Path(original["path"]).read_bytes()
    preparation = _source_preparation or import_module(
        "transfer.joint_flat_programme").FrozenJointFlatSourcePreparation()
    source = preparation.prepare(codec_bytes, expected_sha256=codec["sha256"])
    if source.original_model_content != original_bytes:
        raise ValueError("joint source codec does not contain the original declared source NPZ")
    model = ContinuousDecisionModel.load(io.BytesIO(original_bytes))
    summary = _verify_original_cohort(source, model, source_cohort, target_contract["participant_key"])
    exclusion = {"scope": exclusion_scope, "target_study": target_contract["source_dataset_id"],
        "target_participant_key": target_contract["participant_key"]}
    _assert_exclusion(summary, exclusion)
    source.assert_target(target_contract)
    lines = Path(event_log_path).read_bytes().splitlines(keepends=True)
    if type(completed_event_index) is not int or not 0 <= completed_event_index < len(lines):
        raise ValueError("joint source completed event index is outside its original log")
    events = _artifact(event_log_path, b"".join(lines[:completed_event_index + 1]))
    evidence = {"events": events, "started_event_index": started_event_index,
        "completed_event_index": completed_event_index, "aggregate_terminal": _artifact(aggregate_terminal_path)}
    _read_fit_evidence(evidence, Path.cwd(), original, codec, source_evidence_scope)
    if not isinstance(source_id, str) or not source_id or source_evidence_scope not in _SCOPES:
        raise ValueError("joint source export requires an explicit source ID and evidence scope")
    fit_stage = {"original_model": original, "codec": codec, "fit_evidence": evidence,
        "source_evidence_scope": source_evidence_scope}
    return {"schema": ENTRY_SCHEMA, "source_id": source_id, "actual_source_seed": None,
        "source_evidence_scope": source_evidence_scope, "exclusion": exclusion,
        "codec": codec, "original_model": original, "source_summary": plain_metadata(summary),
        "fit_evidence": evidence, "fit_stage": fit_stage,
        "export": _export_receipt(source, fit_stage=fit_stage, exclusion=exclusion, kind="original_saved_codec")}


def reexport_joint_source_entry(original_entry, source_id, *, source_cohort, target_contract,
        exclusion_scope, output_codec_path, base_directory=None, _source_preparation=None):
    """Use the original saved-model export owner for a new excluded target."""
    from research.joint_flat_authority import (
        source_parameters_from_saved_model,
        source_record_parameters_from_saved_model,
    )
    from transfer.ensemble_views import CalibrationPrefixCompiler, RecordCalibrationPrefixCompiler

    if not isinstance(source_id, str) or not source_id:
        raise ValueError("joint source reexport requires its explicit new source ID")
    base_directory = Path.cwd() if base_directory is None else Path(base_directory)
    catalogue = JointFlatSourceCatalog([original_entry], base_directory=base_directory,
        evidence_scope="software_correctness", _source_preparation=_source_preparation)
    original = catalogue._sources[original_entry["source_id"]]
    stage = original_entry["fit_stage"]
    cohort = tuple(source_cohort)
    compilers = {type(compiler) for compiler, _ in cohort}
    if compilers == {RecordCalibrationPrefixCompiler}:
        export_owner = source_record_parameters_from_saved_model
    elif compilers == {CalibrationPrefixCompiler}:
        export_owner = source_parameters_from_saved_model
    else:
        raise TypeError("joint reexport requires the original registered source cohort")
    parameters, _model = export_owner(cohort, target_participant_key=target_contract["participant_key"],
        degrees_of_freedom=original.parameters.hyperparameters.degrees_of_freedom,
        relative_noise_shape=original.parameters.hyperparameters.relative_noise_shape,
        source_model_path=Path(base_directory) / stage["original_model"]["path"], frozen_source=original)
    programme = import_module("transfer.joint_flat_programme")
    exported = programme.FrozenJointFlatSource.from_parameters(parameters,
        original_model_content=original.original_model_content)
    if semantic_sha256(_numeric_source_identity(exported)) != semantic_sha256(_numeric_source_identity(original)):
        raise ValueError("joint read-only export changed the original fitted numerical source law")
    exclusion = {"scope": exclusion_scope, "target_study": target_contract["source_dataset_id"],
        "target_participant_key": target_contract["participant_key"]}
    summary = _source_summary(exported, _model)
    _assert_exclusion(summary, exclusion)
    exported.assert_target(target_contract)
    with Path(output_codec_path).open("xb") as stream:
        stream.write(exported.content)
    return {"schema": ENTRY_SCHEMA, "source_id": source_id, "actual_source_seed": None,
        "source_evidence_scope": original_entry["source_evidence_scope"], "exclusion": exclusion,
        "codec": _artifact(output_codec_path, exported.content), "original_model": plain_metadata(stage["original_model"]),
        "source_summary": plain_metadata(summary), "fit_evidence": plain_metadata(stage["fit_evidence"]),
        "fit_stage": plain_metadata(stage),
        "export": _export_receipt(exported, fit_stage=stage, exclusion=exclusion, kind="same_NPZ_new_target_zero_fit")}


@dataclass(frozen=True)
class ResolvedJointSource:
    source: object
    receipt: dict


@dataclass(frozen=True)
class _SoftwareSourceIssuer:
    """Explicit software anchor substitute for the array-join contract tests."""
    cohort: tuple
    anchor: dict


@dataclass(frozen=True, init=False)
class _OriginalSourceDataAdmission:
    record: dict
    _owner: object

    def __init__(self, record, owner):
        if owner is not _ADMISSION_OWNER:
            raise TypeError("original source data admission must come from its live issuer")
        object.__setattr__(self, "record", freeze_metadata(record))
        object.__setattr__(self, "_owner", owner)


def _target_identity(target):
    return {key: target.get(key) for key in ("participant_key", "source_dataset_id", "source_sha256",
        "source_record_roster", "channel_names", "channel_types", "sample_rate_hz", "reference", "signal_unit")}


def huebner_original_source_metadata(source_requests):
    """Read the independently admitted original roster and selected declarations.

    This is metadata only. It grants no mathematical/data admission until the
    registered factory and actual source arrays have been joined below.
    """
    roster_bytes = _read_artifact(_HUEBNER_PUBLIC_ROSTER, Path.cwd())
    roster = json.loads(roster_bytes)
    metadata = roster["original_metadata_manifest"]
    original_metadata = Path(metadata["path"]).read_bytes()
    if hashlib.sha256(original_metadata).hexdigest() != metadata["sha256"]:
        raise ValueError("Huebner original metadata differs from the independently admitted roster")
    admitted = {row["subject"]: row for row in roster["participant_declarations"]}
    declarations, receipts = [], []
    for request in source_requests:
        row = admitted[request["subject"]]
        payload = Path(row["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("Huebner source declaration differs from the original admitted participant")
        declaration = json.loads(payload)
        if (declaration["dataset_id"] != "Huebner2017"
                or declaration["source_format"] != "original_vmrk_plus_NEMAR_BDF"
                or declaration["source_subject_id"] != request["subject"]):
            raise ValueError("Huebner source declaration changed its registered original identity")
        selectors = request["record_selectors"]
        selected = []
        for selector in selectors:
            matches = [record for record in declaration["record_sources"]
                if all(record[key] == selector[key] for key in ("block", "run"))]
            if len(matches) != 1:
                raise ValueError("Huebner source selector requires one exact original acquisition")
            selected.append(matches[0])
        if not selected or len({semantic_sha256(row) for row in selected}) != len(selected):
            raise ValueError("Huebner source requires its nonempty unique original record selection")
        declaration = {**declaration, "record_sources": selected}
        declaration.pop("evaluation_session_ids", None)
        declarations.append((declaration, request["label_budget"]))
        receipts.append({"participant_declaration": _artifact(row["path"], payload), "request": request})
    return tuple(declarations), freeze_metadata({"roster": _HUEBNER_PUBLIC_ROSTER,
        "original_metadata_manifest": _artifact(metadata["path"], original_metadata),
        "source_declarations": receipts})


def admit_huebner_joint_source_data(entry, *, target_contract, exclusion_scope,
        source_requests=(), authorization_preprocessing=None, response_seconds=1.2, pre_roll_seconds=.2,
        base_directory=None, _software_issuer=None, _source_preparation=None):
    """New data admission; the old fit's evidence scope remains its old value.

    The production route is rooted in the separately reviewed original Huebner
    roster. The private software substitute always yields software evidence.
    A serialized admission record is descriptive and cannot replace this owner.
    """
    from transfer.continuous_decision import ContinuousDecisionModel
    from transfer.ensemble_views import CalibrationPrefixCompiler

    base_directory = Path.cwd() if base_directory is None else Path(base_directory)
    catalogue = JointFlatSourceCatalog([entry], base_directory=base_directory,
        evidence_scope="software_correctness", _source_preparation=_source_preparation)
    source = catalogue._sources[entry["source_id"]]
    if _software_issuer is None:
        declarations, anchor = huebner_original_source_metadata(source_requests)
        if not declarations or authorization_preprocessing is None:
            raise ValueError("Huebner data admission requires actual source requests and its preprocessing contract")
        cohort = []
        for declaration, budget in declarations:
            compiler = CalibrationPrefixCompiler.from_dataset_source(declaration, budgets=(budget,), profiles=(),
                authorization_preprocessing=authorization_preprocessing, response_seconds=response_seconds,
                pre_roll_seconds=pre_roll_seconds)
            cohort.append((compiler, compiler.compile(budget)))
        data_scope = "original_public_data_development"
    else:
        if type(_software_issuer) is not _SoftwareSourceIssuer:
            raise TypeError("software data admission requires its explicit software issuer substitute")
        cohort, anchor = _software_issuer.cohort, _software_issuer.anchor
        data_scope = "software_correctness"
    model = ContinuousDecisionModel.load(io.BytesIO(source.original_model_content))
    summary = _verify_original_cohort(source, model, cohort, target_contract["participant_key"])
    exclusion = {"scope": exclusion_scope, "target_study": target_contract["source_dataset_id"],
        "target_participant_key": target_contract["participant_key"]}
    if semantic_sha256(exclusion) != semantic_sha256(entry["exclusion"]):
        raise ValueError("source data admission changed its current target or exclusion programme")
    _assert_exclusion(summary, exclusion)
    source.assert_target(target_contract)
    record = {"schema": "n2p3_joint_original_source_data_admission/1", "source_id": entry["source_id"],
        "entry_sha256": _entry_identity(entry), "data_admission_scope": data_scope,
        "source_programme_evidence_scope": entry["source_evidence_scope"],
        "original_fit_stage_sha256": semantic_sha256(entry["fit_stage"]),
        "original_model_sha256": entry["original_model"]["sha256"], "codec_sha256": source.digest,
        "source_summary_sha256": semantic_sha256(summary), "source_records": summary["source_records"],
        "source_prefixes": summary["source_prefixes"], "geometry": summary["geometry"],
        "target_identity": _target_identity(target_contract), "exclusion": exclusion, "original_data_anchor": anchor,
        "registered_factory_sha256": [compiler.adapter.original_factory_source_sha256
            for compiler, _ in cohort],
        "admission_issuer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_refits": 0, "target_fits": 0}
    return _OriginalSourceDataAdmission(record, _ADMISSION_OWNER)


class JointFlatSourceCatalog:
    """Admit immutable original source states; rebind every actual target call."""

    def __init__(self, entries, *, base_directory, evidence_scope, data_admissions=(), _source_preparation=None):
        from transfer.continuous_decision import ContinuousDecisionModel

        if evidence_scope not in _SCOPES:
            raise ValueError("joint source consumer evidence scope is unsupported")
        self._entries, self._sources = {}, {}
        self._data_admissions = {}
        self.evidence_scope = evidence_scope
        for admission in data_admissions:
            if type(admission) is not _OriginalSourceDataAdmission or admission._owner is not _ADMISSION_OWNER:
                raise TypeError("joint data admission requires the actual private original-data issuer owner")
            key = admission.record["source_id"]
            if key in self._data_admissions:
                raise ValueError("joint source has duplicate current data admissions")
            self._data_admissions[key] = admission.record
        preparation = _source_preparation or import_module(
            "transfer.joint_flat_programme").FrozenJointFlatSourcePreparation()
        for declared in entries:
            entry = freeze_metadata(declared)
            source_id = entry["source_id"]
            if (entry["schema"] != ENTRY_SCHEMA or not isinstance(source_id, str) or not source_id
                    or source_id in self._entries or entry["actual_source_seed"] is not None):
                raise ValueError("joint source catalogue requires unique typed IDs and its actual null source seed")
            scope = entry["source_evidence_scope"]
            if scope not in _SCOPES:
                raise ValueError("joint source programme evidence scope is unsupported")
            admission = self._data_admissions.get(source_id)
            if evidence_scope != "software_correctness" and (
                    admission is None or admission["data_admission_scope"] != "original_public_data_development"):
                raise ValueError("joint public consumer requires a new original-public-data admission")
            if admission is not None and admission["entry_sha256"] != _entry_identity(entry):
                raise ValueError("joint current data admission belongs to different source/export bytes")
            _assert_exclusion(entry["source_summary"], entry["exclusion"])
            codec = _read_artifact(entry["codec"], base_directory)
            original = _read_artifact(entry["original_model"], base_directory)
            stage = entry["fit_stage"]
            if (stage["source_evidence_scope"] != scope
                    or semantic_sha256(stage["original_model"]) != semantic_sha256(entry["original_model"])
                    or semantic_sha256(stage["fit_evidence"]) != semantic_sha256(entry["fit_evidence"])):
                raise ValueError("joint export changed its original source stage or source NPZ")
            original_codec = _read_artifact(stage["codec"], base_directory)
            producer = _read_fit_evidence(stage["fit_evidence"], base_directory,
                stage["original_model"], stage["codec"], scope)
            source = preparation.prepare(codec, expected_sha256=entry["codec"]["sha256"])
            if source.original_model_content != original:
                raise ValueError("joint source codec contains a different original source NPZ")
            model = ContinuousDecisionModel.load(io.BytesIO(original))
            summary = _source_summary(source, model)
            if semantic_sha256(summary) != semantic_sha256(entry["source_summary"]):
                raise ValueError("joint source catalogue changed the original parameter or cohort authority")
            _assert_exclusion(summary, entry["exclusion"])
            original_source = (source if stage["codec"]["sha256"] == entry["codec"]["sha256"] else
                preparation.prepare(original_codec, expected_sha256=stage["codec"]["sha256"]))
            numeric = _numeric_source_identity(source)
            if semantic_sha256(numeric) != semantic_sha256(_numeric_source_identity(original_source)):
                raise ValueError("joint export changed the original fitted numerical source law")
            exported = entry["export"]
            expected_export = {"schema": "n2p3_joint_readonly_source_export/1",
                "fit_stage_sha256": semantic_sha256(stage), "original_model_sha256": entry["original_model"]["sha256"],
                "original_fit_codec_sha256": stage["codec"]["sha256"], "export_codec_sha256": source.digest,
                "parameter_authority_sha256": source.parameters.digest, "numeric_source": numeric,
                "exclusion": entry["exclusion"], "source_refits": 0, "target_fits": 0}
            if any(semantic_sha256(exported[key]) != semantic_sha256(value) for key, value in expected_export.items()):
                raise ValueError("joint read-only export receipt differs from its original source and current codec")
            self._entries[source_id] = freeze_metadata({**entry, "producer": producer,
                "producer_digest": semantic_sha256(producer), "data_admission": admission,
                "consumer_evidence_scope": evidence_scope})
            self._sources[source_id] = source

    def record(self):
        return tuple(self._entries.values())

    def resolve(self, source_id, *, target_contract, exclusion_scope, pipeline_pair=None):
        if source_id not in self._entries:
            raise ValueError("joint source binding names an unregistered source ID")
        entry, source = self._entries[source_id], self._sources[source_id]
        expected = {"scope": exclusion_scope, "target_study": target_contract["source_dataset_id"],
            "target_participant_key": target_contract["participant_key"]}
        if semantic_sha256(expected) != semantic_sha256(entry["exclusion"]):
            raise ValueError("joint source binding differs from this actual target or exclusion programme")
        source.assert_target(target_contract)
        admission = self._data_admissions.get(source_id)
        if admission is not None and semantic_sha256(admission["target_identity"]) != semantic_sha256(_target_identity(target_contract)):
            raise ValueError("joint current data admission differs from this actual target identity")
        pair = None if pipeline_pair is None else tuple(pipeline_pair)
        if pair is not None and (len(pair) != 2 or any(type(seed) is not int or seed < 0 for seed in pair)):
            raise ValueError("joint pipeline coordinates require the declared nonnegative pair")
        receipt = freeze_metadata({"schema": RECEIPT_SCHEMA, "source_id": source_id,
            "role": SOURCE_ROLE, "checkpoint_sha256": source.digest,
            "original_model_sha256": entry["original_model"]["sha256"],
            "parameter_authority_sha256": entry["source_summary"]["parameter_authority_sha256"],
            "producer_digest": entry["producer_digest"], "exclusion": expected,
            "source_evidence_scope": entry["source_evidence_scope"],
            "data_admission": admission, "readonly_export": entry["export"],
            "consumer_evidence_scope": self.evidence_scope, "actual_source_seed": None,
            "pipeline_pair": pair, "source_randomness": "deterministic_original_continuous_source_fit",
            "resolution_source_fits": 0, "resolution_target_fits": 0})
        return ResolvedJointSource(source, receipt)


def resolve_joint_candidate_sources(expert_programs, source_bindings, catalog, *, target_contract,
        pipeline_pair=None):
    """Resolve just the joint subset; the established NN owner keeps its sources."""
    joints = tuple(spec for spec in expert_programs if spec.family == "joint_flat")
    specs = {spec.expert_id: spec for spec in joints}
    if len(specs) != len(joints):
        raise ValueError("joint source resolver requires the complete unique expert roster")
    if set(source_bindings) != set(specs) or any(not isinstance(value, str) or not value for value in source_bindings.values()):
        raise ValueError("joint candidate sources require exactly one declared binding for every joint expert")
    if any(spec.source_checkpoint_role != SOURCE_ROLE for spec in specs.values()):
        raise ValueError("joint candidate source binding differs from its actual expert family role")
    resolved = {name: catalog.resolve(source_bindings[name], target_contract=target_contract,
        exclusion_scope=spec.joint_flat.source_exclusion_scope, pipeline_pair=pipeline_pair)
        for name, spec in specs.items()}
    return {name: row.source for name, row in resolved.items()}, {name: row.receipt for name, row in resolved.items()}
