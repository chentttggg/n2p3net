"""Complete original source bytes, shared on disk and admitted per ingress.

A reference locates evidence. The original checkpoint and target/compiler gates
issue its authority; neither a digest nor a caller's checked flag does so.
"""
from __future__ import annotations

import hashlib
import io
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from research.frozen_metadata import freeze_metadata, plain_metadata

SOURCE_CONTRACT_REF_SCHEMA = "n2p3_source_contract_reference/1"
SOURCE_STORE_SCHEMA = "n2p3_complete_source_store/1"
_SOURCE_WRITER = ContextVar("complete_source_metadata_writer", default=None)


def current_source_metadata_store():
    return _SOURCE_WRITER.get()


@contextmanager
def source_metadata_scope(store):
    """Choose shared storage only inside the actual scientific writer scope."""
    token = _SOURCE_WRITER.set(store)
    try:
        yield
    finally:
        _SOURCE_WRITER.reset(token)


def source_contract_reference(content, training_contract_digest, target_binding, evidence_scope):
    if not isinstance(content, bytes) or not content:
        raise ValueError("source authority requires complete immutable checkpoint bytes")
    return freeze_metadata({
        "schema": SOURCE_CONTRACT_REF_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "checkpoint_bytes": len(content),
        "training_contract_digest": training_contract_digest,
        "target_binding": target_binding,
        "consumer_evidence_scope": evidence_scope,
    })


def _reference(record):
    required = {"schema", "checkpoint_sha256", "checkpoint_bytes", "training_contract_digest",
                "target_binding", "consumer_evidence_scope"}
    if set(record) != required or record["schema"] != SOURCE_CONTRACT_REF_SCHEMA:
        raise ValueError("unsupported or incomplete source authority reference")
    for name in ("checkpoint_sha256", "training_contract_digest"):
        value = record[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid original source authority digest")
    if type(record["checkpoint_bytes"]) is not int or record["checkpoint_bytes"] < 1:
        raise ValueError("source reference requires the actual complete checkpoint size")
    return record


class SourceMetadataStore:
    """Publish each original checkpoint once through the existing disk owner."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def publish(self, content):
        from research.source_execution import process_file_lock

        if not isinstance(content, bytes) or not content:
            raise ValueError("source store requires complete immutable checkpoint bytes")
        digest = hashlib.sha256(content).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (digest + ".pt")
        with process_file_lock(path.with_suffix(".lock")):
            self._publish_locked(path, content, digest)
        return digest

    def _publish_locked(self, path, content, digest):
        from research.source_execution import current_input_admission

        if path.exists():
            if path.is_symlink() or path.read_bytes() != content:
                raise ValueError("published original source checkpoint bytes changed")
            return
        temporary = self.directory / ("." + digest + "." + uuid4().hex + ".tmp")
        admission = current_input_admission()
        from contextlib import nullcontext

        allocation = (admission.artifact_writer(path, len(content), temporary_paths=(temporary,))
                      if admission is not None else nullcontext())
        with allocation:
            try:
                with temporary.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if path.is_symlink() or path.read_bytes() != content:
                        raise ValueError("concurrent source publisher supplied different bytes") from None
            finally:
                temporary.unlink(missing_ok=True)

    def read(self, reference):
        reference = _reference(reference)
        path = self.directory / (reference["checkpoint_sha256"] + ".pt")
        if path.is_symlink():
            raise ValueError("source authority blob must be an original regular file")
        return path.read_bytes()


class _SourceMetadataAdmission:
    """Private transaction: original bytes and full procedure admitted once.

    Public model/report loads create a fresh instance. No target EEG, fitted BN,
    target statistics or caller-created proof can enter this cache.
    """

    def __init__(self, *, store=None, embedded=None):
        self._store = store
        self._embedded = {} if embedded is None else embedded
        self._sources = {}
        self.source_admissions = 0
        self.target_bindings = 0

    def resolve(self, reference):
        from data.epochs import PreprocessingSpec, assert_physical_epoch_dataset
        from data.identity import DatasetIdentityTable
        from data.study import study_ids_for_identity
        from research.source_selection import validate_source_procedure
        from transfer.checkpoint import (
            _AdmittedTransferSource,
            load_checkpoint_payload,
        )
        from transfer.ensemble_views import TrialCheckpointTarget

        reference = _reference(reference)
        binding = plain_metadata(reference["target_binding"])
        if set(binding) != {"subject", "identity", "study_membership", "preprocessing",
                            "channel_names", "provenance"}:
            raise ValueError("source reference lacks its actual target input authority")
        identity = DatasetIdentityTable.from_payload(binding["identity"])
        target_studies = study_ids_for_identity(identity, binding["study_membership"])
        key = reference["checkpoint_sha256"]
        scope = reference["consumer_evidence_scope"]
        source = self._sources.get(key)
        if source is None:
            content = self._embedded.get(key)
            if content is None:
                if self._store is None:
                    raise ValueError("shared source model requires its complete metadata store")
                content = self._store.read(reference)
            if (not isinstance(content, bytes) or len(content) != reference["checkpoint_bytes"]
                    or hashlib.sha256(content).hexdigest() != key):
                raise ValueError("original source authority bytes or size changed")
            payload = load_checkpoint_payload(io.BytesIO(content))
            kind = "selection_refit" if payload.get("source_epoch_selection") is not None else "fixed_recipe"
            for study in target_studies:
                validate_source_procedure(payload, source_seed=payload["training_contract"]["seed"],
                    outer_target_study=study, evidence_scope=scope, kind=kind)
            excluded = payload.get("source_fit_ledger", {}).get("excluded_source_datasets")
            source = _AdmittedTransferSource(content, _decoded_payload=payload)
            source._storage_excluded_studies = None if excluded is None else frozenset(excluded)
            source._storage_scope = scope
            self._sources[key] = source
            self.source_admissions += 1
        if len(source.content) != reference["checkpoint_bytes"]:
            raise ValueError("original source authority bytes or size changed")
        if (source._storage_scope != scope
                or source.metadata["training_contract_digest"] != reference["training_contract_digest"]):
            raise ValueError("source reference changed its original contract or consumer scope")
        target = TrialCheckpointTarget(PreprocessingSpec(**binding["preprocessing"]),
            tuple(binding["channel_names"]), identity, binding["provenance"])
        assert_physical_epoch_dataset(target)
        bound = source.bind(target, binding["subject"])
        bound.assert_library_exclusion(target_membership=binding["study_membership"])
        if source._storage_excluded_studies is not None and not set(target_studies) <= source._storage_excluded_studies:
            raise ValueError("target study differs from the original source procedure exclusion")
        self.target_bindings += 1
        return source.metadata["training_contract"]

    def content(self, reference):
        if reference["checkpoint_sha256"] not in self._sources:
            self.resolve(reference)
        return self._sources[reference["checkpoint_sha256"]].content

    def validate_predictor(self, reference, record, state):
        from dataclasses import asdict

        import numpy as np
        import torch

        from research.contracts import semantic_sha256
        from transfer.binary import plain_trunk_configuration

        source = self._sources[reference["checkpoint_sha256"]]
        trunk = source._trunk
        expected_input = {"kind": "n2p3", "configuration": asdict(plain_trunk_configuration(trunk)),
            "architecture": trunk.architecture_record(), "n_channels": trunk.n_channels,
            "n_times": trunk.n_times, "sfreq": trunk.sfreq, "tmin_s": trunk.tmin_s, "pooling_mode": trunk.pooling_mode}
        if semantic_sha256({name: record[name] for name in expected_input}) != semantic_sha256(expected_input):
            raise ValueError("source-only model programme differs from its original checkpoint")
        if semantic_sha256(record["source_calibration"]) != semantic_sha256(source.metadata["source_calibration"]):
            raise ValueError("source-only probability calibration differs from its original checkpoint")
        expected = {"trunk." + name: value for name, value in source._trunk.state_dict().items()}
        expected.update(input_mean=torch.as_tensor(source._input_statistics[0]),
                        input_std=torch.as_tensor(source._input_statistics[1]))
        if set(state) != set(expected) or any(value.dtype != state[name].dtype
                or not np.array_equal(value.numpy(), state[name].numpy()) for name, value in expected.items()):
            raise ValueError("source-only numerical state differs from its original checkpoint")


def source_contract_reference_in(record):
    calibration = record.get("class_calibration")
    if calibration is None or calibration.get("origin") != "source_mixture":
        return None
    training = calibration["fit_contract"]["training_contract"]
    return training if training.get("schema") == SOURCE_CONTRACT_REF_SCHEMA else None


def expand_source_record(record, admission):
    reference = source_contract_reference_in(record)
    if reference is None:
        return record
    calibration = record["class_calibration"]
    return {**record, "class_calibration": {**calibration, "fit_contract": {
        **calibration["fit_contract"], "training_contract": admission.resolve(reference)}}}


def compact_source_record(record, reference):
    calibration = record["class_calibration"]
    return {**record, "class_calibration": {**calibration, "fit_contract": {
        **calibration["fit_contract"], "training_contract": plain_metadata(reference)}}}


def compact_source_semantics(record):
    """Storage projection; full in-memory scientific records retain their values."""
    if isinstance(record, dict):
        reference = record.get("source_contract_reference")
        result = {name: compact_source_semantics(value) for name, value in record.items()
                  if reference is None or name != "probability_semantics"}
        if reference is not None and "probability_semantics" in record:
            result["probability_semantics"] = compact_source_record(
                {"class_calibration": record["probability_semantics"]}, reference)["class_calibration"]
        return result
    if isinstance(record, (list, tuple)):
        return [compact_source_semantics(value) for value in record]
    return record


def expand_source_semantics(record, admission, *, require_references=False):
    if isinstance(record, dict):
        result = {name: expand_source_semantics(value, admission, require_references=require_references)
                  for name, value in record.items()}
        if "probability_semantics" in record:
            wrapper = {"class_calibration": record["probability_semantics"]}
            reference = source_contract_reference_in(wrapper)
            if ("source_contract_reference" in record or require_references
                    and record["probability_semantics"].get("origin") == "source_mixture") and reference is None:
                raise ValueError("source probability origin lost its explicit storage dependency")
            if reference is not None:
                if record.get("source_contract_reference") != reference:
                    raise ValueError("probability origin differs from its complete source reference")
                result["probability_semantics"] = expand_source_record(wrapper, admission)["class_calibration"]
        return result
    if isinstance(record, list):
        return [expand_source_semantics(value, admission, require_references=require_references) for value in record]
    return record


def source_dependency_descriptor(directory, store, references):
    """Relative locators keep a complete run and its source directory relocatable."""
    from research.contracts import semantic_sha256

    unique = {semantic_sha256(reference): plain_metadata(freeze_metadata(reference)) for reference in references}
    return {"schema": SOURCE_STORE_SCHEMA,
            "directory": Path(os.path.relpath(store.directory, Path(directory).resolve())).as_posix(),
            "references": [unique[key] for key in sorted(unique)]}


def admit_source_dependencies(descriptor, directory, *, store=None, admission=None):
    if (set(descriptor) != {"schema", "directory", "references"}
            or descriptor["schema"] != SOURCE_STORE_SCHEMA
            or not isinstance(descriptor["directory"], str) or Path(descriptor["directory"]).is_absolute()
            or not isinstance(descriptor["references"], list) or not descriptor["references"]):
        raise ValueError("package lacks its complete explicit source dependency closure")
    actual_store = SourceMetadataStore(Path(directory) / descriptor["directory"])
    if store is not None and actual_store.directory != store.directory:
        raise ValueError("source dependency store differs from its actual package locator")
    owner = admission or _SourceMetadataAdmission(store=actual_store)
    if owner._store is None:
        owner._store = actual_store
    elif owner._store.directory != actual_store.directory:
        raise ValueError("one source admission cannot change its actual package store")
    from research.contracts import semantic_sha256

    keys = set()
    for reference in descriptor["references"]:
        key = semantic_sha256(reference)
        if key in keys:
            raise ValueError("package repeats its source authority dependency")
        keys.add(key)
        owner.resolve(reference)
    return owner


def collect_source_dependencies(directory, package_directories, *, include_joint_prefixes=False):
    """Collect numerical dependencies, optionally including sealed joint transport.

    Numerical/source loaders keep the default and independently verify typed
    transport references. Whole-package export and parent closures explicitly
    include joint prefix descriptors, including packages without an ensemble.
    """
    import json

    from research.contracts import semantic_sha256

    references, store = {}, None
    manifests = ("ensemble.json", "prefix_inner_replay.json") if include_joint_prefixes else ("ensemble.json",)
    for package in dict.fromkeys(Path(package) for package in package_directories):
        for name in manifests:
            path = package / name
            if not path.exists():
                continue
            payload = json.loads(path.read_bytes())
            record = payload["record"]
            if name == "prefix_inner_replay.json" and record.get("schema") != "n2p3_joint_flat_prefix_inner_replay/1":
                continue
            if payload["record_sha256"] != semantic_sha256(record):
                raise ValueError("source closure collection requires the actual sealed package")
            descriptor = record.get("source_metadata")
            if descriptor is None:
                continue
            current = SourceMetadataStore(package / descriptor["directory"])
            if store is not None and current.directory != store.directory:
                raise ValueError("one package closure must use one actual complete source store")
            store = current
            for reference in descriptor["references"]:
                references[semantic_sha256(reference)] = reference
    return None if not references else source_dependency_descriptor(directory, store, references.values())


def export_source_package_closure(directory, destination):
    """Copy a package/graph plus complete shared sources with original seals.

    Preserve all relative locators and file bytes. The returned entry is inside
    the exported closure and can be loaded without the original run or checkout.
    """
    import shutil

    from research.execution import atomic_write_json
    from research.source_execution import current_input_admission

    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    if destination.is_relative_to(directory):
        raise ValueError("source closure export must have an independent destination")
    packages = {path.parent for name in ("ensemble.json", "prefix_inner_replay.json")
                for path in directory.rglob(name)}
    descriptor = collect_source_dependencies(directory, packages, include_joint_prefixes=True)
    roots = [directory]
    if descriptor is not None:
        roots.append((directory / descriptor["directory"]).resolve())
    common = Path(os.path.commonpath(roots))
    files = {}
    for path in directory.rglob("*"):
        if path.is_file():
            if path.is_symlink():
                raise ValueError("package closure requires original regular artifacts")
            files[path.relative_to(common)] = path
    if descriptor is not None:
        admission = admit_source_dependencies(descriptor, directory)
        store = SourceMetadataStore(directory / descriptor["directory"])
        for reference in descriptor["references"]:
            path = store.directory / (reference["checkpoint_sha256"] + ".pt")
            if hashlib.sha256(admission.content(reference)).hexdigest() != reference["checkpoint_sha256"]:
                raise ValueError("source closure differs from its admitted original bytes")
            files[path.relative_to(common)] = path
    destination.mkdir(parents=True, exist_ok=False)
    owner = current_input_admission()
    from contextlib import nullcontext

    receipts = {}
    source_seals = ({} if descriptor is None else {
        (store.directory / (reference["checkpoint_sha256"] + ".pt")).relative_to(common): reference["checkpoint_sha256"]
        for reference in descriptor["references"]})
    with owner.artifact_directory(destination, sum(path.stat().st_size for path in files.values()) + 2**20) if owner else nullcontext():
        for relative, source in files.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            receipts[relative.as_posix()] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                            "bytes": target.stat().st_size}
            if relative in source_seals and receipts[relative.as_posix()]["sha256"] != source_seals[relative]:
                raise ValueError("source closure changed while its original bytes were copied")
        entry = destination / directory.relative_to(common)
        atomic_write_json(destination / "source_closure.json", {"schema": "n2p3_source_package_closure/1",
            "entry": entry.relative_to(destination).as_posix(), "files": receipts})
    return entry
