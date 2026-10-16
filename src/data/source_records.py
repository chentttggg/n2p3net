"""Explicit local public-ERP records using audited parsers and the shared raw pipeline."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from data.channel import canonical_channel_name
from data.contract import BRAIN_SYNC_CAUSAL_DATA_CONTRACT
from data.dataset import build_subject
from data.domain import adapt_observed_montage, annotate_source_domain
from data.epochs import (
    PreprocessingSpec,
    _cache_record_path,
    _read_epoch_cache_record,
    concatenate_epoch_datasets,
    load_epoch_dataset,
    loaded_epoch_cache_attestation,
    materialize_dataset_lineage,
    parse_preprocessing_spec,
    preprocessing_spec_from_contract,
    save_epoch_dataset,
    validate_preprocessing_execution,
)
from data.exclusions import source_participant_excluded
from data.lineage import DataLineage
from data.manifest import _subject_epoch_dataset
from data.preprocess import RawSampleScope, prefix_raw_scopes, source_epoch_support
from data.source_parsers import read_source_record as _read_record
from data.source_parsers import source_companions

SIX = ("Fz", "P3", "Pz", "P4", "PO8", "Oz")
SOURCE_EOG_PREPROCESSING = preprocessing_spec_from_contract(BRAIN_SYNC_CAUSAL_DATA_CONTRACT)
SOURCE_PREPARATION_SCHEMA = "n2p3_source_preparation/2"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_prepared_cache(path, *, cache_sha256, cache_record_sha256):
    """Re-admit the exact NPZ and full-validation record published together.

    The preparation ledger binds both files after save_epoch_dataset validated
    the complete dataset. Matching both original byte identities preserves the
    record's participant/lineage contract without decompressing every shard.
    Downstream consumers still perform their normal attested immutable load.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError("completed source cache is missing; preserve the run and use a new output directory")
    _, attestation = _read_epoch_cache_record(path)
    if not cache_record_sha256 or file_sha256(_cache_record_path(path)) != cache_record_sha256:
        raise ValueError("completed source cache record changed or lacks its original digest")
    if (attestation["sha256"] != cache_sha256 or path.stat().st_size != attestation["byte_size"]
            or file_sha256(path) != cache_sha256):
        raise ValueError("completed source cache no longer matches its original attestation")


def source_preparation_method(preprocessing: PreprocessingSpec | None = None):
    """Bind the data implementation and parser environment used by this preparation."""
    directory = Path(__file__).parent
    spec = SOURCE_EOG_PREPROCESSING if preprocessing is None else preprocessing
    spec.validate()
    return {"schema": "n2p3_source_record_method/2",
            "preprocessing": spec.record(),
            "data_modules_sha256": {path.name: file_sha256(path) for path in sorted(directory.glob("*.py"))},
            "runner_sha256": file_sha256(directory.parents[1] / "experiments/prepare_source_records.py"),
            "packages": {name: version(name) for name in ("numpy", "scipy", "mne", "moabb")}}


def verify_source_record_inputs(item, *, root, extracted_member=None):
    """Verify declared primary bytes and bind actual parser companions before use.

    Undeclared companion hashes are a new preparation observation, never evidence
    retroactively attached to an old completed shard. Resume compares this entire
    record with the observation made before the original parse.
    """
    root = Path(root)
    source = root / item["path"]
    if item.get("member"):
        relative = Path(item["member"])
        if relative.anchor or ".." in relative.parts:
            raise ValueError("invalid archive member")
        if item["study"] in {"Huebner2017", "GTN", "Chailloux2020"}:
            raise ValueError("multi-file source parsers require explicit extracted inputs")
        if extracted_member is None:
            with ZipFile(source) as archive, archive.open(item["member"]) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            digest = file_sha256(extracted_member)
    else:
        digest = file_sha256(source)
    if digest != item["sha256"]:
        raise ValueError("source file or archive member changed from preparation manifest")
    inputs = [{"role": "raw", "path": item["path"], "member": item.get("member"),
               "sha256": digest, "identity_basis": "manifest"}]
    companions = source_companions(item, source)
    declared = item.get("sidecars", {})
    if not isinstance(declared, dict) or set(declared) - set(companions):
        raise ValueError("sidecar declarations must identify actual parser companions")
    for role, path in companions.items():
        actual = file_sha256(path)
        declaration = declared.get(role)
        if declaration is not None and (
            not isinstance(declaration, dict)
            or (root / declaration.get("path", "")).resolve() != path.resolve()
            or declaration.get("sha256") != actual
        ):
            raise ValueError(f"source {role} sidecar changed from preparation manifest")
        inputs.append({"role": role, "path": str(path.relative_to(root)), "sha256": actual,
                       "identity_basis": "manifest" if declaration is not None else "preparation_start"})
    return inputs


def _source_channel_admission(raw):
    wanted = {canonical_channel_name(c) for c in (*SIX, "Cz")}
    originals = list(raw.ch_names)
    types = raw.get_channel_types()
    canonical = [canonical_channel_name(name) for name in originals]
    by_identity = {}
    for index, name in enumerate(canonical):
        by_identity.setdefault(name, []).append(index)
    if any(len(by_identity.get(name, ())) > 1 for name in wanted):
        raise ValueError("source channels collide after canonical normalization")
    bad_indices = set()
    for name in raw.info["bads"]:
        matches = by_identity.get(canonical_channel_name(name), ())
        if len(matches) != 1:
            raise ValueError("source bad-channel identity is missing or ambiguous")
        bad_indices.add(matches[0])
    reference = by_identity.get("CZ", ())
    if len(reference) != 1 or types[reference[0]] != "eeg" or reference[0] in bad_indices:
        raise ValueError("source requires a measured, EEG-typed, non-bad Cz reference")
    for index, name in enumerate(canonical):
        if name in wanted and types[index] != "eeg":
            raise ValueError("source target/reference electrode must have the EEG role")
    selected = [index for index, name in enumerate(canonical) if name in wanted and index not in bad_indices]
    if len(selected) < 2:
        raise ValueError("source requires an observed target EEG channel besides Cz")
    names = [originals[index] for index in selected]
    return names, {"schema": "n2p3_source_channel_admission/1",
        "original_names": originals, "original_types": types, "original_bads": list(raw.info["bads"]),
        "canonical_names": canonical, "selected_original_names": names,
        "removed_bad_channels": [originals[index] for index in sorted(bad_indices) if canonical[index] in wanted],
        "reference": "CZ", "missing_value_policy": "exclude_before_filter_then_masked_montage_zero"}


def prepare_source_record(item, *, root, output, preparation_method=None,
                          preprocessing: PreprocessingSpec | None = None,
                          raw_scope: RawSampleScope | None = None):
    """Read verified bytes, preprocess native channels, then map measured Cz differences."""
    item = dict(item)
    if "preprocessing" in item:
        declared = parse_preprocessing_spec(item["preprocessing"])
        if preprocessing is not None and declared != preprocessing:
            raise ValueError("record preprocessing differs from the explicit preparation profile")
        preprocessing = declared
    if "raw_scope" in item:
        declared_scope = RawSampleScope.from_record(item["raw_scope"])
        if raw_scope is not None and declared_scope != raw_scope:
            raise ValueError("record raw scope differs from the explicit preparation scope")
        raw_scope = declared_scope
    spec = SOURCE_EOG_PREPROCESSING if preprocessing is None else preprocessing
    if not isinstance(spec, PreprocessingSpec):
        raise TypeError("source preparation requires a complete PreprocessingSpec")
    spec.validate()
    if raw_scope is not None:
        if not isinstance(raw_scope, RawSampleScope):
            raise TypeError("source preparation raw_scope must be a RawSampleScope")
    if source_participant_excluded(item["study"], item["subject"]):
        raise ValueError("author-excluded source participant")
    root, output = Path(root), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    method = source_preparation_method() if spec == SOURCE_EOG_PREPROCESSING else source_preparation_method(spec)
    if preparation_method is not None and preparation_method != method:
        raise ValueError("source preparation implementation changed before worker execution")
    with _verified_source_record(item, root=root, output=output) as loaded:
        return _prepare_record_view(item, output=output, method=method, spec=spec,
                                    raw_scope=raw_scope, loaded=loaded)


@contextmanager
def _verified_source_record(item, *, root, output):
    """Parse one verified original recording; views never share filtered samples."""
    source = root / item["path"]
    with tempfile.TemporaryDirectory(prefix="n2p3-parser-", dir=output) as temporary:
        if item.get("member"):
            with ZipFile(source) as archive:
                relative = Path(item["member"])
                if relative.anchor or ".." in relative.parts:
                    raise ValueError("invalid archive member")
                path = Path(temporary) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item["member"]) as incoming, path.open("wb") as extracted:
                    shutil.copyfileobj(incoming, extracted)
        else:
            path = source.parent if item["study"] == "GTN" else source
        inputs = verify_source_record_inputs(item, root=root,
            extracted_member=path if item.get("member") else None)
        parsed = _read_record(item, path)
        raw, events, labels = parsed[:3]
        candidate = parsed[3] if len(parsed) == 4 else None
        try:
            yield raw, events, labels, inputs, candidate
        finally:
            raw.close()


def _prepare_record_view(item, *, output, method, spec, raw_scope, loaded, require_both_classes=True):
    original, events, labels, inputs, candidate = loaded
    raw = original.copy()
    if len(events) != len(labels):
        raise ValueError("record requires correctly aligned two-class event evidence")
    if raw_scope is not None:
        raw_scope.validate_source(raw, events)
    authorized = np.arange(len(events)) if raw_scope is None else np.asarray(raw_scope.event_indices)
    classes = set(np.asarray(labels)[authorized])
    if not classes or not classes <= {0, 1} or (require_both_classes and classes != {0, 1}):
        raise ValueError("authorized record scope requires correctly aligned two-class event evidence")
    rate = float(raw.info["sfreq"])
    if rate != item["sample_rate_hz"]:
        raise ValueError("source sampling rate differs from manifest")
    names, admission = _source_channel_admission(raw)
    raw.pick(names)
    coordinate_derivation = None
    if item["study"] == "EPFLP300":
        # MOABB attaches the idealized BioSemi32 template, not measured positions.
        # Map the same named EEG electrodes to the shared model's nominal grid.
        original_montage = raw.get_montage()
        coordinate_derivation = {"original_template": "biosemi32", "model_template": "standard_1005",
            "coordinates_are_measured": False, "channel_labels_unchanged": True,
            "original_positions_m": {key: value.tolist() for key, value in
                original_montage.get_positions()["ch_pos"].items()}}
        raw.set_montage("standard_1005")
    start = 0 if raw_scope is None else raw_scope.start_sample - int(raw.first_samp)
    stop = int(raw.n_times) if raw_scope is None else raw_scope.stop_sample - int(raw.first_samp)
    values = raw.get_data(start=start, stop=stop)
    if not np.isfinite(values).all() or any(np.ptp(c) == 0 for c in values):
        raise ValueError("nonfinite or entire-record constant measured EEG")
    if candidate is not None:
        from data.bnci2014_008_candidate import build_bnci2014_008_subject_dataset

        dataset = build_bnci2014_008_subject_dataset(candidate, preprocessing=spec, raw_scope=raw_scope, raw=original)
    else:
        subject = build_subject(raw, events, labels, subject_id=item["subject"],
            metadata={"dataset_id": item["study"], "session": item["session"], "run": item["run"],
                      "selection_id": item["session"] + ":" + item["run"], "reference": item["reference"]},
            preprocessing=spec, raw_scope=raw_scope, channels=names, montage="standard_1005")
        dataset = _subject_epoch_dataset(subject, spec, dataset_name=item["study"])
    dataset.provenance.update(source="explicit_audited_public_raw", dataset_class=item["study"],
        source_sample_rate_hz=rate, source_reference=item["reference"], raw_record=item,
        original_event_count=len(events),
        **({"original_target_count": int(labels.sum())} if raw_scope is None else
           {"authorized_target_count": int(np.asarray(labels)[authorized].sum()),
            "authorized_event_count": len(authorized)}),
        source_parser="data.source_parsers.read_source_record", source_qc="raw_finite_and_constant_gate; learned_epoch_policy_not_applied")
    preparation = {"schema": "n2p3_source_record_preparation/2", "method": method,
                   "consumed_inputs": inputs, "channel_admission": admission,
                   "label_admission": "binary_two_classes" if require_both_classes else "binary_labels_allow_single_class",
                   **({"raw_scope": raw_scope.record()} if raw_scope is not None else {})}
    dataset.provenance.update(source_preparation=preparation, source_channel_admission=admission)
    if raw_scope is not None:
        dataset.provenance["target_raw_record_scopes"] = [{
            "raw_identity": {key: item[key] for key in ("study", "subject", "session", "run")} | {
                "inputs": [{"role": value["role"], "sha256": value["sha256"]} for value in inputs]},
            "raw_scope": raw_scope.record(),
            "receipt_sha256": dataset.provenance["preprocessing_execution"]["receipt_sha256"],
        }]
    dataset.lineage = DataLineage.derive([materialize_dataset_lineage(dataset)],
        operation="verified_source_record_preparation", parameters=preparation)
    if coordinate_derivation is not None:
        dataset.provenance["source_coordinate_derivation"] = coordinate_derivation
    if item["study"] == "GTN":
        dataset.provenance["reference_evidence"] = "10.1038/sdata.2016.121: common reference on bridge of nose; native metadata absence preserved in raw_record"
    dataset = annotate_source_domain(adapt_observed_montage(dataset, SIX, reference_channel="Cz"), item["study"])
    token = hashlib.sha256(json.dumps({"record": item, "preparation": preparation}, sort_keys=True).encode()).hexdigest()[:20]
    path = output / (token + ".npz")
    if path.exists():
        previous = load_epoch_dataset(path, validation="attested")
        if previous.provenance.get("source_preparation") != preparation:
            raise FileExistsError("existing source shard has a different preparation identity; use a new output")
    else:
        save_epoch_dataset(path, dataset)
    timeline = dataset.event_timeline
    purged = np.flatnonzero(timeline.evidence_indices < 0)
    scope_evidence = {} if raw_scope is None else {
        "exposed_raw_samples": raw_scope.stop_sample - raw_scope.start_sample,
        "authorized_label_count": len(raw_scope.event_indices),
        "retained_label_count": dataset.n_epochs,
        "purged_event_ids": timeline.event_ids[purged].tolist(),
        "purged_event_details": timeline.status_details[purged].tolist(),
    }
    raw.close()
    return {"record": item, "cache": str(path), "cache_sha256": file_sha256(path),
            "cache_record_sha256": file_sha256(_cache_record_path(path)),
            "preparation": preparation,
            "n_epochs": dataset.n_epochs, "scheduled": dataset.event_timeline.n_events, "target_epochs": int(dataset.y.sum()),
            "channels": list(dataset.channel_names), "observed": dataset.channel_mask.tolist(),
            **scope_evidence}


def prepare_source_record_prefixes(item, *, root, output, budgets, minimum_test_trials=200,
                                   preprocessing=None, preparation_method=None, prepare_inner=True):
    """Materialize independent per-K calibration operators and one fixed test.

    Budgets allocate original labels before boundary/artifact purge. The shared
    resolver owns limited-data fallback; eligibility never uses class scores.
    Each numeric view starts from the same unfiltered reader result.
    """
    from transfer.binary import BinaryTimeSplit, resolve_calibration_splits

    item = dict(item)
    if "raw_scope" in item:
        raise ValueError("prefix preparation requires a complete original record, not a preselected raw_scope")
    spec = SOURCE_EOG_PREPROCESSING if preprocessing is None else preprocessing
    if "preprocessing" in item:
        declared = parse_preprocessing_spec(item["preprocessing"])
        if preprocessing is not None and declared != preprocessing:
            raise ValueError("record preprocessing differs from prefix preparation profile")
        spec = declared
    spec.validate()
    method = source_preparation_method() if spec == SOURCE_EOG_PREPROCESSING else source_preparation_method(spec)
    if preparation_method is not None and preparation_method != method:
        raise ValueError("source preparation implementation changed before prefix execution")
    if source_participant_excluded(item["study"], item["subject"]):
        raise ValueError("author-excluded source participant")
    root, output = Path(root), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _verified_source_record(item, root=root, output=output) as loaded:
        raw, events, _, inputs, _ = loaded
        starts, stops = source_epoch_support(events, float(raw.info["sfreq"]), spec)
        if np.any(np.diff(events[:, 0]) <= 0):
            raise ValueError("prefix preparation requires unique chronological physical events")
        rows = np.arange(len(events), dtype=np.int64)
        first, end = int(raw.first_samp), int(raw.first_samp + raw.n_times)

        def scopes(k, maximum):
            if maximum == 0:
                return None, RawSampleScope("recording:heldout", first, end, tuple(rows))
            return prefix_raw_scopes(raw, events, spec, calibration_events=k,
                                     maximum_calibration_events=maximum)

        def builder(k, maximum):
            _, test_scope = scopes(k, maximum)
            later = rows >= maximum
            eligible = later & (starts >= test_scope.start_sample) & (stops <= test_scope.stop_sample)
            return BinaryTimeSplit(rows[:k], rows[eligible], rows[later & ~eligible])

        splits, adjustments = resolve_calibration_splits(tuple(budgets), builder,
            minimum_test_trials=minimum_test_trials, available_trials=len(events))
        maximum = max(splits)
        _, test_scope = scopes(0, maximum)
        test = _prepare_record_view(item, output=output, method=method, spec=spec,
                                   raw_scope=test_scope, loaded=loaded)
        calibration = {}
        early_stopping_inner = {}
        for budget in splits:
            calibration_scope, _ = scopes(budget, maximum)
            calibration[str(budget)] = None if calibration_scope is None else _prepare_record_view(
                item, output=output, method=method, spec=spec, raw_scope=calibration_scope, loaded=loaded)
            if prepare_inner and calibration_scope is not None:
                early_stopping_inner[str(budget)] = _prepare_scope_inner_views(item, loaded=loaded,
                    calibration_scope=calibration_scope, output=output, method=method, spec=spec)
    return {"schema": "n2p3_target_raw_prefixes/1", "record": item,
            "preprocessing": spec.record(), "preparation_method": method, "consumed_inputs": inputs,
            "protocol": {"kind": "single_recording_prefix", "minimum_test_trials": minimum_test_trials},
            "requested_budgets": list(budgets), "effective_budgets": list(splits),
            "adjustments": {str(key): value for key, value in adjustments.items()},
            "maximum_calibration_events": maximum, "calibration": calibration, "test": test,
            "early_stopping_inner": early_stopping_inner, "prepare_inner": prepare_inner}


def _prepare_scope_inner_views(item, *, loaded, calibration_scope, output, method, spec):
    """Rebuild nested raw train/validation operators inside one authorized C."""
    raw, events = loaded[:2]
    try:
        train_scope, validation_scope = prefix_raw_scopes(raw, events, spec,
            calibration_events=max(1, int(len(calibration_scope.event_indices) * .8)),
            parent_scope=calibration_scope)
        training = _prepare_record_view(item, output=output, method=method, spec=spec,
                                       raw_scope=train_scope, loaded=loaded)
        validation = _prepare_record_view(item, output=output, method=method, spec=spec,
                                         raw_scope=validation_scope, loaded=loaded)
        if any(not 0 < view["target_epochs"] < view["n_epochs"] for view in (training, validation)):
            raise ValueError("inner train and validation must each retain both binary classes after raw-boundary purge")
        return {"status": "ready", "mode": "within_recording_raw_80_20",
                "training": training, "validation": validation}
    except ValueError as error:
        return {"status": "unavailable", "reason": str(error)}


def verify_prepared_prefixes(record, *, item, root, method, budgets, minimum_test_trials, protocol=None, prepare_inner=True):
    """Admit completed scope caches without repeating raw parsing or filtering."""
    if record.get("prepare_inner", True) != prepare_inner:
        raise ValueError("target prefix inner preparation mode changed")
    inner_views = []
    for inner in record.get("early_stopping_inner", {}).values():
        if inner["status"] == "ready":
            for role in ("training", "validation"):
                inner_views.extend(inner[role] if isinstance(inner[role], list) else [inner[role]])
    if protocol is not None and protocol.get("kind") == "last_session_holdout":
        expected_protocol = {**protocol, "minimum_test_trials": protocol.get("minimum_test_trials", minimum_test_trials)}
        if (record.get("schema") != "n2p3_target_raw_prefixes/1" or record.get("records") != item
                or record.get("preparation_method") != method or record.get("preprocessing") != method["preprocessing"]
                or record.get("requested_budgets") != list(budgets) or record.get("protocol") != expected_protocol):
            raise ValueError("target subject prefix preparation contract changed")
        early, tests = _ordered_target_records(item, protocol)
        inventory = record["record_inventory"]
        if [row["record"] for row in inventory] != [*early, *tests]:
            raise ValueError("target prefix inventory order changed")
        inputs = {}
        for row in inventory:
            current = verify_source_record_inputs(row["record"], root=root)
            if current != row["consumed_inputs"]:
                raise ValueError("target prefix original inputs changed")
            inputs[json.dumps(row["record"], sort_keys=True)] = current
        for view in [*record["test"], *(view for values in record["calibration"].values() for view in values), *inner_views]:
            preparation = view["preparation"]
            if (preparation.get("method") != method
                    or preparation.get("consumed_inputs") != inputs.get(json.dumps(view["record"], sort_keys=True))):
                raise ValueError("target prefix shard method or original input changed")
            verify_prepared_cache(view["cache"], cache_sha256=view["cache_sha256"],
                                  cache_record_sha256=view.get("cache_record_sha256"))
        return
    if (record.get("schema") != "n2p3_target_raw_prefixes/1" or record.get("record") != item
            or record.get("preparation_method") != method
            or record.get("preprocessing") != method["preprocessing"]
            or record.get("requested_budgets") != list(budgets)
            or record.get("protocol") != {"kind": "single_recording_prefix", "minimum_test_trials": minimum_test_trials}
            or record.get("consumed_inputs") != verify_source_record_inputs(item, root=root)):
        raise ValueError("target prefix preparation contract or original inputs changed")
    calibration = record["calibration"]
    if set(calibration) != {str(k) for k in record["effective_budgets"]}:
        raise ValueError("target prefix calibration budgets differ from their cache views")
    heldout = RawSampleScope.from_record(record["test"]["preparation"]["raw_scope"])
    for key, view in [("test", record["test"]), *calibration.items()]:
        if key == "0":
            if view is not None:
                raise ValueError("K0 cannot own target fit samples")
            continue
        if view is None or view.get("record") != item:
            raise ValueError("target prefix cache has no matching original record")
        preparation = view["preparation"]
        scope = RawSampleScope.from_record(preparation["raw_scope"])
        if (preparation.get("method") != method or preparation.get("consumed_inputs") != record["consumed_inputs"]
                or (key != "test" and (len(scope.event_indices) != int(key) or scope.stop_sample > heldout.start_sample))):
            raise ValueError("target prefix cache does not match the authorized scope contract")
        verify_prepared_cache(view["cache"], cache_sha256=view["cache_sha256"],
                              cache_record_sha256=view.get("cache_record_sha256"))
    inner_views = record.get("early_stopping_inner", {})
    if set(inner_views) != (set(calibration) - {"0"} if prepare_inner else set()):
        raise ValueError("completed target prefixes lack the current inner raw preparation")
    for key, inner in inner_views.items():
        if inner.get("status") == "unavailable" and inner.get("reason"):
            continue
        if inner.get("status") != "ready":
            raise ValueError("invalid target inner preparation status")
        parent = RawSampleScope.from_record(calibration[key]["preparation"]["raw_scope"])
        pair = []
        for role in ("training", "validation"):
            view = inner[role]
            preparation = view["preparation"]
            scope = RawSampleScope.from_record(preparation["raw_scope"])
            if (view.get("record") != item or preparation.get("method") != method
                    or preparation.get("consumed_inputs") != record["consumed_inputs"]):
                raise ValueError("target inner preparation differs from its original inputs")
            verify_prepared_cache(view["cache"], cache_sha256=view["cache_sha256"],
                                  cache_record_sha256=view.get("cache_record_sha256"))
            pair.append(scope)
        a, b = pair
        boundary = max(1, int(int(key) * .8))
        if (a.start_sample != parent.start_sample or b.stop_sample != parent.stop_sample
                or a.stop_sample != b.start_sample or a.event_indices != parent.event_indices[:boundary]
                or b.event_indices != parent.event_indices[boundary:]):
            raise ValueError("target inner preparation escaped the declared calibration scope")


def _ordered_target_records(items, protocol):
    if protocol.get("kind") != "last_session_holdout":
        raise ValueError("multiple target records require explicit last_session_holdout")
    sessions, runs = protocol.get("session_order"), protocol.get("run_order")
    if any(not isinstance(axis, list) or not axis or len(set(axis)) != len(axis)
           or any(not isinstance(value, str) or not value for value in axis) for axis in (sessions, runs)):
        raise ValueError("target recording order requires unique explicit session/run strings")
    if protocol.get("test_session") != sessions[-1] or len(sessions) < 2:
        raise ValueError("target test_session must be the last of multiple declared sessions")
    if not items or len({(item["study"], str(item["subject"])) for item in items}) != 1:
        raise ValueError("target record group must belong to one study and participant")
    physical = [(item["path"], item.get("member")) for item in items]
    pairs = [(item["session"], item["run"]) for item in items]
    if len(set(physical)) != len(physical) or len(set(pairs)) != len(pairs):
        raise ValueError("target protocol cannot duplicate a physical file or recording identity")
    if any(session not in sessions or run not in runs for session, run in pairs):
        raise ValueError("target recording is absent from the declared protocol order")
    ordered = sorted(items, key=lambda item: (sessions.index(item["session"]), runs.index(item["run"])))
    early = [item for item in ordered if item["session"] != protocol["test_session"]]
    tests = [item for item in ordered if item["session"] == protocol["test_session"]]
    if not early or not tests:
        raise ValueError("target protocol requires earlier calibration and later test recordings")
    return early, tests


def prepare_source_subject_prefixes(items, *, root, output, budgets, protocol,
                                    minimum_test_trials=200, preprocessing=None, preparation_method=None, prepare_inner=True):
    """Allocate labels over declared earlier records; filter each raw scope alone.

    A bounded metadata pass resolves the shared budget fallback. A second pass
    holds one original recording and reuses that reader for all its numeric K
    views. Complete earlier records are consumed whole; a partial record ends
    at the last authorized epoch's exact source support.
    """
    from transfer.binary import BinaryTimeSplit, resolve_calibration_splits

    items = [dict(item) for item in items]
    early, tests = _ordered_target_records(items, protocol)
    spec = SOURCE_EOG_PREPROCESSING if preprocessing is None else preprocessing
    spec.validate()
    for item in items:
        if "raw_scope" in item:
            raise ValueError("target recording protocol requires original complete raw records")
        if "preprocessing" in item and parse_preprocessing_spec(item["preprocessing"]) != spec:
            raise ValueError("target recording preprocessing differs from the common profile")
        if source_participant_excluded(item["study"], item["subject"]):
            raise ValueError("author-excluded source participant")
    method = source_preparation_method() if spec == SOURCE_EOG_PREPROCESSING else source_preparation_method(spec)
    if preparation_method is not None and preparation_method != method:
        raise ValueError("source preparation implementation changed before subject prefix execution")
    root, output = Path(root), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for item in [*early, *tests]:
        with _verified_source_record(item, root=root, output=output) as loaded:
            raw, events, _, inputs, _ = loaded
            starts, stops = source_epoch_support(events, float(raw.info["sfreq"]), spec)
            if not len(events) or np.any(np.diff(events[:, 0]) <= 0):
                raise ValueError("target recording requires unique chronological events")
            first, end = int(raw.first_samp), int(raw.first_samp + raw.n_times)
            inventory.append({"record": item, "consumed_inputs": inputs, "event_count": len(events),
                              "first_sample": first, "stop_sample": end,
                              "eligible_count": int(((starts >= first) & (stops <= end)).sum())})
    available = sum(row["event_count"] for row in inventory[:len(early)])
    n_test = sum(row["eligible_count"] for row in inventory[len(early):])

    def builder(k, maximum):
        if not 0 <= k <= maximum <= available:
            raise ValueError("calibration allocation exceeds earlier recordings")
        return BinaryTimeSplit(np.arange(k), np.arange(available, available + n_test), np.empty(0, dtype=np.int64))

    minimum = protocol.get("minimum_test_trials", minimum_test_trials)
    splits, adjustments = resolve_calibration_splits(tuple(budgets), builder,
        minimum_test_trials=minimum, available_trials=available, separate_test_recordings=True)
    calibration = {str(k): [] for k in splits}
    early_stopping_inner = {}
    test_views, offset = [], 0
    for index, observed in enumerate(inventory):
        item, count = observed["record"], observed["event_count"]
        local_counts = sorted({min(count, max(0, k - offset)) for k in splits}) if index < len(early) else [count]
        if not any(local_counts):
            offset += count
            continue
        with _verified_source_record(item, root=root, output=output) as loaded:
            raw, events, _, inputs, _ = loaded
            if (inputs != observed["consumed_inputs"] or len(events) != count
                    or int(raw.first_samp) != observed["first_sample"]
                    or int(raw.first_samp + raw.n_times) != observed["stop_sample"]):
                raise ValueError("target raw identity or geometry changed between planning and execution")
            _, stops = source_epoch_support(events, float(raw.info["sfreq"]), spec)
            views = {}
            for allocated in local_counts:
                if not allocated:
                    continue
                stop = observed["stop_sample"] if allocated == count else int(stops[:allocated].max())
                scope = RawSampleScope("recording:heldout" if index >= len(early) else "recording:calibration",
                                       observed["first_sample"], stop, tuple(range(allocated)))
                views[allocated] = _prepare_record_view(item, output=output, method=method, spec=spec,
                    raw_scope=scope, loaded=loaded, require_both_classes=False)
                if prepare_inner and index == 0 and allocated in splits:
                    early_stopping_inner[str(allocated)] = _prepare_scope_inner_views(item, loaded=loaded,
                        calibration_scope=scope, output=output, method=method, spec=spec)
            if index >= len(early):
                test_views.append(views[count])
            else:
                for k in splits:
                    allocated = min(count, max(0, k - offset))
                    if allocated:
                        calibration[str(k)].append(views[allocated])
        if index < len(early):
            offset += count
    for key, views in calibration.items():
        if prepare_inner and len(views) > 1:
            split = len(views) - max(1, int(np.ceil(.2 * len(views))))
            training, validation = views[:split], views[split:]
            if any(not 0 < view["target_epochs"] < view["n_epochs"] for view in views):
                early_stopping_inner[key] = {"status": "unavailable",
                    "reason": "each independent inner recording must retain both binary classes"}
            else:
                early_stopping_inner[key] = {"status": "ready", "mode": "complete_recordings_80_20",
                                             "training": training, "validation": validation}
    return {"schema": "n2p3_target_raw_prefixes/1", "records": items,
            "subject": str(items[0]["subject"]), "preprocessing": spec.record(),
            "preparation_method": method, "protocol": {**protocol, "minimum_test_trials": minimum},
            "record_inventory": inventory, "requested_budgets": list(budgets),
            "effective_budgets": list(splits), "maximum_calibration_events": max(splits),
            "adjustments": {str(key): value for key, value in adjustments.items()},
            "calibration": calibration, "test": test_views, "early_stopping_inner": early_stopping_inner, "prepare_inner": prepare_inner}


def _load_subject_prefixes(entry, path, spec, method, *, load_inner=True):
    """Validate ordered multi-record allocations, then join independently processed views."""
    from transfer.binary import validate_calibration_budgets

    early, tests = _ordered_target_records(entry["records"], entry["protocol"])
    subject = str(early[0]["subject"])
    if entry.get("subject") != subject:
        raise ValueError("target prefix subject differs from its recording identities")
    budgets = entry["effective_budgets"]
    validate_calibration_budgets(tuple(budgets))
    if set(entry["calibration"]) != {str(k) for k in budgets} or entry["maximum_calibration_events"] != max(budgets):
        raise ValueError("target prefix budgets do not match the available views")
    inventory = entry["record_inventory"]
    if [row["record"] for row in inventory] != [*early, *tests]:
        raise ValueError("target raw inventory differs from the declared protocol order")
    by_record = {json.dumps(row["record"], sort_keys=True): row for row in inventory}
    loaded = {}

    def load_views(views, allocation, *, test=False):
        expected, remaining = [], allocation
        for item in tests if test else early:
            observed = by_record[json.dumps(item, sort_keys=True)]
            take = observed["event_count"] if test else min(remaining, observed["event_count"])
            if take:
                expected.append((item, observed, take))
            remaining -= take
        if (not test and remaining != 0) or len(views) != len(expected):
            raise ValueError("target prefix view count differs from the ordered label allocation")
        datasets = []
        for view, (item, observed, take) in zip(views, expected, strict=True):
            if view.get("record") != item:
                raise ValueError("target prefix views are not in declared physical recording order")
            cache = Path(view["cache"])
            cache = cache if cache.is_absolute() else path.parent / cache
            token = (str(cache), view["cache_sha256"], view["cache_record_sha256"])
            if token not in loaded:
                verify_prepared_cache(cache, cache_sha256=token[1], cache_record_sha256=token[2])
                loaded[token] = load_epoch_dataset(cache, validation="attested")
            dataset = loaded[token]
            preparation = view["preparation"]
            if (dataset.preprocessing != spec or dataset.provenance.get("source_preparation") != preparation
                    or preparation.get("method") != method
                    or preparation.get("consumed_inputs") != observed["consumed_inputs"]
                    or set(dataset.subject_ids.astype(str)) != {subject}):
                raise ValueError("target prefix cache differs from its attested original recording")
            scope = RawSampleScope.from_record(preparation["raw_scope"])
            receipts = validate_preprocessing_execution(dataset.provenance.get("preprocessing_execution"), spec)
            if len(receipts) != 1:
                raise ValueError("each target shard requires one original raw execution receipt")
            context = receipts[0]["source_context"]
            parent = context.get("raw_scope", {})
            if (parent.get("declaration") != scope.record()
                    or parent.get("parent_first_samp") != observed["first_sample"]
                    or parent.get("parent_n_times") != observed["stop_sample"] - observed["first_sample"]
                    or parent.get("parent_event_count") != observed["event_count"]):
                raise ValueError("target prefix scope differs from the original raw geometry")
            rate = context["sample_rate_hz"]
            last_support = int(np.rint(dataset.event_timeline.onset_times_s * rate).max()) + round(spec.tmax_ms / 1000 * rate) + 1
            expected_stop = observed["stop_sample"] if take == observed["event_count"] else last_support
            if (scope.start_sample != observed["first_sample"] or scope.stop_sample != expected_stop
                    or scope.event_indices != tuple(range(take)) or view["authorized_label_count"] != take
                    or view["retained_label_count"] != dataset.n_epochs
                    or view["scheduled"] != dataset.event_timeline.n_events
                    or view["exposed_raw_samples"] != scope.stop_sample - scope.start_sample):
                raise ValueError("target prefix view exposes samples or labels outside its ordered budget")
            datasets.append(dataset)
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return _combine_target_views(datasets, entry["protocol"])

    test = load_views(entry["test"], 0, test=True)
    calibration = {k: load_views(entry["calibration"][str(k)], k) for k in budgets}
    inner = {}
    for key, declaration in (entry.get("early_stopping_inner", {}) if load_inner else {}).items():
        if int(key) not in calibration or not int(key):
            raise ValueError("target inner view lacks an authorized calibration budget")
        if declaration["status"] == "unavailable":
            inner[int(key)] = {"status": "unavailable", "reason": declaration["reason"]}
            continue
        if declaration["status"] != "ready":
            raise ValueError("unknown target inner raw view status")
        parent_views = entry["calibration"][key]
        if len(parent_views) == 1:
            parent = parent_views[0]
            scope = RawSampleScope.from_record(parent["preparation"]["raw_scope"])
            observed = by_record[json.dumps(parent["record"], sort_keys=True)]
            temporary_entry = {"record": parent["record"],
                "consumed_inputs": parent["preparation"]["consumed_inputs"],
                "early_stopping_inner": {key: declaration}}
            inner.update(_load_target_inner_prefixes(temporary_entry, path, spec, method, subject, {key: scope},
                (observed["first_sample"], observed["stop_sample"] - observed["first_sample"], observed["event_count"])))
            continue
        split = len(parent_views) - max(1, int(np.ceil(.2 * len(parent_views))))
        if (declaration.get("mode") != "complete_recordings_80_20"
                or declaration["training"] != parent_views[:split] or declaration["validation"] != parent_views[split:]):
            raise ValueError("inner raw recording partition differs from the authorized ordered calibration views")
        views = {}
        for role in ("training", "validation"):
            datasets = []
            for view in declaration[role]:
                cache = Path(view["cache"])
                cache = cache if cache.is_absolute() else path.parent / cache
                datasets.append(loaded[(str(cache), view["cache_sha256"], view["cache_record_sha256"])])
            views[role] = _combine_target_views(datasets, entry["protocol"])
        inner[int(key)] = {"status": "ready", **views}
    return {"records": entry["records"], "manifest": entry, "test": test,
            "calibration": calibration, "early_stopping_inner": inner}


def _combine_target_views(datasets, protocol):
    if len(datasets) == 1:
        return datasets[0]
    shared = {}
    for field in ("source_reference", "event_contract", "candidate_task_contract", "candidate_vocabulary", "session_schema"):
        value = datasets[0].provenance.get(field)
        if any(dataset.provenance.get(field) != value for dataset in datasets[1:]):
            raise ValueError(f"target recording views have different physical contracts: {field}")
        if value is not None:
            shared[field] = value
    return concatenate_epoch_datasets(datasets, name=datasets[0].name,
        provenance={**shared,
                    "target_raw_prefix_protocol": protocol,
                    "target_raw_view_components": [target_raw_view_identity(dataset) for dataset in datasets],
                    "target_raw_record_scopes": [record for dataset in datasets
                                                  for record in dataset.provenance["target_raw_record_scopes"]]})


def target_raw_view_identity(dataset):
    """Identity of verified bytes or their declared ordered numeric composition."""
    if dataset.verified_cache_attestation is not None:
        attestation = loaded_epoch_cache_attestation(dataset)
        return {"schema": "n2p3_target_raw_view_identity/1", "kind": "attested_cache",
                "sha256": attestation["sha256"]}
    components = dataset.provenance.get("target_raw_view_components")
    scopes = dataset.provenance.get("target_raw_record_scopes")
    if not isinstance(components, list) or not components or not isinstance(scopes, list) or not scopes:
        raise ValueError("target raw view lacks verified cache or ordered component identities")
    identity = {"schema": "n2p3_target_raw_view_identity/1", "kind": "ordered_attested_cache_composition",
                "components": components, "raw_scopes": scopes,
                "protocol": dataset.provenance["target_raw_prefix_protocol"]}
    return {**identity, "sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}


def load_target_prefixes(path, *, subjects=None, load_inner=True):
    """Load attested per-subject views without changing recording identities.

    Relative cache paths resolve beside the manifest for explicit relocation.
    Subject filtering validates the collection structure first and loads only
    requested cache views, so parallel workers do not decompress a whole study.
    """
    from transfer.binary import validate_calibration_budgets

    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "n2p3_target_raw_prefix_collection/1":
        raise ValueError("unsupported target raw prefix collection")
    spec = parse_preprocessing_spec(manifest["preprocessing"])
    method = manifest["preparation_method"]
    if method.get("preprocessing") != spec.record():
        raise ValueError("target prefix collection method differs from its preprocessing")
    wanted = None
    if subjects is not None:
        if isinstance(subjects, (str, bytes)):
            raise ValueError("target subjects must be a nonempty explicit sequence")
        requested = list(subjects)
        if (not requested or any(not isinstance(subject, str) or not subject for subject in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("target subjects must be unique nonempty strings")
        wanted = set(requested)
    identified = []
    for entry in manifest["records"]:
        if (entry.get("schema") != "n2p3_target_raw_prefixes/1" or entry.get("preprocessing") != spec.record()
                or entry.get("preparation_method") != method):
            raise ValueError("target prefix entry lacks the declared preparation/protocol")
        kind = entry.get("protocol", {}).get("kind")
        if kind == "last_session_holdout":
            early, _ = _ordered_target_records(entry["records"], entry["protocol"])
            subject = str(early[0]["subject"])
            if entry.get("subject") != subject:
                raise ValueError("target prefix subject differs from its recording identities")
        elif kind == "single_recording_prefix":
            subject = str(entry["record"]["subject"])
        else:
            raise ValueError("unsupported target raw prefix protocol")
        budgets = entry["effective_budgets"]
        validate_calibration_budgets(tuple(budgets))
        if entry["maximum_calibration_events"] != max(budgets) or set(entry["calibration"]) != {str(k) for k in budgets}:
            raise ValueError("target prefix budgets do not match materialized views")
        identified.append((subject, entry))
    declared_subjects = [subject for subject, _ in identified]
    if not declared_subjects or len(set(declared_subjects)) != len(declared_subjects):
        raise ValueError("target prefix collection must contain unique participant entries")
    if wanted is not None and wanted - set(declared_subjects):
        raise ValueError(f"requested target subjects absent from prefix collection: {sorted(wanted - set(declared_subjects))}")
    output = {}
    for subject, entry in identified:
        if wanted is not None and subject not in wanted:
            continue
        if entry.get("protocol", {}).get("kind") == "last_session_holdout":
            output[subject] = _load_subject_prefixes(entry, path, spec, method, load_inner=load_inner)
            continue
        item = entry["record"]
        views, scopes, parents = {}, {}, []
        calibration = entry["calibration"]
        budgets = entry["effective_budgets"]
        maximum = entry["maximum_calibration_events"]
        for key, view in [("test", entry["test"]), *calibration.items()]:
            if key == "0":
                if view is not None:
                    raise ValueError("K0 cannot own calibration evidence")
                views[key] = None
                continue
            if not isinstance(view, dict) or view.get("record") != item:
                raise ValueError("target prefix view differs from its physical source record")
            cache = Path(view["cache"])
            cache = cache if cache.is_absolute() else path.parent / cache
            verify_prepared_cache(cache, cache_sha256=view["cache_sha256"],
                                  cache_record_sha256=view.get("cache_record_sha256"))
            dataset = load_epoch_dataset(cache, validation="attested")
            preparation = view["preparation"]
            if (dataset.preprocessing != spec or dataset.provenance.get("source_preparation") != preparation
                    or preparation.get("method") != method
                    or preparation.get("consumed_inputs") != entry["consumed_inputs"]
                    or set(dataset.subject_ids.astype(str)) != {subject}):
                raise ValueError("target prefix cache does not attest the requested original input")
            scope = RawSampleScope.from_record(preparation["raw_scope"])
            receipts = validate_preprocessing_execution(dataset.provenance.get("preprocessing_execution"), spec)
            if len(receipts) != 1:
                raise ValueError("one prefix view must derive from exactly one original raw")
            context = receipts[0]["source_context"].get("raw_scope", {})
            if context.get("declaration") != scope.record():
                raise ValueError("target prefix declaration differs from the executed numeric scope")
            parent = (context["parent_first_samp"], context["parent_n_times"], context["parent_event_count"])
            parents.append(parent)
            expected = tuple(range(maximum, parent[2])) if key == "test" else tuple(range(int(key)))
            if (scope.event_indices != expected or view["authorized_label_count"] != len(expected)
                    or view["retained_label_count"] != dataset.n_epochs
                    or view["scheduled"] != dataset.event_timeline.n_events
                    or view["exposed_raw_samples"] != scope.stop_sample - scope.start_sample):
                raise ValueError("target prefix original-label allocation or surviving counts differ")
            if key != "test" and scope.start_sample != parent[0]:
                raise ValueError("calibration raw must start at the original recording origin")
            views[key], scopes[key] = dataset, scope
        if len(set(parents)) != 1:
            raise ValueError("target prefix views disagree on the original raw geometry")
        test_scope = scopes["test"]
        if (test_scope.stop_sample != parents[0][0] + parents[0][1]
                or (maximum and scopes[str(maximum)].stop_sample != test_scope.start_sample)
                or (not maximum and test_scope.start_sample != parents[0][0])
                or any(scope.stop_sample > test_scope.start_sample for key, scope in scopes.items() if key != "test")):
            raise ValueError("target calibration and fixed test raw supports overlap or leave an undeclared boundary")
        authority = views["test"].identity_table.record_for(subject).authority_key()
        if any(dataset.identity_table.record_for(subject).authority_key() != authority
               for dataset in views.values() if dataset is not None):
            raise ValueError("target prefix views disagree on participant identity")
        inner = (_load_target_inner_prefixes(entry, path, spec, method, subject, scopes, parents[0])
                 if load_inner else {})
        output[subject] = {"record": item, "manifest": entry, "test": views["test"],
                           "calibration": {int(key): views[key] for key in calibration},
                           "early_stopping_inner": inner}
    if not output:
        raise ValueError("target prefix collection is empty")
    return output


def _load_target_inner_prefixes(entry, path, spec, method, subject, parent_scopes, parent_geometry):
    """Admit independent inner raw operators within each complete calibration scope."""
    result = {}
    for key, inner in entry.get("early_stopping_inner", {}).items():
        if key not in parent_scopes or key == "test":
            raise ValueError("early-stopping inner view has no calibration parent")
        if inner["status"] == "unavailable":
            result[int(key)] = {"status": "unavailable", "reason": inner["reason"]}
            continue
        if inner["status"] != "ready":
            raise ValueError("unknown early-stopping inner view status")
        views, scopes = {}, {}
        for role in ("training", "validation"):
            view = inner[role]
            cache = Path(view["cache"])
            cache = cache if cache.is_absolute() else path.parent / cache
            verify_prepared_cache(cache, cache_sha256=view["cache_sha256"],
                                  cache_record_sha256=view.get("cache_record_sha256"))
            dataset = load_epoch_dataset(cache, validation="attested")
            preparation = view["preparation"]
            scope = RawSampleScope.from_record(preparation["raw_scope"])
            if (view.get("record") != entry["record"] or dataset.preprocessing != spec
                    or dataset.provenance.get("source_preparation") != preparation
                    or preparation.get("method") != method
                    or preparation.get("consumed_inputs") != entry["consumed_inputs"]
                    or set(dataset.subject_ids.astype(str)) != {subject}):
                raise ValueError("early-stopping raw view differs from its declared original input")
            receipts = validate_preprocessing_execution(dataset.provenance.get("preprocessing_execution"), spec)
            if len(receipts) != 1:
                raise ValueError("early-stopping raw view requires exactly one original recording")
            context = receipts[0]["source_context"].get("raw_scope", {})
            geometry = (context["parent_first_samp"], context["parent_n_times"], context["parent_event_count"])
            if context.get("declaration") != scope.record() or geometry != parent_geometry:
                raise ValueError("early-stopping raw view does not match its executed support")
            if (view["authorized_label_count"] != len(scope.event_indices)
                    or view["retained_label_count"] != dataset.n_epochs
                    or view["scheduled"] != dataset.event_timeline.n_events
                    or view["exposed_raw_samples"] != scope.stop_sample - scope.start_sample):
                raise ValueError("early-stopping raw view counts differ from executed support")
            views[role], scopes[role] = dataset, scope
        parent = parent_scopes[key]
        training, validation = scopes["training"], scopes["validation"]
        boundary = max(1, int(int(key) * .8))
        if (training.start_sample != parent.start_sample or validation.stop_sample != parent.stop_sample
                or training.stop_sample != validation.start_sample
                or training.event_indices != parent.event_indices[:boundary]
                or validation.event_indices != parent.event_indices[boundary:]):
            raise ValueError("early-stopping raw train/validation scopes overlap or escape calibration")
        result[int(key)] = {"status": "ready", **views}
    return result


def merge_source_shards(paths, output):
    from data.epochs import load_epoch_dataset

    records = [load_epoch_dataset(path, validation="attested") for path in paths]
    if not records:
        raise ValueError("cannot merge an empty source study")
    provenance = {key: value for key, value in records[0].provenance.items()
                  if key not in {"raw_record", "original_event_count", "original_target_count",
                                 "authorized_event_count", "authorized_target_count",
                                 "preprocessing_execution", "source_channel_admission", "source_preparation"}}
    provenance["source_record_count"] = len(records)
    return save_epoch_dataset(output, concatenate_epoch_datasets(
        records, name=records[0].name, provenance=provenance))
