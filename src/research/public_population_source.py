"""Original full-source records -> shared person likelihoods -> native forecast.

The source issuer is the original registered-record compiler in a named
source-only role. Numeric codecs preserve source dependencies and grant no
raw or label access. A live target admission is issued separately.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from data.candidate_input import resolve_candidate_dataset_adapter
from research import population_response_forecast as population
from research import source_likelihood_export as likelihood_export
from research.channel_noise_mechanism import FittedChannelNoise, NoiseGeometry
from research.gaussian_local_factor import SharedResponseChart
from research.joint_response_noise_reference import (
    TwoExposureWithinReference,
    two_exposure_gls_reference,
)
from research.population_noise_reference import (
    SourceStationaryMoments,
    moments_from_original_spatial_model,
    publish_fold_noise_reference,
)
from research.public_native_source_admission import _owned_target
from transfer.ensemble_views import RecordSourceTrainingCompiler, calibration_prefix_identity
from transfer.native_response_dictionary import NativeResponseDictionary
from transfer.noise_model import array_digest, freeze_metadata, metadata_digest
from transfer.spatial_programme import SpatialDecisionModel

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PERSONS, _NATIVES, _ADMISSIONS = WeakKeyDictionary(), WeakKeyDictionary(), WeakKeyDictionary()
_EXECUTIONS = WeakKeyDictionary()
_PERSON_PUBLICATIONS, _FOLD_PUBLICATIONS = {}, {}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bytes(reference):
    content = Path(reference["path"]).read_bytes()
    if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError("the explicit original source dependency changed")
    return content


def _artifact(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


@dataclass(frozen=True, eq=False)
class FullSourcePerson:
    compiler: RecordSourceTrainingCompiler
    prefix: object
    records: tuple
    receipts: tuple
    record: dict


def open_frozen_source_person(freeze_reference, participant_key, *, profiles,
                             authorization_preprocessing, response_seconds, pre_roll_seconds):
    """Reissue the selected source records from their registered original reader.

    This is the actual source-EEG read boundary, used only when the reviewed
    complete source programme is launched. Session 2 is never attached.
    """
    plan = json.loads(_bytes(freeze_reference))
    if plan["schema"] != "n2p3_H_LOSO31_population_source_freeze_candidate/1":
        raise ValueError("the explicit complete source freeze is required")
    person = next(row for row in plan["persons"] if row["participant_key"] == participant_key)
    rows = tuple(row for row in plan["source_record_ledger"] if row["participant_key"] == participant_key
                 and row["common_source_admission_candidate"])
    declaration = json.loads(_bytes(person["declaration"]))
    selected = {(int(row["session"]), int(row["run"])) for row in rows}
    source_rows = [row for row in declaration["record_sources"]
                   if (int(row["block"]), int(row["run"])) in selected]
    source_declaration = {key: value for key, value in declaration.items()
                          if key not in {"record_sources", "evaluation_session_ids"}}
    source_declaration["record_sources"] = source_rows
    adapter = resolve_candidate_dataset_adapter(source_declaration)
    geometry = plan["geometry"]
    actual = tuple(adapter.recordings)
    expected = {row["record_id"]: row for row in rows}
    if adapter.participant_key != participant_key or {s.record_id for s in actual} != set(expected):
        raise ValueError("the registered source participant/record roster differs from its freeze")
    for source in actual:
        row = expected[source.record_id]
        if (source.source_sha256 != row["original_BDF_reference"]["sha256"]
                or len(source.native_onsets) != row["requested_labels"]
                or np.any(source.native_onsets < 0)
                or source.session_id not in {"0", "1"}
                or tuple(source.channel_names) != tuple(geometry["channels"])
                or source.sample_rate_hz != geometry["sfreq"] or source.reference != geometry["reference"]):
            raise ValueError("source bytes, full label/clock scope or native geometry differs from the freeze")
    compiler = RecordSourceTrainingCompiler(adapter, profiles=profiles,
        authorization_preprocessing=authorization_preprocessing, response_seconds=response_seconds,
        pre_roll_seconds=pre_roll_seconds)
    prefix, records, receipts = compiler.source_fit_operands()
    record = freeze_metadata({"schema": "original_registered_full_source_person/1",
        "participant_key": participant_key, "source_freeze": freeze_reference,
        "source_declaration": person["declaration"], "original_adapter": adapter.record(),
        "source_prefix_identity": calibration_prefix_identity(prefix),
        "source_record_ids": tuple(r.record_id for r in records),
        "source_labels": sum(len(r.target_mask) for r in records), "source_sessions": (0, 1),
        "future_records_attached": False, "geometry": geometry,
        "requested_unavailable_source_records": tuple(row for row in plan["source_record_ledger"]
            if row["participant_key"] == participant_key and not row["common_source_admission_candidate"]),
        "producer_sha256": IMPLEMENTATION_SHA256})
    value = FullSourcePerson(compiler, prefix, records, receipts, record)
    _PERSONS[value] = record
    return value


def require_full_source_person(value):
    if type(value) is not FullSourcePerson or _PERSONS.get(value) is not value.record:
        raise TypeError("the live original full-source person issuer is required")
    value.compiler.assert_original_prefix(value.prefix)
    return value


def fit_and_export_source_person(person, *, signal_programme, dictionary, noise_options,
                                 output_directory, shared_store):
    """Fit one person once and export only the actual same-fit data factors."""
    person = require_full_source_person(person)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    geometry = person.record["geometry"]
    model, blocks = SpatialDecisionModel.fit_with_blocks(person.records, person.receipts, signal_programme,
        sfreq=geometry["sfreq"], channels=tuple(geometry["channels"]),
        candidate_ids=tuple(person.prefix.target_contract["candidate_vocabulary"]),
        operator_identity=geometry["operator_identity"], dictionary=dictionary,
        fit_scope="source_training", **noise_options)
    model_path = directory/"original_source_model.json"
    model.save(model_path)
    reference = _artifact(model_path)
    handle = likelihood_export.admit_original_source_model(_bytes(reference),
        expected_sha256=reference["sha256"], locator=reference)
    operands = likelihood_export.bind_original_fit_blocks(handle, blocks)
    likelihood = likelihood_export.export_original_source_likelihood(operands)
    compact = shared_store.put(likelihood)
    moments = moments_from_original_spatial_model(handle, likelihood_reference=compact)
    receipt = {"schema": "same_fit_source_person_likelihood_publication/1",
        "source_person": person.record, "original_source_model": reference,
        "shared_likelihood": compact.record(), "source_export_metadata": likelihood.metadata,
        "stationary_moments": moments.record(),
        "noise_fit_programmes": 1, "fixed_noise_response_refits": 1,
        "same_fit_block_count": len(blocks), "individual_posterior_used_as_measurement": False,
        "producer_sha256": IMPLEMENTATION_SHA256}
    content = (_json(receipt)+"\n").encode()
    (directory/"person_likelihood.json").write_bytes(content)
    publication = _artifact(directory/"person_likelihood.json")
    _PERSON_PUBLICATIONS[publication["sha256"]] = freeze_metadata(receipt)
    return compact, publication


def fit_shared_population_fold(store, fold, references, *, reference, templates, scale_program,
                               frozen_policy_reference):
    """Only the fold's declared source people enter variance and mean learning."""
    policy = json.loads(_bytes(frozen_policy_reference))
    bare_reference = {key: value for key, value in reference.record().items() if key != "policy_sha256"}
    if (policy["schema"] != "fixed_full_response_population_reference_policy/1"
            or reference.policy_sha256 != frozen_policy_reference["sha256"]
            or policy["reference"] != bare_reference
            or policy["templates"] != [value.record() for value in templates]
            or policy["scale_program"] != scale_program.record()):
        raise ValueError("the mean/reference/template policy differs from its predeclared source policy")
    fold_reference = store.put_fold_references(fold, references)
    likelihoods = store.fold_inputs(fold, references, population_core=population)
    actual_sources = sorted(row.original_likelihood_artifact_sha256 for row in likelihoods)
    if policy["origin"] == "fold_source_only":
        if policy["source_likelihoods"] != actual_sources:
            raise ValueError("the fixed source reference uses a different participant fold")
    elif policy["origin"] != "external_fixed" or policy["source_likelihoods"]:
        raise ValueError("the reference policy needs an explicit external or exact fold-source-only origin")
    forecast = population.fit_population_variance(likelihoods, fold=fold, reference=reference,
        templates=templates, program=scale_program)
    return forecast, fold_reference


def prepare_fold_response_reference(store, fold, references, *, person_publications, dictionary,
                                     conditioning_samples, scale_program, output_directory):
    """Freeze the named source-derived reference before source hyperlearning.

    m0=0, F0=Kref.factor and the sole between template also uses that full
    factor. Every source direction is retained without Gram formation or 1/n.
    """
    likelihoods = store.fold_inputs(fold, references, population_core=population)
    by_person = {row.participant_key: row for row in likelihoods}
    moments = []
    publications = {}
    for publication in person_publications:
        receipt = json.loads(_bytes(publication))
        key = receipt["source_person"]["participant_key"]
        if key not in by_person or key in publications:
            raise ValueError("reference publication roster differs from the exact source fold")
        moment = SourceStationaryMoments.from_record(receipt["stationary_moments"])
        source_ref = next(row for row in references if row.participant_key == key)
        if (receipt["shared_likelihood"] != source_ref.record()
                or moment.dependency["source_likelihood"] != source_ref.record()
                or moment.participant_key != key):
            raise ValueError("source moments differ from the actual same-fit person likelihood")
        publications[key] = publication
        moments.append(moment)
    if set(publications) != set(by_person):
        raise ValueError("the fixed source reference needs every actual source person once")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    noise, noise_dependency = publish_fold_noise_reference(moments, fold=fold,
        conditioning_samples=conditioning_samples, output_path=directory/"noise-reference.json")
    source_ids = sorted(row.original_likelihood_artifact_sha256 for row in likelihoods)
    within = two_exposure_gls_reference(dictionary.matrix, noise, noise_scale=1.,
        dictionary_identity=dictionary.digest(), source_reference_binding={"origin": "fold_source_only",
        "source_likelihoods": source_ids}, strategy="shared_parameter_law_kronecker", block_native_rows=128)
    within_path = directory/"within-reference.json"
    within.save(within_path)
    chart = likelihoods[0].chart
    if chart.dimension != len(within.factor):
        raise ValueError("the complete source reference differs from the actual response coefficient chart")
    zero = np.zeros(chart.dimension)
    template = population.VarianceTemplate("source_two_exposure_full_response", chart, within.factor)
    bare = {"chart_identity": chart.identity, "mean_identity": population._array_id(zero),
        "factor_identity": population._array_id(within.factor)}
    policy = {"schema": "fixed_full_response_population_reference_policy/1", "origin": "fold_source_only",
        "source_likelihoods": source_ids, "reference": bare, "templates": [template.record()],
        "scale_program": scale_program.record(), "reference_mean": "fixed_zero",
        "source_person_publications": [publications[key] for key in fold.source_participant_keys],
        "noise_reference": noise_dependency, "within_reference": _artifact(within_path),
        "reference_factor": "full_two_independent_unit_exposure_common_increment_GLS_factor",
        "factor_divisor": 1, "source_posterior_factor_used": False,
        "producer_sha256": IMPLEMENTATION_SHA256}
    policy_path = directory/"reference-policy.json"
    policy_path.write_text(_json(policy)+"\n", encoding="utf8", newline="\n")
    policy_dependency = _artifact(policy_path)
    reference = population.MeanReference(chart, zero, within.factor, policy_dependency["sha256"])
    return reference, (template,), policy_dependency, noise_dependency, _artifact(within_path)


def save_shared_forecast(forecast, directory, *, store, fold_reference):
    """Save one fold state and references; A/r are kept once in the person store."""
    fold, likelihoods = store.load_fold_inputs(fold_reference, population_core=population)
    if fold.record() != forecast.fold.record() or [v.record() for v in likelihoods] != [v.record() for v in forecast.likelihoods]:
        raise ValueError("the shared fold references differ from the actual fitted population inputs")
    state, arrays = population.forecast_numeric_state(forecast)
    small = {name: value for name, value in arrays.items() if not name.startswith("source_")}
    stream = io.BytesIO()
    np.savez_compressed(stream, **small)
    payload = stream.getvalue()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"state.npz").write_bytes(payload)
    witness = population.save_forecast_qr_witness(forecast, directory/"qr-witness.npz")
    record = {"schema": "shared_person_full_response_population_forecast/1", "state": state,
        "fold_reference": fold_reference, "shared_store": str(store.directory.resolve()),
        "state_arrays": {"file": "state.npz", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        "qr_witness": {**witness, "path": "qr-witness.npz"},
        "person_matrices_stored_here": False, "raw_or_label_authority_granted": False,
        "codec_producer_sha256": IMPLEMENTATION_SHA256}
    (directory/"forecast.json").write_text(_json(record)+"\n", encoding="utf8", newline="\n")
    publication = _artifact(directory/"forecast.json")
    _FOLD_PUBLICATIONS[publication["sha256"]] = freeze_metadata({"fold": forecast.fold.record(),
        "input_identity": forecast.input_identity, "qr_witness": record["qr_witness"]})
    return publication


def load_shared_forecast(reference, *, shared_store=None):
    """Cold restore checks every person's A/r/c and fold, with zero fit/QR."""
    record = json.loads(_bytes(reference))
    if (record["schema"] != "shared_person_full_response_population_forecast/1"
            or record["person_matrices_stored_here"] is not False
            or record["raw_or_label_authority_granted"] is not False
            or record["state_arrays"]["file"] != "state.npz"):
        raise ValueError("the compact forecast changed its shared-source or numeric-only role")
    store = shared_store or likelihood_export.SharedLikelihoodStore(record["shared_store"])
    fold, likelihoods = store.load_fold_inputs(record["fold_reference"], population_core=population)
    state = record["state"]
    if fold.record() != state["fold"] or [v.record() for v in likelihoods] != state["likelihoods"]:
        raise ValueError("the original complete person/fold likelihood dependencies changed")
    payload_path = Path(reference["path"]).parent/"state.npz"
    payload = _bytes({"path": str(payload_path), **{k: record["state_arrays"][k] for k in ("bytes", "sha256")}})
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(name.startswith("source_") for name in arrays):
        raise ValueError("the compact forecast must reference its original person matrices once")
    for i, likelihood in enumerate(likelihoods):
        arrays[f"source_{i}_matrix"], arrays[f"source_{i}_response"] = likelihood.matrix, likelihood.response
    witness = population.load_forecast_qr_witness(record["qr_witness"],
        base_directory=Path(reference["path"]).parent)
    return population.restore_forecast_state(state, arrays, qr_witness=witness)


@dataclass(frozen=True, eq=False)
class SourcePopulationExecution:
    """Accepted complete source execution, independent of numeric chart objects."""

    record: dict
    execution_reference: dict
    acceptance_reference: dict
    matrix_reference: dict

    @property
    def identity(self):
        return metadata_digest({"execution": self.execution_reference,
            "independent_acceptance": self.acceptance_reference, "matrix": self.matrix_reference})


def publish_source_population_execution(matrix_reference, *, person_publications, fold_publications,
                                         recipe_references, output_path):
    """Publish current actual source/fold outputs for independent acceptance.

    No live source admission is minted here. This runtime issuer accepts only
    publications produced by this programme's original-reader/fit owners.
    """
    matrix = json.loads(_bytes(matrix_reference))
    persons, folds = [], []
    for reference in person_publications:
        original = _PERSON_PUBLICATIONS.get(reference["sha256"])
        if original is None:
            raise TypeError("the current original-reader same-fit person publication is required")
        publication = json.loads(_bytes(reference))
        if metadata_digest(publication) != metadata_digest(original):
            raise ValueError("the current same-fit person publication changed")
        persons.append({"participant_key": original["source_person"]["participant_key"], "publication": reference,
            "shared_likelihood": publication["shared_likelihood"]})
    for entry in fold_publications:
        reference = entry["forecast"]
        original = _FOLD_PUBLICATIONS.get(reference["sha256"])
        if original is None:
            raise TypeError("the current actual complete fold fit publication is required")
        saved = json.loads(_bytes(reference))
        for key in ("within_reference", "noise_reference"):
            _bytes(entry[key])
        fold = original["fold"]
        folds.append({**entry, "fold_id": fold["fold_id"],
            "target_participant_key": fold["target_identity_keys"][0],
            "source_participant_keys": list(fold["source_participant_keys"]),
            "source_likelihoods": [row for person in persons
                if person["participant_key"] in fold["source_participant_keys"]
                for row in (person["shared_likelihood"],)],
            "input_identity": original["input_identity"], "qr_witness": saved["qr_witness"]})
    if (len(persons) != 13 or len(folds) != 13
            or {row["participant_key"] for row in persons} != {row["source_person_id"] for row in matrix["person_programmes"]}
            or {row["fold_id"] for row in folds} != {row["fold_id"] for row in matrix["fold_programmes"]}):
        raise ValueError("the complete runtime publication retains exactly thirteen people and folds")
    people = {row["participant_key"]: row for row in persons}
    for fold in folds:
        fold["source_likelihoods"] = [people[key]["shared_likelihood"] for key in fold["source_participant_keys"]]
    for reference in recipe_references:
        _bytes(reference)
    record = {"schema": "complete_original_population_source_execution/1", "source_matrix": matrix_reference,
        "source_freeze": matrix["source_freeze"], "programme_seed": matrix["programme_seed"],
        "persons": persons, "folds": folds, "recipes": list(recipe_references), "complete": True,
        "future_source_reads": 0, "individual_posterior_used_as_measurement": False,
        "producer_sha256": IMPLEMENTATION_SHA256}
    with Path(output_path).open("x", encoding="utf8", newline="\n") as stream:
        stream.write(_json(record)+"\n")
    return _artifact(output_path)


def load_source_population_execution(execution_reference, *, trusted_acceptance_reference,
                                     expected_matrix_reference):
    """Admit a reviewed complete execution against an externally pinned receipt.

    The caller supplies the independent review's pinned acceptance reference.
    Neither a numeric forecast nor its self-signed metadata supplies that trust
    anchor. All thirteen person/fold publications are bound here; selected fold
    numeric dependencies are verified once by the native loader.
    """
    matrix = json.loads(_bytes(expected_matrix_reference))
    execution = json.loads(_bytes(execution_reference))
    accepted = json.loads(_bytes(trusted_acceptance_reference))
    if (accepted["schema"] != "independent_complete_population_source_execution_acceptance/1"
            or accepted["accepted"] is not True or accepted["independent_source_review"] is not True
            or not accepted["reviewer"] or not accepted["evidence"]
            or accepted["execution"] != execution_reference
            or accepted["source_matrix"] != expected_matrix_reference):
        raise ValueError("a separately trusted complete-source execution acceptance is required")
    for evidence in accepted["evidence"]:
        _bytes(evidence)
    if (matrix["schema"] != "public_population_source_execution_matrix_candidate/1"
            or execution["schema"] != "complete_original_population_source_execution/1"
            or execution["source_matrix"] != expected_matrix_reference
            or execution["source_freeze"] != matrix["source_freeze"]
            or execution["programme_seed"] != matrix["programme_seed"]
            or execution["complete"] is not True or execution["future_source_reads"] != 0
            or execution["individual_posterior_used_as_measurement"] is not False):
        raise ValueError("the accepted source execution differs from the complete frozen programme")
    expected_people = {row["source_person_id"]: row for row in matrix["person_programmes"]}
    expected_folds = {row["fold_id"]: row for row in matrix["fold_programmes"]}
    people = {row["participant_key"]: row for row in execution["persons"]}
    folds = {row["fold_id"]: row for row in execution["folds"]}
    if (len(expected_people) != 13 or len(expected_folds) != 13 or len(people) != 13 or len(folds) != 13
            or len(execution["persons"]) != 13 or len(execution["folds"]) != 13
            or set(people) != set(expected_people) or set(folds) != set(expected_folds)):
        raise ValueError("live population provenance requires all thirteen original people and folds")
    for key, entry in people.items():
        receipt = json.loads(_bytes(entry["publication"]))
        source = receipt["source_person"]
        expected = expected_people[key]
        if (receipt["schema"] != "same_fit_source_person_likelihood_publication/1"
                or source["participant_key"] != key or source["source_freeze"] != matrix["source_freeze"]
                or source["source_declaration"] != expected["declaration"]
                or set(source["source_record_ids"]) != set(expected["selected_source_records"])
                or source["source_labels"] != expected["selected_label_count"]
                or source["future_records_attached"] is not False
                or receipt["noise_fit_programmes"] != 1
                or receipt["individual_posterior_used_as_measurement"] is not False
                or receipt["shared_likelihood"] != entry["shared_likelihood"]):
            raise ValueError("a person publication differs from its actual complete source programme")
    for key, entry in folds.items():
        expected = expected_folds[key]
        if (entry["target_participant_key"] != expected["target_participant_key"]
                or entry["source_participant_keys"] != expected["source_person_references"]
                or len(entry["source_participant_keys"]) != 12
                or entry["source_likelihoods"] != [people[p]["shared_likelihood"]
                    for p in entry["source_participant_keys"]]):
            raise ValueError("an accepted fold differs from its exact twelve-person source programme")
        state = json.loads(_bytes(entry["forecast"]))
        if (state["schema"] != "shared_person_full_response_population_forecast/1"
                or state["state"]["input_identity"] != entry["input_identity"]
                or state["state"]["fold"]["fold_id"] != key
                or state["state"]["fold"]["source_participant_keys"] != entry["source_participant_keys"]
                or state["qr_witness"] != entry["qr_witness"]):
            raise ValueError("the reviewed fold forecast differs from its actual publication")
    record = freeze_metadata(execution)
    value = SourcePopulationExecution(record, freeze_metadata(execution_reference),
        freeze_metadata(trusted_acceptance_reference), freeze_metadata(expected_matrix_reference))
    _EXECUTIONS[value] = record
    return value


def require_source_population_execution(value, native):
    if type(value) is not SourcePopulationExecution or _EXECUTIONS.get(value) is not value.record:
        raise TypeError("the independently accepted complete original-source execution is required")
    fold = native.forecast.fold
    matches = [row for row in value.record["folds"] if row["fold_id"] == fold.fold_id]
    if len(matches) != 1:
        raise ValueError("the selected native forecast is absent from the accepted source execution")
    row = matches[0]
    if (metadata_digest(row["forecast"]) != metadata_digest(native.forecast_reference)
            or row["input_identity"] != native.forecast.input_identity
            or metadata_digest(row["within_reference"]) != metadata_digest(native.within_reference_dependency)
            or metadata_digest(row["noise_reference"]) != metadata_digest(native.noise_reference_dependency)
            or tuple(row["source_participant_keys"]) != fold.source_participant_keys
            or (row["target_participant_key"],) != fold.target_identity_keys):
        raise ValueError("native forecast/reference dependencies differ from the accepted source execution")
    return value


@dataclass(frozen=True, eq=False)
class PopulationNativeResponse:
    forecast: population.PopulationForecast
    dictionary: NativeResponseDictionary
    geometry: NoiseGeometry
    forecast_reference: dict
    within_reference: TwoExposureWithinReference
    within_reference_dependency: dict
    noise_reference: FittedChannelNoise
    noise_reference_dependency: dict

    def __post_init__(self):
        definition = json.loads(self.forecast.chart.physical_definition_json)
        binding = definition["original_spatial_design_binding"]
        expected_names = tuple(f"{group}:basis:{i}:channel:{channel}"
            for group in ("common", "target_increment") for i in range(self.dictionary.matrix.shape[1])
            for channel in self.geometry.channels)
        if (definition["schema"] != "native_full_response_coefficient_chart/1"
                or definition["coefficient_order"] != "coefficient_major_C_order_P_by_C"
                or tuple(definition["channels"]) != self.geometry.channels
                or definition["signal_unit"] != self.geometry.unit
                or definition["acquisition_reference"] != self.geometry.reference
                or binding["sample_rate_hz"] != self.geometry.sfreq
                or binding["operator_identity"] != self.geometry.operator_identity
                or self.dictionary.sample_rate_hz != self.geometry.sfreq
                or array_digest(self.dictionary.matrix) != binding["actual_basis_sha256"]
                or 2*self.dictionary.matrix.shape[1]*len(self.geometry.channels) != self.forecast.chart.dimension
                or self.forecast.chart.coefficient_names != expected_names):
            raise ValueError("population native consumption requires the full original basis, clock and physical chart")
        within = self.within_reference
        noise = self.noise_reference
        if (type(noise) is not FittedChannelNoise or noise.geometry.record() != self.geometry.record()
                or noise._runtime_state_digest() != within.binding["source_noise_identity"]):
            raise ValueError("the native noise reference differs from the actual within reference noise law")
        noise.assert_integrity()
        if (type(within) is not TwoExposureWithinReference
                or within.factor.shape != (self.forecast.chart.dimension, self.forecast.chart.dimension)
                or within.binding["basis_sha256"] != array_digest(self.dictionary.matrix)
                or metadata_digest(within.binding["noise_geometry"]) != metadata_digest(self.geometry.record())
                or within.binding["dictionary_identity"] != self.dictionary.digest()
                or within.binding["source_posterior_factor_used"] is not False):
            raise ValueError("the fixed within reference differs from the full native source function/noise coordinates")
        origin = within.binding["source_reference_binding"]
        manifests = sorted(row.original_likelihood_artifact_sha256 for row in self.forecast.likelihoods)
        if origin["origin"] == "fold_source_only":
            if sorted(origin["source_likelihoods"]) != manifests:
                raise ValueError("the fixed within reference includes a different source participant fold")
            noise_origin = noise.provenance["source_reference_binding"]
            noise_manifests = sorted(row["source_likelihood"]["sha256"]
                for row in noise_origin["source_dependencies"])
            if (metadata_digest(noise_origin["source_fold"]) != metadata_digest(self.forecast.fold.record())
                    or noise_manifests != manifests):
                raise ValueError("the fixed noise reference includes a different source participant fold")
        elif origin["origin"] != "external_fixed" or origin["source_likelihoods"]:
            raise ValueError("within reference requires an explicit source-only or external fixed origin")

    @cached_property
    def source_chart(self):
        factor = self.forecast.factor
        channels = len(self.geometry.channels)
        width = 2*self.dictionary.matrix.shape[1]
        names = tuple(f"population:{self.forecast.input_identity}:between:{i}"
            for i in range(self.forecast.between_factor.shape[1]))+tuple(
            f"population:{self.forecast.input_identity}:mean_uncertainty:{i}"
            for i in range(self.forecast.mean_factor.shape[1]))
        return SharedResponseChart(self.forecast.mean.reshape(width, channels),
            factor.reshape(width, channels, factor.shape[1]), names, self.geometry.channels,
            {"schema": "complete_native_population_response_chart/1", "forecast": self.forecast_reference,
             "source_fold": self.forecast.fold.record(), "source_input_identity": self.forecast.input_identity,
             "population_producer_sha256": self.forecast.producer_sha256,
             "coefficient_chart": self.forecast.chart.record(),
             "covariance": "between_person_plus_population_mean_uncertainty",
             "full_latent_columns_preserved": True, "source_posterior_as_measurement": False,
             "dictionary_digest": self.dictionary.digest(), "native_geometry": self.geometry.record(),
             "raw_or_label_authority_granted": False})

    @property
    def identity(self):
        return self.source_chart.identity


def load_population_native_response(reference, *, dictionary, geometry, within_reference_dependency,
                                    noise_reference_dependency, shared_store=None):
    """Load the complete source fold through its numeric-only artifact owner."""
    forecast = load_shared_forecast(reference, shared_store=shared_store)
    if forecast.restoration_validation != "complete_same_fit_source_and_population_qr_witness":
        raise ValueError("new population use requires its complete original same-fit source/forecast QR witness")
    within = TwoExposureWithinReference.load(within_reference_dependency["path"],
        expected_sha256=within_reference_dependency["sha256"])
    _bytes(noise_reference_dependency)
    noise = FittedChannelNoise.load(noise_reference_dependency["path"])
    value = PopulationNativeResponse(forecast, dictionary, geometry, freeze_metadata(reference), within,
        freeze_metadata(within_reference_dependency), noise, freeze_metadata(noise_reference_dependency))
    _NATIVES[value] = value.identity
    return value


@dataclass(frozen=True, eq=False)
class PublicPopulationSourceAdmission:
    native: PopulationNativeResponse
    compiler: object
    prefixes: tuple
    calibration_inputs: tuple
    record: dict
    source_execution: SourcePopulationExecution

    @property
    def source_chart(self):
        return self.native.source_chart

    @property
    def basis(self):
        return self.native.dictionary.matrix

    @property
    def dictionary(self):
        return self.native.dictionary

    @property
    def geometry(self):
        return self.native.geometry

    @property
    def within_reference(self):
        return self.native.within_reference

    @property
    def noise_reference(self):
        return self.native.noise_reference

    @property
    def identity(self):
        return metadata_digest(self.record)


def admit_public_population_source(native, compiler, prefixes, calibration_inputs, *, source_execution):
    """Join the numeric source fold to original live C and complete held input."""
    if (type(native) is not PopulationNativeResponse or _NATIVES.get(native) != native.identity):
        raise TypeError("the full original-chart population numeric source is required")
    source_execution = require_source_population_execution(source_execution, native)
    prefixes, inputs, geometry, target = _owned_target(compiler, prefixes, calibration_inputs)
    fold = native.forecast.fold
    if (target["participant_key"] not in fold.target_identity_keys
            or geometry.record() != native.geometry.record()):
        raise ValueError("the source forecast belongs to another target or native observation contract")
    target_eeg = {row["source_sha256"] for row in target["source_record_roster"].values()}
    target_records = set(target["source_record_roster"])
    for likelihood in native.forecast.likelihoods:
        if set(likelihood.identity_keys) & set(fold.target_identity_keys):
            raise ValueError("the actual target entered the source population")
        if any(r.eeg_sha256 in target_eeg or r.record_id in target_records for r in likelihood.records):
            raise ValueError("the actual target C/held acquisition entered the source population")
    record = freeze_metadata({"schema": "live_public_full_population_source_consumption/1",
        "source_native_identity": native.identity, "source_fold": fold.record(),
        "accepted_source_execution_identity": source_execution.identity,
        "source_person_likelihoods": tuple(row.record() for row in native.forecast.likelihoods),
        "new_target_union": target, "source_fit_count_here": 0,
        "fixed_within_reference": native.within_reference_dependency,
        "fixed_within_reference_identity": native.within_reference.identity,
        "fixed_noise_reference": native.noise_reference_dependency,
        "fixed_noise_reference_identity": native.noise_reference._runtime_state_digest(),
        "target_labels_entering_source_hyperlearning": 0, "source_role": "new_participant_full_response_forecast",
        "single_person_C68_development_prior": False, "producer_sha256": IMPLEMENTATION_SHA256})
    value = PublicPopulationSourceAdmission(native, compiler, prefixes, inputs, record, source_execution)
    _ADMISSIONS[value] = record
    return value


def require_public_population_source(value):
    if type(value) is not PublicPopulationSourceAdmission or _ADMISSIONS.get(value) is not value.record:
        raise TypeError("the live original-source-fold/public-target admission is required")
    require_source_population_execution(value.source_execution, value.native)
    _, _, _, target = _owned_target(value.compiler, value.prefixes, value.calibration_inputs)
    if (metadata_digest(target) != metadata_digest(value.record["new_target_union"])
            or value.native.identity != value.record["source_native_identity"]):
        raise ValueError("the original source forecast or live target-union relation changed")
    return value
