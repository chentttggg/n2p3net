"""Export original SpatialDecisionModel source likelihoods, never posteriors.

The original saved model is admitted by its canonical owner and external SHA.
Actual original blocks must match its fit receipts. New numerical output grants
no EEG/label authority; each compact person likelihood can be shared by folds.
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
from scipy.linalg import solve_triangular

from transfer.noise_model import array_digest, metadata_digest
from transfer.spatial_programme import SpatialDecisionModel, _require_original_fit
from transfer.spatial_serialization import same_record_owners, saved_record_identity
from transfer.spatial_signal_prior import SpatialResponseBlock

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_MODELS, _BLOCKS = WeakKeyDictionary(), WeakKeyDictionary()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _freeze(array):
    value = np.asarray(array, dtype=float)
    if not np.isfinite(value).all():
        raise ArithmeticError("finite original likelihood operands are required")
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


def _relation(actual, expected, magnitude, count):
    if actual.shape != expected.shape or np.any(np.abs(actual-expected) > 128*np.finfo(float).eps*max(1, count)*magnitude):
        raise ValueError("original source likelihood response differs from its saved compression")


def squared_norm_record(values):
    """Keep a scale representation even when the final squared float underflows.

    This constant is not used to decide response rank. The original weak A/r
    entries remain present and can become visible through a later functional.
    """
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values), initial=0.))
    unit_square = 0. if scale == 0 else float(np.sum((values/scale)**2))
    represented = Fraction.from_float(scale)**2 * Fraction.from_float(unit_square)
    try:
        rounded = float(represented)
    except OverflowError as error:
        raise ArithmeticError("source residual energy is not representable") from error
    return {"scale_hex": scale.hex(), "unit_square_sum_hex": unit_square.hex(),
            "scaled_squared_norm_rational": str(represented), "rounded_squared_norm": rounded,
            "residual_array_sha256": array_digest(values),
            "nonzero_energy_rounds_to_zero": bool(represented and rounded == 0)}


def native_column_permutation(response_coefficients, channels):
    """native[i*C+j] takes original spatial[j*P+i], with every column kept."""
    if response_coefficients < 1 or channels < 1:
        raise ValueError("positive physical response and channel dimensions are required")
    return np.arange(response_coefficients*channels).reshape(channels, response_coefficients).T.ravel()


def block_log_constant(rows, channels, logdet_noise, logdet_nuisance, outside, temporal_log_abs_det):
    return float(-.5*(rows*channels*np.log(2*np.pi)+rows*logdet_noise
        +channels*logdet_nuisance+outside)+channels*temporal_log_abs_det)


def compress_numeric_likelihood(matrix, response, log_constant):
    """General numeric thin QR; a helper, not original-data admission."""
    matrix, response = _freeze(matrix), _freeze(response)
    if matrix.ndim != 2 or response.shape != (len(matrix),) or not np.isfinite(log_constant):
        raise ValueError("a complete finite data likelihood is required")
    if len(matrix) <= matrix.shape[1]:
        residual = np.zeros(0)
        output, rhs = matrix, response
    else:
        q, output = np.linalg.qr(matrix, mode="reduced")
        rhs = q.T @ response
        residual = response-q@rhs
    energy = squared_norm_record(residual)
    constant = float(log_constant)-.5*energy["rounded_squared_norm"]
    if not np.isfinite(constant):
        raise ArithmeticError("the complete source likelihood constant is not representable")
    return _freeze(output), _freeze(rhs), constant, {
        "program": "thin_QR_keep_all_coefficient_columns_no_rank_threshold",
        "original_rows": len(matrix), "compressed_rows": len(output), "columns": matrix.shape[1],
        "original_information_sha256": array_digest(matrix, response),
        "compressed_information_sha256": array_digest(output, rhs), "outside": energy,
        "all_physical_columns_retained": True}


@dataclass(frozen=True, eq=False)
class SourceModelHandle:
    source_file_sha256: str

    @property
    def model(self):
        return _require_model(self)[0]


def admit_original_source_model(content, *, expected_sha256, locator):
    """Decode the original producer with its canonical owner, without fitting."""
    if not isinstance(content, bytes) or _sha(content) != expected_sha256:
        raise ValueError("original source model bytes differ from their external identity")
    model = SpatialDecisionModel.load(io.BytesIO(content))
    return _admit_source_model(model, expected_sha256=expected_sha256, locator=locator)


def admit_saved_original_source_model(model, path, *, expected_sha256, locator):
    """Admit the same fit by comparing every original saved byte to its owner.

    The cold byte ingress remains available for replay. This same-fit ingress
    avoids decoding a second complete model while the fit blocks are resident.
    """
    if type(model) is not SpatialDecisionModel:
        raise TypeError("the actual original SpatialDecisionModel producer is required")
    _require_original_fit(model)
    identity = saved_record_identity(path, model._record_view())
    if (identity["sha256"] != expected_sha256 or identity["sha256"] != locator["sha256"]
            or identity["bytes"] != locator["bytes"]
            or Path(path).resolve() != Path(locator["path"]).resolve()):
        raise ValueError("saved original source model differs from its external file identity")
    return _admit_source_model(model, expected_sha256=expected_sha256, locator=locator)


def _admit_source_model(model, *, expected_sha256, locator):
    if (model.provenance["fit_scope"] != "source_training"
            or any(row["authorization"]["fit_scope"] != "source_training" for row in model.posterior.fit_records)):
        raise ValueError("data-likelihood export requires an original source-training producer")
    people = {row["authorization"]["original_view"]["participant_key"] for row in model.posterior.fit_records}
    if len(people) != 1:
        raise ValueError("one person likelihood cannot pool different independent source people")
    handle = SourceModelHandle(expected_sha256)
    _MODELS[handle] = (model, model.digest(), _json(locator), next(iter(people)), model._serialization_owners())
    return handle


def _require_model(handle):
    row = _MODELS.get(handle) if type(handle) is SourceModelHandle else None
    if (row is None or not same_record_owners(row[0]._serialization_owners(), row[4])
            or row[0].digest() != row[1]):
        raise TypeError("the original admitted source model and fit state are required")
    return row[:4]


@dataclass(frozen=True, eq=False)
class OriginalBlockOperands:
    source_file_sha256: str


def bind_original_fit_blocks(handle, blocks):
    """Join actual block arrays to the exact original fit receipts and factors."""
    model, _, _, _ = _require_model(handle)
    blocks = tuple(blocks)
    if len(blocks) != len(model.posterior.fit_records):
        raise ValueError("every original fitted source record must be present")
    for block, saved, physical in zip(blocks, model.posterior.fit_records, model.posterior.structural_factors, strict=True):
        if type(block) is not SpatialResponseBlock:
            raise TypeError("only original canonical SpatialResponseBlock operands are accepted")
        original = {key: value for key, value in saved.items() if key != "compression"}
        if (metadata_digest(block.record()) != metadata_digest(original)
                or not np.array_equal(block.design, physical.design)
                or (block.nuisance is None) != (physical.nuisance is None)
                or block.nuisance is not None and not np.array_equal(block.nuisance, physical.nuisance)):
            raise ValueError("source block arrays/permissions differ from the original same-fit producer")
    operands = OriginalBlockOperands(handle.source_file_sha256)
    _BLOCKS[operands] = (handle, blocks)
    return operands


def _chart(model):
    prior = model.posterior.prior
    return full_response_chart(prior.layout, prior.design_binding, prior.mean.shape[0])


def full_response_chart(layout, design_binding, response_coefficients):
    """The common physical chart, independent of an auxiliary posterior."""
    p = response_coefficients
    if p % 2 or tuple(design_binding["coefficient_groups"]) != ("common", "target_increment"):
        raise ValueError("the physical common/target_increment coefficient programme must be explicit")
    names = tuple(f"{group}:basis:{i}:channel:{channel}" for group in ("common", "target_increment")
                  for i in range(p//2) for channel in layout.channels)
    layout = layout.record()
    definition = {"schema": "native_full_response_coefficient_chart/1",
        "coefficient_order": "coefficient_major_C_order_P_by_C",
        "channels": layout["channels"], "response_coefficients": p,
        "acquisition_reference": layout["acquisition_reference"], "signal_unit": layout["signal_unit"],
        "channel_types": layout["channel_types"],
        "original_spatial_design_binding": json.loads(_json(design_binding)),
        "individual_prior_not_part_of_likelihood": True}
    return names, definition


def _recover_block(model, block, saved, physical, information, response):
    """Use the original saved compression; no noise factorization or QR rerun."""
    compression = physical.compression
    observation = physical.observation
    values = (block.samples/observation.equilibration_scales) @ observation.normalized_support_basis
    white = solve_triangular(compression.observed_noise_root, values.T, lower=True).T
    if compression.nuisance_q is not None:
        white = white-compression.nuisance_q @ (compression.nuisance_q.T @ white)
    projected = compression.design_q.T @ white
    expected_response = projected.flatten(order="F")
    magnitude = (np.abs(compression.design_q).T @ np.abs(white)).flatten(order="F") + np.abs(response)
    _relation(expected_response, response, magnitude, len(white))
    # A/r are the original saved arrays, not the re-evaluated projection.
    outside = white-compression.design_q @ projected
    energy = squared_norm_record(outside)
    nuisance_dimensions = 0 if compression.nuisance_root is None else len(compression.nuisance_root)
    rows, channels = len(white)-nuisance_dimensions, compression.observed_noise_root.shape[0]
    noise_logdet = 2*float(np.log(np.diag(compression.observed_noise_root)).sum())+2*observation.coordinate_log_volume
    nuisance_logdet = (0. if compression.nuisance_root is None else
        2*float(np.log(np.abs(np.diag(compression.nuisance_root))).sum()))
    temporal = float(block.temporal_log_abs_det)
    constant = block_log_constant(rows, channels, noise_logdet, nuisance_logdet,
                                  energy["rounded_squared_norm"], temporal)
    authorization = saved["authorization"]
    record = {"record_id": authorization["record_id"], "fit_scope": authorization["fit_scope"],
        "source_sha256": authorization["source_sha256"], "raw_sample_support": list(authorization["raw_sample_support"]),
        "original_fit_labels_sha256": authorization["original_fit_labels_sha256"],
        "original_fit_onsets_sha256": authorization["original_fit_onsets_sha256"],
        "labels": model.baseline.provenance["records"][
            [r["authorization"]["record_id"] for r in model.posterior.fit_records].index(authorization["record_id"])]["n_events"],
        "original_fit_record_sha256": metadata_digest(saved),
        "original_input_sha256": saved["actual_input_sha256"],
        "original_information_sha256": array_digest(information, response),
        "original_noise_binding_sha256": metadata_digest(model.posterior.noise_binding),
        "original_compression_sha256": compression._record_sha256,
        "original_temporal_program_sha256": authorization["temporal_program_sha256"],
        "rows_after_nuisance": rows, "observed_spatial_dimensions": channels,
        "logdet_noise": noise_logdet, "logdet_nuisance": nuisance_logdet,
        "temporal_log_abs_det": temporal, "outside": energy, "log_constant": float(constant),
        "constant_formula": "-.5*(rows*channels*log(2pi)+rows*logdet_noise+channels*logdet_nuisance+outside)+channels*temporal_log_abs_det"}
    return float(constant), record


@dataclass(frozen=True, eq=False)
class ExportedSourceLikelihood:
    matrix: np.ndarray = field(repr=False)
    response: np.ndarray = field(repr=False)
    log_constant: float
    metadata_json: str

    def __post_init__(self):
        matrix, response = _freeze(self.matrix), _freeze(self.response)
        metadata = json.loads(self.metadata_json)
        if (matrix.ndim != 2 or response.shape != (len(matrix),) or len(matrix) > matrix.shape[1]
                or not np.isfinite(self.log_constant)
                or array_digest(matrix, response) != metadata["compression"]["compressed_information_sha256"]
                or self.log_constant != metadata["log_constant"]
                or sorted(metadata["column_permutation"]) != list(range(matrix.shape[1]))
                or len(metadata["coefficient_names"]) != matrix.shape[1]
                or metadata["prior_information_in_data_likelihood"] is not False
                or metadata["raw_or_label_authority_granted"] is not False):
            raise ValueError("the full-column compact source likelihood differs from its numerical receipt")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "metadata_json", _json(metadata))

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    def log_kernel(self, native_coefficients):
        native_coefficients = np.asarray(native_coefficients, dtype=float)
        if native_coefficients.shape != (self.matrix.shape[1],) or not np.isfinite(native_coefficients).all():
            raise ValueError("a finite one-dimensional native vec_C coefficient vector is required")
        residual = self.response-self.matrix@native_coefficients
        return self.log_constant-.5*squared_norm_record(residual)["rounded_squared_norm"]

    def population_input(self, core, *, compact_manifest_sha256):
        metadata = self.metadata
        chart = core.ResponseChart.from_definition(metadata["coefficient_names"], metadata["physical_chart"])
        dependencies = tuple(core.SourceRecordDependency(row["record_id"], row["source_sha256"],
            row["original_fit_labels_sha256"], tuple(row["raw_sample_support"]), row["labels"])
            for row in metadata["block_constants"])
        return core.FullResponseLikelihood(metadata["participant_key"], tuple(metadata["identity_keys"]), chart,
            self.matrix, self.response, self.log_constant, dependencies,
            compact_manifest_sha256, metadata["exporter_sha256"])


def export_original_source_likelihood(operands):
    registered = _BLOCKS.get(operands) if type(operands) is OriginalBlockOperands else None
    if registered is None:
        raise TypeError("original admitted same-fit source block operands are required")
    handle, blocks = registered
    model, model_identity, locator, person = _require_model(handle)
    # Rejoin at consumption; metadata state is never accepted as a capability.
    bind_original_fit_blocks(handle, blocks)
    posterior = model.posterior
    p, channels = posterior.prior.mean.shape
    permutation = native_column_permutation(p, channels)
    constants, records = [], []
    for block, saved, physical, information, response in zip(blocks, posterior.fit_records,
            posterior.structural_factors, posterior.information_roots, posterior.responses, strict=True):
        constant, record = _recover_block(model, block, saved, physical, information, response)
        constants.append(constant)
        records.append(record)
    original = np.vstack(posterior.information_roots)[:, permutation]
    response = np.concatenate(posterior.responses)
    matrix, values, constant, compression = compress_numeric_likelihood(original, response, sum(constants))
    names, definition = _chart(model)
    metadata = {"schema": "native_full_response_source_likelihood_export/1", "participant_key": person,
        "identity_keys": [person], "coefficient_names": list(names), "physical_chart": definition,
        "column_permutation": permutation.tolist(),
        "column_rule": "A_native[:,i*C+j]=A_spatial[:,j*P+i]",
        "source_file_sha256": handle.source_file_sha256, "source_locator": json.loads(locator),
        "source_model_artifact_sha256": model_identity,
        "source_model_producer_sha256": model.producer_implementation_sha256,
        "source_model_consumer_sha256": model.consumer_implementation_sha256,
        "source_posterior_artifact_sha256": posterior.digest(),
        "source_prior_numeric_chart_sha256": posterior.prior.digest(),
        "prior_information_in_data_likelihood": False, "block_constants": records,
        "block_log_constants_sum": sum(constants), "compression": compression,
        "log_constant": constant, "exporter_sha256": IMPLEMENTATION_SHA256,
        "raw_or_label_authority_granted": False}
    return ExportedSourceLikelihood(matrix, values, constant, _json(metadata))


@dataclass(frozen=True)
class SharedLikelihoodReference:
    manifest_name: str
    sha256: str
    bytes: int
    participant_key: str

    def record(self):
        return {"manifest_name": self.manifest_name, "sha256": self.sha256, "bytes": self.bytes,
                "participant_key": self.participant_key}


class SharedLikelihoodStore:
    """One immutable A/r per person artifact, referenced across all source folds."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._population_inputs = {}

    def _write_once(self, name, value):
        path = self.directory/name
        if path.exists():
            if path.read_bytes() != value:
                raise ValueError("immutable shared likelihood path has conflicting bytes")
        else:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".publishing-", delete=False) as temporary:
                temporary.write(value)
                staged = Path(temporary.name)
            staged.replace(path)

    def put(self, likelihood):
        stream = io.BytesIO()
        np.savez_compressed(stream, matrix=likelihood.matrix, response=likelihood.response)
        payload = stream.getvalue()
        array_sha = _sha(payload)
        record = {"schema": "shared_full_response_source_likelihood/1", "metadata": likelihood.metadata,
            "log_constant": likelihood.log_constant, "arrays": {"file": array_sha+".npz",
                "bytes": len(payload), "sha256": array_sha}, "raw_or_label_authority_granted": False}
        content = (_json(record)+"\n").encode()
        manifest_sha = _sha(content)
        for name, value in ((array_sha+".npz", payload), (manifest_sha+".json", content)):
            self._write_once(name, value)
        reference = SharedLikelihoodReference(manifest_sha+".json", manifest_sha, len(content), likelihood.metadata["participant_key"])
        self._cache[manifest_sha] = reference, likelihood
        return reference

    def get(self, reference):
        if reference.sha256 in self._cache:
            original_reference, value = self._cache[reference.sha256]
            if reference != original_reference:
                raise ValueError("shared source reference changed participant identity")
            return value
        if reference.manifest_name != reference.sha256+".json":
            raise ValueError("shared likelihood manifest must retain its content identity")
        content = (self.directory/reference.manifest_name).read_bytes()
        if len(content) != reference.bytes or _sha(content) != reference.sha256:
            raise ValueError("shared source likelihood manifest bytes changed")
        record = json.loads(content)
        if (record["schema"] != "shared_full_response_source_likelihood/1"
                or record["raw_or_label_authority_granted"] is not False
                or record["arrays"]["file"] != record["arrays"]["sha256"]+".npz"):
            raise ValueError("shared likelihood numerical-only scope or array identity changed")
        payload = (self.directory/record["arrays"]["file"]).read_bytes()
        if len(payload) != record["arrays"]["bytes"] or _sha(payload) != record["arrays"]["sha256"]:
            raise ValueError("shared likelihood numeric arrays changed")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            if set(arrays.files) != {"matrix", "response"}:
                raise ValueError("shared source payload contains unexpected operands")
            value = ExportedSourceLikelihood(arrays["matrix"], arrays["response"], record["log_constant"], _json(record["metadata"]))
        if value.metadata["participant_key"] != reference.participant_key:
            raise ValueError("shared source reference changed participant identity")
        self._cache[reference.sha256] = reference, value
        return value

    def fold_inputs(self, fold, references, *, population_core):
        by_person = {ref.participant_key: ref for ref in references}
        if len(by_person) != len(references) or not set(fold.source_participant_keys) <= set(by_person):
            raise ValueError("the shared source roster lacks complete independent fold participants")
        # Only small references are repeated; matrices are cached once per store.
        result = []
        for person in fold.source_participant_keys:
            reference = by_person[person]
            likelihood = self.get(reference)
            cache_key = reference.sha256, population_core.FullResponseLikelihood
            if cache_key not in self._population_inputs:
                self._population_inputs[cache_key] = likelihood.population_input(population_core,
                    compact_manifest_sha256=reference.sha256)
            result.append(self._population_inputs[cache_key])
        return tuple(result)

    def put_fold_references(self, fold, references):
        """Persist only the fold and shared dependency refs, never copies of A/r."""
        by_person = {r.participant_key: r for r in references}
        if len(by_person) != len(references) or not set(fold.source_participant_keys) <= set(by_person):
            raise ValueError("shared likelihood references lack the complete source fold")
        selected = [by_person[key] for key in fold.source_participant_keys]
        actual = []
        for ref in selected:
            value = self.get(ref)
            metadata = value.metadata
            if set(metadata["identity_keys"]) & set(fold.target_identity_keys):
                raise ValueError("the excluded target entered shared source fold references")
            for row in metadata["block_constants"]:
                if row["source_sha256"] in fold.excluded_eeg_sha256s or row["record_id"] in fold.excluded_record_ids:
                    raise ValueError("excluded original target bytes/record entered shared source references")
                actual.append((ref.participant_key, row["record_id"], row["source_sha256"], row["labels"]))
        if set(actual) != set(fold.expected_source_records):
            raise ValueError("shared source record/label dependencies differ from the original fold")
        record = {"schema": "shared_full_response_source_fold_references/1", "fold": fold.record(),
                  "likelihood_references": [ref.record() for ref in selected], "raw_or_label_authority_granted": False}
        content = (_json(record)+"\n").encode()
        digest = _sha(content)
        self._write_once(digest+".fold.json", content)
        return {"file": digest+".fold.json", "sha256": digest, "bytes": len(content)}

    def load_fold_inputs(self, reference, *, population_core):
        if reference["file"] != reference["sha256"]+".fold.json":
            raise ValueError("the shared source fold file changed its content identity")
        content = (self.directory/reference["file"]).read_bytes()
        if len(content) != reference["bytes"] or _sha(content) != reference["sha256"]:
            raise ValueError("shared source fold dependency bytes changed")
        record = json.loads(content)
        f = dict(record["fold"])
        if (record["schema"] != "shared_full_response_source_fold_references/1"
                or record["raw_or_label_authority_granted"] is not False
                or f.pop("exclusion_scope") != "target_participant_all_records_and_source_hyperlearning"):
            raise ValueError("saved shared source fold roles changed")
        fold = population_core.ParticipantFold(**f)
        references = tuple(SharedLikelihoodReference(**r) for r in record["likelihood_references"])
        inputs = self.fold_inputs(fold, references, population_core=population_core)
        # Recheck all source dependency links at the new ingress; no fitting.
        self.put_fold_references(fold, references)
        return fold, inputs
