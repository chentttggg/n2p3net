"""Same-fit source data factors, without an unused physical posterior.

Live publication authority comes only from the original FullSourcePerson and
its actual noise fit/blocks. Cold products contain numerical state and original
dependencies; they cannot mint reader, label or live-fit authority.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from research.execution import _atomic_write_stream
from research.population_noise_reference import (
    SourceStationaryMoments,
    moments_from_source_baseline,
)
from research.source_likelihood_export import (
    IMPLEMENTATION_SHA256 as EXPORTER_SHA256,
)
from research.source_likelihood_export import (
    ExportedSourceLikelihood,
    SharedLikelihoodReference,
    SharedLikelihoodStore,
    _freeze,
    block_log_constant,
    compress_numeric_likelihood,
    full_response_chart,
    native_column_permutation,
    squared_norm_record,
)
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.noise_model import array_digest, freeze_metadata, metadata_digest
from transfer.spatial_programme import (
    _fit_controller_receipt,
    _layout,
    _observation,
    _response_block,
)
from transfer.spatial_signal_prior import (
    SpatialLayout,
    SpatialObservation,
    SpatialResponseBlock,
    _ExactRowSpace,
    _spd,
    _temporal_structural_rowspace,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PRODUCT_KIND = "same_fit_full_response_data_factor_v2"
_LIVE, _SAVED = WeakKeyDictionary(), WeakKeyDictionary()


@dataclass(frozen=True)
class SourceDataFactorProgram:
    observation: str = "native"
    reference_channel: str | None = None

    def __post_init__(self):
        if (self.observation not in {"native", "car", "measured_reference"}
                or self.observation == "measured_reference" and not self.reference_channel
                or self.observation != "measured_reference" and self.reference_channel is not None):
            raise ValueError("source data compression needs its actual physical observation")

    def record(self):
        return {"schema": "same_fit_full_response_data_programme/1", "observation": self.observation,
            "reference_channel": self.reference_channel, "noise_programme": "original_continuous_fit_then_freeze",
            "response_programme": "full_column_data_factor_without_auxiliary_posterior",
            "individual_proper_prior_in_data_information": False}

    @classmethod
    def from_record(cls, record):
        value = cls(record["observation"], record["reference_channel"])
        if value.record() != record:
            raise ValueError("the source data-factor programme has a different declared role")
        return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _artifact(path):
    path = Path(path).resolve()
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _read(reference):
    content = Path(reference["path"]).read_bytes()
    if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError("the source data-factor dependency differs from its external bytes")
    return content


def _write(path, content):
    if Path(path).exists():
        raise FileExistsError(path)
    return _atomic_write_stream(path, lambda: iter((content,)), len(content))


def _write_record(path, value):
    _write(path, (_json(value) + "\n").encode())
    return _artifact(path)


def _physical_map(observation):
    matrix = (observation._structural_matrix()
              if observation.declaration["kind"] == "average_actual_EEG" else observation.matrix)
    return [[str(Fraction.from_float(float(value))) for value in row] for row in matrix]


def _support_component(block, input_record):
    _, temporal = _temporal_structural_rowspace(block.design, block.nuisance)
    return {"record_id": block.authorization["record_id"], "observation": block.observation.record(),
        "spatial_map": _physical_map(block.observation), "temporal_rowspace": temporal,
        "design_shape": list(block.design.shape),
        "nuisance_columns": 0 if block.nuisance is None else block.nuisance.shape[1],
        "actual_design_sha256": array_digest(block.design),
        "actual_nuisance_sha256": None if block.nuisance is None else array_digest(block.nuisance),
        "actual_input_sha256": input_record["actual_input_sha256"],
        "coverage": "all_actual_binary64_rows_or_exact_full_column_span"}


def validate_structural_support(support, *, columns):
    """Check compact exact tensor components; raw correspondence is issuer-bound.

    A rank-deficient component explicitly records a complete-row traversal.
    A full-column span permits the original owner's mathematically exact early
    stop. No union of unrelated spatial/temporal spaces is substituted.
    """
    if (support["schema"] != "source_exact_tensor_space_sum/1"
            or support["columns"] != columns
            or support["coefficient_order"] != "column_major_response_by_native_EEG"
            or not support["components"]):
        raise ValueError("the complete exact source support has a different role")
    for component in support["components"]:
        observation = SpatialObservation.from_record(component["observation"])
        temporal = component["temporal_rowspace"]
        n, p = component["design_shape"]
        nuisance = component["nuisance_columns"]
        width = p + nuisance
        if (p * len(observation.layout.channels) != columns or n < 1
                or tuple(map(tuple, component["spatial_map"])) != tuple(map(tuple, _physical_map(observation)))
                or temporal["schema"] != "n2p3_spatial_temporal_rowspace/1"
                or temporal["original_rows"] != n or temporal["joint_columns"] != width
                or component["coverage"] != "all_actual_binary64_rows_or_exact_full_column_span"):
            raise ValueError("source tensor support differs from its physical original coordinates")
        exact = _ExactRowSpace(width)
        original = tuple(tuple(Fraction(value) for value in row) for row in temporal["joint_rref"])
        for row in original:
            exact.add(row)
        quotient = tuple(tuple(row[nuisance:]) for row in exact.rows if not any(row[:nuisance]))
        selected = temporal["selected_original_rows"]
        full = len(exact.pivots) == width
        if (tuple(tuple(row) for row in exact.rows) != original
                or len(original) != temporal["joint_rank"]
                or temporal["full_column_span"] != full
                or len(selected) != len(original) or len(set(selected)) != len(selected)
                or any(type(i) is not int or i < 0 or i >= temporal["rows_examined"] for i in selected)
                or not len(original) <= temporal["rows_examined"] <= n
                or not full and temporal["rows_examined"] != n
                or tuple(tuple(Fraction(v) for v in row) for row in temporal["response_quotient"]) != quotient
                or temporal["response_quotient_rank"] != len(quotient)
                or temporal["nuisance_rank"] != len(original) - len(quotient)):
            raise ValueError("the exact source quotient or complete-row coverage witness changed")
    return support


def _energy_value(record):
    scale, unit = float.fromhex(record["scale_hex"]), float.fromhex(record["unit_square_sum_hex"])
    if not np.isfinite([scale, unit]).all() or min(scale, unit) < 0:
        raise ValueError("source residual energy needs its finite nonnegative original scale")
    exact = Fraction.from_float(scale)**2 * Fraction.from_float(unit)
    rounded = float(exact)
    if (Fraction(record["scaled_squared_norm_rational"]) != exact
            or record["rounded_squared_norm"] != rounded
            or record["nonzero_energy_rounds_to_zero"] != bool(exact and rounded == 0)):
        raise ValueError("the source residual energy differs from its scaled representation")
    return rounded


def _validate_constants(anchor):
    components = anchor["structural_support"]["components"]
    records = anchor["block_constants"]
    if len(components) != len(records):
        raise ValueError("each source constant needs its own actual record-local support")
    for record, component in zip(records, components, strict=True):
        observed = SpatialObservation.from_record(component["observation"])
        n, _ = component["design_shape"]
        if (record["record_id"] != component["record_id"]
                or record["original_input_sha256"] != component["actual_input_sha256"]
                or record["rows_after_nuisance"] != n-component["nuisance_columns"]
                or record["observed_spatial_dimensions"] != observed.normalized_range_map.shape[0]):
            raise ValueError("source physical normalizer changed its actual rows or observed support")
        expected = block_log_constant(record["rows_after_nuisance"], record["observed_spatial_dimensions"],
            record["logdet_noise"], record["logdet_nuisance"], _energy_value(record["outside"]), record["temporal_log_abs_det"])
        if record["log_constant"] != expected:
            raise ValueError("the source block lost its absolute physical normalizer")
    compression = anchor["compression"]
    expected = float(sum(row["log_constant"] for row in records))-.5*_energy_value(compression["outside"])
    if (anchor["log_constant"] != expected or compression["columns"] != anchor["matrix_shape"][1]
            or compression["compressed_rows"] != anchor["matrix_shape"][0]
            or compression["all_physical_columns_retained"] is not True):
        raise ValueError("the compact source factor lost its full-column absolute data likelihood")


@dataclass(frozen=True)
class BlockDataFactor:
    information: np.ndarray = field(repr=False)
    response: np.ndarray = field(repr=False)
    log_constant: float
    constants: dict
    support: dict


def compile_block_data_factor(block, native_noise, native_root=None):
    """Numerical core only; it deliberately issues no live-source capability."""
    if type(block) is not SpatialResponseBlock:
        raise TypeError("the original canonical response block is required")
    information, response, outside, n, channels, noise_logdet, nuisance_logdet = (
        block._compressed_components(native_noise, native_root, return_factors=False))
    energy = squared_norm_record(outside)
    constant = block_log_constant(n, channels, noise_logdet, nuisance_logdet,
                                  energy["rounded_squared_norm"], block.temporal_log_abs_det)
    components = {"rows_after_nuisance": n, "observed_spatial_dimensions": channels,
        "logdet_noise": noise_logdet, "logdet_nuisance": nuisance_logdet,
        "temporal_log_abs_det": float(block.temporal_log_abs_det), "outside": energy,
        "log_constant": constant,
        "constant_formula": "-.5*(rows*channels*log(2pi)+rows*logdet_noise+channels*logdet_nuisance+outside)+channels*temporal_log_abs_det"}
    return BlockDataFactor(_freeze(information), _freeze(response), constant,
                           freeze_metadata(components), freeze_metadata(_support_component(block, block.record())))


@dataclass(frozen=True, eq=False)
class SourceDataFactorFit:
    participant_key: str


@dataclass(frozen=True)
class _FitState:
    baseline: ContinuousDecisionModel
    baseline_content: bytes = field(repr=False)
    layout: SpatialLayout
    matrix: np.ndarray = field(repr=False)
    response: np.ndarray = field(repr=False)
    log_constant: float
    anchor: dict

    def signature(self):
        # Arrays own immutable bytes, metadata is recursively frozen. Retain
        # original owners/view coordinates instead of rehashing their payloads
        # at each same-fit publication gate.
        return (id(self.baseline), self.baseline.digest(), id(self.baseline_content), id(self.layout),
            id(self.matrix), self.matrix.shape, self.matrix.strides, self.matrix.dtype.str,
            id(self.response), self.response.shape, self.response.strides, self.response.dtype.str,
            self.log_constant, id(self.anchor))


def _require_live(value):
    row = _LIVE.get(value) if type(value) is SourceDataFactorFit else None
    if row is None or row[0].signature() != row[1] or value.participant_key != row[0].anchor["source_person"]["participant_key"]:
        raise TypeError("the unchanged original full-source data-factor fit issuer is required")
    return row[0]


def fit_source_data_factor(person, *, data_programme, dictionary, noise_options):
    """One original noise fit and one per-record data compression, no posterior."""
    from research.public_population_source import require_full_source_person

    person = require_full_source_person(person)
    if type(data_programme) is not SourceDataFactorProgram:
        raise TypeError("the named original-noise-fixed data-factor declaration is required")
    source_person = person.record
    prefix = person.prefix
    geometry = source_person["geometry"]
    records = person.records
    receipts = tuple(_fit_controller_receipt(record, receipt)
                     for record, receipt in zip(records, person.receipts, strict=True))
    if not records:
        raise ValueError("the full source programme requires its original records")
    layouts = tuple(_layout(receipt) for receipt in receipts)
    for record, receipt, layout in zip(records, receipts, layouts, strict=True):
        if (record.record_id != receipt["record_id"]
                or len(record.samples) != receipt["raw_scope"][1] - receipt["raw_scope"][0]
                or tuple(receipt["channel_names"]) != tuple(geometry["channels"])
                or receipt["sample_rate_hz"] != geometry["sfreq"]
                or layout.record() != layouts[0].record()
                or receipt["participant_key"] != source_person["participant_key"]):
            raise ValueError("same-fit source records differ from their original physical authority")
    baseline = ContinuousDecisionModel.fit(records, sfreq=geometry["sfreq"], channels=tuple(geometry["channels"]),
        candidate_ids=tuple(prefix.target_contract["candidate_vocabulary"]),
        operator_identity=geometry["operator_identity"], dictionary=dictionary,
        fit_scope="source_training", **noise_options)
    if baseline.noise is None or baseline.noise.startup_mode == "conditional_prefix":
        raise ValueError("the complete original stationary source-noise programme is required")
    content = baseline.to_bytes()
    native_noise, native_root = _spd(baseline.innovation_covariance, "original source native noise")
    observation = _observation(layouts[0], data_programme.observation, data_programme.reference_channel)
    from transfer.continuous_decision import _event_design

    block_factors, constants, supports = [], [], []
    for record, receipt, original in zip(records, receipts, baseline.provenance["records"], strict=True):
        block = _response_block(record.samples, _event_design(record.onsets, record.target_mask,
            len(record.samples), baseline.basis), observation, baseline, receipt,
            scope="source_training", original_record=record)
        if (original["record_id"] != record.record_id
                or original["raw_input_sha256"] != block.authorization["original_raw_fit_input_sha256"]
                or original["n_events"] != len(record.onsets)):
            raise ValueError("source compression differs from the original noise-fit inputs")
        factor = compile_block_data_factor(block, native_noise, native_root)
        block_factors.append((factor.information, factor.response))
        authority = block.authorization
        controller_scope = {name: receipt[name] for name in ("schema", "record_id", "participant_key",
            "source_dataset_id", "source_subject_id", "source_sha256", "sample_rate_hz", "channel_names",
            "channel_types", "reference", "signal_unit", "raw_scope", "raw_input_digest", "source_prefix_identity",
            "original_compiler_source_sha256", "label_use", "onsets")}
        constants.append({**factor.constants, "record_id": record.record_id, "fit_scope": "source_training",
            "source_sha256": authority["source_sha256"], "raw_sample_support": list(authority["raw_sample_support"]),
            "original_fit_labels_sha256": authority["original_fit_labels_sha256"],
            "original_fit_onsets_sha256": authority["original_fit_onsets_sha256"], "labels": original["n_events"],
            "original_temporal_program_sha256": authority["temporal_program_sha256"],
            "original_input_sha256": factor.support["actual_input_sha256"],
            "original_raw_fit_input_sha256": authority["original_raw_fit_input_sha256"],
            "original_information_sha256": array_digest(factor.information, factor.response),
            "original_controller_sha256": metadata_digest(receipt), "controller_scope": controller_scope})
        supports.append(factor.support)
    p, channels = 2 * baseline.basis.shape[1], len(baseline.channels)
    permutation = native_column_permutation(p, channels)
    matrix, response, constant, compression = compress_numeric_likelihood(
        np.vstack([a for a, _ in block_factors])[:, permutation],
        np.concatenate([r for _, r in block_factors]), sum(row["log_constant"] for row in constants))
    design_binding = {"dictionary_sha256": baseline.dictionary.digest(), "actual_basis_sha256": array_digest(baseline.basis),
        "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": baseline.sfreq,
        "operator_identity": baseline.operator_identity, "coefficient_groups": ["common", "target_increment"]}
    names, chart = full_response_chart(layouts[0], design_binding, p)
    support = {"schema": "source_exact_tensor_space_sum/1", "columns": p*channels,
        "coefficient_order": "column_major_response_by_native_EEG", "components": supports}
    validate_structural_support(support, columns=matrix.shape[1])
    anchor = freeze_metadata({"schema": "source_same_fit_full_response_data_anchor/1", "source_product_kind": PRODUCT_KIND,
        "source_person": source_person, "source_data_programme": data_programme.record(),
        "layout": layouts[0].record(), "coefficient_names": names, "physical_chart": chart,
        "column_permutation": permutation.tolist(), "column_rule": "A_native[:,i*C+j]=A_spatial[:,j*P+i]",
        "block_constants": constants, "structural_support": support, "compression": compression,
        "matrix_shape": list(matrix.shape), "response_shape": list(response.shape), "log_constant": constant,
        "baseline_artifact_sha256": baseline.digest(), "prior_information_in_data_likelihood": False,
        "raw_or_label_authority_granted": False, "producer_sha256": IMPLEMENTATION_SHA256})
    state = _FitState(baseline, content, layouts[0], matrix, response, constant, anchor)
    value = SourceDataFactorFit(source_person["participant_key"])
    _LIVE[value] = state, state.signature()
    return value


@dataclass(frozen=True)
class ColdSourceDataFactor:
    likelihood: ExportedSourceLikelihood
    moments: SourceStationaryMoments
    anchor: dict
    product: dict
    reference: dict


def inspect_source_data_factor(reference):
    """Read small role/dependency manifests; issue no live-source capability."""
    product = json.loads(_read(reference))
    if (product["schema"] != "same_fit_source_data_factor_product/1"
            or product["source_product_kind"] != PRODUCT_KIND
            or product["noise_fit_programmes"] != 1
            or product["fixed_noise_response_compression_passes"] != 1
            or product["auxiliary_physical_posterior_fits"] != 0
            or product["raw_or_label_authority_granted"] is not False):
        raise ValueError("the source product differs from its declared data-only same-fit role")
    anchor = json.loads(_read(product["source_anchor"]))
    if (anchor["schema"] != "source_same_fit_full_response_data_anchor/1"
            or anchor["source_product_kind"] != PRODUCT_KIND
            or anchor["prior_information_in_data_likelihood"] is not False
            or anchor["raw_or_label_authority_granted"] is not False
            or len(anchor["block_constants"]) != product["same_fit_block_count"]
            or product["shared_likelihood"]["participant_key"] != anchor["source_person"]["participant_key"]):
        raise ValueError("source product dependencies differ from the original complete person")
    SourceDataFactorProgram.from_record(anchor["source_data_programme"])
    return product, anchor


def load_source_data_factor(reference, *, shared_store):
    """Cold numeric restore; neither refits nor acquires live-fit authority."""
    product, anchor = inspect_source_data_factor(reference)
    baseline = ContinuousDecisionModel.load(io.BytesIO(_read(anchor["baseline"])))
    if baseline.digest() != anchor["baseline_artifact_sha256"] or baseline.provenance["fit_scope"] != "source_training":
        raise ValueError("the original same-fit source noise dependency changed")
    layout = SpatialLayout.from_record(anchor["layout"])
    binding = {"dictionary_sha256": baseline.dictionary.digest(), "actual_basis_sha256": array_digest(baseline.basis),
        "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": baseline.sfreq,
        "operator_identity": baseline.operator_identity, "coefficient_groups": ["common", "target_increment"]}
    names, chart = full_response_chart(layout, binding, 2*baseline.basis.shape[1])
    permutation = native_column_permutation(2*baseline.basis.shape[1], len(baseline.channels))
    if (layout.channels != baseline.channels or chart != anchor["physical_chart"]
            or list(names) != anchor["coefficient_names"] or permutation.tolist() != anchor["column_permutation"]):
        raise ValueError("the complete native source chart differs from its actual original noise/dictionary programme")
    likelihood = shared_store.get(SharedLikelihoodReference(**product["shared_likelihood"]))
    metadata = likelihood.metadata
    if (metadata["schema"] != "native_full_response_source_data_factor_export/2"
            or metadata["source_anchor"] != product["source_anchor"]
            or metadata["source_product_kind"] != PRODUCT_KIND
            or metadata["block_constants"] != anchor["block_constants"]
            or metadata["compression"] != anchor["compression"]
            or metadata["physical_chart"] != anchor["physical_chart"]
            or metadata["coefficient_names"] != anchor["coefficient_names"]
            or metadata["column_permutation"] != anchor["column_permutation"]
            or list(likelihood.matrix.shape) != anchor["matrix_shape"]
            or list(likelihood.response.shape) != anchor["response_shape"]
            or likelihood.log_constant != anchor["log_constant"]):
        raise ValueError("the saved complete A/r/c differs from its original source anchor")
    validate_structural_support(anchor["structural_support"], columns=likelihood.matrix.shape[1])
    _validate_constants(anchor)
    moments = SourceStationaryMoments.from_record(product["stationary_moments"])
    expected = moments_from_source_baseline(baseline, layout=layout,
        participant_key=anchor["source_person"]["participant_key"], dependency={
            "source_product_anchor": product["source_anchor"], "source_baseline": anchor["baseline"],
            "source_baseline_artifact_sha256": baseline.digest(), "source_likelihood": product["shared_likelihood"]})
    saved_dependency = {k: v for k, v in moments.dependency.items() if k != "moment_producer_sha256"}
    expected_dependency = {k: v for k, v in expected.dependency.items() if k != "moment_producer_sha256"}
    if (moments.geometry.record() != expected.geometry.record()
            or not np.array_equal(moments.zero_root, expected.zero_root)
            or not np.array_equal(moments.lag_factor, expected.lag_factor)
            or saved_dependency != expected_dependency):
        raise ValueError("source stationary moments differ from actual saved PARCOR/full Q")
    return ColdSourceDataFactor(likelihood, moments, freeze_metadata(anchor), freeze_metadata(product), freeze_metadata(reference))


def save_source_data_factor(value, directory, *, shared_store):
    state = _require_live(value)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory/"original_noise.npz", state.baseline_content)
    baseline_reference = _artifact(directory/"original_noise.npz")
    anchor = {**state.anchor, "baseline": baseline_reference}
    anchor_reference = _write_record(directory/"source-fit-anchor.json", anchor)
    metadata = {"schema": "native_full_response_source_data_factor_export/2", "source_product_kind": PRODUCT_KIND,
        "participant_key": value.participant_key, "identity_keys": [value.participant_key],
        **{name: anchor[name] for name in ("coefficient_names", "physical_chart", "column_permutation", "column_rule",
            "block_constants", "compression", "log_constant")},
        "source_anchor": anchor_reference, "source_producer_sha256": IMPLEMENTATION_SHA256,
        "prior_information_in_data_likelihood": False, "raw_or_label_authority_granted": False,
        "block_log_constants_sum": sum(row["log_constant"] for row in anchor["block_constants"]),
        "exporter_sha256": EXPORTER_SHA256}
    likelihood = ExportedSourceLikelihood(state.matrix, state.response, state.log_constant, _json(metadata))
    compact = shared_store.put(likelihood)
    moments = moments_from_source_baseline(state.baseline, layout=state.layout, participant_key=value.participant_key,
        dependency={"source_product_anchor": anchor_reference, "source_baseline": baseline_reference,
            "source_baseline_artifact_sha256": state.baseline.digest(), "source_likelihood": compact.record()})
    product = {"schema": "same_fit_source_data_factor_product/1", "source_product_kind": PRODUCT_KIND,
        "source_anchor": anchor_reference, "shared_likelihood": compact.record(), "stationary_moments": moments.record(),
        "noise_fit_programmes": 1, "fixed_noise_response_compression_passes": 1, "auxiliary_physical_posterior_fits": 0,
        "same_fit_block_count": len(anchor["block_constants"]), "raw_or_label_authority_granted": False,
        "producer_sha256": IMPLEMENTATION_SHA256}
    reference = _write_record(directory/"source-data-factor.json", product)
    # Force the actual cold byte/array/noise/constant path, not the live cache.
    cold_store = SharedLikelihoodStore(shared_store.directory)
    cold = load_source_data_factor(reference, shared_store=cold_store)
    if (not np.array_equal(cold.likelihood.matrix, state.matrix)
            or not np.array_equal(cold.likelihood.response, state.response)
            or cold.likelihood.log_constant != state.log_constant):
        raise ValueError("source data-factor publication changed the actual live operands")
    _SAVED[value] = _json(reference), cold
    return compact, reference


def require_saved_source_data_factor(value, reference):
    """Join the actual saved product to its still-live original fit owner."""
    _require_live(value)
    saved = _SAVED.get(value)
    if saved is None or saved[0] != _json(reference):
        raise TypeError("the actual same-fit saved source product is required for publication")
    product, anchor = inspect_source_data_factor(reference)
    if metadata_digest(product) != metadata_digest(saved[1].product) or metadata_digest(anchor) != metadata_digest(saved[1].anchor):
        raise ValueError("the original same-fit source publication changed after saving")
    return saved[1]
