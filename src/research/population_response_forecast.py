"""Numeric full-response population forecasting; this module grants no data rights.

Inputs are already compiled fixed-noise likelihood roots, not individual
posteriors.  Every fold learns from exactly its declared independent people.
The forecast factor keeps between-person and mean-estimation columns separate.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("an external SHA256 identity is required")
    return value


def _array(value, ndim):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError("finite arrays in the declared physical chart are required")
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)


def _array_id(value):
    value = np.asarray(value, dtype=np.float64)
    return _digest({"shape": list(value.shape), "dtype": "<f8",
        "bytes_sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest()})


def _generated_relation(actual, expected, magnitude):
    # Verify a saved producer relation in its own component scales.  The saved
    # values remain the law; a cold consumer never substitutes recomputation.
    operations = max(1, *(actual.shape or (1,)))
    bound = 64 * np.finfo(float).eps * operations * magnitude
    if actual.shape != expected.shape or np.any(np.abs(actual-expected) > bound):
        raise ValueError("saved forecast decomposition differs from its original saved factors")


def _proper_square_factor(factor, dimension):
    # An invertible saved square factor proves full support.  There is no
    # singular-value cutoff, covariance Gram matrix, jitter, or weak-column drop.
    if factor.shape != (dimension, dimension) or np.linalg.slogdet(factor)[0] == 0:
        raise ValueError("the declared full-support factor must be square and invertible")


@dataclass(frozen=True)
class ResponseChart:
    coefficient_names: tuple[str, ...]
    physical_definition_json: str

    def __post_init__(self):
        names = tuple(self.coefficient_names)
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("the complete response chart needs unique named coefficients")
        object.__setattr__(self, "coefficient_names", names)
        object.__setattr__(self, "physical_definition_json", _json(json.loads(self.physical_definition_json)))

    @classmethod
    def from_definition(cls, names, definition):
        return cls(tuple(names), _json(definition))

    @property
    def dimension(self):
        return len(self.coefficient_names)

    def record(self):
        return {"coefficient_names": list(self.coefficient_names),
                "physical_definition": json.loads(self.physical_definition_json)}

    @property
    def identity(self):
        return _digest(self.record())


@dataclass(frozen=True)
class SourceRecordDependency:
    record_id: str
    eeg_sha256: str
    label_scope_sha256: str
    native_support: tuple[int, int]
    labels: int

    def __post_init__(self):
        _sha(self.eeg_sha256)
        _sha(self.label_scope_sha256)
        support = tuple(self.native_support)
        if (not self.record_id or len(support) != 2 or any(type(i) is not int for i in support)
                or not 0 <= support[0] < support[1] or type(self.labels) is not int or self.labels < 0):
            raise ValueError("original record, label count and native support must be explicit")
        object.__setattr__(self, "native_support", support)

    def record(self):
        return {"record_id": self.record_id, "eeg_sha256": self.eeg_sha256,
                "label_scope_sha256": self.label_scope_sha256,
                "native_support": list(self.native_support), "labels": self.labels}


@dataclass(frozen=True, eq=False)
class FullResponseLikelihood:
    participant_key: str
    identity_keys: tuple[str, ...]
    chart: ResponseChart
    matrix: np.ndarray = field(repr=False)
    response: np.ndarray = field(repr=False)
    log_constant: float
    records: tuple[SourceRecordDependency, ...]
    original_likelihood_artifact_sha256: str
    original_producer_sha256: str

    def __post_init__(self):
        matrix, response = _array(self.matrix, 2), _array(self.response, 1)
        identities = tuple(sorted(set((*self.identity_keys, self.participant_key))))
        records = tuple(self.records)
        if (not self.participant_key or any(not key for key in identities)
                or matrix.shape != (len(response), self.chart.dimension)
                or not np.isfinite(self.log_constant) or not records
                or len({r.record_id for r in records}) != len(records)):
            raise ValueError("one original participant and its complete likelihood/record dependencies are required")
        _sha(self.original_likelihood_artifact_sha256)
        _sha(self.original_producer_sha256)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "identity_keys", identities)
        object.__setattr__(self, "records", records)

    def record(self):
        return {"participant_key": self.participant_key, "identity_keys": list(self.identity_keys),
            "chart_identity": self.chart.identity, "matrix_identity": _array_id(self.matrix),
            "response_identity": _array_id(self.response), "log_constant": float(self.log_constant),
            "likelihood": "exp(log_constant - 0.5 * norm(response - matrix @ theta)**2)",
            "records": [r.record() for r in self.records],
            "original_likelihood_artifact_sha256": self.original_likelihood_artifact_sha256,
            "original_producer_sha256": self.original_producer_sha256,
            "individual_proper_prior_in_data_information": False,
            "raw_or_label_authority_granted": False}


@dataclass(frozen=True)
class ParticipantFold:
    fold_id: str
    target_identity_keys: tuple[str, ...]
    source_participant_keys: tuple[str, ...]
    # (person, original record id, original EEG SHA, common authorized labels)
    expected_source_records: tuple[tuple[str, str, str, int], ...]
    excluded_eeg_sha256s: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]
    source_plan_sha256: str

    def __post_init__(self):
        source = tuple(self.source_participant_keys)
        target = tuple(sorted(set(self.target_identity_keys)))
        expected = tuple(tuple(r) for r in self.expected_source_records)
        if (not self.fold_id or not source or len(set(source)) != len(source) or not target
                or set(source) & set(target) or {r[0] for r in expected} != set(source)
                or len({r[1] for r in expected}) != len(expected)):
            raise ValueError("a fold needs its complete independent source roster and excluded target")
        for row in expected:
            if len(row) != 4 or type(row[3]) is not int or row[3] < 0:
                raise ValueError("expected source record dependencies have explicit label counts")
            _sha(row[2])
        for digest in self.excluded_eeg_sha256s:
            _sha(digest)
        _sha(self.source_plan_sha256)
        object.__setattr__(self, "source_participant_keys", source)
        object.__setattr__(self, "target_identity_keys", target)
        object.__setattr__(self, "expected_source_records", expected)
        object.__setattr__(self, "excluded_eeg_sha256s", tuple(self.excluded_eeg_sha256s))
        object.__setattr__(self, "excluded_record_ids", tuple(self.excluded_record_ids))

    def record(self):
        return {"fold_id": self.fold_id, "target_identity_keys": list(self.target_identity_keys),
            "source_participant_keys": list(self.source_participant_keys),
            "expected_source_records": [list(row) for row in self.expected_source_records],
            "excluded_eeg_sha256s": list(self.excluded_eeg_sha256s),
            "excluded_record_ids": list(self.excluded_record_ids),
            "source_plan_sha256": self.source_plan_sha256,
            "exclusion_scope": "target_participant_all_records_and_source_hyperlearning"}


def fold_from_freeze_candidate(path, *, expected_sha256, target_participant_key):
    """Consume saved metadata only; the original acquisition owner issues data rights."""
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != _sha(expected_sha256):
        raise ValueError("source freeze candidate bytes changed")
    plan = json.loads(content)
    if plan["schema"] != "n2p3_H_LOSO31_population_source_freeze_candidate/1":
        raise ValueError("unsupported source freeze candidate")
    selected = next(f for f in plan["folds"] if f["target_participant_key"] == target_participant_key)
    people = set(selected["source_participant_keys"])
    ledger = plan["source_record_ledger"]
    expected = tuple((row["participant_key"], row["record_id"], row["original_BDF_reference"]["sha256"],
                      row["requested_labels"]) for row in ledger
                     if row["participant_key"] in people and row["common_source_admission_candidate"])
    target_records = [row for row in ledger if row["participant_key"] == target_participant_key]
    return ParticipantFold(selected["fold_id"], (target_participant_key,),
        tuple(selected["source_participant_keys"]), expected,
        tuple(row["original_BDF_reference"]["sha256"] for row in target_records if row["original_BDF_reference"]),
        tuple(row["record_id"] for row in target_records), expected_sha256)


@dataclass(frozen=True, eq=False)
class MeanReference:
    chart: ResponseChart
    mean: np.ndarray = field(repr=False)
    factor: np.ndarray = field(repr=False)
    policy_sha256: str

    def __post_init__(self):
        mean, factor = _array(self.mean, 1), _array(self.factor, 2)
        if len(mean) != self.chart.dimension:
            raise ValueError("mean reference differs from the full physical response chart")
        _proper_square_factor(factor, self.chart.dimension)
        _sha(self.policy_sha256)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "factor", factor)

    def record(self):
        return {"chart_identity": self.chart.identity, "mean_identity": _array_id(self.mean),
                "factor_identity": _array_id(self.factor), "policy_sha256": self.policy_sha256}


@dataclass(frozen=True, eq=False)
class VarianceTemplate:
    name: str
    chart: ResponseChart
    factor: np.ndarray = field(repr=False)

    def __post_init__(self):
        factor = _array(self.factor, 2)
        if not self.name or factor.shape[0] != self.chart.dimension or factor.shape[1] < 1:
            raise ValueError("variance templates need named physical factors")
        object.__setattr__(self, "factor", factor)

    def record(self):
        return {"name": self.name, "chart_identity": self.chart.identity,
                "factor_identity": _array_id(self.factor)}


@dataclass(frozen=True)
class LogScaleProgram:
    prior_mean: tuple[float, ...]
    prior_sd: tuple[float, ...]
    initial: tuple[float, ...]
    maximum_iterations: int
    gradient_tolerance: float
    policy_sha256: str

    def __post_init__(self):
        for name in ("prior_mean", "prior_sd", "initial"):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        values = np.array((self.prior_mean, self.prior_sd, self.initial))
        if (values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all()
                or min(self.prior_sd) <= 0 or type(self.maximum_iterations) is not int
                or self.maximum_iterations < 1 or not np.isfinite(self.gradient_tolerance)
                or self.gradient_tolerance <= 0):
            raise ValueError("a proper Normal log-variance prior and explicit optimizer programme are required")
        _sha(self.policy_sha256)

    def record(self):
        return {"kind": "Normal_density_with_respect_to_log_variance",
            "prior_mean": list(self.prior_mean), "prior_sd": list(self.prior_sd),
            "initial": list(self.initial), "maximum_iterations": self.maximum_iterations,
            "gradient_tolerance": self.gradient_tolerance, "policy_sha256": self.policy_sha256,
            "optimizer": "scipy.optimize.minimize_BFGS_forward_finite_difference_no_bounds",
            "variance_clip_or_jitter": False, "global_optimality_claimed": False}


def _inputs(likelihoods, fold, reference, templates):
    by_person = {row.participant_key: row for row in likelihoods}
    if len(by_person) != len(likelihoods) or set(by_person) != set(fold.source_participant_keys):
        raise ValueError("the source fold must retain exactly its complete requested participant roster")
    ordered = tuple(by_person[key] for key in fold.source_participant_keys)
    seen_identities, seen_eeg = set(), set()
    observed_records = []
    for row in ordered:
        if row.chart.identity != reference.chart.identity:
            raise ValueError("source likelihood and mean reference have different physical response charts")
        if (set(row.identity_keys) & (set(fold.target_identity_keys) | seen_identities)
                or row.participant_key in fold.target_identity_keys):
            raise ValueError("source participant aliases include the target or repeat an independent person")
        seen_identities.update(row.identity_keys)
        person_eeg = {record.eeg_sha256 for record in row.records}
        if person_eeg & (set(fold.excluded_eeg_sha256s) | seen_eeg):
            raise ValueError("source EEG bytes include the target or duplicate another participant")
        seen_eeg.update(person_eeg)
        for record in row.records:
            if record.record_id in fold.excluded_record_ids:
                raise ValueError("an excluded original target recording entered source learning")
            observed_records.append((row.participant_key, record.record_id, record.eeg_sha256, record.labels))
    if set(observed_records) != set(fold.expected_source_records):
        raise ValueError("source records/label grants differ from the predeclared common admission roster")
    templates = tuple(templates)
    if (not templates or len({t.name for t in templates}) != len(templates)
            or any(t.chart.identity != reference.chart.identity for t in templates)):
        raise ValueError("unique templates must share the complete original response chart")
    _proper_square_factor(templates[0].factor, reference.chart.dimension)
    identity = _digest({"fold": fold.record(), "reference": reference.record(),
        "likelihoods": [row.record() for row in ordered], "templates": [t.record() for t in templates]})
    return ordered, templates, identity


def _between_factor(templates, log_scales):
    logs = np.asarray(log_scales, dtype=float)
    if logs.shape != (len(templates),) or not np.isfinite(logs).all():
        raise ValueError("one finite positive log-variance per named template is required")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        multipliers = np.exp(logs / 2)
    pieces = [s * t.factor for s, t in zip(multipliers, templates, strict=True)]
    if any(np.any((t.factor != 0) & (part == 0)) for t, part in zip(templates, pieces, strict=True)):
        raise ArithmeticError("a nonzero physical variance-factor entry underflowed")
    result = np.column_stack(pieces)
    if not np.isfinite(result).all():
        raise ArithmeticError("the physical between-person factor is not representable")
    return result


def _positive_qr(matrix):
    q, upper = np.linalg.qr(matrix, mode="reduced")
    signs = np.where(np.diag(upper) < 0, -1., 1.)
    return q * signs[None, :], signs[:, None] * upper


@dataclass(frozen=True, eq=False)
class PopulationForecast:
    chart: ResponseChart
    fold: ParticipantFold
    reference: MeanReference
    templates: tuple[VarianceTemplate, ...]
    likelihoods: tuple[FullResponseLikelihood, ...]
    input_identity: str
    log_scales: np.ndarray = field(repr=False)
    mean: np.ndarray = field(repr=False)
    between_factor: np.ndarray = field(repr=False)
    mean_factor: np.ndarray = field(repr=False)
    posterior_latent_upper: np.ndarray = field(repr=False)
    posterior_latent_rhs: np.ndarray = field(repr=False)
    log_source_marginal: float
    variance_fit_json: str
    producer_sha256: str

    def __post_init__(self):
        dimensions = {"log_scales": 1, "mean": 1, "between_factor": 2, "mean_factor": 2,
                      "posterior_latent_upper": 2, "posterior_latent_rhs": 1}
        for name, ndim in dimensions.items():
            object.__setattr__(self, name, _array(getattr(self, name), ndim))
        _sha(self.producer_sha256)
        if not np.isfinite(self.log_source_marginal):
            raise ArithmeticError("the original source marginal must be finite")
        object.__setattr__(self, "variance_fit_json", _json(json.loads(self.variance_fit_json)))

    @property
    def factor(self):
        # Full rectangular factor, with the two variance sources still named.
        return _array(np.column_stack((self.between_factor, self.mean_factor)), 2)

    @property
    def numeric_only(self):
        return True

    def linear_image(self, matrix):
        matrix = _array(matrix, 2)
        if matrix.shape[1] != self.chart.dimension:
            raise ValueError("response functionals must use the original complete chart")
        return matrix @ self.mean, matrix @ self.factor


def _infer(ordered, fold, reference, templates, input_identity, log_scales, variance_fit):
    between = _between_factor(templates, log_scales)
    dimension = reference.chart.dimension
    stack, rhs, log_determinants = [np.eye(dimension)], [np.zeros(dimension)], 0.
    constants = 0.
    for row in ordered:
        b = row.matrix @ between
        # [I, A F_Omega] [I, A F_Omega]^T, represented by its QR square root.
        _, upper = _positive_qr(np.column_stack((np.eye(len(row.response)), b)).T)
        lower = upper.T
        stack.append(solve_triangular(lower, row.matrix @ reference.factor, lower=True))
        rhs.append(solve_triangular(lower, row.response - row.matrix @ reference.mean, lower=True))
        log_determinants += float(np.log(np.diag(lower)).sum())
        constants += float(row.log_constant)
    design, data = np.vstack(stack), np.concatenate(rhs)
    q, upper = _positive_qr(design)
    transformed = q.T @ data
    z_mean = solve_triangular(upper, transformed)
    mean = reference.mean + reference.factor @ z_mean
    mean_factor = solve_triangular(upper.T, reference.factor.T, lower=True).T
    # Evaluate the augmented residual itself, avoiding norm(data)^2 - norm(Q'data)^2.
    residual = data - design @ z_mean
    marginal = constants - log_determinants - float(np.log(np.diag(upper)).sum()) - .5 * float(residual @ residual)
    return PopulationForecast(reference.chart, fold, reference, templates, ordered, input_identity,
        np.asarray(log_scales), mean, between, mean_factor, upper, transformed,
        marginal, _json(variance_fit), IMPLEMENTATION_SHA256)


def infer_population(likelihoods, *, fold, reference, templates, log_scales, fitted_variance=None):
    """Conditional proper forecast at a declared fixed or source-fitted variance."""
    ordered, templates, identity = _inputs(tuple(likelihoods), fold, reference, templates)
    record = {"kind": "predeclared_fixed_log_variances", "input_identity": identity,
              "source_only": True, "variance_hyperparameter_uncertainty_integrated": False}
    if fitted_variance is not None:
        record = json.loads(_json(fitted_variance))
        if (record["input_identity"] != identity or record["log_scales"] != list(log_scales)
                or record["kind"] != "source_only_log_variance_plugin"
                or record["source_only"] is not True or record["success"] is not True):
            raise ValueError("a fitted variance cannot be reused after changing the source fold or likelihoods")
    return _infer(ordered, fold, reference, templates, identity, log_scales, record)


class VarianceOptimizationError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__("the declared source log-scale optimization did not converge")


def fit_population_variance(likelihoods, *, fold, reference, templates, program):
    """Fit positive variance scales using source likelihood and a proper log-scale prior.

    This is a numeric plug-in learner. Real source EEG fitting is upstream.
    Each invocation re-optimizes its own fold; no all-person hyperparameter cache.
    """
    ordered, templates, identity = _inputs(tuple(likelihoods), fold, reference, templates)
    if len(program.prior_mean) != len(templates):
        raise ValueError("log-scale programme and actual named templates differ")
    location, scale = np.array(program.prior_mean), np.array(program.prior_sd)

    def objective(log_scales):
        forecast = _infer(ordered, fold, reference, templates, identity, log_scales,
                          {"kind": "numeric_source_optimization_evaluation"})
        standardized = (log_scales - location) / scale
        prior = -.5 * float(standardized @ standardized) - float(np.log(scale).sum()) - .5 * len(scale) * np.log(2*np.pi)
        return -float(forecast.log_source_marginal + prior)

    result = minimize(objective, np.array(program.initial), method="BFGS",
        options={"maxiter": program.maximum_iterations, "gtol": program.gradient_tolerance})
    record = {"kind": "source_only_log_variance_plugin", "program": program.record(),
        "input_identity": identity, "source_only": True,
        "variance_hyperparameter_uncertainty_integrated": False,
        "log_scales": result.x.tolist(), "negative_log_posterior": float(result.fun),
        "success": bool(result.success), "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev), "final_numeric_gradient": result.jac.tolist(),
        "global_optimality_claimed": False, "producer_sha256": IMPLEMENTATION_SHA256}
    if not result.success or not np.isfinite(result.fun):
        raise VarianceOptimizationError(record)
    return _infer(ordered, fold, reference, templates, identity, result.x, record)


def save_forecast(forecast, directory):
    """Save a compact numeric closure, with source dependency identities, once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {name: getattr(forecast, name) for name in ("log_scales", "mean", "between_factor", "mean_factor",
        "posterior_latent_upper", "posterior_latent_rhs")}
    arrays.update(reference_mean=forecast.reference.mean, reference_factor=forecast.reference.factor)
    for i, template in enumerate(forecast.templates):
        arrays[f"template_{i}"] = template.factor
    for i, likelihood in enumerate(forecast.likelihoods):
        arrays[f"source_{i}_matrix"], arrays[f"source_{i}_response"] = likelihood.matrix, likelihood.response
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    (directory / "arrays.npz").write_bytes(payload)
    record = {"schema": "n2p3_full_response_population_forecast/1", "chart": forecast.chart.record(),
        "fold": forecast.fold.record(), "reference": forecast.reference.record(),
        "templates": [t.record() for t in forecast.templates],
        "likelihoods": [row.record() for row in forecast.likelihoods],
        "input_identity": forecast.input_identity, "log_source_marginal": forecast.log_source_marginal,
        "variance_fit": json.loads(forecast.variance_fit_json), "producer_sha256": forecast.producer_sha256,
        "factor_order": "between_person_columns_then_population_mean_uncertainty_columns",
        "raw_or_label_authority_granted": False,
        "arrays": {"file": "arrays.npz", "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
        "array_identities": {name: _array_id(value) for name, value in arrays.items()}}
    content = (_json(record) + "\n").encode()
    (directory / "forecast.json").write_bytes(content)
    return {"path": str((directory / "forecast.json").resolve()),
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def load_forecast(path, *, expected_sha256):
    """Cold numeric-only load: no optimizer, QR, original source file or raw read."""
    path = Path(path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != _sha(expected_sha256):
        raise ValueError("saved population forecast differs from its external identity")
    record = json.loads(content)
    if (record["schema"] != "n2p3_full_response_population_forecast/1"
            or record["raw_or_label_authority_granted"] is not False
            or record["factor_order"] != "between_person_columns_then_population_mean_uncertainty_columns"
            or record["arrays"]["file"] != "arrays.npz"):
        raise ValueError("saved forecast roles or numeric payload identity differ")
    payload = (path.parent / "arrays.npz").read_bytes()
    if len(payload) != record["arrays"]["bytes"] or hashlib.sha256(payload).hexdigest() != record["arrays"]["sha256"]:
        raise ValueError("saved forecast numeric arrays changed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if {name: _array_id(value) for name, value in arrays.items()} != record["array_identities"]:
        raise ValueError("saved forecast array dependency identities differ")
    chart = ResponseChart.from_definition(record["chart"]["coefficient_names"], record["chart"]["physical_definition"])
    f = record["fold"]
    if f["exclusion_scope"] != "target_participant_all_records_and_source_hyperlearning":
        raise ValueError("saved participant exclusion scope changed")
    fold = ParticipantFold(f["fold_id"], tuple(f["target_identity_keys"]), tuple(f["source_participant_keys"]),
        tuple(tuple(row) for row in f["expected_source_records"]), tuple(f["excluded_eeg_sha256s"]),
        tuple(f["excluded_record_ids"]), f["source_plan_sha256"])
    reference = MeanReference(chart, arrays["reference_mean"], arrays["reference_factor"], record["reference"]["policy_sha256"])
    templates = tuple(VarianceTemplate(row["name"], chart, arrays[f"template_{i}"]) for i, row in enumerate(record["templates"]))
    likelihoods = []
    for i, row in enumerate(record["likelihoods"]):
        dependencies = tuple(SourceRecordDependency(r["record_id"], r["eeg_sha256"], r["label_scope_sha256"],
            tuple(r["native_support"]), r["labels"]) for r in row["records"])
        likelihood = FullResponseLikelihood(row["participant_key"], tuple(row["identity_keys"]), chart,
            arrays[f"source_{i}_matrix"], arrays[f"source_{i}_response"], row["log_constant"], dependencies,
            row["original_likelihood_artifact_sha256"], row["original_producer_sha256"])
        if likelihood.record() != row:
            raise ValueError("saved original likelihood/chart/authority role changed")
        likelihoods.append(likelihood)
    ordered, templates, identity = _inputs(tuple(likelihoods), fold, reference, templates)
    if (identity != record["input_identity"] or reference.record() != record["reference"]
            or [t.record() for t in templates] != record["templates"]
            or record["variance_fit"].get("input_identity") != identity):
        raise ValueError("saved forecast input/fold/variance dependency closure changed")
    between = _between_factor(templates, arrays["log_scales"])
    upper, rhs = arrays["posterior_latent_upper"], arrays["posterior_latent_rhs"]
    if (upper.shape != (chart.dimension, chart.dimension) or not np.array_equal(upper, np.triu(upper))
            or np.any(np.diag(upper) <= 0)):
        raise ValueError("saved posterior latent factor is not the original positive upper chart")
    z = solve_triangular(upper, rhs)
    expected_mean = reference.mean + reference.factor @ z
    _generated_relation(arrays["mean"], expected_mean,
                        np.abs(reference.mean) + np.abs(reference.factor) @ np.abs(z))
    _generated_relation(arrays["mean_factor"] @ upper, reference.factor,
                        np.abs(arrays["mean_factor"]) @ np.abs(upper) + np.abs(reference.factor))
    _generated_relation(arrays["between_factor"], between,
                        np.abs(arrays["between_factor"]) + np.abs(between))
    if record["variance_fit"]["kind"] == "source_only_log_variance_plugin":
        if record["variance_fit"]["log_scales"] != arrays["log_scales"].tolist():
            raise ValueError("saved fitted variance scales changed")
    return PopulationForecast(chart, fold, reference, templates, ordered, identity,
        arrays["log_scales"], arrays["mean"], arrays["between_factor"], arrays["mean_factor"],
        upper, rhs, record["log_source_marginal"], _json(record["variance_fit"]), record["producer_sha256"])
