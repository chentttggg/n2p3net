"""Proper integration proposals with a shared head and local record drifts.

The original coordinates are (u, a_1, ..., a_R). Each record uses its saved
physical-flat chart v=R a only inside the numerical proposal. Conditional
Gaussian blocks retain shared-u correlations without a global d-by-d matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import gammaln, logsumexp

from research.collapsed_continuous_inference import GaussianMixtureProposal, GaussianProposal
from research.continuous_integration_pilot import PilotResult
from transfer.noise_model import frozen_array


def _lower_root(value, dimension):
    root = np.asarray(value, dtype=float)
    if (root.shape != (dimension, dimension) or not np.isfinite(root).all()
            or np.any(np.triu(root, 1) != 0) or np.any(np.diag(root) <= 0)):
        raise ValueError("proposal requires its actual positive lower Cholesky root")
    return frozen_array(root)


@dataclass(frozen=True)
class ConditionalDriftBlock:
    record_id: str
    chart_root: np.ndarray
    mean: np.ndarray
    regression: np.ndarray
    root: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        chart = np.asarray(self.chart_root, dtype=float)
        regression = np.asarray(self.regression, dtype=float)
        if (not self.record_id or mean.ndim != 1 or not len(mean)
                or not np.isfinite(mean).all() or chart.shape != (len(mean), len(mean))
                or not np.isfinite(chart).all() or np.any(np.tril(chart, -1) != 0)
                or np.any(np.diag(chart) == 0) or regression.ndim != 2
                or regression.shape[0] != len(mean) or not np.isfinite(regression).all()):
            raise ValueError("record proposal must retain its physical chart and conditional coordinates")
        object.__setattr__(self, "mean", frozen_array(mean))
        object.__setattr__(self, "chart_root", frozen_array(chart))
        object.__setattr__(self, "regression", frozen_array(regression))
        object.__setattr__(self, "root", _lower_root(self.root, len(mean)))

    def record(self):
        return {"record_id": self.record_id, "chart_root": self.chart_root.tolist(),
                "mean": self.mean.tolist(), "regression": self.regression.tolist(),
                "root": self.root.tolist()}


@dataclass(frozen=True)
class RecordConditionalGaussian:
    head_mean: np.ndarray
    head_root: np.ndarray
    blocks: tuple[ConditionalDriftBlock, ...]

    def __post_init__(self):
        mean = np.asarray(self.head_mean, dtype=float)
        blocks = tuple(self.blocks)
        if (mean.ndim != 1 or not np.isfinite(mean).all() or not blocks
                or len({block.record_id for block in blocks}) != len(blocks)
                or any(block.regression.shape[1] != len(mean) for block in blocks)):
            raise ValueError("one shared proper head and unique local record blocks are required")
        object.__setattr__(self, "head_mean", frozen_array(mean))
        object.__setattr__(self, "head_root", _lower_root(self.head_root, len(mean)))
        object.__setattr__(self, "blocks", blocks)

    @property
    def rank(self):
        return len(self.head_mean)

    @property
    def dimension(self):
        return self.rank+sum(len(block.mean) for block in self.blocks)

    @property
    def mean(self):
        return np.concatenate((self.head_mean, *(solve_triangular(
            block.chart_root, block.mean, lower=False) for block in self.blocks)))

    @property
    def parameter_bytes(self):
        return self.head_mean.nbytes+self.head_root.nbytes+sum(
            block.mean.nbytes+block.chart_root.nbytes+block.regression.nbytes+block.root.nbytes
            for block in self.blocks)

    def _points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.isfinite(points).all():
            raise ValueError("proposal points must retain the full original physical coordinate chart")
        return points

    def draw(self, rng, count):
        head = self.head_mean+rng.normal(size=(count, self.rank))@self.head_root.T
        parts = [head]
        for block in self.blocks:
            value = (block.mean+(head-self.head_mean)@block.regression.T
                +rng.normal(size=(count, len(block.mean)))@block.root.T)
            parts.append(solve_triangular(block.chart_root, value.T, lower=False).T)
        return np.concatenate(parts, axis=1)

    def log_density(self, points):
        points = self._points(points)
        delta = points[:, :self.rank]-self.head_mean
        if self.rank:
            white = solve_triangular(self.head_root, delta.T, lower=True).T
            result = -.5*np.sum(white*white, axis=1)-np.log(np.diag(self.head_root)).sum()
        else:
            result = np.zeros(len(points))
        result -= .5*self.dimension*np.log(2*np.pi)
        offset = self.rank
        for block in self.blocks:
            stop = offset+len(block.mean)
            value = points[:, offset:stop]@block.chart_root.T
            residual = value-block.mean-delta@block.regression.T
            white = solve_triangular(block.root, residual.T, lower=True).T
            result += (np.log(np.abs(np.diag(block.chart_root))).sum()
                -np.log(np.diag(block.root)).sum()-.5*np.sum(white*white, axis=1))
            offset = stop
        return result

    def reversible_move(self, rng, points, scale):
        points = self._points(points)
        if not 0 < scale <= 1:
            raise ValueError("declare a positive pCN scale at most one")
        keep = np.sqrt(1-scale**2)
        old_delta = points[:, :self.rank]-self.head_mean
        head = (self.head_mean+keep*old_delta
            +scale*rng.normal(size=(len(points), self.rank))@self.head_root.T)
        parts, offset = [head], self.rank
        for block in self.blocks:
            stop = offset+len(block.mean)
            old_residual = (points[:, offset:stop]@block.chart_root.T-block.mean
                -old_delta@block.regression.T)
            value = (block.mean+(head-self.head_mean)@block.regression.T+keep*old_residual
                +scale*rng.normal(size=(len(points), len(block.mean)))@block.root.T)
            parts.append(solve_triangular(block.chart_root, value.T, lower=False).T)
            offset = stop
        return np.concatenate(parts, axis=1)

    def record(self):
        return {"kind": "conditional_gaussian_record_blocks/1",
                "head_mean": self.head_mean.tolist(), "head_root": self.head_root.tolist(),
                "blocks": [block.record() for block in self.blocks]}


@dataclass(frozen=True)
class RecordConditionalStudent:
    """A finite-covariance t reference with the same sparse physical factor.

    The Gaussian covariance factor defines the t scale matrix. Its covariance
    is scale * nu/(nu-2). One numerical Gamma scale is shared by every block;
    it is separate from any noise-scale variable in the scientific model.
    """

    base: RecordConditionalGaussian
    degrees_of_freedom: float

    def __post_init__(self):
        if (not isinstance(self.base, RecordConditionalGaussian)
                or not np.isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 2):
            raise ValueError("declare a record-conditional t reference with finite covariance")

    @property
    def mean(self):
        return self.base.mean

    @property
    def parameter_bytes(self):
        return self.base.parameter_bytes

    def draw(self, rng, count):
        normal = self.base.draw(rng, count)
        precision = rng.gamma(self.degrees_of_freedom/2, 2/self.degrees_of_freedom, size=count)
        return self.mean+(normal-self.mean)/np.sqrt(precision[:, None])

    def log_density(self, points):
        base = self.base
        points = base._points(points)
        delta = points[:, :base.rank]-base.head_mean
        radius = np.zeros(len(points))
        log_factor_determinant = float(np.log(np.diag(base.head_root)).sum())
        if base.rank:
            white = solve_triangular(base.head_root, delta.T, lower=True).T
            radius = np.hypot.reduce(white, axis=1)
        offset = base.rank
        for block in base.blocks:
            stop = offset+len(block.mean)
            residual = (points[:, offset:stop]@block.chart_root.T-block.mean
                -delta@block.regression.T)
            white = solve_triangular(block.root, residual.T, lower=True).T
            radius = np.hypot(radius, np.hypot.reduce(white, axis=1))
            log_factor_determinant += (np.log(np.diag(block.root)).sum()
                -np.log(np.abs(np.diag(block.chart_root))).sum())
            offset = stop
        nu, dimension = self.degrees_of_freedom, base.dimension
        with np.errstate(divide="ignore"):
            radial_log = np.logaddexp(0., 2*np.log(radius)-np.log(nu))
        return (gammaln((nu+dimension)/2)-gammaln(nu/2)
            -.5*dimension*np.log(nu*np.pi)-log_factor_determinant
            -.5*(nu+dimension)*radial_log)

    def reversible_move(self, rng, points, scale):
        # Independent draws form a reversible kernel for this component law.
        return self.draw(rng, len(self.base._points(points)))

    def record(self):
        return {"kind": "conditional_student_record_blocks/1", "base": self.base.record(),
                "degrees_of_freedom": float(self.degrees_of_freedom),
                "matrix_role": "scale_not_covariance", "Gamma_role": "numerical_proposal_only"}


def with_heavy_reference(proposal, reference, *, weight, degrees_of_freedom):
    """Give the original Gaussian target a positive polynomial-tail reference.

    For any proper finite Gaussian-mixture target Psi, Psi/t is bounded. A
    mixture g >= weight*t retains that bound without changing Psi or its prior.
    The bound's existence does not turn a replicate SE into a certificate.
    """
    if not 0 < weight < 1:
        raise ValueError("declare a positive reference mass strictly below one")
    components = proposal.components if isinstance(proposal, GaussianMixtureProposal) else (proposal,)
    logs = proposal.log_weights if isinstance(proposal, GaussianMixtureProposal) else np.zeros(1)
    tail = RecordConditionalStudent(reference, degrees_of_freedom)
    return GaussianMixtureProposal((*components, tail), np.r_[logs+np.log1p(-weight), np.log(weight)])


def proposal_from_record(record):
    """Restore the complete proposal law for saved-model replay."""
    if record["kind"] == "proper_gaussian":
        return GaussianProposal(np.array(record["mean"]), np.array(record["root"]))
    if record["kind"] == "conditional_gaussian_record_blocks/1":
        return RecordConditionalGaussian(np.array(record["head_mean"]), np.array(record["head_root"]),
            tuple(ConditionalDriftBlock(row["record_id"], np.array(row["chart_root"]),
                np.array(row["mean"]), np.array(row["regression"]), np.array(row["root"]))
                for row in record["blocks"]))
    if record["kind"] == "conditional_student_record_blocks/1":
        return RecordConditionalStudent(proposal_from_record(record["base"]),
            float(record["degrees_of_freedom"]))
    if record["kind"] == "proper_finite_gaussian_mixture":
        return GaussianMixtureProposal(tuple(proposal_from_record(c) for c in record["components"]),
            np.array(record["log_weights"]))
    raise ValueError("unsupported saved continuous proposal kind")


def reference_from_prepared(target):
    """A proper numerical reference in du da using each actual flat QR chart."""
    blocks = []
    for model, _, stop in target.records:
        admission = model.flat_admission[stop]
        if not admission["identified"]:
            raise ValueError("a numerical reference requires the original flat admission")
        chart = np.array(admission["physical_root"])
        linear = 2*model.endpoints[stop].common[1+model.rank:, 0]
        mean = solve_triangular(chart.T, linear, lower=True)
        blocks.append(ConditionalDriftBlock(model.record_id, chart, mean,
            np.zeros((len(mean), target.rank)), np.eye(len(mean))))
    return RecordConditionalGaussian(np.zeros(target.rank), np.eye(target.rank), tuple(blocks))


def pilot_from_structured_cloud(target, reference, *, pilot_seed, particles,
                                covariance_reference_fraction):
    """Independent positive pilot with record-local conditional moments.

    Covariance blending changes the numerical proposal only. The full density
    in original physical coordinates corrects every pilot/production weight.
    """
    fraction = covariance_reference_fraction
    if not 0 < fraction < 1 or particles < 2 or reference.dimension != target.dimension:
        raise ValueError("declare a proper structured reference and positive pilot configuration")
    rng = np.random.default_rng(pilot_seed)
    points = reference.draw(rng, particles)
    log_g = reference.log_density(points)
    joint = target.evaluate_many(points)
    importance = joint-log_g[:, None, None]
    scales = logsumexp(importance, axis=0)-np.log(particles)
    head = points[:, :reference.rank]
    reference_covariance = reference.head_root@reference.head_root.T
    components, ess_rows = [], []
    for endpoint in range(len(target.endpoints)):
        ess_row = []
        for candidate in range(len(target.q_values)):
            logs = importance[:, endpoint, candidate]
            weights = np.exp(logs-logsumexp(logs))
            head_mean = weights@head
            delta = head-head_mean
            covariance = (1-fraction)*(delta.T*weights)@delta+fraction*reference_covariance
            head_root = np.linalg.cholesky(covariance)
            blocks, offset = [], reference.rank
            for original in reference.blocks:
                stop = offset+len(original.mean)
                values = points[:, offset:stop]@original.chart_root.T
                mean = weights@values
                centered = values-mean
                cross = ((1-fraction)*(centered.T*weights)@delta
                    +fraction*original.regression@reference_covariance)
                regression = (cho_solve((head_root, True), cross.T).T
                    if reference.rank else np.empty((len(mean), 0)))
                residual = centered-delta@regression.T
                difference = original.regression-regression
                conditional = ((1-fraction)*(residual.T*weights)@residual
                    +fraction*(original.root@original.root.T
                        +difference@reference_covariance@difference.T))
                blocks.append(ConditionalDriftBlock(original.record_id, original.chart_root,
                    mean, regression, np.linalg.cholesky(conditional)))
                offset = stop
            components.append(RecordConditionalGaussian(head_mean, head_root, tuple(blocks)))
            ess_row.append(float(1/(weights@weights)))
        ess_rows.append(ess_row)
    proposal = GaussianMixtureProposal(tuple(components), np.full(len(components), -np.log(len(components))))
    return PilotResult(proposal, scales,
        {"schema": "n2p3_structured_continuous_proposal_pilot/1", "target_identity": target.identity,
         "pilot_seed": pilot_seed, "particles": particles, "proper_reference": reference.record(),
         "component_log_scales": scales.tolist(), "covariance_reference_fraction": fraction,
         "candidate_support": list(target.q_values), "importance_ess_diagnostic": ess_rows,
         "proposal_parameter_bytes": sum(component.parameter_bytes for component in components),
         "proposal_storage_order": "Tq_times_head_squared_plus_sum_record_head_cross_and_local_squared",
         "coordinate_measure": "original_du_da_with_each_record_chart_Jacobian",
         "truth_read": False, "source_fits": 0, "target_model_fits": 0,
         "finite_sample_error_certificate": False},
        {"pilot_points": points, "pilot_log_g": log_g, "pilot_joint": joint})
