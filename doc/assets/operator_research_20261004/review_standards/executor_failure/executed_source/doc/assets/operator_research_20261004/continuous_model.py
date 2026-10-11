"""Frozen research GLS: explicit finite mean operator and segmented AR noise.

This is an experiment implementation, not a product default. The filtered
arm fixes the response design but retains an approximate AR1 residual model
and shared post-processing drift nuisance. It is NOT exact F Sigma F.T.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from data.finite_filter_operator import FiniteFilterOperator
from transfer.continuous_decision import ContinuousDecisionResult, _event_design, _validate_record
from transfer.noise_model import array_digest, freeze_metadata, frozen_array, metadata_digest
from transfer.temporal_noise import TemporalNoise


def project(noise, values):
    whitened = noise.whiten(values)
    nuisance = np.column_stack([np.ones(len(values)), np.linspace(-1, 1, len(values))])
    Q, _ = np.linalg.qr(noise.whiten(nuisance), mode="reduced")
    return whitened - Q @ (Q.T @ whitened)


def residual_diagnostics(records):
    centered = [values - values.mean(axis=0) for values in records]
    variance = sum(np.sum(values**2, axis=0) for values in centered)
    usable = variance > np.finfo(float).tiny
    return {"mean_channel_acf": {str(lag): float(np.mean(
        sum(np.sum(values[lag:, usable] * values[:-lag, usable], axis=0)
            for values in centered if lag < len(values)) / variance[usable]))
        for lag in (1, 2, 8, 32, 128) if any(lag < len(values) for values in centered)},
        "mean_squared_innovation": float(sum(np.sum(v**2) for v in records) / sum(v.size for v in records)),
        "cross_record_lag_pairs": 0}


def spatial_estimate(residual, method):
    if method == "pooled_lw":
        fitted = LedoitWolf(assume_centered=True).fit(residual)
        return fitted.covariance_, float(fitted.shrinkage_)
    if method != "standardized_lw":
        raise ValueError("unknown declared spatial shrinkage")
    scale = np.sqrt(np.mean(residual**2, axis=0))
    if np.any(scale <= np.finfo(float).tiny):
        raise ValueError("cannot standardize a zero-energy observed channel")
    fitted = LedoitWolf(assume_centered=True).fit(residual / scale)
    return fitted.covariance_ * scale[:, None] * scale[None, :], float(fitted.shrinkage_)


@dataclass(frozen=True)
class OperatorGLS:
    basis: np.ndarray
    mean: np.ndarray
    row_covariance: np.ndarray
    channel_covariance: np.ndarray
    noise: TemporalNoise
    operator: FiniteFilterOperator | None
    filter_design: bool
    sfreq: float
    channels: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    provenance: dict

    def __post_init__(self):
        for field in ("basis", "mean", "row_covariance", "channel_covariance"):
            value = np.asarray(getattr(self, field), dtype=float)
            if not np.isfinite(value).all():
                raise ValueError("nonfinite response state")
            object.__setattr__(self, field, frozen_array(value))
        p, c = 2 * self.basis.shape[1], len(self.channels)
        if self.mean.shape != (p, c) or self.row_covariance.shape != (p, p) or self.channel_covariance.shape != (c, c):
            raise ValueError("response state geometry differs")
        if (not isinstance(self.provenance, dict) or self.provenance.get("fit_scope") != "target_calibration"
                or not self.provenance.get("records") or type(self.provenance.get("ridge")) not in (int, float)
                or not np.isfinite(self.provenance["ridge"]) or self.provenance["ridge"] <= 0):
            raise ValueError("research GLS artifact requires a nonempty target calibration ledger")
        required = {"record_id", "samples", "events", "target_count", "input_sha256", "signal_sha256", "right_censored"}
        for record in self.provenance["records"]:
            if set(record) != required or type(record["samples"]) is not int or type(record["events"]) is not int:
                raise ValueError("research GLS training record ledger is incomplete")
            if record["samples"] < len(self.basis) + 3 or record["events"] < 1:
                raise ValueError("research GLS training record ledger has invalid dimensions")
            for field in ("input_sha256", "signal_sha256"):
                value = record[field]
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("research GLS training record digest is invalid")
        if self.provenance.get("kernel_prior_scale") not in {"innovation", "marginal"}:
            raise ValueError("research GLS prior scale declaration is missing")
        if self.provenance.get("startup_mode") not in {"stationary", "stationary_innovations", "conditional_prefix"}:
            raise ValueError("research GLS startup declaration is missing")
        if self.provenance.get("noise_method") not in {"burg", "legacy_clipped_ar1"}:
            raise ValueError("research GLS noise estimator declaration is missing")
        implementation = self.provenance.get("implementation_sha256")
        current_implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if implementation != current_implementation:
            raise ValueError("research GLS artifact was fitted by a different implementation")
        if self.noise.provenance.get("fit_scope") != "target_calibration":
            raise ValueError("research GLS noise scope differs from model fit scope")
        if tuple(r["record_id"] for r in self.noise.provenance.get("records", ())) != tuple(
                r["record_id"] for r in self.provenance["records"]):
            raise ValueError("research GLS noise and model ledgers differ")
        covariance = (self.channel_covariance + self.channel_covariance.T) / 2
        if not np.allclose(self.channel_covariance, covariance, rtol=1e-10, atol=1e-12):
            raise ValueError("research GLS channel covariance must be symmetric")
        np.linalg.cholesky(covariance)
        np.linalg.cholesky(self.row_covariance)
        np.linalg.cholesky(self.channel_covariance)
        object.__setattr__(self, "provenance", freeze_metadata(self.provenance))
        object.__setattr__(self, "_certificate", metadata_digest(self.record()))

    @classmethod
    def fit(cls, records, *, basis, sfreq, channels, candidate_ids, order,
            operator=None, filter_design=True, noise_method="burg", ridge=1.,
            feasible_iterations=1, channel_weighted_noise=False, spatial_shrinkage="pooled_lw",
            kernel_prior_scale="innovation", startup_mode="stationary", implementation_sha256=None):
        records = tuple(records)
        if not records or len({r.record_id for r in records}) != len(records):
            raise ValueError("fit requires distinct legal records")
        if noise_method not in {"burg", "legacy_clipped_ar1"} or (noise_method == "legacy_clipped_ar1" and order != 1):
            raise ValueError("unknown fixed noise estimator")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("positive fixed ridge required")
        if type(feasible_iterations) is not int or not 1 <= feasible_iterations <= 10 or type(channel_weighted_noise) is not bool:
            raise ValueError("fixed bounded feasible GLS iterations and weighting required")
        if noise_method != "burg" and (feasible_iterations != 1 or channel_weighted_noise):
            raise ValueError("legacy reference keeps its original one-pass unweighted estimator")
        if kernel_prior_scale not in {"innovation", "marginal"}:
            raise ValueError("kernel prior scale must be explicit")
        if startup_mode not in {"stationary", "conditional_prefix", "stationary_innovations"}:
            raise ValueError("unknown startup protocol")
        if noise_method != "burg" and startup_mode != "stationary":
            raise ValueError("legacy reference retains its original stationary startup")
        if startup_mode == "conditional_prefix" and (operator is not None or noise_method != "burg"):
            raise ValueError("conditional prefix is restricted to unfiltered distributed signals with Burg noise")
        current_implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if implementation_sha256 is not None and implementation_sha256 != current_implementation:
            raise ValueError("fit implementation digest differs from the loaded source")
        basis = np.asarray(basis, dtype=float)
        samples, designs, ledger, labels = [], [], [], []
        for record in records:
            x, onset = _validate_record(record.samples, record.onsets, len(channels), len(basis),
                                       allow_right_censor=record.right_censored)
            if startup_mode == "conditional_prefix" and onset[0] < order:
                raise ValueError("conditional prefix contains a modeled stimulus response")
            y = np.asarray(record.target_mask)
            if y.shape != onset.shape or not np.isin(y, [0, 1]).all():
                raise ValueError("aligned calibration labels required")
            d = _event_design(onset, y, len(x), basis)
            samples.append(x if operator is None else operator.apply(x))
            designs.append(d if operator is None or not filter_design else operator.apply(d))
            labels.extend(y.tolist())
            ledger.append({"record_id": record.record_id, "samples": len(x), "events": len(onset),
                "target_count": int(y.sum()), "input_sha256": array_digest(x, onset, y),
                "signal_sha256": array_digest(x, onset), "right_censored": record.right_censored})
        if set(labels) != {0, 1}:
            raise ValueError("both classes required")

        def solve(noise):
            design = np.vstack([project(noise, d) for d in designs])
            observed = np.vstack([project(noise, x) for x in samples])
            effective_ridge = ridge / noise.marginal_variance_per_unit_innovation if kernel_prior_scale == "marginal" else ridge
            # QR of the augmented ridge system avoids squaring its condition
            # number in D.T @ D and keeps a valid posterior covariance for
            # high-order, marginal-scale priors.
            augmented = np.vstack((design, np.sqrt(effective_ridge) * np.eye(design.shape[1])))
            _, factor = np.linalg.qr(augmented, mode="reduced")
            if np.min(np.abs(np.diag(factor))) <= np.finfo(float).tiny:
                raise ValueError("ridge posterior design is numerically rank deficient")
            inverse_factor = np.linalg.solve(factor, np.eye(factor.shape[0]))
            row_cov = inverse_factor @ inverse_factor.T
            mean = row_cov @ (design.T @ observed)
            gram = design.T @ design
            return design, observed, mean, (row_cov + row_cov.T) / 2, gram

        white = TemporalNoise.from_parameters([], innovation_variance=1., reference_id="initial_OLS")
        d, x, initial, _, _ = solve(white)
        initial_residual = x - d @ initial
        offsets = np.cumsum([0] + [len(x) for x in samples])
        segments = [initial_residual[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        channel_metric, _ = spatial_estimate(initial_residual, spatial_shrinkage)
        channel_metric += np.eye(len(channels)) * max(float(np.trace(channel_metric) / len(channels)), np.finfo(float).tiny) * 1e-8
        iterations = []
        for iteration in range(feasible_iterations):
            metric_digest = None
            fitting_segments = segments
            if channel_weighted_noise:
                root = np.linalg.cholesky(channel_metric)
                fitting_segments = [solve_triangular(root, r.T, lower=True).T for r in segments]
                metric_digest = array_digest(channel_metric)
            if noise_method == "burg":
                noise = TemporalNoise.fit(fitting_segments, order=order, record_ids=[r.record_id for r in records],
                                          fit_scope="target_calibration", startup_mode=startup_mode)
            else:
                numerator = sum(float(np.sum(r[1:] * r[:-1])) for r in segments)
                denominator = sum(float(np.sum(r[:-1]**2)) for r in segments)
                phi = float(np.clip(numerator / max(denominator, np.finfo(float).tiny), -.95, .95))
                noise = TemporalNoise.from_parameters([phi], innovation_variance=1.,
                    reference_id="legacy_AR1_fitted_from_outer_ledger_residuals")
            d, x, mean, row_cov, gram = solve(noise)
            likelihood_offsets = np.cumsum([0] + [len(value) - (noise.order if startup_mode == "conditional_prefix" else 0)
                                                   for value in samples])
            residual = x - d @ mean
            effective = float(np.trace(gram @ row_cov))
            df = len(x) - 2 * len(records) - effective
            if df <= 1:
                raise ValueError("insufficient residual degrees of freedom")
            estimate, shrinkage = spatial_estimate(residual, spatial_shrinkage)
            covariance = estimate * len(x) / df
            scale = max(float(np.trace(covariance) / len(channels)), float(np.mean(x**2)) * np.finfo(float).eps, np.finfo(float).tiny)
            covariance += np.eye(len(channels)) * scale * 1e-8
            iterations.append({"iteration": iteration + 1, "temporal_coefficients": noise.coefficients.tolist(),
                "temporal_fit_channel_metric_sha256": metric_digest, "channel_covariance_sha256": array_digest(covariance),
                "marginal_variance_per_unit_innovation": noise.marginal_variance_per_unit_innovation,
                "effective_ridge": ridge / noise.marginal_variance_per_unit_innovation if kernel_prior_scale == "marginal" else ridge,
                "residual_diagnostics": residual_diagnostics([
                    residual[a:b] for a, b in zip(likelihood_offsets[:-1], likelihood_offsets[1:], strict=True)])})
            if iteration + 1 < feasible_iterations:
                # Refit mean and nuisance, then return to the pre-whitening
                # coordinate before estimating a new temporal shape.
                segments = []
                for observed, design in zip(samples, designs, strict=True):
                    nuisance = np.column_stack([np.ones(len(observed)), np.linspace(-1, 1, len(observed))])
                    raw_residual = observed - design @ mean
                    beta = np.linalg.lstsq(noise.whiten(nuisance), noise.whiten(raw_residual), rcond=None)[0]
                    segments.append(raw_residual - nuisance @ beta)
                channel_metric = covariance
        return cls(basis, mean, row_cov, covariance, noise, operator, filter_design, sfreq,
            tuple(channels), tuple(candidate_ids), {"fit_scope": "target_calibration", "records": ledger,
                "ridge": ridge, "noise_method": noise_method, "noise_fit_passes": feasible_iterations,
                "kernel_prior_scale": kernel_prior_scale, "startup_mode": startup_mode,
                "prefix_conditioning_is_a_distinct_estimand": startup_mode == "conditional_prefix",
                "conditioning_samples_per_record": noise.order if startup_mode == "conditional_prefix" else 0,
                "channel_weighted_temporal_fit": channel_weighted_noise, "feasible_GLS_iterations": iterations,
                "monotone_likelihood_optimization_claimed": False,
                "effective_kernel_parameters": effective, "residual_degrees_of_freedom": df,
                "channel_shrinkage": shrinkage, "spatial_shrinkage_method": spatial_shrinkage,
                "train_residual_diagnostics": residual_diagnostics([
                    residual[a:b] for a, b in zip(likelihood_offsets[:-1], likelihood_offsets[1:], strict=True)]),
                "nuisance": "shared_post_processing_constant_and_linear",
                "hyperparameter_uncertainty_integrated": False,
                "random_physiological_kernel_variation_integrated": False,
                "implementation_sha256": current_implementation})

    def score(self, samples, onsets, membership, *, record_id, sfreq, channels, candidate_ids):
        if metadata_digest(self.record()) != self._certificate:
            raise ValueError("frozen response state modified")
        self.noise.assert_integrity()
        if sfreq != self.sfreq or tuple(channels) != self.channels or tuple(candidate_ids) != self.candidate_ids:
            raise ValueError("frozen signal or vocabulary mismatch")
        x, onset = _validate_record(samples, onsets, len(channels), len(self.basis))
        if self.noise.startup_mode == "conditional_prefix" and (self.operator is not None or onset[0] < self.noise.order):
            raise ValueError("prediction prefix must precede all modeled responses without project filtering")
        if record_id in {r["record_id"] for r in self.provenance["records"]} or array_digest(x, onset) in {
            r["signal_sha256"] for r in self.provenance["records"]
        }:
            raise ValueError("training evidence is not held-out prediction")
        member = np.asarray(membership)
        if (member.shape != (len(onset), len(self.candidate_ids)) or member.dtype != np.bool_
                or np.any(member.sum(axis=0) == 0) or np.any(member.sum(axis=1) == 0)
                or np.any(member.sum(axis=1) == len(self.candidate_ids))):
            raise ValueError("complete boolean membership required")
        processed = x if self.operator is None else self.operator.apply(x)
        observed = project(self.noise, processed)
        channel_root = np.linalg.cholesky(self.channel_covariance)
        kernel_root = np.linalg.cholesky(self.row_covariance)
        evidence = []
        for candidate in range(len(self.candidate_ids)):
            design = _event_design(onset, member[:, candidate], len(x), self.basis)
            if self.operator is not None and self.filter_design:
                design = self.operator.apply(design)
            design = project(self.noise, design)
            residual = solve_triangular(channel_root, (observed - design @ self.mean).T, lower=True).T
            u = design @ kernel_root
            root = np.linalg.cholesky(np.eye(u.shape[1]) + u.T @ u)
            correction = solve_triangular(root, u.T @ residual, lower=True)
            quadratic = float(np.sum(residual**2) - np.sum(correction**2))
            if quadratic < -1e-8 * max(1., float(np.sum(residual**2))):
                raise ValueError("unstable posterior solve")
            logdet = len(channels) * 2 * float(np.log(np.diag(root)).sum())
            evidence.append(-.5 * (max(0., quadratic) + logdet))
        evidence = np.asarray(evidence)
        if not np.isfinite(evidence).all():
            raise ValueError("nonfinite evidence")
        best = int(np.argmax(evidence))
        tied = np.count_nonzero(np.isclose(evidence, evidence[best], rtol=0, atol=1e-10)) > 1
        return ContinuousDecisionResult(self.candidate_ids, frozen_array(evidence),
            frozen_array(np.exp(evidence - logsumexp(evidence))), None if tied else self.candidate_ids[best],
            "tied_evidence" if tied else "ok", len(x), array_digest(x, onset, member))

    def record(self):
        return {"schema": "n2p3_operator_gls_research/1", "sfreq": self.sfreq,
            "channels": self.channels, "candidate_ids": self.candidate_ids,
            "noise": self.noise.record(), "operator": None if self.operator is None else self.operator.record(),
            "filter_design": self.filter_design, "provenance": self.provenance,
            "state_sha256": array_digest(self.basis, self.mean, self.row_covariance, self.channel_covariance)}

    def save(self, path):
        metadata = self.record()
        if metadata_digest(metadata) != self._certificate:
            raise ValueError("frozen state changed")
        metadata["artifact_sha256"] = metadata_digest(metadata)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, basis=self.basis, mean=self.mean, row_covariance=self.row_covariance,
                channel_covariance=self.channel_covariance, metadata=np.asarray(json.dumps(metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            state = [z[key] for key in ("basis", "mean", "row_covariance", "channel_covariance")]
            meta = json.loads(str(z["metadata"]))
        if meta.pop("artifact_sha256", None) != metadata_digest(meta):
            raise ValueError("research artifact metadata changed")
        if meta["schema"] != "n2p3_operator_gls_research/1" or array_digest(*state) != meta["state_sha256"]:
            raise ValueError("invalid research artifact")
        op = meta["operator"]
        operator = None if op is None else FiniteFilterOperator.from_record(op)
        return cls(*state, TemporalNoise.from_record(meta["noise"]), operator, meta["filter_design"],
                   meta["sfreq"], tuple(meta["channels"]), tuple(meta["candidate_ids"]), meta["provenance"])
