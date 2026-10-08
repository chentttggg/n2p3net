"""Independent dense oracles for the reference-aware general Gaussian owner."""
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from transfer.noise_model import array_digest, metadata_digest
from transfer.spatial_signal_prior import (
    SpatialLayout,
    SpatialObservation,
    SpatialPriorIdentificationError,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
)


def auth(record_id, rows, *, scope="target_calibration", dictionary="b" * 64):
    source = hashlib.sha256(record_id.removesuffix("-prefix").encode()).hexdigest()
    return {"record_id": record_id, "fit_scope": scope, "source_sha256": source,
            "dictionary_sha256": dictionary, "temporal_program_sha256": "c" * 64,
            "raw_sample_support": [0, rows]}


def layout(unit="V", positions=True):
    return SpatialLayout(("Fz", "Cz", "Pz"), "left earlobe", "d" * 64, ("EEG",) * 3, unit,
                         np.array([[0, .05, .07], [0, 0, .1], [0, -.05, .07]]) if positions else None,
                         "head" if positions else None)


def prior(geometry=None, *, temporal=None, signal=None, mean=None, dictionary="b" * 64):
    geometry = layout() if geometry is None else geometry
    return SpatialSignalPrior(geometry, np.array([[.3, -.2, .5], [.1, .4, -.1]]) if mean is None else mean,
        np.array([[1.2, .2], [.2, .7]]) if temporal is None else temporal,
        np.array([[2., .2, -.1], [.2, .9, .3], [-.1, .3, 1.7]]) if signal is None else signal,
        {"dictionary_sha256": dictionary, "vectorization": "column_major_response_by_native_EEG"},
        {"role": "predeclared", "authority_sha256": "e" * 64})


def block(rng, geometry, record_id="C1", observation=None, *, nuisance=False, rows=12,
          dictionary="b" * 64, scope="target_calibration"):
    observation = SpatialObservation.native(geometry) if observation is None else observation
    y = observation.project(rng.normal(size=(rows, 3)))
    design = rng.normal(size=(rows, 2))
    z = np.column_stack((np.ones(rows), np.linspace(-1, 1, rows))) if nuisance else None
    return SpatialResponseBlock(y, design, observation,
                                auth(record_id, rows, scope=scope, dictionary=dictionary), z)


def dense_posterior(p, blocks, noise):
    covariance = np.kron(p.signal_covariance, p.temporal_covariance)
    precision = np.linalg.inv(covariance)
    rhs = precision @ p.mean.flatten(order="F")
    for values in blocks:
        u = values.observation.support_basis
        a = u.T @ values.observation.matrix
        y = (values.samples @ u).flatten(order="F")
        x = np.kron(a, values.design)
        n = len(values.samples)
        likelihood_precision = np.linalg.inv(np.kron(a @ noise @ a.T, np.eye(n)))
        if values.nuisance is not None:
            z = np.kron(np.eye(a.shape[0]), values.nuisance)
            projected = likelihood_precision @ z
            likelihood_precision -= projected @ np.linalg.solve(z.T @ projected, projected.T)
        precision += x.T @ likelihood_precision @ x
        rhs += x.T @ likelihood_precision @ y
    covariance = np.linalg.inv(precision)
    return (covariance @ rhs).reshape(p.mean.shape, order="F"), covariance


def dense_density(model, values):
    u = values.observation.support_basis
    a = u.T @ values.observation.matrix
    n, channels = len(values.samples), a.shape[0]
    design = np.kron(a, values.design)
    covariance = np.kron(a @ model.native_noise_covariance @ a.T, np.eye(n))
    covariance += design @ model.covariance @ design.T
    centered = (values.samples @ u).flatten(order="F") - design @ model.mean.flatten(order="F")
    precision = np.linalg.inv(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    dimensions = n * channels
    if values.nuisance is not None:
        z = np.kron(np.eye(channels), values.nuisance)
        projected = precision @ z
        information = z.T @ projected
        logdet += np.linalg.slogdet(information)[1]
        precision -= projected @ np.linalg.solve(information, projected.T)
        dimensions -= z.shape[1]
    return float(-.5 * (dimensions * np.log(2 * np.pi) + logdet + centered @ precision @ centered)
                 + channels * values.temporal_log_abs_det)


def resign_posterior(record):
    names = ("mean", "covariance", "native_noise_covariance", "precision_root", "response",
             "orthogonal_factor", "qr_column_scales", "native_noise_root")
    record["numeric_sha256"] = array_digest(*(np.asarray(record[name]) for name in names))
    record["artifact_sha256"] = metadata_digest({key: value for key, value in record.items()
                                                 if key != "artifact_sha256"})
    return record


class TestSpatialSignalPrior(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261008)
        self.geometry = layout()
        self.prior = prior(self.geometry)
        self.noise = np.array([[1.1, .2, .1], [.2, 1.7, -.3], [.1, -.3, .8]])
        self.noise_binding = {"authority_sha256": "f" * 64, "role": "frozen_C_noise"}

    def fit(self, p=None, blocks=None, noise=None):
        return SpatialSignalPosterior.fit(self.prior if p is None else p,
            (block(self.rng, self.geometry),) if blocks is None else blocks,
            native_noise_covariance=self.noise if noise is None else noise,
            noise_binding=self.noise_binding)

    def test_exact_dense_posterior_and_predictive_oracle(self):
        records = [block(self.rng, self.geometry, "C1"), block(self.rng, self.geometry, "C2")]
        fitted = self.fit(blocks=records)
        mean, covariance = dense_posterior(self.prior, records, self.noise)
        np.testing.assert_allclose(fitted.mean, mean, atol=3e-14)
        np.testing.assert_allclose(fitted.covariance, covariance, atol=3e-14)
        held = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        self.assertAlmostEqual(fitted.log_density(held), dense_density(fitted, held), places=11)
        # An independent signal covariance is not a single H_posterior x Sigma_noise.
        whiten = np.linalg.inv(np.linalg.cholesky(self.noise))
        change = np.kron(whiten, np.eye(2))
        white_covariance = change @ fitted.covariance @ change.T
        self.assertGreater(np.linalg.norm(white_covariance[:2, 2:4]), 1e-6)

    def test_diffuse_nuisance_and_conditional_prefix_oracle(self):
        records = [block(self.rng, self.geometry, "C1", nuisance=True),
                   block(self.rng, self.geometry, "C2", nuisance=True)]
        fitted = self.fit(blocks=records)
        mean, covariance = dense_posterior(self.prior, records, self.noise)
        np.testing.assert_allclose(fitted.mean, mean, atol=3e-14)
        np.testing.assert_allclose(fitted.covariance, covariance, atol=3e-14)
        held = block(self.rng, self.geometry, "held", nuisance=True, scope="read_only_prediction")
        prefix = replace(held, samples=held.samples[:5], design=held.design[:5], nuisance=held.nuisance[:5],
                         authorization=auth("held-prefix", 5, scope="read_only_prediction"))
        expected = dense_density(fitted, held) - dense_density(fitted, prefix)
        self.assertAlmostEqual(fitted.conditional_log_density(held, prefix), expected, places=11)
        # Both response and nuisance are conditioned, not only AR/noise history.
        self.assertNotAlmostEqual(fitted.log_density(held), expected, places=3)

    def test_nonorthogonal_response_basis_change(self):
        values = block(self.rng, self.geometry, nuisance=True)
        held = block(self.rng, self.geometry, "held", nuisance=True, scope="read_only_prediction")
        fitted = self.fit(blocks=(values,))
        np.testing.assert_allclose(fitted.native_noise_root @ fitted.native_noise_root.T,
                                   self.noise, atol=3e-15)
        a = np.array([[2., .8], [.1, .4]])
        inverse = np.linalg.inv(a)
        p = prior(self.geometry, mean=inverse @ self.prior.mean,
                  temporal=inverse @ self.prior.temporal_covariance @ inverse.T,
                  signal=self.prior.signal_covariance, dictionary="1" * 64)
        def change(record):
            return replace(record, design=record.design @ a,
                           authorization={**record.authorization, "dictionary_sha256": "1" * 64})
        transformed = self.fit(p=p, blocks=(change(values),))
        self.assertAlmostEqual(fitted.log_density(held), transformed.log_density(change(held)), places=11)
        np.testing.assert_allclose(a @ transformed.mean, fitted.mean, atol=3e-14)

    def test_voltage_units_transform_signal_and_noise_separately(self):
        values = block(self.rng, self.geometry)
        held = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        fitted = self.fit(blocks=(values,))
        scale = 1e6
        geometry = layout(unit="uV")
        p = prior(geometry, mean=scale * self.prior.mean,
                  temporal=self.prior.temporal_covariance, signal=scale**2 * self.prior.signal_covariance)
        observation = SpatialObservation.native(geometry)
        def change(record):
            return replace(record, samples=scale * record.samples, observation=observation)
        transformed = self.fit(p=p, blocks=(change(values),), noise=scale**2 * self.noise)
        expected = fitted.log_density(held) - held.samples.size * np.log(scale)
        self.assertAlmostEqual(transformed.log_density(change(held)), expected, places=10)

    def test_reference_null_is_exact_and_cz_range_is_full(self):
        car = SpatialObservation.average_reference(self.geometry)
        cz = SpatialObservation.measured_reference(self.geometry, "Cz")
        self.assertEqual(car.rank_ledger["rank"], 2)
        self.assertEqual(cz.rank_ledger["rank"], 2)
        self.assertEqual(cz.rank_ledger["output_dimensions"], 2)
        values = block(self.rng, self.geometry, observation=car)
        fitted = self.fit(blocks=(values,))
        mean, covariance = dense_posterior(self.prior, [values], self.noise)
        np.testing.assert_allclose(fitted.mean, mean, atol=3e-14)
        np.testing.assert_allclose(fitted.covariance, covariance, atol=3e-14)
        held = block(self.rng, self.geometry, "held", observation=car, scope="read_only_prediction")
        self.assertAlmostEqual(fitted.log_density(held), dense_density(fitted, held), places=11)
        with self.assertRaisesRegex(SpatialPriorIdentificationError, "outside_observation_range"):
            fitted.log_density(replace(held, samples=held.samples + .01))
        with self.assertRaisesRegex(SpatialPriorIdentificationError, "zero_filled"):
            SpatialObservation(self.geometry, np.array([[1., 0., 0.], [0., 0., 0.]]),
                               ("Fz", "missing-P4"), "left earlobe")

    def test_invertible_spatial_change_preserves_log_bayes_factor(self):
        values = block(self.rng, self.geometry)
        fitted = self.fit(blocks=(values,))
        h0 = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        h1 = replace(h0, design=h0.design + .2)
        a = np.array([[1., 0., -1.], [0., 1., -1.], [.2, 0., 1.]])
        observation = SpatialObservation(self.geometry, a, ("contrast0", "contrast1", "native-extension"),
                                         "invertible_native_contrasts")
        def change(record):
            return replace(record, samples=record.samples @ a.T, observation=observation)
        transformed = self.fit(blocks=(change(values),))
        before = fitted.log_density(h0) - fitted.log_density(h1)
        after = transformed.log_density(change(h0)) - transformed.log_density(change(h1))
        self.assertAlmostEqual(before, after, places=10)

    def test_extreme_output_row_gain_preserves_information_and_density_jacobian(self):
        values = block(self.rng, self.geometry)
        baseline = self.fit(blocks=(values,))
        gains = np.array([1., 1e-20, -1e20])
        observation = SpatialObservation(self.geometry, np.diag(gains),
            ("Fz-coordinate", "Cz-coordinate", "Pz-coordinate"), "same_native_reference_with_output_gains")
        self.assertEqual(observation.rank_ledger["structural_rank"], 3)
        self.assertEqual(observation.rank_ledger["lost_native_directions"], 0)
        transformed = self.fit(blocks=(replace(values, samples=values.samples * gains,
                                              observation=observation),))
        np.testing.assert_allclose(transformed.mean, baseline.mean, atol=3e-14)
        np.testing.assert_allclose(transformed.covariance, baseline.covariance, atol=3e-14)
        held = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        scaled = replace(held, samples=held.samples * gains, observation=observation)
        expected = baseline.log_density(held) - len(held.samples) * np.log(np.abs(gains)).sum()
        self.assertAlmostEqual(transformed.log_density(scaled), expected, places=10)

    def test_near_dependent_coordinates_are_unresolved_not_declared_lost(self):
        geometry = SpatialLayout(("Fz", "Pz"), "left earlobe", "a" * 64, ("EEG", "EEG"))
        with self.assertRaises(SpatialPriorIdentificationError) as captured:
            SpatialObservation(geometry, np.array([[1., 0.], [1., 1e-20]]),
                               ("first", "near-first"), "native_derived_coordinates")
        evidence = captured.exception.record()
        self.assertEqual(evidence["reason"], "observation_resolution_unresolved")
        self.assertNotIn("lost_native_directions", evidence)

    def test_car_structural_declaration_survives_roundoff_and_output_gains(self):
        observation = SpatialObservation.average_reference(self.geometry)
        self.assertEqual(observation.rank_ledger["structural_rank"], 2)
        self.assertEqual(observation.declaration["kind"], "average_actual_EEG")
        with self.assertRaisesRegex(SpatialPriorIdentificationError, "observation_resolution_unresolved"):
            SpatialObservation(self.geometry, observation.matrix, observation.output_names,
                               observation.output_reference)
        changed_matrix = observation.matrix.copy()
        changed_matrix[0, 0] += 1e-8
        with self.assertRaisesRegex(ValueError, "declaration"):
            SpatialObservation(self.geometry, changed_matrix, observation.output_names,
                               observation.output_reference, dict(observation.declaration))
        values = block(self.rng, self.geometry, observation=observation)
        baseline = self.fit(blocks=(values,))
        scaled_observation = observation.with_output_gains((1., 1e-20, 1e-20))
        scaled_values = replace(values, samples=values.samples * np.array([1., 1e-20, 1e-20]),
                                observation=scaled_observation)
        changed = self.fit(blocks=(scaled_values,))
        np.testing.assert_allclose(changed.mean, baseline.mean, atol=3e-14)
        np.testing.assert_allclose(changed.covariance, baseline.covariance, atol=3e-14)
        held = block(self.rng, self.geometry, "held", observation=observation, scope="read_only_prediction")
        scaled_held = replace(held, samples=held.samples * np.array([1., 1e-20, 1e-20]),
                              observation=scaled_observation)
        jacobian = scaled_observation.coordinate_log_volume - observation.coordinate_log_volume
        self.assertAlmostEqual(changed.log_density(scaled_held),
                               baseline.log_density(held) - len(held.samples) * jacobian, places=10)

    def test_weak_precision_coordinated_relabel_cannot_pass_frozen_likelihood(self):
        geometry = SpatialLayout(("Fz", "Pz"), "left earlobe", "a" * 64, ("EEG", "EEG"))
        p = SpatialSignalPrior(geometry, np.zeros((1, 2)), np.eye(1), np.diag([1., 1e16]),
            dict(self.prior.design_binding), dict(self.prior.source))
        values = SpatialResponseBlock(np.zeros((4, 2)), np.zeros((4, 1)),
            SpatialObservation.native(geometry), auth("C", 4))
        fitted = self.fit(p=p, blocks=(values,), noise=np.eye(2))
        record = fitted.record()
        root = np.asarray(record["precision_root"])
        root[1, 1] *= 2
        inverse = np.linalg.solve(root, np.eye(2))
        record["precision_root"], record["covariance"] = root.tolist(), (inverse @ inverse.T).tolist()
        resign_posterior(record)
        with self.assertRaises(ValueError):
            SpatialSignalPosterior.from_record(record)
        # Re-signing Q as well cannot preserve both actual QR and orthogonality.
        q = np.asarray(record["orthogonal_factor"])
        q[:, 1] *= .5
        record["orthogonal_factor"] = q.tolist()
        resign_posterior(record)
        with self.assertRaises(ValueError):
            SpatialSignalPosterior.from_record(record)
        # A coordinated weak native-noise root change must still contradict
        # the actual frozen Sigma. It cannot be hidden by its strong variance.
        weak_noise = np.diag([1., 1e-40])
        weak_fitted = self.fit(p=p, blocks=(values,), noise=weak_noise)
        record = weak_fitted.record()
        noise_root = np.asarray(record["native_noise_root"])
        noise_root[1, 1] *= 2
        record["native_noise_root"] = noise_root.tolist()
        for entry in record["fit_records"]:
            entry["compression"]["native_noise_root_sha256"] = array_digest(noise_root)
        record["fit_sha256"] = metadata_digest(record["fit_records"])
        resign_posterior(record)
        with self.assertRaises(ValueError):
            SpatialSignalPosterior.from_record(record)

    def test_posterior_covariance_small_direction_and_tiny_rhs_are_checked(self):
        geometry = SpatialLayout(("Fz", "Pz"), "left earlobe", "a" * 64, ("EEG", "EEG"))
        p = SpatialSignalPrior(geometry, np.zeros((1, 2)), np.eye(1), np.diag([1., 1e16]),
            dict(self.prior.design_binding), dict(self.prior.source))
        values = SpatialResponseBlock(np.zeros((4, 2)), np.zeros((4, 1)),
            SpatialObservation.native(geometry), auth("C", 4))
        fitted = self.fit(p=p, blocks=(values,), noise=np.eye(2))
        record = fitted.record()
        record["covariance"][0][0] *= 4
        resign_posterior(record)
        with self.assertRaises(ValueError):
            SpatialSignalPosterior.from_record(record)
        record = fitted.record()
        record["response"][1] = 1e-20
        root = np.asarray(record["precision_root"])
        record["mean"] = np.linalg.solve(root, np.asarray(record["response"])).reshape((1, 2), order="F").tolist()
        resign_posterior(record)
        with self.assertRaises(ValueError):
            SpatialSignalPosterior.from_record(record)

    def test_signal_prior_changes_do_not_reweight_noise(self):
        values = block(self.rng, self.geometry)
        roi = self.prior.signal_covariance.copy()
        roi[2, 2] *= 9
        altered = prior(self.geometry, mean=self.prior.mean,
                        temporal=self.prior.temporal_covariance, signal=roi)
        baseline = self.fit(blocks=(values,))
        fitted = self.fit(p=altered, blocks=(values,))
        np.testing.assert_array_equal(fitted.native_noise_covariance, baseline.native_noise_covariance)
        self.assertGreater(np.linalg.norm(fitted.mean - baseline.mean), .001)

    def test_prior_factor_gauge_is_not_a_voltage_change(self):
        values = block(self.rng, self.geometry)
        held = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        baseline = self.fit(blocks=(values,))
        changed_prior = prior(self.geometry, mean=self.prior.mean,
                              temporal=50 * self.prior.temporal_covariance,
                              signal=self.prior.signal_covariance / 50)
        changed = self.fit(p=changed_prior, blocks=(values,))
        np.testing.assert_allclose(baseline.mean, changed.mean, atol=3e-14)
        np.testing.assert_allclose(baseline.covariance, changed.covariance, atol=3e-14)
        self.assertAlmostEqual(baseline.log_density(held), changed.log_density(held), places=11)

    def test_null_prior_mean_does_not_create_car_information(self):
        observation = SpatialObservation.average_reference(self.geometry)
        values = block(self.rng, self.geometry, observation=observation)
        held = block(self.rng, self.geometry, "held", observation=observation, scope="read_only_prediction")
        baseline = self.fit(blocks=(values,))
        changed_prior = prior(self.geometry, mean=self.prior.mean + np.array([[7.], [-3.]]),
                              temporal=self.prior.temporal_covariance, signal=self.prior.signal_covariance)
        changed = self.fit(p=changed_prior, blocks=(values,))
        np.testing.assert_allclose(baseline.mean @ observation.matrix.T,
                                   changed.mean @ observation.matrix.T, atol=3e-14)
        self.assertAlmostEqual(baseline.log_density(held), changed.log_density(held), places=11)

    def test_graph_positions_and_learning_fail_closed(self):
        p = SpatialSignalPrior.graph(layout=self.geometry, mean=self.prior.mean,
            temporal_covariance=self.prior.temporal_covariance, variance=3., length_scale_m=.1,
            graph_strength=2., design_binding=dict(self.prior.design_binding), source=dict(self.prior.source))
        self.assertGreater(np.linalg.eigvalsh(p.signal_covariance).min(), 0)
        np.testing.assert_allclose(p.signal_covariance @ np.ones(3), 3 * np.ones(3))
        with self.assertRaisesRegex(SpatialPriorIdentificationError, "actual_positions"):
            SpatialSignalPrior.graph(layout=layout(positions=False), mean=self.prior.mean,
                temporal_covariance=self.prior.temporal_covariance, variance=3., length_scale_m=.1,
                graph_strength=2., design_binding=dict(self.prior.design_binding), source=dict(self.prior.source))
        with self.assertRaises(SpatialPriorIdentificationError) as captured:
            SpatialSignalPrior.learn(mechanism="dprime", independent_groups=("only",))
        self.assertEqual(captured.exception.record()["reason"], "independent_training_groups_required")

    def test_read_only_prediction_cannot_fit_and_authority_mismatch_rejected(self):
        held = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        with self.assertRaisesRegex(ValueError, "actual input"):
            self.fit(blocks=(held,))
        wrong = replace(block(self.rng, self.geometry),
                        authorization=auth("C1", 12, dictionary="0" * 64))
        with self.assertRaisesRegex(ValueError, "actual input"):
            self.fit(blocks=(wrong,))
        first = block(self.rng, self.geometry)
        reused = replace(first, authorization={**first.authorization, "record_id": "reused-as-second-C"})
        with self.assertRaisesRegex(ValueError, "reuse actual raw"):
            self.fit(blocks=(first, reused))
        with self.assertRaisesRegex(ValueError, "measured EEG"):
            replace(self.geometry, channel_types=("EEG", "EEG", "EOG"))

    def test_save_load_hashes_relations_and_immutable_state(self):
        fitted = self.fit()
        record = fitted.record()
        with patch("numpy.linalg.qr", side_effect=AssertionError("load must preserve producer QR")):
            loaded = SpatialSignalPosterior.from_record(record)
        self.assertEqual(record, loaded.record())
        with self.assertRaises(ValueError):
            fitted.mean.setflags(write=True)
        with self.assertRaises(ValueError):
            fitted.native_noise_root.setflags(write=True)
        with self.assertRaises(TypeError):
            fitted.fit_records[0]["authorization"]["fit_scope"] = "test"
        tampered = copy.deepcopy(record)
        tampered["mean"][0][0] += .05
        resign_posterior(tampered)
        with self.assertRaisesRegex(ValueError, "actual C information"):
            SpatialSignalPosterior.from_record(tampered)
        tampered = copy.deepcopy(record)
        tampered["covariance"][0][0] *= 1.01
        resign_posterior(tampered)
        with self.assertRaisesRegex(ValueError, "actual C information"):
            SpatialSignalPosterior.from_record(tampered)
        with tempfile.TemporaryDirectory() as directory:
            path = fitted.save(Path(directory) / "spatial.json")
            reloaded = SpatialSignalPosterior.load(path, expected_artifact_sha256=record["artifact_sha256"])
            self.assertEqual(record, reloaded.record())

    def test_pure_coefficient_unit_scaling_does_not_discard_precision_direction(self):
        values = block(self.rng, self.geometry)
        held = block(self.rng, self.geometry, "held", scope="read_only_prediction")
        baseline = self.fit(blocks=(values,))
        a = np.diag([1., 1e-20])
        inverse = np.diag([1., 1e20])
        p = prior(self.geometry, mean=inverse @ self.prior.mean,
                  temporal=inverse @ self.prior.temporal_covariance @ inverse.T,
                  signal=self.prior.signal_covariance, dictionary="1" * 64)
        def change(record):
            return replace(record, design=record.design @ a,
                           authorization={**record.authorization, "dictionary_sha256": "1" * 64})
        transformed = self.fit(p=p, blocks=(change(values),))
        np.testing.assert_allclose(a @ transformed.mean, baseline.mean, atol=3e-14)
        self.assertAlmostEqual(transformed.log_density(change(held)), baseline.log_density(held), places=10)


if __name__ == "__main__":
    unittest.main()
