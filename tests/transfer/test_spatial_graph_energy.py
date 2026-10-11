"""Attributable fixed-graph energy and complete-receipt owner contracts."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from research.channel_noise_mechanism import FrozenERPResponse, NoiseGeometry
from transfer.noise_model import array_digest, metadata_digest
from transfer.spatial_signal_prior import (
    SpatialLayout,
    SpatialObservation,
    SpatialResponseBlock,
    SpatialSignalPosterior,
    SpatialSignalPrior,
)


@pytest.fixture
def graph_c():
    rng = np.random.default_rng(20261010)
    layout = SpatialLayout(("Fz", "Cz", "Pz"), "right earlobe", "a" * 64,
        ("EEG",) * 3, "V", np.array([[0, .07, .065], [0, .015, .095], [0, -.065, .08]]), "head")
    source = {"role": "predeclared", "authority_sha256": "c" * 64}
    basis = np.array([[1.], [.5], [-.2]])
    binding = {"dictionary_sha256": "b" * 64, "actual_basis_sha256": array_digest(basis),
        "vectorization": "column_major_response_by_native_EEG", "sample_rate_hz": 128.,
        "operator_identity": "native_raw_identity"}
    reference = SpatialSignalPrior(layout, np.zeros((2, 3)),
        np.array([[1.3, .4], [.4, .7]]), .8 * np.eye(3), binding, source)
    noise = np.array([[1.5, .3, -.1], [.3, .6, .05], [-.1, .05, 1.1]])
    blocks = []
    for index in range(2):
        rows = 18
        authority = {"record_id": f"C{index}", "fit_scope": "target_calibration",
            "source_sha256": "d" * 64, "dictionary_sha256": "b" * 64,
            "temporal_program_sha256": "e" * 64,
            "raw_sample_support": [30 * index, 30 * index + rows],
            "original_view": {"participant_key": "participant-A"}}
        blocks.append(SpatialResponseBlock(rng.normal(size=(rows, 3)), rng.normal(size=(rows, 2)),
            SpatialObservation.native(layout), authority,
            np.column_stack((np.ones(rows), np.linspace(-1, 1, rows)))))
    return reference, blocks, noise, source


def energy_oracle(prior, blocks, noise):
    result = 0.
    for block in blocks:
        a = block.observation.support_basis.T @ block.observation.matrix
        d, z = block.design, block.nuisance
        if z is not None:
            d = d - z @ np.linalg.solve(z.T @ z, z.T @ d)
        result += np.trace(prior.temporal_covariance @ (d.T @ d)) * np.trace(
            np.linalg.solve(a @ noise @ a.T, a @ prior.signal_covariance @ a.T))
    return float(result)


def test_same_c_noise_energy_matches_trace_but_not_effective_df(graph_c):
    reference, blocks, noise, source = graph_c
    graph, receipt = reference.energy_matched_graph(blocks, native_noise_covariance=noise,
        length_scale_m=.06, graph_strength=1., source=source)
    target = energy_oracle(reference, blocks, noise)
    assert receipt["actual_matched_energy"] == pytest.approx(target, rel=2e-13)
    assert not receipt["effective_response_df_matched"]
    assert graph.expected_whitened_signal_energy(blocks,
        native_noise_covariance=noise) == pytest.approx(target, rel=2e-13)
    assert graph.signal_covariance @ np.ones(3) == pytest.approx(
        graph.source["graph"]["variance"] * np.ones(3), rel=1e-13)
    zero_samples = [replace(b, samples=np.zeros_like(b.samples)) for b in blocks]
    assert graph.expected_whitened_signal_energy(zero_samples, native_noise_covariance=noise) == (
        graph.expected_whitened_signal_energy(blocks, native_noise_covariance=noise))
    gram = np.zeros((2, 2))
    for block in blocks:
        d, z = block.design, block.nuisance
        d = d - z @ np.linalg.solve(z.T @ z, z.T @ d)
        gram += d.T @ d
    information = np.kron(np.linalg.inv(noise), gram)

    def df(prior):
        precision = np.linalg.inv(np.kron(prior.signal_covariance, prior.temporal_covariance))
        return float(np.trace(np.linalg.solve(precision + information, information)))

    assert abs(df(reference) - df(graph)) > 1e-3
    assert not np.allclose(graph.signal_covariance / graph.signal_covariance[0, 0], noise / noise[0, 0])


@pytest.mark.parametrize("coordinate_change", ["units", "basis", "permutation"])
def test_energy_match_respects_physical_coordinate_changes(graph_c, coordinate_change):
    reference, blocks, noise, source = graph_c
    graph, _ = reference.energy_matched_graph(blocks, native_noise_covariance=noise,
        length_scale_m=.06, graph_strength=1., source=source)
    if coordinate_change == "units":
        layout = replace(reference.layout, signal_unit="uV")
        changed_reference = replace(reference, layout=layout, mean=reference.mean * 1e6,
                                    signal_covariance=reference.signal_covariance * 1e12)
        changed_blocks = [replace(b, samples=b.samples * 1e6,
            observation=SpatialObservation.native(layout)) for b in blocks]
        changed_noise, expected = noise * 1e12, graph.signal_covariance * 1e12
    elif coordinate_change == "basis":
        t = np.array([[2., .8], [-.2, .6]])
        inv_t = np.linalg.inv(t)
        changed_reference = replace(reference, mean=inv_t @ reference.mean,
            temporal_covariance=inv_t @ reference.temporal_covariance @ inv_t.T)
        changed_blocks = [replace(b, design=b.design @ t) for b in blocks]
        changed_noise, expected = noise, graph.signal_covariance
        assert not np.isclose(energy_oracle(reference, changed_blocks, noise),
                              energy_oracle(reference, blocks, noise), rtol=.1)
    else:
        indices = np.array([2, 0, 1])
        layout = replace(reference.layout,
            channels=tuple(reference.layout.channels[i] for i in indices),
            positions_m=reference.layout.positions_m[indices])
        changed_reference = replace(reference, layout=layout, mean=reference.mean[:, indices])
        changed_blocks = [replace(b, samples=b.samples[:, indices],
            observation=SpatialObservation.native(layout)) for b in blocks]
        changed_noise = noise[np.ix_(indices, indices)]
        expected = graph.signal_covariance[np.ix_(indices, indices)]
    changed_graph, receipt = changed_reference.energy_matched_graph(changed_blocks,
        native_noise_covariance=changed_noise, length_scale_m=.06, graph_strength=1., source=source)
    np.testing.assert_allclose(changed_graph.signal_covariance, expected, rtol=2e-13)
    assert receipt["target_energy"] == pytest.approx(energy_oracle(reference, blocks, noise), rel=2e-13)


def test_push_same_prior_through_reference_or_missing_without_rebuilding(graph_c):
    reference, blocks, noise, source = graph_c
    graph, receipt = reference.energy_matched_graph(blocks, native_noise_covariance=noise,
        length_scale_m=.06, graph_strength=1., source=source)
    cz = SpatialObservation.measured_reference(graph.layout, "Cz")
    kept = [0, 2]
    smaller = replace(graph.layout, channels=("Fz", "Pz"), channel_types=("EEG",) * 2,
                      positions_m=graph.layout.positions_m[kept])
    rebuilt = SpatialSignalPrior.graph(layout=smaller, mean=np.zeros((2, 2)),
        temporal_covariance=graph.temporal_covariance, variance=graph.source["graph"]["variance"],
        length_scale_m=.06, graph_strength=1., design_binding=dict(graph.design_binding), source=source)
    pushed = cz.matrix @ graph.signal_covariance @ cz.matrix.T
    assert np.linalg.norm(pushed - rebuilt.signal_covariance) / np.linalg.norm(pushed) > .1
    assert not np.allclose(graph.signal_covariance[np.ix_(kept, kept)], rebuilt.signal_covariance)
    projected = [replace(b, observation=cz, samples=cz.project(b.samples)) for b in blocks]
    actual = graph.expected_whitened_signal_energy(projected, native_noise_covariance=noise)
    assert actual == pytest.approx(energy_oracle(graph, projected, noise), rel=2e-13)
    assert not np.isclose(actual, reference.expected_whitened_signal_energy(projected,
                                                                          native_noise_covariance=noise))
    assert graph.source["C_plugin_energy_matching"]["target_energy"] == receipt["target_energy"]
    assert not receipt["projected_graph_rebuilt"] and not receipt["projected_energy_rematched"]
    gained = cz.with_output_gains([1e-5, -1e5])
    gained_c = [replace(b, observation=gained, samples=gained.project(b.samples)) for b in blocks]
    assert graph.expected_whitened_signal_energy(gained_c,
        native_noise_covariance=noise) == pytest.approx(actual, rel=2e-13)
    car = SpatialObservation.average_reference(graph.layout)
    car_c = [replace(b, observation=car, samples=car.project(b.samples)) for b in blocks]
    assert graph.expected_whitened_signal_energy(car_c,
        native_noise_covariance=noise) == pytest.approx(actual, rel=2e-13)
    missing = SpatialObservation(graph.layout, np.eye(3)[kept], ("Fz", "Pz"),
                                 graph.layout.acquisition_reference)
    missing_c = [replace(b, observation=missing, samples=missing.project(b.samples)) for b in blocks]
    assert graph.expected_whitened_signal_energy(missing_c,
        native_noise_covariance=noise) == pytest.approx(energy_oracle(graph, missing_c, noise), rel=2e-13)


def test_complete_receipt_roundtrip_and_mutation_isolation(graph_c, tmp_path):
    reference, blocks, noise, source = graph_c
    graph, receipt = reference.energy_matched_graph(blocks, native_noise_covariance=noise,
        length_scale_m=.06, graph_strength=1., source=source)
    expected = json.loads(json.dumps(receipt))
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(graph.record()), encoding="utf-8")
    restored = SpatialSignalPrior.from_record(json.loads(path.read_text(encoding="utf-8")))
    assert json.loads(json.dumps(restored.source["C_plugin_energy_matching"])) == expected
    digest_payload = dict(expected)
    digest = digest_payload.pop("match_binding_sha256")
    assert digest == metadata_digest(digest_payload)
    receipt["relative_energy_error"] = -1.
    assert json.loads(json.dumps(graph.source["C_plugin_energy_matching"])) == expected
    np.testing.assert_array_equal(restored.precision_root, graph.precision_root)
    np.testing.assert_array_equal(restored.signal_covariance, graph.signal_covariance)


def test_matched_graph_consumes_existing_posterior_and_physical_factor_owner(graph_c, tmp_path):
    reference, blocks, noise, source = graph_c
    graph, receipt = reference.energy_matched_graph(blocks, native_noise_covariance=noise,
        length_scale_m=.06, graph_strength=1., source=source)
    posterior = SpatialSignalPosterior.fit(graph, blocks, native_noise_covariance=noise,
                                          noise_binding={"authority_sha256": "f" * 64})
    path = tmp_path / "posterior.json"
    posterior.save(path)
    restored = SpatialSignalPosterior.load(path,
        expected_artifact_sha256=posterior.record()["artifact_sha256"])
    assert json.loads(json.dumps(restored.prior.source["C_plugin_energy_matching"])) == receipt
    geometry = NoiseGeometry(graph.layout.channels, 128., "V", graph.layout.acquisition_reference,
                             "native_raw_identity", 3)
    ledger = [{"record_id": b.authorization["record_id"], "participant_key": "participant-A",
        "recording_id": "recording-A", "source_sha256": "d" * 64,
        "read_support": list(b.authorization["raw_sample_support"]), "fit_scope": "target_calibration"}
        for b in blocks]
    response = FrozenERPResponse.from_spatial_posterior(restored, geometry=geometry,
        source_model_sha256="1" * 64, response_id="graph_C", training_records=ledger)
    # Independent definition of column-major to row-major coefficient indices.
    order = [channel * 2 + row for row in range(2) for channel in range(3)]
    np.testing.assert_allclose(response.coefficient_factor @ response.coefficient_factor.T,
                              posterior.covariance[np.ix_(order, order)], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(restored.native_noise_covariance, noise)


@pytest.mark.parametrize("failure", ["empty", "held", "duplicate", "mixed_scopes", "dictionary", "zero"])
def test_energy_matching_rejects_invalid_c_without_fallback(graph_c, failure):
    reference, blocks, noise, source = graph_c
    if failure == "empty":
        requested = []
    elif failure == "held":
        requested = [replace(blocks[0], authorization={**blocks[0].authorization,
                                                      "fit_scope": "read_only_prediction"})]
    elif failure == "duplicate":
        requested = [blocks[0], blocks[0]]
    elif failure == "mixed_scopes":
        requested = [blocks[0], replace(blocks[1], authorization={**blocks[1].authorization,
                                                               "fit_scope": "source_training"})]
    elif failure == "dictionary":
        requested = [replace(blocks[0], authorization={**blocks[0].authorization,
                                                      "dictionary_sha256": "0" * 64})]
    else:
        requested = [replace(blocks[0], design=np.zeros_like(blocks[0].design))]
    with pytest.raises(ValueError):
        reference.energy_matched_graph(requested, native_noise_covariance=noise,
            length_scale_m=.06, graph_strength=1., source=source)
    with pytest.raises(ValueError, match="unknown source-learning mechanism"):
        SpatialSignalPrior.learn(mechanism="graph", estimates=(), policy=None, source_authority={})
