"""Behavioral contracts for calibration information priors."""
import numpy as np
import pytest

from transfer.kernel_prior import (
    CompiledKernelPrior,
    KernelPriorRankError,
    KernelPriorSpec,
    compile_kernel_prior,
    resolve_kernel_prior,
)


def designs():
    result = []
    for n in (43, 47):
        t = np.linspace(-1, 1, n)
        result.append(np.column_stack((np.sin(4*t), np.cos(5*t), np.sin(7*t), np.cos(9*t))))
    return result


def compile_prior(values, spec=None):
    return compile_kernel_prior(KernelPriorSpec.calibration_g(9.) if spec is None else spec,
        values, record_ids=("C1", "C2"), fit_scope="target_calibration",
        binding={"design_source": "authorized_synthetic_C"})


def dense_information(values):
    total = np.zeros((values[0].shape[1], values[0].shape[1]))
    for d in values:
        z = np.column_stack((np.ones(len(d)), np.linspace(-1, 1, len(d))))
        projected = d-z@np.linalg.solve(z.T@z, z.T@d)
        total += projected.T@projected
    return total


def test_full_c_incremental_information_root_matches_independent_nuisance_projection():
    values = designs()
    prior = compile_prior(values)
    np.testing.assert_allclose(prior.reference_root.T@prior.reference_root, dense_information(values), atol=2e-13)
    np.testing.assert_allclose(prior.precision_root.T@prior.precision_root, dense_information(values)/9, atol=2e-13)
    assert prior.rank_ledger["rank"] == 4
    assert prior.rank_ledger["observed_rows"] == 90
    assert prior.rank_ledger["nuisance_dimensions"] == 4
    assert prior.spec.record()["information_support"] == "all_authorized_C"


def test_general_basis_gauge_and_shared_program_do_not_use_coefficient_ridge():
    values = designs()
    change = np.array([[2., .3, 0., 0.], [0., .4, 0., 0.], [0., 0., 1.2, -.2], [0., 0., .1, .8]])
    original, transformed = compile_prior(values), compile_prior([d@change for d in values])
    inverse = np.linalg.inv(change)
    np.testing.assert_allclose(transformed.row_covariance, inverse@original.row_covariance@inverse.T, atol=3e-13)


def test_independent_record_drifts_keep_confounding_and_structured_rank_failure():
    values = [np.column_stack((d[:, 0], np.full(len(d), (-1.)**i))) for i, d in enumerate(designs())]
    with pytest.raises(KernelPriorRankError) as caught:
        compile_prior(values)
    failure = caught.value.record()
    assert failure["kind"] == "kernel_prior_rank_failure"
    assert failure["rank_ledger"]["rank"] == 1
    assert failure["rank_ledger"]["requested_dimensions"] == 2
    assert failure["prior_spec"]["rank_policy"] == "require_full_rank"


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_pure_nuisance_roundoff_is_not_identified_information(columns, scale):
    values = [scale*np.column_stack((np.ones(n), np.linspace(-1, 1, n)))[:, :columns] for n in (43, 47)]
    with pytest.raises(KernelPriorRankError) as caught:
        compile_prior(values)
    ledger = caught.value.rank_ledger
    assert ledger["rank"] == 0
    assert ledger["original_design_scale"] > 0
    assert ledger["absolute_threshold"] > max(ledger["singular_values"])


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_uniformly_scaled_real_information_and_resolvable_weak_contrast_remain_supported(scale):
    values = [scale*np.column_stack((np.sin(4*np.linspace(-1, 1, n)), np.cos(5*np.linspace(-1, 1, n))))
              for n in (43, 47)]
    prior = compile_prior(values)
    assert prior.rank_ledger["rank"] == 2
    replay = CompiledKernelPrior.from_record(prior.record())
    assert replay.digest() == prior.digest()
    weak = [np.column_stack((np.ones(n)+1e-10*np.sin(4*np.linspace(-1, 1, n)),
                            np.cos(5*np.linspace(-1, 1, n)))) for n in (43, 47)]
    assert compile_prior(weak).rank_ledger["rank"] == 2


def test_compiled_root_and_provenance_are_immutable_and_replayable():
    original = compile_prior(designs())
    replay = CompiledKernelPrior.from_record(original.record())
    assert replay.digest() == original.digest()
    np.testing.assert_array_equal(replay.precision_root, original.precision_root)
    with pytest.raises(ValueError):
        original.precision_root[0, 0] = 3
    with pytest.raises(TypeError):
        original.provenance["fit_scope"] = "held_test"


def test_spec_kind_parameters_are_mutually_exclusive_and_prior_is_not_inverse_ridge():
    assert resolve_kernel_prior(None, None) == KernelPriorSpec.isotropic_ridge(1.)
    with pytest.raises(ValueError, match="mutually"):
        resolve_kernel_prior(KernelPriorSpec.calibration_g(9), 1.)
    with pytest.raises(ValueError):
        KernelPriorSpec("calibration_g", g=9., precision=1.)
    with pytest.raises(ValueError):
        KernelPriorSpec.from_record({**KernelPriorSpec.calibration_g(9.).record(), "precision": 1.})


def test_forged_g_cannot_relabel_the_saved_precision_root():
    prior = compile_prior(designs())
    payload = prior.record()
    payload["spec"]["g"] = 8.
    # Recompute the outer content checksum; numeric relation must still reject.
    from transfer.noise_model import metadata_digest
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="precision root"):
        CompiledKernelPrior.from_record(payload)


def test_load_recomputes_rank_resolution_from_saved_original_design_root():
    prior = compile_prior(designs())
    payload = prior.record()
    payload["provenance"]["rank_ledger"]["absolute_threshold"] = 1e-100
    from transfer.noise_model import metadata_digest
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="rank ledger"):
        CompiledKernelPrior.from_record(payload)


def test_trace_isotropic_control_matches_g_energy_and_scales_with_g():
    values = [d@np.diag([.25, 1., 3., 5.]) for d in designs()]
    information = dense_information(values)
    k = len(information)
    controls = []
    for g in (2., 8.):
        control = compile_prior(values, KernelPriorSpec.calibration_trace_isotropic(g))
        g_prior = compile_prior(values, KernelPriorSpec.calibration_g(g))
        expected_precision = np.trace(information)/(k*g)
        np.testing.assert_allclose(control.derived_precision, expected_precision, rtol=3e-15)
        np.testing.assert_allclose(control.row_covariance, np.eye(k)/expected_precision, rtol=3e-15)
        for prior in (control, g_prior):
            np.testing.assert_allclose(np.trace(information@prior.row_covariance), k*g, rtol=3e-15)
            np.testing.assert_allclose(prior.reference_root.T@prior.reference_root, information, atol=1e-12)
        assert not np.isclose(control.derived_precision, 1/g)
        assert not np.allclose(control.row_covariance, g_prior.row_covariance)
        assert g_prior.derived_precision is None
        controls.append(control)
    np.testing.assert_allclose(controls[1].row_covariance, 4*controls[0].row_covariance, rtol=3e-15)
    assert controls[1].derived_precision == pytest.approx(controls[0].derived_precision/4)


def test_trace_control_is_coordinate_fixed_under_general_basis_change():
    values = designs()
    change = np.array([[2., .3, 0., 0.], [0., .4, 0., 0.], [0., 0., 1.2, -.2], [0., 0., .1, .8]])
    original = compile_prior(values, KernelPriorSpec.calibration_trace_isotropic(9.))
    transformed = compile_prior([d@change for d in values], original.spec)
    inverse = np.linalg.inv(change)
    assert not np.allclose(transformed.row_covariance, inverse@original.row_covariance@inverse.T)
    assert not np.allclose(values[0]@original.row_covariance@values[0].T,
                           values[0]@change@transformed.row_covariance@change.T@values[0].T)
    np.testing.assert_allclose(np.trace(dense_information([d@change for d in values])@transformed.row_covariance), 36.)


@pytest.mark.parametrize("case", ["zero", "affine", "duplicate"])
def test_trace_control_preserves_g_singular_reference_failure_lineage(case):
    if case == "zero":
        values = [np.zeros((len(d), 2)) for d in designs()]
    elif case == "affine":
        values = [np.column_stack((np.ones(len(d)), np.linspace(-1, 1, len(d)))) for d in designs()]
    else:
        values = [np.column_stack((d[:, 0], d[:, 0])) for d in designs()]
    failures = []
    for kind in (KernelPriorSpec.calibration_g, KernelPriorSpec.calibration_trace_isotropic):
        with pytest.raises(KernelPriorRankError) as caught:
            compile_prior(values, kind(9.))
        failure = caught.value.record()
        assert failure["kind"] == "kernel_prior_rank_failure"
        assert failure["prior_spec"]["kind"] == kind(9.).kind
        assert failure["provenance"]["fit_scope"] == "target_calibration"
        failures.append(failure["rank_ledger"])
    assert failures[0] == failures[1]
    assert failures[0]["rank"] == (1 if case == "duplicate" else 0)


def test_trace_control_compiled_state_replays_and_rejects_resigned_wrong_scale():
    from transfer.noise_model import metadata_digest

    control = compile_prior(designs(), KernelPriorSpec.calibration_trace_isotropic(9.))
    replay = CompiledKernelPrior.from_record(control.record())
    assert replay.digest() == control.digest()
    assert replay.derived_precision == control.derived_precision
    with pytest.raises(ValueError):
        replay.precision_root[0, 0] = 1.
    payload = control.record()
    payload["spec"]["g"] = 8.
    payload["artifact_sha256"] = metadata_digest({k: v for k, v in payload.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="precision root"):
        CompiledKernelPrior.from_record(payload)


def test_trace_control_rejects_external_precision_and_held_fit_scope():
    spec = KernelPriorSpec.calibration_trace_isotropic(9.)
    assert KernelPriorSpec.from_record(spec.record()) == spec
    with pytest.raises(ValueError, match="mutually"):
        resolve_kernel_prior(spec, 1.)
    with pytest.raises(ValueError):
        KernelPriorSpec.from_record({**spec.record(), "precision": 1.})
    with pytest.raises(ValueError, match="fit scope"):
        compile_kernel_prior(spec, designs(), record_ids=("C1", "C2"), fit_scope="held_test",
                             binding={"design_source": "held"})
