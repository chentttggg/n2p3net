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
