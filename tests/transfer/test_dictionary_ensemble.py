import numpy as np
import pytest

from tests.transfer.test_ensemble_fit import selections
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram, FittedEnsemble, fit_calibration_ensemble
from transfer.kernel_prior import KernelPriorSpec
from transfer.response_dictionary import ResponseDictionarySpec


def program():
    return CalibrationExpertProgram("cos_hard", "continuous_ar", noise_order=1, conditioning_samples=32,
        dictionary=ResponseDictionarySpec("cosine", .5, 5, "hard_zero"), kernel_prior=KernelPriorSpec.calibration_g(4.))


def test_owned_program_rejects_competing_n_basis_and_reconstructs_wire():
    spec = program()
    assert "n_basis" not in spec.record() and "response_seconds" not in spec.record()
    assert CalibrationExpertProgram.from_record(spec.record()).dictionary == spec.dictionary
    with pytest.raises(ValueError, match="conflict|owned"):
        CalibrationExpertProgram("bad", "continuous_ar", dictionary=spec.dictionary, n_basis=6)


def test_dictionary_real_grouped_oof_and_whole_package_readonly_replay(tmp_path):
    groups = selections()
    spec = program()
    mixture = EnsembleProgram.equal_weights("cos", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture, calibration_strategy="grouped_temperature")
    before = fitted.score(groups[3].view)
    path = fitted.save(tmp_path/"model")
    restored = FittedEnsemble.load(path)
    assert restored.score(groups[3].view).record() == before.record()
    predictor = restored._experts[0].predictor
    assert predictor.dictionary.spec == spec.dictionary
    assert predictor.dictionary_geometry["kernel_dimensions"] == 6
    assert len(fitted.temperature_record["OOF_groups"]) == 3
    assert predictor.kernel_prior.provenance["binding"]["dictionary_digest"] == predictor.dictionary.digest()
    np.testing.assert_array_equal(predictor.basis, predictor.dictionary.matrix)
