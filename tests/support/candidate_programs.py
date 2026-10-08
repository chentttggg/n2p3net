"""Real official-shape candidate inputs shared by scientific and cloud callers."""

import hashlib
import json
from dataclasses import replace

import numpy as np
from scipy.io import savemat

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.ensemble_views import SelectionEpochProfile
from transfer.recipes import TargetRecipe, erp_target_recipe


def study_plan(tmp_path):
    path = tmp_path / "A01.mat"
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(39).normal(0, 4, fields["X"].shape).astype(np.float32)
    savemat(path, {"data": fields})
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    kwargs = dict(source_sample_rate_hz=256., channel_names=BNCI2014_008_CHANNELS,
                  reference=BNCI2014_008_SOURCE_REFERENCE)
    none = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"), **kwargs)
    baseline = SelectionEpochProfile.from_preprocessing(spec, **kwargs)
    experts = [CalibrationExpertProgram("raw", "continuous_ar", noise_order=0, n_basis=2),
               CalibrationExpertProgram("toeplitz", "trial", erp_target_recipe("toeplitz_lda", bin_width=16), none),
               CalibrationExpertProgram("scratch", "trial", TargetRecipe("scratch", initialization="scratch",
                    epochs=1, batch_size=128, bn_buffers="calibration", dropout="off"), baseline)]
    mixtures = [EnsembleProgram.equal_weights("classic", ("raw", "toeplitz")),
                EnsembleProgram.equal_weights("with-scratch", ("raw", "toeplitz", "scratch"))]
    plan = {"schema": "n2p3_candidate_ensemble_study/1", "evidence_scope": "software_correctness",
            "records": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "authorized_labels": 800, "group_count": 2, "observation_response_seconds": 308 / 256,
            "pre_roll_seconds": .2, "experts": [row.record() for row in experts],
            "mixtures": [row.record() for row in mixtures], "baseline_mixture": "classic", "seeds": [23],
            "device": "cpu", "workers": 1, "cpu_threads_per_worker": 1,
            "calibration_strategy": "fixed", "source_checkpoints": {}}
    output = tmp_path / "plan.json"
    output.write_text(json.dumps(plan), encoding="utf-8")
    return output
