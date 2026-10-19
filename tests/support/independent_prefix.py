"""Deterministic raw BNCI fixture for independent prefix caller tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
from scipy.io import savemat

from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from tests.support.bnci import bnci_mat_fields
from transfer.ensemble_fit import CalibrationExpertProgram
from transfer.ensemble_views import SelectionEpochProfile, prepare_bnci_independent_prefixes
from transfer.recipes import erp_target_recipe


def independent_prefix_case(root):
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(39).normal(0, 4, fields["X"].shape).astype(np.float32)
    path = root / "A01.mat"
    savemat(path, {"data": fields})
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    profile = SelectionEpochProfile.from_preprocessing(replace(spec, baseline_mode="none"),
        source_sample_rate_hz=256., channel_names=BNCI2014_008_CHANNELS, reference=BNCI2014_008_SOURCE_REFERENCE)
    experts = (CalibrationExpertProgram("raw", "continuous_ar", noise_order=0, n_basis=2),
               CalibrationExpertProgram("toeplitz", "trial", erp_target_recipe("toeplitz_lda", bin_width=16), profile))
    views = prepare_bnci_independent_prefixes(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        budgets=(0, 200), profiles=(profile,), authorization_preprocessing=spec)
    return root, fields, path, spec, experts, views
