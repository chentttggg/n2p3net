import copy
from itertools import combinations

import numpy as np
import pytest
import torch

from research.binary_probability import arithmetic_binary_pool, binary_probability_metrics
from tests.support.datasets import epoch_dataset
from transfer.calibration_candidates import fit_calibration_candidate
from transfer.calibration_state import FrozenTrialPredictor
from transfer.class_calibration import calibration_trial_loss, semantic_trial_logits
from transfer.ensemble_views import observed_spatial_projection, project_raw_observations
from transfer.recipes import TargetRecipe


def test_fixed_class_mass_minibatches_and_partial_tail_match_full_calibration_gradient():
    labels = np.asarray([0, 0, 0, 1])
    fixed, record = calibration_trial_loss(labels, reduction="calibration_balanced_fixed_mass", device="cpu")
    old, _ = calibration_trial_loss(labels, reduction="batch_class_mass", device="cpu")
    fixed.double()
    old.double()
    assert record["pos_weight"] == 3. and record["prior"] == .25 and record["class_mass"] == 1.5
    z = torch.tensor([.4, -.2, 1.3, -.8], dtype=torch.float64, requires_grad=True)
    logits = torch.stack((torch.zeros_like(z), z), dim=1)
    y = torch.as_tensor(labels)
    full = fixed(logits, y)
    expected = .5 * (torch.nn.functional.softplus(z[:3]).mean() + torch.nn.functional.softplus(-z[3]))
    torch.testing.assert_close(full, expected, rtol=0, atol=1e-15)
    expected_gradient = torch.sigmoid(z.detach()) - y
    expected_gradient *= torch.tensor([1/6, 1/6, 1/6, .5], dtype=torch.float64)
    torch.testing.assert_close(torch.autograd.grad(full, z)[0], expected_gradient, rtol=0, atol=1e-15)
    minibatches = [torch.as_tensor(pair) for pair in combinations(range(4), 2)]
    unbiased = torch.stack([fixed(logits.index_select(0, rows), y[rows]) for rows in minibatches]).mean()
    biased = torch.stack([old(logits.index_select(0, rows), y[rows]) for rows in minibatches]).mean()
    torch.testing.assert_close(unbiased, expected, rtol=0, atol=1e-15)
    assert not torch.isclose(biased, expected, rtol=1e-6, atol=1e-6)
    # A final one-row batch uses its actual length, and retains the C mass.
    tail = fixed(logits[-1:], y[-1:])
    torch.testing.assert_close(tail, 2 * torch.nn.functional.softplus(-z[-1]), rtol=0, atol=1e-15)


def test_real_fixed_mass_nn_exports_actual_probability_and_reloads(tmp_path):
    data = epoch_dataset()
    data.y = np.asarray([0, 0, 0, 0, 0, 1])
    recipe = TargetRecipe("current", initialization="scratch", epochs=1, batch_size=4, seed=17,
        trial_loss_reduction="calibration_balanced_fixed_mass")
    fitted = fit_calibration_candidate(data, recipe, np.ones(data.n_epochs, dtype=bool), device="cpu")
    frozen = FrozenTrialPredictor.from_fitted(fitted)
    scores = frozen.probability_scores(data.X)
    calibration = scores["semantics"]
    assert calibration["pos_weight"] == 5. and calibration["prior"] == 1/6
    assert calibration["fit_contract"]["class_risk"]["class_counts"] == (5, 1)
    np.testing.assert_allclose(scores["posterior_logits"], scores["raw_logits"] - np.log(5), atol=1e-14)
    np.testing.assert_allclose(scores["llr"], scores["raw_logits"], atol=1e-14)
    frozen.save(tmp_path / "fitted.pt")
    reloaded = FrozenTrialPredictor.load(tmp_path / "fitted.pt")
    replay = reloaded.probability_scores(data.X)
    for name in ("raw_logits", "posterior_logits", "llr", "probability"):
        np.testing.assert_array_equal(scores[name], replay[name])
    payload = torch.load(tmp_path / "fitted.pt", weights_only=True)
    original = copy.deepcopy(payload)
    payload["record"]["class_calibration"]["prior"] = .5
    torch.save(payload, tmp_path / "wrong_prior.pt")
    with pytest.raises(ValueError, match="actual fitted risk"):
        FrozenTrialPredictor.load(tmp_path / "wrong_prior.pt")
    wrong_schema = copy.deepcopy(original)
    wrong_schema["schema"] = "n2p3_frozen_trial_predictor/1"
    torch.save(wrong_schema, tmp_path / "wrong_schema.pt")
    with pytest.raises(ValueError, match="legacy predictor"):
        FrozenTrialPredictor.load(tmp_path / "wrong_schema.pt")
    wrong_kind = copy.deepcopy(original)
    wrong_kind["record"]["class_calibration"]["origin"] = "target_generative"
    torch.save(wrong_kind, tmp_path / "wrong_kind.pt")
    with pytest.raises(ValueError, match="numerical producer"):
        FrozenTrialPredictor.load(tmp_path / "wrong_kind.pt")
    legacy = copy.deepcopy(payload)
    legacy["schema"] = "n2p3_frozen_trial_predictor/1"
    legacy["record"].pop("class_calibration")
    torch.save(legacy, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="legacy trial score"):
        FrozenTrialPredictor.load(tmp_path / "legacy.pt").probability_scores(data.X)
    with pytest.raises(ValueError):
        fitted.adapter.fit(np.full_like(data.X, np.nan), data.y)
    assert fitted.adapter.training_class_risk_ is None


def test_zero_information_probabilities_use_the_actual_fit_population():

    _, risk = calibration_trial_loss(np.asarray([0] * 9 + [1]), reduction="calibration_balanced_fixed_mass", device="cpu")
    calibration = {"origin": "target_calibration", "pos_weight": 9., "prior": .1, "temperature": 1.,
        "fit_contract": {"class_risk": risk, "class_counts": [9, 1], "prior": .1,
                         "recipe": {"trial_loss_reduction": "calibration_balanced_fixed_mass"}}}
    llr, posterior = semantic_trial_logits([0.], calibration)
    np.testing.assert_allclose(llr, [0.], atol=1e-14)
    assert 1/(1+np.exp(-posterior[0])) == pytest.approx(.1)


def test_executed_source_checkpoint_accepts_compiled_k0_and_actual_selection_ingress(tmp_path):
    import hashlib
    import json
    from dataclasses import replace

    import mne

    from data.candidate_task import MembershipDecision
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import (
        load_epoch_dataset,
        preprocessing_execution_binding,
        preprocessing_spec_from_contract,
        save_epoch_dataset,
    )
    from data.identity import DatasetIdentityTable, origin_subject_key
    from data.preprocess import execute_preprocessing
    from research.contracts import semantic_sha256
    from tests.support.checkpoints import write_checkpoint
    from transfer.checkpoint import PreparedTransferCheckpoint, checkpoint_training_contract
    from transfer.ensemble_views import (
        CalibrationSelection,
        PredictionDecisionView,
        SelectionEpochProfile,
        TrialCheckpointTarget,
        build_calibration_epoch_dataset,
    )

    preprocessing = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none")
    channels = ("Fz", "Cz", "Pz")
    native = np.random.default_rng(361).normal(0, 1e-6, (3, 3000))
    raw = mne.io.RawArray(native, mne.create_info(list(channels), 256., "eeg"), verbose=False)
    events = np.asarray([[600, 0, 1], [1000, 0, 2], [1500, 0, 1], [2000, 0, 2]])
    actual = execute_preprocessing(raw, events, preprocessing)
    profile = SelectionEpochProfile.from_preprocessing(preprocessing, source_sample_rate_hz=256.,
        channel_names=channels, reference="native common")
    decision = MembershipDecision.direct(("a", "b", "a", "b"), ("a", "b"),
        event_ids=tuple(f"e{i}" for i in range(4)), repetition_indices=np.asarray([0, 0, 1, 1]), repetitions=2)
    view = PredictionDecisionView("actual", origin_subject_key("target", "one"), "decision", native.T,
        events[:, 0], decision, "a" * 64, (0, 3000), 256., channels, ("EEG",) * 3, "native common",
        "fixture.raw", "target", "one", 3000)
    data = build_calibration_epoch_dataset((CalibrationSelection(view, np.asarray([False, True, False, True]), "b"),), profile)
    checkpoint = write_checkpoint(tmp_path / "source.pt", data, source_snapshot_sha256="e" * 64)
    payload = torch.load(checkpoint, weights_only=True)
    payload["input_preprocessing_execution"] = actual.execution_receipt
    contract = checkpoint_training_contract(payload)
    contract = replace(contract, preprocessing={**contract.preprocessing,
        **preprocessing_execution_binding(actual.execution_receipt, preprocessing)})
    payload["training_contract"], payload["training_contract_digest"] = contract.record(), contract.digest()
    torch.save(payload, checkpoint)
    target = TrialCheckpointTarget(preprocessing, data.channel_names, DatasetIdentityTable.from_source_rows(["one"], ["target"]),
        {"source_reference": "native common", "selection_epoch_capability": profile.execution_capability()})
    PreparedTransferCheckpoint(checkpoint, target)
    PreparedTransferCheckpoint(checkpoint, data)
    saved = save_epoch_dataset(tmp_path / "OOF.npz", data)
    PreparedTransferCheckpoint(checkpoint, load_epoch_dataset(saved, validation="attested"))
    changed = json.loads(json.dumps(data.provenance))
    changed["selection_epoch_execution"][0]["source_epoch_scopes"][0][0] += 1
    with pytest.raises(ValueError, match="authorized observation"):
        PreparedTransferCheckpoint(checkpoint, replace(data, provenance=changed))
    missing = replace(target, provenance={"source_reference": "native common"})
    with pytest.raises(ValueError, match="compiled selection capability"):
        PreparedTransferCheckpoint(checkpoint, missing)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert semantic_sha256(data.provenance["selection_epoch_capability"])
    training = {"optimizer": {"config": {"pos_weight": 1.}}}
    source = {"origin": "source_mixture", "pos_weight": 1., "prior": .1, "temperature": 1.,
        "fit_contract": {"training_contract": training, "training_contract_digest": semantic_sha256(training)}}
    llr, posterior = semantic_trial_logits([np.log(.1/.9)], source)
    np.testing.assert_allclose(llr, [0.], atol=1e-14)
    assert 1/(1+np.exp(-posterior[0])) == pytest.approx(.1)


def test_raw_cz_projection_retains_observed_rank_and_rejects_false_matrix():
    native = ("Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8")
    slots = ("Fz", "P3", "Pz", "P4", "PO8", "Oz")
    samples = np.arange(80, dtype=np.float64).reshape(10, 8) * 1e-6
    before = samples.copy()
    declaration = {"mode": "observed_measured_reference", "slot_channels": list(slots),
        "reference_channel": "Cz", "observed_only_for_raw_covariance": True, "native_sample_rate_hz": 256.}
    values, channels, operator, receipt = project_raw_observations(samples, native, "right earlobe", declaration,
                                                                 sample_rate_hz=256.)
    expected = samples[:, [0, 4, 2, 5, 7, 3]] - samples[:, [1]]
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(samples, before)
    assert channels == slots and receipt["rank"] == 6 and all(receipt["observed_slots"])
    matrix, mask = observed_spatial_projection(("Fz", "Cz", "Pz"), slots, reference_channel="Cz")
    assert mask.tolist() == [True, False, True, False, False, False]
    reduced, reduced_channels, _, reduced_receipt = project_raw_observations(samples[:, :3], native[:3],
        "right earlobe", declaration, sample_rate_hz=256.)
    assert reduced.shape == (10, 2) and reduced_channels == ("Fz", "Pz") and reduced_receipt["rank"] == 2
    with pytest.raises(ValueError, match="spatial matrix"):
        project_raw_observations(samples, native, "right earlobe", {**declaration, "matrix": np.zeros((6, 8)).tolist()},
                                 sample_rate_hz=256.)
    assert "observed_spatial" in operator and matrix.shape == (6, 3)


def test_binary_probability_pool_retains_stable_risk_at_extreme_scores():
    logits = np.array([[-1000., 1000.], [1000., -1000.]])
    result = arithmetic_binary_pool(logits, (.5, .5))
    np.testing.assert_allclose(result, [0., 0.], atol=0)
    metrics = binary_probability_metrics([0, 1], result)
    assert metrics["nll"] == pytest.approx(np.log(2)) and metrics["brier"] == .25
    same = arithmetic_binary_pool(logits[:1], (1.,))
    np.testing.assert_array_equal(same, logits[0])


def test_native_binary_caller_exports_three_seed_proper_pools_and_classic_aliases(tmp_path):
    import hashlib
    import json
    from dataclasses import replace

    from data.epochs import load_epoch_dataset
    from data.source_records import prepare_source_record_prefixes
    from experiments.run_erp_comparator import run_comparison
    from research.contracts import semantic_sha256
    from tests.support.checkpoints import write_checkpoint
    from tests.support.raw_prefix import write_real_record

    item, spec, _ = write_real_record(tmp_path)
    prefix = prepare_source_record_prefixes(item, root=tmp_path, output=tmp_path / "prepared",
        preprocessing=replace(spec, baseline_mode="none"), budgets=(8,), minimum_test_trials=2)
    collection = tmp_path / "prefix.json"
    collection.write_text(json.dumps({"schema": "n2p3_target_raw_prefix_collection/1",
        "preprocessing": prefix["preprocessing"], "preparation_method": prefix["preparation_method"], "records": [prefix]}))
    calibration = load_epoch_dataset(prefix["calibration"]["8"]["cache"], validation="attested")
    pairs, catalog = [[11, 21], [12, 22], [13, 23]], []
    for source_seed, _ in pairs:
        checkpoint = write_checkpoint(tmp_path / f"source{source_seed}.pt", calibration, source_snapshot_sha256="e" * 64)
        payload = torch.load(checkpoint, weights_only=True)
        payload["training_contract"]["seed"] = source_seed
        payload["training_contract_digest"] = semantic_sha256(payload["training_contract"])
        torch.save(payload, checkpoint)
        catalog.append({"outer_target_study": "Huebner2017", "source_seed": source_seed, "path": checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "procedure": {"kind": "fixed_recipe", "selection": "not_applicable"}})
    arms = [{"name": name, "family": family, "prefix_collection": str(collection),
             **({"epochs": 1, "batch_size": 4, "trial_loss_reduction": "calibration_balanced_fixed_mass"}
                if family in {"full_fine", "scratch"} else {"bin_width": 16} if family != "waveform_lda" else {})}
            for name, family in (("T", "full_fine"), ("S", "scratch"), ("P", "toeplitz_lda"),
                                 ("X", "xdawn_erp"), ("wave", "waveform_lda"))]
    plan = {"schema": "n2p3_erp_comparator_plan/1", "evidence_scope": "software_correctness",
        "subjects": ["1"], "budgets": [8], "arms": arms, "allowed_input_axes": [],
        "seed_pairs": pairs, "source_seed_catalog": catalog, "require_probability_output": True,
        "probability_pools": [{"name": "TRE", "expert_ids": ["T", "P", "X"], "weights": [.5, .25, .25]},
                              {"name": "SRE", "expert_ids": ["S", "P", "X"], "weights": [.5, .25, .25]}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = run_comparison(path, tmp_path / "run")
    assert len(report["cells"]) == 15 and report["physical_target_fits"] == 9
    assert all(cell["status"] == cell["probability_status"] == "completed" for cell in report["cells"])
    assert len(report["probability_pools"]) == 6
    assert all(cell["status"] == "completed" and cell["probability_scored_rows"] > 0 for cell in report["probability_pools"])
    assert sum(cell["physical_target_fit"] for cell in report["cells"] if cell["family"] not in {"full_fine", "scratch"}) == 3
    for cell in report["cells"]:
        with np.load(tmp_path / "run" / cell["prediction_artifact"]["path"]) as saved:
            assert {"raw_logits", "llr", "posterior_logits", "probability"} <= set(saved.files)
            assert cell["nll"] == binary_probability_metrics(saved["y"], saved["posterior_logits"])["nll"]


def test_bnci_software_mat_canonical_raw_prefix_fits_and_reloads_on_original_physical_support(tmp_path):
    import hashlib
    from dataclasses import replace

    from scipy.io import savemat

    from data.bnci2014_008_candidate import BNCI2014_008_CHANNELS, BNCI2014_008_SOURCE_REFERENCE
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from tests.support.bnci import bnci_mat_fields
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        FittedEnsemble,
        fit_calibration_prefix,
    )
    from transfer.ensemble_views import prepare_bnci_independent_prefixes

    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(377).normal(0, 4, fields["X"].shape).astype(np.float32)
    path = tmp_path / "A01.mat"
    savemat(path, {"data": fields})
    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none")
    views = prepare_bnci_independent_prefixes(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        budgets=(200,), profiles=(), authorization_preprocessing=spec)
    declaration = {"mode": "observed_measured_reference", "slot_channels": ["Fz", "P3", "Pz", "P4", "PO8", "Oz"],
        "reference_channel": "Cz", "observed_only_for_raw_covariance": True, "native_sample_rate_hz": 256.}
    programme = CalibrationExpertProgram("raw6", "continuous_ar", noise_order=8, n_basis=8,
        conditioning_samples=32, spatial=declaration)
    with pytest.raises(TypeError):
        programme.spatial["reference_channel"] = "Fz"
    fit = fit_calibration_prefix(views.calibration[200], (programme,), EnsembleProgram.equal_weights("raw", ("raw6",)))
    assert fit.expert_fit_available["raw6"]
    held = views.decisions[-1]
    scored = fit.score(held)
    assert scored.probability_available and scored.physical_support_digest == held.physical_support_digest
    fit.save(tmp_path / "package")
    replay = FittedEnsemble.load(tmp_path / "package").score(held)
    assert replay.record() == scored.record()
    model = fit._experts[0].predictor
    assert model.channels == ("Fz", "P3", "Pz", "P4", "PO8", "Oz") and model.sfreq == 256.
    receipt = model.provenance["raw_spatial_execution"][0]
    assert receipt["native_channels"] == BNCI2014_008_CHANNELS
    assert receipt["native_reference"] == BNCI2014_008_SOURCE_REFERENCE and receipt["rank"] == 6
    assert views.calibration[200].raw.samples.shape[1] == 8 and held.samples.shape[1] == 8
    assert scored.expert_identities[0].numerical_input_digest != held.raw_input_digest


def test_temperature_subset_keeps_fixed_controls_out_of_oof_and_restores_binding(tmp_path, monkeypatch):
    import json
    from dataclasses import replace

    from data.candidate_task import MembershipDecision
    from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
    from data.epochs import preprocessing_spec_from_contract
    from data.identity import origin_subject_key
    from research.contracts import semantic_sha256
    from transfer import ensemble_fit as owner
    from transfer.calibration_candidates import waveform_lda_recipe
    from transfer.ensemble import EnsembleProgram
    from transfer.ensemble_fit import (
        CalibrationExpertProgram,
        FittedEnsemble,
        fit_calibration_ensemble,
    )
    from transfer.ensemble_views import (
        CalibrationSelection,
        PredictionDecisionView,
        SelectionEpochProfile,
    )
    from transfer.recipes import erp_target_recipe

    groups = []
    for group in range(4):
        decision = MembershipDecision.direct(("a", "b", "b", "a"), ("a", "b"),
            event_ids=tuple(f"g{group}/e{row}" for row in range(4)),
            repetition_indices=np.asarray([0, 0, 1, 1]), repetitions=2)
        view = PredictionDecisionView(record_id=f"r{group}", participant_key=origin_subject_key("fixture", "one"),
            decision_id=f"d{group}", samples=np.random.default_rng(group + 617).normal(0, 1e-6, (551, 3)),
            onsets=np.asarray([51, 115, 179, 243]), decision=decision, source_sha256="a" * 64,
            raw_scope=(group * 600, group * 600 + 551), sample_rate_hz=256., channel_names=("Fz", "Cz", "Pz"),
            channel_types=("EEG",) * 3, reference="right earlobe", source_relative_path="fixture.mat",
            source_dataset_id="fixture", source_subject_id="one", available_sample=group * 600 + 551)
        groups.append(CalibrationSelection(view, np.asarray([True, False, False, True]), "a"))
    spec = replace(preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT), baseline_mode="none")
    profile = SelectionEpochProfile.from_preprocessing(spec, source_sample_rate_hz=256., channel_names=("Fz", "Pz"),
                                                      reference="right earlobe", reference_channel="Cz")
    experts = (CalibrationExpertProgram("toe", "trial", erp_target_recipe("toeplitz_lda", bin_width=16), profile),
               CalibrationExpertProgram("control", "trial", waveform_lda_recipe(), profile))
    actual_fit = owner._fit_expert

    def refuse_control_oof(spec, data, *arguments, **keywords):
        if spec.expert_id == "control" and data.n_epochs < 12:
            raise RuntimeError("unrelated control has no OOF programme")
        return actual_fit(spec, data, *arguments, **keywords)

    monkeypatch.setattr(owner, "_fit_expert", refuse_control_oof)
    fitted = fit_calibration_ensemble(groups[:3], experts, EnsembleProgram.equal_weights("both", ("toe", "control")),
        calibration_strategy="grouped_temperature", temperature_expert_ids=("toe",))
    scores = fitted.score_experts(groups[3].view)
    assert scores["toe"].expert_identities[0].probability_calibrated
    assert not scores["control"].expert_identities[0].probability_calibrated
    assert fitted.temperature_record["temperatures"]["control"] == 1.
    assert all(set(row["expert_fit_digests"]) == {"toe"} for row in fitted.temperature_record["OOF_groups"])
    fitted.save(tmp_path / "programme")
    restored = FittedEnsemble.load(tmp_path / "programme")
    assert restored.score(groups[3].view).record() == fitted.score(groups[3].view).record()
    path = tmp_path / "programme/ensemble.json"
    payload = json.loads(path.read_text())
    payload["record"]["temperature_calibration"]["temperature_expert_ids"] = ["toe", "control"]
    payload["record"]["temperature_calibration"]["fixed_control_expert_ids"] = []
    payload["record_sha256"] = semantic_sha256(payload["record"])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="actual grouped OOF"):
        FittedEnsemble.load(tmp_path / "programme")
