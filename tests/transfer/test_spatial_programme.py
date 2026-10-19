"""Real source/C programme factories, exact scope, grouped OOF and frozen replay."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.identity import DatasetIdentityTable, origin_subject_key
from data.lineage import DataLineage
from data.study import _from_metadata
from research.prefix_inner_replay import check_native_fit_inputs, prefix_native_fit_inputs
from tests.support.bnci import bnci_mat_fields
from tests.support.ensemble_fixtures import selections
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_fit import (
    CalibrationExpertProgram,
    FittedEnsemble,
    fit_calibration_ensemble,
    fit_calibration_prefix,
)
from transfer.ensemble_views import (
    CalibrationSelection,
    prepare_bnci_calibration_groups,
    prepare_bnci_independent_prefixes,
)
from transfer.noise_model import metadata_digest
from transfer.response_dictionary import ResponseDictionary, ResponseDictionarySpec
from transfer.spatial_programme import (
    SpatialDecisionModel,
    SpatialSignalProgram,
    _layout,
    declared_spatial_effect,
    fit_spatial_source_group,
)
from transfer.spatial_signal_prior import (
    SourceGroupPolicy,
    SourceGroupPrior,
    SpatialLikelihoodDependence,
)


def specification(programme=None):
    return CalibrationExpertProgram("physical", "continuous_ar", noise_order=0, n_basis=2,
        conditioning_samples=32, signal_prior=SpatialSignalProgram(9e-12) if programme is None else programme)


def cohort_fixture():
    source_groups = []
    names = ("source-1", "source-2", "source-3")
    for person, name in enumerate(names):
        groups = []
        for original in selections()[:2]:
            samples = original.view.samples*(1+.2*person)
            source_sha = hashlib.sha256((name+original.view.source_sha256).encode()).hexdigest()
            view = replace(original.view, record_id=f"{name}:{original.view.record_id}",
                decision_id=f"{name}:{original.view.decision_id}", samples=samples, source_sha256=source_sha,
                participant_key=origin_subject_key("fixture-source", name), source_dataset_id="fixture-source",
                source_subject_id=name, source_relative_path=name+".mat")
            groups.append(CalibrationSelection(view, original.labels, original.target_candidate))
        source_groups.append(tuple(groups))
    source_identity = DatasetIdentityTable.from_source_rows(local_subject_ids=names, source_dataset_ids=("fixture-source",)*3)
    target_identity = DatasetIdentityTable.from_source_rows(local_subject_ids=("one",), source_dataset_ids=("fixture",))
    source_study = _from_metadata(source_identity, DataLineage.source(parameters={"source_dataset_ids":["fixture-source"]}),
        {"dataset_class":"source-study"})
    target_study = _from_metadata(target_identity, DataLineage.source(parameters={"source_dataset_ids":["fixture"]}),
        {"dataset_class":"target-study"})
    dictionary = ResponseDictionary.compile(ResponseDictionarySpec("gaussian", 1.2, 2), sample_rate_hz=256.)
    common = declared_spatial_effect(_layout(selections()[0].view.physical_record()), dictionary,
                                     observation="measured_reference", reference_channel="Cz")
    policy = SourceGroupPolicy(("independent",), np.eye(2)[None,...], 9e-12*np.eye(2), np.array([9e-12]),
        np.array([1e-8]), 20., np.zeros(2), 9e-12*np.eye(2), 100., "f"*64)
    return source_groups, source_identity, target_identity, source_study, target_study, dictionary, common, policy


@pytest.fixture(scope="module")
def source_bundle():
    groups, identity, target, membership, target_membership, dictionary, common, policy = cohort_fixture()
    return fit_spatial_source_group(groups, identity=identity, study_membership=membership,
        target_identity=target, target_membership=target_membership, programme=SpatialSignalProgram(9e-12),
        dictionary=dictionary, common_effect=common, policy=policy, noise_order=0, ridge=1., conditioning_samples=32)


def test_real_grouped_oof_physical_response_and_full_package_replay(tmp_path, monkeypatch):
    import transfer.spatial_programme as spatial_programme

    groups = selections()
    spec = specification()
    mixture = EnsembleProgram.equal_weights("physical", (spec.expert_id,))
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture, calibration_strategy="grouped_temperature")
    assert fitted.fit_available
    assert len(fitted.temperature_record["OOF_groups"]) == 3
    for row in fitted.temperature_record["OOF_groups"]:
        assert row["held_record_id"] not in row["training_record_ids"]
    actual = fitted._experts[0].predictor
    assert isinstance(actual, SpatialDecisionModel)
    assert actual.baseline.noise.innovation_variance != 1.
    assert actual.provenance["response_fit_passes_after_noise_freeze"] == 1
    assert actual.provenance["noise_refitted_after_freeze"] is False
    before = fitted.score(groups[3].view)
    assert before.probability_available
    package = fitted.save(tmp_path/"physical-package")
    assert (package/"expert_000.json").is_file()
    original_state = actual.record()
    assert actual.producer_implementation_sha256 == spatial_programme.IMPLEMENTATION_SHA256
    monkeypatch.setattr(spatial_programme, "IMPLEMENTATION_SHA256", "f"*64)
    restored = FittedEnsemble.load(package)
    restored_model = restored._experts[0].predictor
    assert restored_model.record() == original_state
    assert restored_model.digest() == actual.digest()
    assert restored_model.consumer_implementation_sha256 == "f"*64
    assert replace(restored_model).producer_implementation_sha256 == original_state["implementation_sha256"]
    after = restored.score(groups[3].view)
    assert before.record() == after.record()


def test_source_actual_fitted_noise_blue_forecast_target_and_saved_replay(source_bundle, tmp_path):
    group, sources, measurements = source_bundle
    assert len(sources) == len(measurements) == 3
    assert group.fit_binding["source_authority"]["independent_unit"] == "participant"
    for source, measured in zip(sources, measurements, strict=True):
        assert source.baseline.provenance["fit_scope"] == "source_training"
        assert source.baseline.noise.innovation_variance != 1.
        assert measured.receipt["temporal_noise_sha256"] == source.baseline.noise.record()["artifact_sha256"]
        assert measured.receipt["posterior_mean_used_as_measurement"] is False
        assert measured.receipt["resolution"]["prior_information_used"] is False
        path = tmp_path / (measured.receipt["participant_key"].replace(":", "_")+".json")
        source.save(path)
        restored = SpatialDecisionModel.load(path)
        assert restored.digest() == source.digest()
    spec = specification(SpatialSignalProgram(9e-12, source_group=group))
    mixture = EnsembleProgram.equal_weights("source-physical", (spec.expert_id,))
    groups = selections()
    fitted = fit_calibration_ensemble(groups[:3], (spec,), mixture, calibration_strategy="grouped_temperature")
    assert fitted.fit_available
    actual = fitted._experts[0].predictor
    effect = group.effect
    prior = actual.posterior.prior
    np.testing.assert_allclose(effect.functional@prior.mean.ravel(order="F"), group.mean, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(effect.functional@np.kron(prior.signal_covariance, prior.temporal_covariance)@effect.functional.T,
                               group.forecast_covariance, rtol=1e-10, atol=1e-22)
    before = fitted.score(groups[3].view)
    package = fitted.save(tmp_path/"source-target")
    restored = FittedEnsemble.load(package)
    assert restored.score(groups[3].view).record() == before.record()
    assert restored._experts[0].predictor.programme.source_group.record() == group.record()


def test_mirror_prior_declaration_cannot_relabel_another_fitted_physical_prior(tmp_path):
    spec = specification()
    model = fit_calibration_ensemble(selections()[:2], (spec,), EnsembleProgram.equal_weights("fixed", (spec.expert_id,)))
    predictor = model._experts[0].predictor
    saved = json.loads(json.dumps(predictor.record()))
    saved["implementation_sha256"] = "f"*64
    saved["signal_programme"]["native_signal_variance"] *= 2
    saved["provenance"]["signal_programme"] = copy.deepcopy(saved["signal_programme"])
    saved["artifact_sha256"] = metadata_digest({key:value for key,value in saved.items() if key!="artifact_sha256"})
    path = tmp_path/"relabelled.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="primitive signal programme"):
        SpatialDecisionModel.load(path)


def test_physical_raw_bridge_cannot_relabel_other_noise_training_inputs(tmp_path):
    spec = specification()
    model = fit_calibration_ensemble(selections()[:2], (spec,), EnsembleProgram.equal_weights("fixed", (spec.expert_id,)))
    saved = json.loads(json.dumps(model._experts[0].predictor.record()))
    posterior = saved["posterior"]
    posterior["fit_records"][0]["authorization"]["original_raw_fit_input_sha256"] = "0"*64
    posterior["fit_sha256"] = metadata_digest(posterior["fit_records"])
    posterior["artifact_sha256"] = metadata_digest({key:value for key,value in posterior.items() if key!="artifact_sha256"})
    saved["provenance"]["physical_response_artifact_sha256"] = posterior["artifact_sha256"]
    saved["artifact_sha256"] = metadata_digest({key:value for key,value in saved.items() if key!="artifact_sha256"})
    path = tmp_path/"wrong-raw-bridge.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="controller/noise authority"):
        SpatialDecisionModel.load(path)


def test_saved_source_cohort_cannot_include_the_independently_excluded_target_study(source_bundle):
    group, *_ = source_bundle
    saved = json.loads(json.dumps(group.record()))
    authority = saved["fit_binding"]["source_authority"]
    identity = DatasetIdentityTable.from_payload(authority["source_identity"])
    authority["source_study_membership"] = _from_metadata(identity,
        DataLineage.source(parameters={"source_dataset_ids":["fixture-source"]}), {"dataset_class":"target-study"})
    fields = ("source_identity", "source_study_membership", "target_identity", "target_study_membership",
              "outer_excluded_studies", "participant_units", "original_source_records", "source_programme_digests",
              "source_programme_receipts", "source_measurements")
    authority["authority_sha256"] = metadata_digest({key:authority[key] for key in fields})
    saved["artifact_sha256"] = metadata_digest({key:value for key,value in saved.items() if key!="artifact_sha256"})
    rebound = SourceGroupPrior.from_record(saved, expected_artifact_sha256=saved["artifact_sha256"])
    with pytest.raises(ValueError, match="outer study"):
        SpatialSignalProgram(9e-12, source_group=rebound)


def test_scalar_innovation_diagnostic_never_multiplies_source_effect_noise_again(source_bundle):
    _, sources, _ = source_bundle
    source = sources[0]
    noise = source.baseline.noise
    assert noise.innovation_variance != 1.
    effect = declared_spatial_effect(source.posterior.prior.layout, source.dictionary,
                                     observation="measured_reference", reference_channel="Cz")
    dependence = SpatialLikelihoodDependence.independent(source.posterior, authority_sha256="e"*64)
    measured = source.posterior.estimate_effect(effect, dependence, temporal_noise=noise)
    w = effect.waveform_weights
    inverse = np.linalg.solve(noise.whiten(np.eye(len(w))), np.eye(len(w)))
    expected = np.sqrt(np.sum((w@inverse)**2)*np.diag(effect.spatial_map@source.baseline.innovation_covariance@effect.spatial_map.T))
    np.testing.assert_allclose(measured.noise_sd, expected, rtol=1e-13, atol=0.)
    assert not np.allclose(measured.noise_sd, expected*np.sqrt(noise.innovation_variance), rtol=1e-5, atol=0.)


def test_source_truth_receipt_must_match_the_original_fit_labels(source_bundle):
    group, *_ = source_bundle
    saved = json.loads(json.dumps(group.record()))
    authority = saved["fit_binding"]["source_authority"]
    authority["original_source_records"][0][0]["labels_sha256"] = "0"*64
    fields = ("source_identity", "source_study_membership", "target_identity", "target_study_membership",
              "outer_excluded_studies", "participant_units", "original_source_records", "source_programme_digests",
              "source_programme_receipts", "source_measurements")
    authority["authority_sha256"] = metadata_digest({key:authority[key] for key in fields})
    saved["artifact_sha256"] = metadata_digest({key:value for key,value in saved.items() if key!="artifact_sha256"})
    rebound = SourceGroupPrior.from_record(saved, expected_artifact_sha256=saved["artifact_sha256"])
    with pytest.raises(ValueError, match="calibration labels"):
        SpatialSignalProgram(9e-12, source_group=rebound)


def test_actual_mat_long_source_transport_target_oof_and_saved_replay(tmp_path, monkeypatch):
    from transfer.spatial_signal_prior import SpatialSignalPosterior

    def diagnostic_null_is_not_a_consumer(*args):
        raise AssertionError("actual source/OOF/save-load must use compact exact membership")

    monkeypatch.setattr(SpatialSignalPosterior, "structural_null_basis", property(diagnostic_null_is_not_a_consumer))
    participants = []
    for index, name in enumerate(("A01", "A02")):
        fields = bnci_mat_fields()
        fields["X"] = np.random.default_rng(334+index).normal(0, 5, fields["X"].shape).astype(np.float32)
        path = tmp_path/(name+".mat")
        savemat(path, {"data": fields}, do_compression=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        parsed = prepare_bnci_calibration_groups(path, source_sha256=digest, authorized_labels=400, group_count=2)
        assert parsed.ledger["used_label_count"] == 240
        assert all(len(group.view.samples) > 7500 for group in parsed.groups)
        participants.append(parsed.groups)
    _, _, target, _, target_membership, dictionary, common, policy = cohort_fixture()
    identity = DatasetIdentityTable.from_source_rows(local_subject_ids=("A01", "A02"),
        source_dataset_ids=("BNCI2014-008",)*2)
    membership = _from_metadata(identity, DataLineage.source(parameters={"source_dataset_ids":["BNCI2014-008"]}),
        {"dataset_class":"BNCI2014-008"})
    group, sources, measurements = fit_spatial_source_group(participants, identity=identity,
        study_membership=membership, target_identity=target, target_membership=target_membership,
        programme=SpatialSignalProgram(9e-12), dictionary=dictionary, common_effect=common, policy=policy,
        noise_order=0, ridge=1., conditioning_samples=32)
    for source, measured in zip(sources, measurements, strict=True):
        assert source.channels == ("Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8")
        assert len(measured.mean) == 2
        assert measured.receipt["temporal_noise_sha256"] == source.baseline.noise.record()["artifact_sha256"]
    programme = specification(SpatialSignalProgram(9e-12, source_group=group))
    mixture = EnsembleProgram.equal_weights("source-to-other-layout", (programme.expert_id,))
    target_groups = selections()
    target_model = fit_calibration_ensemble(target_groups[:3], (programme,), mixture,
        calibration_strategy="grouped_temperature")
    assert target_model.fit_available
    prediction = target_model.score(target_groups[3].view)
    assert prediction.probability_available
    restored = FittedEnsemble.load(target_model.save(tmp_path/"mat-source-target"))
    assert restored.score(target_groups[3].view).record() == prediction.record()
    (tmp_path/"actual_source_group.json").write_text(json.dumps(group.record()), encoding="utf-8")
    (tmp_path/"actual_source_blue.json").write_text(json.dumps([item.record() for item in measurements]), encoding="utf-8")


def test_actual_mat_partial_prefix_native_inputs_and_loaded_future_replay(tmp_path):
    fields = bnci_mat_fields()
    fields["X"] = np.random.default_rng(336).normal(0, 5, fields["X"].shape).astype(np.float32)
    path = tmp_path/"A01.mat"
    savemat(path, {"data": fields}, do_compression=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    preprocessing = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    views = prepare_bnci_independent_prefixes(path, source_sha256=digest, budgets=(0, 200, 800), profiles=(),
        authorization_preprocessing=preprocessing)
    spec = specification()
    fixed = EnsembleProgram.equal_weights("physical-prefix", (spec.expert_id,))
    future = next(view for view in views.decisions if view.samples is not None)
    for budget in (200, 800):
        prefix = views.calibration[budget]
        actual = fit_calibration_prefix(prefix, (spec,), fixed)
        assert actual.fit_available
        expected = prefix_native_fit_inputs(prefix, (spec,))
        check_native_fit_inputs(expected, actual, actual_K=budget)
        assert actual._experts[0].predictor.posterior.fit_records[0]["authorization"]["right_censored"]
        score = actual.score(future)
        assert score.probability_available
        loaded = FittedEnsemble.load(actual.save(tmp_path/f"physical-prefix-{budget}"))
        check_native_fit_inputs(expected, loaded, actual_K=budget)
        assert loaded.score(future).record() == score.record()
