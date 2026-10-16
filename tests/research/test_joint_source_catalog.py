"""Real tiny source/issuer checks for the standalone joint-source slice.

Until the joint expert is integrated, run against the explicitly frozen joint
code closure documented in tmp/JOINT_SOURCE_CATALOG_IMPLEMENTATION_HANDOFF.md.
The fixture uses two channels and a short synthetic acquisition, not public EEG.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.candidate_input import (
    CandidateDatasetAdapter,
    OriginalCandidateRecording,
    OriginalCandidateSelection,
)
from data.candidate_task import MembershipDecision
from data.contract import DEFAULT_P300_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from research.joint_flat_authority import build_source_record_parameters
from research.joint_source_catalog import (
    SOURCE_ROLE,
    JointFlatSourceCatalog,
    _SoftwareSourceIssuer,
    admit_huebner_joint_source_data,
    issue_joint_source_entry,
    reexport_joint_source_entry,
    resolve_joint_candidate_sources,
)
from transfer.continuous_decision import ContinuousDecisionModel
from transfer.ensemble_views import RecordCalibrationPrefixCompiler

_joint_programme = import_module("transfer.joint_flat_programme")
FrozenJointFlatSource = _joint_programme.FrozenJointFlatSource
FrozenJointFlatSourcePreparation = _joint_programme.FrozenJointFlatSourcePreparation


def _record_issuer(subject, *, study="tiny-software-study", seed=79):
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 5e-6, (192, 2))
    onsets = np.arange(8, 168, 8, dtype=np.int64)
    labels = np.empty(len(onsets), dtype=bool)
    selections = []
    for index in range(len(onsets) // 2):
        rows = (2 * index, 2 * index + 1)
        q = "A" if index % 2 else "B"
        decision = MembershipDecision.direct(("A", "B"), ("A", "B"),
            event_ids=(f"{subject}:{index}:0", f"{subject}:{index}:1"),
            repetition_indices=np.zeros(2, dtype=np.int64), repetitions=1)
        labels[list(rows)] = decision.membership[:, decision.candidate_vocabulary.index(q)]
        selections.append(OriginalCandidateSelection(f"{subject}:selection:{index}", rows, decision, q))
    raw_digest = hashlib.sha256(samples.tobytes() + onsets.tobytes() + labels.tobytes()).hexdigest()
    participant = f"n2p3-origin://{study}/participant/{subject}"
    recording = OriginalCandidateRecording(f"{study}:{subject}:session:0:run:0", raw_digest,
        f"{subject}.synthetic", participant, subject, "0", "0", 16., ("Cz", "Pz"), "nose", len(samples),
        onsets, labels, tuple(selections), {"scope": "software_correctness", "sha256": raw_digest},
        lambda start, stop: samples[start:stop])
    adapter = CandidateDatasetAdapter(study, participant, subject, "synthetic_software_fixture",
        (recording,), ({"sha256": raw_digest},), {"software_fixture": True}, "synthetic_direct_selection",
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    preprocessing = replace(preprocessing_spec_from_contract(DEFAULT_P300_DATA_CONTRACT),
        tmin_ms=-62.5, tmax_ms=250.)
    compiler = RecordCalibrationPrefixCompiler(adapter, budgets=(0, 8), profiles=(),
        authorization_preprocessing=preprocessing, response_seconds=.25, pre_roll_seconds=.0625)
    return compiler, samples


@pytest.fixture(scope="module")
def original_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("joint-original-source")
    compiler, samples = _record_issuer("source")
    target, _ = _record_issuer("target", seed=91)
    prefix = compiler.compile(8)
    model_path, codec_path = directory / "source.npz", directory / "codec.json"
    parameters, model = build_source_record_parameters(((compiler, prefix),),
        target_participant_key=target.target["participant_key"], degrees_of_freedom=4., relative_noise_shape=1.,
        basis=np.ones((4, 1)), ridge=1., noise_order=0, conditioning_samples=4, source_model_path=model_path)
    source = FrozenJointFlatSource.from_parameters(parameters, original_model_content=model_path.read_bytes())
    codec_path.write_bytes(source.content)
    def artifact(path):
        content = path.read_bytes()
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    events = [
        {"phase": "source_fit_started", "pid": os.getpid(), "time": 1., "actual_public_source_fit_attempts": 1,
            "source_evidence_scope": "software_correctness"},
        {"phase": "source_saved", "pid": os.getpid(), "time": 2., "source_model": artifact(model_path),
            "frozen_parameters": artifact(codec_path), "actual_successful_public_source_fits": 1,
            "noise_refits_after_original_source": 0},
        {"phase": "later_target_failure", "pid": os.getpid(), "time": 3.},
    ]
    event_path, terminal_path = directory / "events.jsonl", directory / "aggregate.json"
    event_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf8")
    terminal = {"schema": "joint_source_software_fixture_terminal/1", "observed_original_fitter_pid": os.getpid(),
        "source_successful_fit_count": 1, "source_model_sha256": artifact(model_path)["sha256"],
        "frozen_source_sha256": source.digest, "source_parameter_refits_after_original_source": 0,
        "source_unchanged": True, "source_after": {"test_fixture": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "monitor_exit": 1, "child_exit_code": None, "monitor_failure": "declared fixture aggregate failure"}
    terminal_path.write_text(json.dumps(terminal), encoding="utf8")
    return SimpleNamespace(directory=directory, compiler=compiler, samples=samples, prefix=prefix,
        target=target.compile(0).target_contract, model=model, source=source, model_path=model_path,
        codec_path=codec_path, event_path=event_path, terminal_path=terminal_path)


def _no_fit(*_args, **_kwargs):
    raise AssertionError("source registration or resolution attempted a new fit")


def _entry(fixture):
    return issue_joint_source_entry("source-C8-to-target", codec_path=fixture.codec_path,
        original_model_path=fixture.model_path, source_cohort=((fixture.compiler, fixture.prefix),),
        target_contract=fixture.target, exclusion_scope="target_participant", event_log_path=fixture.event_path,
        started_event_index=0, completed_event_index=1, aggregate_terminal_path=fixture.terminal_path,
        source_evidence_scope="software_correctness")


def _catalog(entry, *, preparation=None, scope="software_correctness"):
    return JointFlatSourceCatalog([entry], base_directory=Path.cwd(), evidence_scope=scope,
        _source_preparation=preparation)


def test_real_source_stage_survives_failed_aggregate_and_resolution_never_refits(original_source, monkeypatch):
    monkeypatch.setattr(ContinuousDecisionModel, "fit", _no_fit)
    from research.channel_noise_mechanism import FittedChannelNoise
    monkeypatch.setattr(FittedChannelNoise, "fit", _no_fit)
    entry = _entry(original_source)
    class CountPreparation(FrozenJointFlatSourcePreparation):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def prepare(self, *args, **kwargs):
            self.calls += 1
            return super().prepare(*args, **kwargs)
    prep = CountPreparation()
    catalog = _catalog(entry, preparation=prep)
    first = catalog.resolve(entry["source_id"], target_contract=original_source.target,
        exclusion_scope="target_participant", pipeline_pair=(101, 103))
    second = catalog.resolve(entry["source_id"], target_contract=original_source.target,
        exclusion_scope="target_participant", pipeline_pair=(109, 113))
    assert first.source is second.source and prep.calls == 1
    np.testing.assert_array_equal(first.source.parameters.prior.mean, original_source.source.parameters.prior.mean)
    np.testing.assert_array_equal(first.source.parameters.prior.root, original_source.source.parameters.prior.root)
    assert first.source.original_model_content == original_source.model_path.read_bytes()
    assert first.receipt["actual_source_seed"] is None and first.receipt["pipeline_pair"] == (101, 103)
    assert second.receipt["pipeline_pair"] == (109, 113)
    producer = catalog.record()[0]["producer"]
    assert producer["source_fit_count"] == 1 and producer["export_source_fit_count"] == 0
    assert producer["aggregate_terminal"]["monitor_exit"] == 1
    assert producer["aggregate_terminal"]["child_exit_code"] is None
    assert producer["completed"]["phase"] == "source_saved"
    with pytest.raises(TypeError, match="frozen"):
        catalog.record()[0]["source_summary"]["source_records"][0]["n_events"] = 999


def test_joint_binding_requires_source_before_fitting_and_never_uses_nn_decoder(original_source, monkeypatch):
    from transfer import checkpoint
    from transfer.ensemble_fit import CalibrationExpertProgram

    monkeypatch.setattr(checkpoint, "load_checkpoint_payload", _no_fit)
    monkeypatch.setattr(ContinuousDecisionModel, "fit", _no_fit)
    catalog = _catalog(_entry(original_source))
    sampler = _joint_programme.JointFlatSamplingProgram(4., 8, 8, (2, 3), 1, 2, 1., 1.2, 5., 5., .1)
    joint = CalibrationExpertProgram("joint", "joint_flat", joint_flat=_joint_programme.JointFlatProgram(
        sampler, 4, "whole_observed_current_q"))
    nn = SimpleNamespace(expert_id="NN", family="trial", source_checkpoint_role="pretrained_weights")
    with pytest.raises(ValueError, match="every joint expert"):
        resolve_joint_candidate_sources((joint, nn), {}, catalog, target_contract=original_source.target)
    sources, receipts = resolve_joint_candidate_sources((joint, nn), {"joint": "source-C8-to-target"}, catalog,
        target_contract=original_source.target, pipeline_pair=(17, 19))
    assert set(sources) == set(receipts) == {"joint"} and receipts["joint"]["role"] == SOURCE_ROLE
    assert sources["joint"].digest == original_source.source.digest


@pytest.mark.parametrize("change", ["participant", "reference", "fs", "input_sha", "study", "scope"])
def test_warm_source_rechecks_actual_target_and_scope(original_source, change):
    catalog = _catalog(_entry(original_source))
    target = copy.deepcopy(dict(original_source.target))
    scope = "target_participant"
    if change == "participant":
        target["participant_key"] = "another-unseen-target"
    elif change == "reference":
        target["reference"] = "different-actual-reference"
    elif change == "fs":
        target["sample_rate_hz"] = 32.
    elif change == "input_sha":
        target["source_sha256"] = original_source.prefix.target_contract["source_sha256"]
    elif change == "study":
        target["source_dataset_id"] = "another-study"
    else:
        scope = "whole_target_dataset"
    with pytest.raises(ValueError, match="target|physical|source|exclusion"):
        catalog.resolve("source-C8-to-target", target_contract=target, exclusion_scope=scope)


def test_participant_source_cannot_be_relabelled_as_whole_study_or_public(original_source):
    entry = _entry(original_source)
    changed = copy.deepcopy(entry)
    changed["exclusion"]["scope"] = "whole_target_dataset"
    with pytest.raises(ValueError, match="whole-study"):
        _catalog(changed)
    with pytest.raises(ValueError, match="new original-public-data admission"):
        _catalog(entry, scope="public_calibration_development")
    changed = copy.deepcopy(entry)
    changed["source_evidence_scope"] = "public_calibration_development"
    with pytest.raises(ValueError, match="original source stage"):
        _catalog(changed)


def test_original_issuer_checks_actual_source_arrays_against_original_npz(original_source):
    other, _ = _record_issuer("other-source", seed=173)
    with pytest.raises(ValueError, match="record order|samples|identities"):
        issue_joint_source_entry("wrong-source", codec_path=original_source.codec_path,
            original_model_path=original_source.model_path, source_cohort=((other, other.compile(8)),),
            target_contract=original_source.target, exclusion_scope="target_participant",
            event_log_path=original_source.event_path, started_event_index=0, completed_event_index=1,
            aggregate_terminal_path=original_source.terminal_path, source_evidence_scope="software_correctness")


def test_original_issuer_rechecks_raw_samples_even_with_the_same_record_identity(original_source):
    before = original_source.samples[0, 0]
    original_source.samples[0, 0] += 1e-6
    try:
        with pytest.raises(ValueError, match="original bytes|samples"):
            _entry(original_source)
    finally:
        original_source.samples[0, 0] = before


@pytest.mark.parametrize("field", ["codec", "original_model", "completed_event"])
def test_source_artifact_and_fit_stage_identity_are_joined(original_source, tmp_path, field):
    entry = _entry(original_source)
    if field in {"codec", "original_model"}:
        original = Path(entry[field]["path"]).read_bytes()
        changed = tmp_path / field
        changed.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        entry[field]["path"] = str(changed)
    else:
        events = Path(entry["fit_evidence"]["events"]["path"]).read_bytes().splitlines()
        completed = json.loads(events[1])
        completed["source_model"]["sha256"] = "1" * 64
        payload = events[0] + b"\n" + json.dumps(completed).encode() + b"\n"
        changed = tmp_path / "wrong-events.jsonl"
        changed.write_bytes(payload)
        entry["fit_evidence"]["events"] = {"path": str(changed), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}
    with pytest.raises(ValueError, match="artifact bytes|different source artifacts"):
        _catalog(entry)


def test_source_summary_must_match_actual_source_model_not_just_its_own_fields(original_source):
    entry = _entry(original_source)
    entry["source_summary"]["source_records"][0]["n_events"] += 1
    with pytest.raises(ValueError, match="parameter or cohort authority"):
        _catalog(entry)


def _software_admission(entry, fixture, *, target=None, cohort=None):
    issuer = _SoftwareSourceIssuer(tuple(cohort or ((fixture.compiler, fixture.prefix),)),
        {"schema": "explicit_software_source_anchor/1", "test_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()})
    return admit_huebner_joint_source_data(entry, target_contract=fixture.target if target is None else target,
        exclusion_scope="target_participant", _software_issuer=issuer)


def test_same_npz_two_actual_targets_use_new_export_with_original_fit_receipt(original_source, tmp_path, monkeypatch):
    from research import channel_noise_mechanism
    from research.channel_noise_mechanism import FittedChannelNoise
    from research.contracts import semantic_sha256

    entry = _entry(original_source)
    old_entry_digest = semantic_sha256(entry)
    original_files = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        original_source.model_path, original_source.codec_path, original_source.event_path, original_source.terminal_path)}
    target, _ = _record_issuer("second-target", seed=137)
    target_contract = target.compile(0).target_contract
    monkeypatch.setattr(ContinuousDecisionModel, "fit", _no_fit)
    monkeypatch.setattr(FittedChannelNoise, "fit", _no_fit)
    monkeypatch.setattr(FittedChannelNoise, "from_temporal_noise", _no_fit)
    monkeypatch.setattr(channel_noise_mechanism, "stationary_root", _no_fit)
    monkeypatch.setattr(channel_noise_mechanism, "compile_shared_stationary_root", _no_fit)
    exported = reexport_joint_source_entry(entry, "source-C8-to-second-target",
        source_cohort=((original_source.compiler, original_source.prefix),), target_contract=target_contract,
        exclusion_scope="target_participant", output_codec_path=tmp_path / "second-target.json")
    assert semantic_sha256(entry) == old_entry_digest
    assert original_files == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original_files}
    assert exported["codec"]["sha256"] != entry["codec"]["sha256"]
    assert exported["fit_stage"] == entry["fit_stage"]
    assert exported["fit_stage"]["codec"]["sha256"] == entry["codec"]["sha256"]
    assert exported["original_model"] == entry["original_model"]
    catalog = JointFlatSourceCatalog([entry, exported], base_directory=Path.cwd(), evidence_scope="software_correctness")
    first = catalog.resolve(entry["source_id"], target_contract=original_source.target, exclusion_scope="target_participant")
    second = catalog.resolve(exported["source_id"], target_contract=target_contract, exclusion_scope="target_participant")
    np.testing.assert_array_equal(first.source.parameters.prior.mean, second.source.parameters.prior.mean)
    np.testing.assert_array_equal(first.source.parameters.prior.root, second.source.parameters.prior.root)
    np.testing.assert_array_equal(first.source.parameters.noise.compiled_prefix_root,
        second.source.parameters.noise.compiled_prefix_root)
    assert first.source.original_model_content == second.source.original_model_content
    assert first.receipt["producer_digest"] == second.receipt["producer_digest"]
    assert second.receipt["readonly_export"]["source_refits"] == 0
    admission = _software_admission(exported, original_source, target=target_contract)
    admitted = JointFlatSourceCatalog([exported], base_directory=Path.cwd(), evidence_scope="software_correctness",
        data_admissions=(admission,))
    assert admitted.resolve(exported["source_id"], target_contract=target_contract,
        exclusion_scope="target_participant").receipt["data_admission"]["codec_sha256"] == exported["codec"]["sha256"]
    with pytest.raises(ValueError, match="actual target|exclusion"):
        catalog.resolve(exported["source_id"], target_contract=original_source.target, exclusion_scope="target_participant")
    with pytest.raises(ValueError, match="source participant|excluded target"):
        reexport_joint_source_entry(entry, "invalid-source-target",
            source_cohort=((original_source.compiler, original_source.prefix),),
            target_contract=original_source.prefix.target_contract, exclusion_scope="target_participant",
            output_codec_path=tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_live_data_admission_rechecks_arrays_and_retains_old_programme_scope(original_source, monkeypatch):
    from dataclasses import FrozenInstanceError

    monkeypatch.setattr(ContinuousDecisionModel, "fit", _no_fit)
    entry = _entry(original_source)
    admission = _software_admission(entry, original_source)
    catalog = JointFlatSourceCatalog([entry], base_directory=Path.cwd(), evidence_scope="software_correctness",
        data_admissions=(admission,))
    row = catalog.resolve(entry["source_id"], target_contract=original_source.target, exclusion_scope="target_participant")
    assert row.receipt["source_evidence_scope"] == "software_correctness"
    assert row.receipt["data_admission"]["data_admission_scope"] == "software_correctness"
    assert row.receipt["data_admission"]["source_records"][0]["input_sha256"] == original_source.model.provenance["records"][0]["input_sha256"]
    assert row.receipt["data_admission"]["source_refits"] == 0
    with pytest.raises(FrozenInstanceError):
        admission.record = {"data_admission_scope": "original_public_data_development"}
    with pytest.raises(TypeError, match="private original-data issuer"):
        JointFlatSourceCatalog([entry], base_directory=Path.cwd(), evidence_scope="public_calibration_development",
            data_admissions=(dict(admission.record),))
    with pytest.raises(ValueError, match="new original-public-data admission"):
        JointFlatSourceCatalog([entry], base_directory=Path.cwd(), evidence_scope="public_calibration_development",
            data_admissions=(admission,))
    changed = dict(original_source.target)
    changed["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="data admission.*target identity"):
        catalog.resolve(entry["source_id"], target_contract=changed, exclusion_scope="target_participant")


def test_data_admission_joins_actual_source_and_export_proof_cannot_be_faked(original_source, tmp_path):
    entry = _entry(original_source)
    other, _ = _record_issuer("impostor", seed=511)
    with pytest.raises(ValueError, match="record order|samples|identities"):
        _software_admission(entry, original_source, cohort=((other, other.compile(8)),))
    changed = copy.deepcopy(entry)
    changed["export"]["source_refits"] = 1
    with pytest.raises(ValueError, match="read-only export receipt"):
        _catalog(changed)
    admission = _software_admission(entry, original_source)
    changed = copy.deepcopy(entry)
    changed["codec"]["path"] = str(tmp_path / "unrelated-codec.json")
    with pytest.raises(ValueError, match="different source/export bytes"):
        JointFlatSourceCatalog([changed], base_directory=Path.cwd(), evidence_scope="software_correctness",
            data_admissions=(admission,))
