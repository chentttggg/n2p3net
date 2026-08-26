"""Contract and synthetic integration tests for the independent P300 PEC audit."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from audit_p300 import (
    ArrayP300Adapter,
    AuditInputError,
    ClosureConfig,
    CrossCovarianceEraser,
    EraseConfig,
    FeatureTable,
    LayerIntervention,
    LayerProbeAuditor,
    MetricSpec,
    P300AuditData,
    P300ClosureAuditor,
    P300EraseAuditor,
    P300FeatureLexicon,
    P300PECAuditor,
    P300Split,
    ProbeConfig,
    TorchLayerBinding,
    TorchP300Adapter,
    digit_hit,
)


def make_epochs_data(
    *, n_subjects: int = 6, trials_per_digit: int = 2, seed: int = 7
) -> P300AuditData:
    """Create subject-grouped, stimulus-locked epochs with a P300 target effect."""

    rng = np.random.default_rng(seed)
    time_ms = np.arange(-200.0, 801.0, 10.0)
    channels = ("Fz", "Cz", "Pz", "Oz")
    p300 = np.exp(-0.5 * ((time_ms - 450.0) / 55.0) ** 2)
    epochs: list[np.ndarray] = []
    targets: list[int] = []
    subjects: list[int] = []
    digits: list[int] = []
    thoughts: list[int] = []
    for subject in range(n_subjects):
        thought = subject % 9 + 1
        for digit in range(1, 10):
            for _ in range(trials_per_digit):
                target = int(digit == thought)
                trial = 0.02 * rng.standard_normal((len(channels), time_ms.size))
                trial[2] += (1.0 if target else 0.08) * p300
                trial[1] += (0.35 if target else 0.02) * p300
                epochs.append(trial)
                targets.append(target)
                subjects.append(subject)
                digits.append(digit)
                thoughts.append(thought)
    return P300AuditData(
        X=np.asarray(epochs),
        target=np.asarray(targets),
        subjects=np.asarray(subjects),
        time_ms=time_ms,
        digits=np.asarray(digits),
        thought_numbers=np.asarray(thoughts),
        channel_names=channels,
    )


def subject_split(data: P300AuditData) -> P300Split:
    return P300Split(
        train=np.flatnonzero(np.isin(data.subjects, [0, 1, 2])),
        validation=np.flatnonzero(data.subjects == 3),
        test=np.flatnonzero(np.isin(data.subjects, [4, 5])),
    )


def test_audit_data_rejects_nonfinite_input() -> None:
    epochs = np.zeros((4, 2, 8), dtype=float)
    epochs[0, 0, 0] = np.nan
    with pytest.raises(AuditInputError, match="NaN"):
        P300AuditData(
            X=epochs,
            target=np.array([0, 1, 0, 1]),
            subjects=np.arange(4),
            time_ms=np.arange(8),
        )


def test_split_rejects_overlap_and_subject_leakage() -> None:
    with pytest.raises(AuditInputError, match="overlap"):
        P300Split(train=np.array([0, 1]), validation=np.array([1, 2]), test=np.array([3]))

    data = P300AuditData(
        X=np.zeros((6, 2, 8)),
        target=np.array([0, 1, 0, 1, 0, 1]),
        subjects=np.array([0, 0, 1, 1, 2, 2]),
        time_ms=np.arange(8),
    )
    leaking = P300Split(train=np.array([0, 2]), validation=np.array([1, 3]), test=np.array([4, 5]))
    with pytest.raises(AuditInputError, match="subjects overlap"):
        leaking.validate_against(data)

    same_subject = P300Split(
        train=np.array([0, 1]),
        validation=np.array([2, 3]),
        test=np.array([4, 5]),
        require_subject_disjoint=False,
    )
    same_subject.validate_against(data)


def test_p300_feature_lexicon_has_63_finite_features() -> None:
    data = make_epochs_data(n_subjects=1)
    table = P300FeatureLexicon().extract(data)
    assert table.values.shape == (18, 63)
    assert len(table.names) == len(table.families) == 63
    assert set(table.families) == {
        "time",
        "frequency",
        "time_frequency",
        "complexity",
        "cross_frequency",
        "cross_channel",
    }
    assert np.all(np.isfinite(table.values))


def test_cross_covariance_eraser_removes_known_linear_direction() -> None:
    rng = np.random.default_rng(3)
    activation = rng.standard_normal((128, 5))
    target = 3.0 * activation[:, 0] + 0.01 * rng.standard_normal(128)
    eraser = CrossCovarianceEraser.fit(activation, target)
    residual = eraser.transform(activation)
    assert eraser.rank == 1
    assert abs(np.corrcoef(residual[:, 0], target)[0, 1]) < 0.1
    assert np.allclose(residual.mean(axis=0), activation.mean(axis=0), atol=1e-10)


def test_digit_hit_aggregates_subject_and_digit_trials() -> None:
    subjects = np.repeat([10, 11], 18)
    digits = np.tile(np.repeat(np.arange(1, 10), 2), 2)
    thoughts = np.repeat([4, 8], 18)
    score = np.where(digits == thoughts, 3.0, -1.0)
    assert digit_hit(score, digits, thoughts, subjects) == pytest.approx(1.0)
    with pytest.raises(AuditInputError, match="exactly 9"):
        digit_hit(score[:10], digits[:10], thoughts[:10], subjects[:10])


def test_array_adapter_intervention_does_not_mutate_cache() -> None:
    epochs = np.zeros((5, 2, 8))
    cached = {"layer": np.arange(25, dtype=float).reshape(5, 5)}
    adapter = ArrayP300Adapter(cached, lambda activations: activations["layer"][:, 0])
    before = adapter.collect_activations(epochs)["layer"]
    edited = adapter.predict_scores(
        epochs,
        intervention=LayerIntervention("layer", lambda value: np.zeros_like(value)),
    )
    after = adapter.collect_activations(epochs)["layer"]
    assert np.all(edited == 0.0)
    assert np.array_equal(before, cached["layer"])
    assert np.array_equal(after, cached["layer"])


class _TupleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear(value), value


class _TupleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _TupleBlock()
        self.head = nn.Linear(4, 1, bias=False)

    def forward(self, epochs: torch.Tensor) -> torch.Tensor:
        value = epochs.reshape(epochs.shape[0], 4)
        return self.head(self.block(value)[0])


def test_torch_adapter_supports_custom_read_write_binding() -> None:
    model = _TupleModel()
    adapter = TorchP300Adapter(
        model,
        {
            "tuple_layer": TorchLayerBinding(
                name="tuple_layer",
                module=model.block,
                read=lambda output: output[0],
                write=lambda output, tensor: (tensor, output[1]),
            )
        },
        score_fn=lambda output: output,
    )
    epochs = np.arange(20, dtype=float).reshape(5, 1, 4)
    activation = adapter.collect_activations(epochs)["tuple_layer"]
    original_score = adapter.predict_scores(epochs)
    edited_score = adapter.predict_scores(
        epochs,
        intervention=LayerIntervention("tuple_layer", lambda value: np.zeros_like(value)),
    )
    assert activation.shape == (5, 4)
    assert original_score.shape == edited_score.shape == (5,)
    assert not np.allclose(original_score, edited_score)


def test_runner_keeps_no_digit_closure_binary_only() -> None:
    full_data = make_epochs_data(n_subjects=3)
    data = P300AuditData(
        X=full_data.X,
        target=full_data.target,
        subjects=full_data.subjects,
        time_ms=full_data.time_ms,
        channel_names=full_data.channel_names,
    )
    split = P300Split(
        train=np.flatnonzero(data.subjects == 0),
        validation=np.flatnonzero(data.subjects == 1),
        test=np.flatnonzero(data.subjects == 2),
    )
    activation = np.zeros((data.n_trials, 4), dtype=float)
    adapter = ArrayP300Adapter({"layer": activation}, lambda values: values["layer"][:, 0])
    report = P300PECAuditor(
        probe_config=ProbeConfig(alphas=(1.0,), encoding_r2_threshold=10.0)
    ).run(adapter, data, split, model_name="runner_contract")
    assert report.metadata["digit_metric_available"] is False
    assert report.closures == ()
    assert report.primary_erase_metric == "binary_auc"


def test_probe_erase_closure_synthetic_chain() -> None:
    data = make_epochs_data(n_subjects=6, trials_per_digit=8, seed=11)
    split = subject_split(data)
    rng = np.random.default_rng(12)
    signal = data.target.astype(float) + 0.03 * rng.standard_normal(data.n_trials)
    noise = rng.standard_normal(data.n_trials)
    features = FeatureTable(
        values=np.column_stack((signal, noise)),
        names=("target_signal", "noise"),
        families=("time", "complexity"),
    )
    activations = {"readout": np.column_stack((signal, np.zeros((data.n_trials, 15))))}
    adapter = ArrayP300Adapter(
        activations,
        lambda values: values["readout"][:, 0],
    )

    probes = LayerProbeAuditor(
        ProbeConfig(alphas=(1.0,), encoding_r2_threshold=0.05, peak_margin=0.0)
    ).run(activations, features, split, data)
    assert probes[0].selection_encoded
    assert probes[0].test_encoded
    assert not probes[1].selection_encoded
    assert probes[0].second_best_test_r2 == float("-inf")

    erasures = P300EraseAuditor(EraseConfig(n_bootstrap=32, fdr_q=0.2, residual_r2_max=0.4)).run(
        adapter, data, split, features, activations, probes
    )
    target_erase = erasures[0]
    assert target_erase.representation_causal
    assert target_erase.real_drop is not None
    assert target_erase.erased_metric < target_erase.baseline_metric

    model_score = adapter.predict_scores(data.X)[split.test]
    closures = P300ClosureAuditor(ClosureConfig()).run(
        data,
        split,
        features,
        ("target_signal",),
        model_score,
        metrics=(MetricSpec("binary_auc"),),
    )
    assert closures[0].defined
    assert closures[0].transparent_metric > 0.9
    assert closures[0].closure_ratio > 0.8
