"""GLM 协议测试（2026-08-22）：被试级验证早停 + TCN BN 消融 + σ 界透传。

语义：
    - _split_val_subjects：train/val 无被试交集、确定性（同 seed 同划分）、
      clamp 到 [min, max]、被试数 <4 时 fail-closed。
    - fit(subject_ids=...) 端到端：val 曲线存在、早停后 best_state 恢复。
    - evaluate() 只对声明 fit_accepts_subject_ids 的模型传 subject_ids。
    - encoder norm="bn"：可构造、前向形状不变、反向传播非 NaN。
    - N2P3Net sigma_bounds 透传到 ComponentWindow。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.n2p3net import N2P3Net

C = 3
D = 64
T = 256


def make_adapter(**kw):
    from baselines.n2p3net import N2P3NetBaseline

    torch.manual_seed(0)
    trainer_kwargs = dict(epochs=2, batch_size=8, augment=False, lambda_jit=0.0, seed=0)
    trainer_kwargs.update(kw.pop("trainer_kwargs", {}))
    return N2P3NetBaseline(
        model_kwargs=dict(n_channels=3, channel_names=("Fz", "Cz", "Pz"), encoder_depth=1),
        trainer_kwargs=trainer_kwargs,
        device=torch.device("cpu"),
        **kw,
    )


def make_subject_data(n_subjects=8, trials_per_subject=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_subjects * trials_per_subject, C, T)).astype("float32")
    y = rng.integers(0, 2, n_subjects * trials_per_subject).astype("int64")
    s = np.repeat(np.arange(n_subjects), trials_per_subject)
    return X, y, s


def test_adapter_rejects_float_labels_before_training() -> None:
    adapter = make_adapter()
    X = np.zeros((4, C, T), dtype=np.float32)

    with pytest.raises(ValueError, match="integer labels"):
        adapter.fit(X, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))


def test_adapter_rejects_integer_eeg_before_silent_cast() -> None:
    adapter = make_adapter()
    with pytest.raises(ValueError, match="floating dtype"):
        adapter.fit(np.zeros((4, C, T), dtype=np.int16), np.array([0, 1, 0, 1]))


def test_repetition_prediction_forwards_mask_and_rejects_fractional_order() -> None:
    adapter = make_adapter()
    seen: dict[str, np.ndarray] = {}

    class Evidence:
        @staticmethod
        def candidate_log_scores(evidence, quality, digits, *, digit_vocab):
            return torch.tensor([0.0, 1.0]), torch.ones(len(evidence))

    adapter._fitted = True
    adapter.repetition_fitted_ = True
    adapter.repetition_ready_ = True
    adapter.model_ = SimpleNamespace(repetition_evidence=Evidence())

    def repetition_inputs(X, trial_channel_mask=None):
        seen["mask"] = np.asarray(trial_channel_mask).copy()
        return None, torch.zeros(len(X)), torch.zeros(len(X), 1)

    adapter._predict_repetition_inputs = repetition_inputs
    X = np.zeros((2, C, T), dtype=np.float32)
    mask = np.ones((2, C), dtype=bool)
    mask[0, 2] = False
    result = adapter.predict_repetition_candidates(
        X,
        np.array([1, 2]),
        np.array(["s", "s"]),
        digit_vocab=(1, 2),
        evidence_budgets=(1,),
        trial_channel_mask=mask,
    )

    assert np.array_equal(seen["mask"], mask)
    assert result["prefix_minK_chain_llr@1"]["predicted"].tolist() == [2]
    with pytest.raises(ValueError, match="integer dtype"):
        adapter.predict_repetition_candidates(
            X,
            np.array([1, 2]),
            np.array(["s", "s"]),
            digit_vocab=(1, 2),
            evidence_budgets=(1,),
            acquisition_indices=np.array([0.1, 1.0]),
            trial_channel_mask=mask,
        )


def test_fold_runtime_release_drops_previous_model_reference():
    adapter = make_adapter()
    adapter.model_ = torch.nn.Linear(4, 4)
    adapter._runtime_E_chn = torch.ones(2)
    adapter._runtime_channel_mask = torch.ones(3, dtype=torch.bool)
    adapter.generative_profile_ = object()
    adapter.last_history = {"train_losses": [1.0]}

    adapter._release_fold_runtime()

    assert adapter.model_ is None
    assert adapter._runtime_E_chn is None
    assert adapter._runtime_channel_mask is None
    assert adapter.generative_profile_ is None
    assert adapter.last_history is None


def test_epoch_trajectory_prediction_restores_selected_model_state():
    adapter = make_adapter(trainer_kwargs={"epoch_trajectory_audit": True})
    adapter.model_ = torch.nn.Linear(1, 1, bias=False)
    adapter._fitted = True
    with torch.no_grad():
        adapter.model_.weight.fill_(7.0)
    adapter.epoch_trajectory_checkpoints_ = [
        {
            "schema": "n2p3net_epoch_trajectory_checkpoint/1",
            "development_only": True,
            "event": {"epoch": epoch, "task_val_loss": float(3 - epoch)},
            "state_dict": {"weight": torch.tensor([[float(epoch)]])},
        }
        for epoch in (1, 2)
    ]
    adapter._predict_model_logit = lambda X: np.full(
        len(X), float(adapter.model_.weight.detach().item())
    )

    before = adapter._predict_model_logit(np.zeros((3, 1, 1), dtype=np.float32))
    trajectory = adapter.predict_epoch_trajectory_logits(
        np.zeros((3, 1, 1), dtype=np.float32)
    )
    after = adapter._predict_model_logit(np.zeros((3, 1, 1), dtype=np.float32))

    assert [row["epoch"] for row in trajectory] == [1, 2]
    assert [row["logits"][0] for row in trajectory] == [1.0, 2.0]
    assert np.array_equal(before, after)
    assert adapter.model_.weight.detach().item() == 7.0


def test_epoch_trajectory_checkpoint_sink_persists_atomic_fold_files(
    tmp_path, monkeypatch
):
    from baselines.n2p3net import _make_epoch_checkpoint_sink

    monkeypatch.setenv("N2P3NET_EPOCH_PROGRESS_DIR", str(tmp_path))
    sink, references = _make_epoch_checkpoint_sink(enabled=True, fold_id=4)
    assert sink is not None
    sink({"epoch": 2, "phase": "joint"}, {"weight": torch.tensor([2.0])})

    expected = tmp_path / "checkpoints" / "fold_4" / "epoch_002.pt"
    assert references == [expected]
    assert expected.is_file()
    assert not expected.with_suffix(".pt.tmp").exists()
    payload = torch.load(expected, map_location="cpu", weights_only=True)
    assert payload["development_only"] is True
    assert payload["event"]["epoch"] == 2
    assert payload["state_dict"]["weight"].item() == 2.0


# ---------------- _split_val_subjects 语义 ----------------


def test_val_split_no_subject_overlap_and_deterministic():
    adapter = make_adapter(val_subject_frac=0.25, val_subjects_min=2, val_subjects_max=4)
    X, y, s = make_subject_data(n_subjects=8)
    X1, y1, Xv, yv, k = adapter._split_val_subjects(X, y, s)
    assert k == 2  # clamp(0.25*8=2, min=2, max=4)
    assert len(X1) + len(Xv) == len(X)
    assert len(y1) + len(yv) == len(y)
    # 验证侧行数应恰为整数个被试的试次数（6/被试），即切分按被试整体、无被试跨界
    assert len(Xv) % 6 == 0 and len(Xv) == 2 * 6
    # 同 seed 复跑应得到完全一致的划分
    X1b, y1b, Xvb, yvb, kb = adapter._split_val_subjects(X, y, s)
    assert kb == k
    assert np.array_equal(X1, X1b) and np.array_equal(Xv, Xvb)
    # 被试级切分：用相同 rng 逻辑复现 val 被试集合，验证整组被试试次进 val（无跨集泄漏）
    cfg_seed = int(adapter.trainer_kwargs.get("seed", 0))
    rng = np.random.default_rng(cfg_seed)
    val_subj = set(rng.choice(np.unique(s), size=k, replace=False).tolist())
    val_mask = np.isin(s, list(val_subj))
    assert len(Xv) == int(val_mask.sum())
    assert np.array_equal(Xv, X[val_mask])
    assert np.array_equal(X1, X[~val_mask])


def test_val_split_clamps_to_bounds():
    adapter = make_adapter(val_subject_frac=0.5, val_subjects_min=2, val_subjects_max=3)
    X, y, s = make_subject_data(n_subjects=20)
    _, _, _, _, k = adapter._split_val_subjects(X, y, s)
    assert k == 3  # 0.5*20=10 → clamp 到 max=3


def test_val_split_disabled_or_too_few_subjects():
    X, y, s = make_subject_data(n_subjects=3)
    adapter = make_adapter(val_subject_frac=0.5)
    with pytest.raises(ValueError, match="at least four"):
        adapter._split_val_subjects(X, y, s)
    adapter_off = make_adapter(val_subject_frac=None)
    X1, y1, Xv, yv, k = adapter_off._split_val_subjects(X, y, s)
    assert k == 0 and len(Xv) == 0


def test_supervised_reconstruction_profile_excludes_validation_subjects():
    """ERP targets and identifiability gates must not consume validation labels."""

    from types import MethodType

    data, labels, subjects = make_subject_data(n_subjects=8, trials_per_subject=6)
    adapter = make_adapter(val_subject_frac=0.25, val_subjects_min=2, val_subjects_max=2)
    split = adapter._subject_validation_split(subjects)
    captured = {}

    def capture(self, train_data, train_labels, *args, **kwargs):
        captured["X_train"] = train_data
        captured["y_train"] = train_labels
        captured["reconstruction_X"] = kwargs["reconstruction_X"]
        captured["reconstruction_y"] = kwargs["reconstruction_y"]
        captured["audit_X"] = kwargs["audit_X"]
        return self

    adapter._fit_common = MethodType(capture, adapter)
    adapter.fit(data, labels, subject_ids=subjects)

    assert len(captured["X_train"]) < int(split.train_mask.sum())
    assert np.array_equal(captured["X_train"], captured["reconstruction_X"])
    assert np.array_equal(captured["y_train"], captured["reconstruction_y"])
    assert len(captured["audit_X"]) > 0
    train_markers = set(captured["X_train"][:, 0, 0].tolist())
    audit_markers = set(captured["audit_X"][:, 0, 0].tolist())
    validation_markers = set(data[split.validation_mask, 0, 0].tolist())
    assert train_markers.isdisjoint(audit_markers)
    assert train_markers.isdisjoint(validation_markers)
    assert audit_markers.isdisjoint(validation_markers)


def test_prequential_fusion_requires_subject_crossfit_evidence():
    adapter = make_adapter()
    adapter.prequential_variant_ = "m2_diag"
    subjects = np.repeat(np.arange(5), 20)
    labels = np.tile(np.repeat([0, 1], 10), 5).astype(np.float32)
    evidence = (2.0 * labels - 1.0).astype(np.float32)
    X = evidence[:, None, None]
    adapter._prequential_llr = lambda data, variant: data[:, 0, 0].astype(np.float64)
    adapter._predict_model_logit = lambda data: np.zeros(len(data), dtype=np.float32)

    report = adapter._fit_prequential_fusion(X, labels, subjects)

    assert report["crossfit_passed"] is True
    assert report["crossfit_subject_win_fraction"] == 1.0
    assert report["crossfit_fused_nll"] < report["crossfit_base_nll"]
    assert report["coefficient"] > 0.0
    assert report["llr_temperature"] > 0.0
    assert "llr_center" not in report


def test_prequential_fusion_fails_closed_with_too_few_subjects():
    adapter = make_adapter()
    adapter.prequential_variant_ = "m2_diag"
    subjects = np.repeat(np.arange(3), 20)
    labels = np.tile(np.repeat([0, 1], 10), 3).astype(np.float32)
    evidence = (2.0 * labels - 1.0).astype(np.float32)
    X = evidence[:, None, None]
    adapter._prequential_llr = lambda data, variant: data[:, 0, 0].astype(np.float64)
    adapter._predict_model_logit = lambda data: np.zeros(len(data), dtype=np.float32)

    report = adapter._fit_prequential_fusion(X, labels, subjects)

    assert report["crossfit_passed"] is False
    assert report["coefficient"] == 0.0
    assert report["failure"] == "needs_at_least_four_validation_subjects"


def test_prequential_fusion_fails_closed_for_single_class_subject() -> None:
    adapter = make_adapter()
    adapter.prequential_variant_ = "m2_diag"
    subjects = np.repeat(np.arange(5), 20)
    labels = np.tile(np.repeat([0, 1], 10), 5).astype(np.float32)
    labels[subjects == 4] = 0.0
    evidence = (2.0 * labels - 1.0).astype(np.float32)
    X = evidence[:, None, None]
    adapter._prequential_llr = lambda data, variant: data[:, 0, 0].astype(np.float64)
    adapter._predict_model_logit = lambda data: np.zeros(len(data), dtype=np.float32)

    report = adapter._fit_prequential_fusion(X, labels, subjects)

    assert report["crossfit_passed"] is False
    assert report["coefficient"] == 0.0
    assert report["failure"] == "every_validation_subject_needs_both_target_classes"


def test_single_class_subject_stays_in_optimization_not_audit() -> None:
    from baselines.validation import subject_disjoint_audit_split

    subjects = np.repeat(np.arange(8), 4)
    eligible = np.ones(len(subjects), dtype=bool)
    candidate = subjects != 0
    split = subject_disjoint_audit_split(
        subjects,
        eligible_mask=eligible,
        candidate_mask=candidate,
        n_subjects=2,
        seed=3,
    )

    assert split.optimization_mask[subjects == 0].all()
    assert not split.audit_mask[subjects == 0].any()


def test_fusion_selects_complementary_candidate_from_audit_eligible_set():
    adapter = make_adapter()
    subjects = np.repeat(np.arange(5), 20)
    labels = np.tile(np.repeat([0, 1], 10), 5).astype(np.float32)
    evidence = (2.0 * labels - 1.0).astype(np.float32)
    X = evidence[:, None, None]

    def candidate_llr(data, variant):
        sign = 1.0 if variant == "m2_low_rank" else -1.0
        return sign * data[:, 0, 0].astype(np.float64)

    adapter._prequential_llr = candidate_llr
    adapter._predict_model_logit = lambda data: np.zeros(len(data), dtype=np.float32)
    report = adapter._select_prequential_fusion(
        X,
        labels,
        subjects,
        eligible_variants=("m2_low_rank", "m2_diag"),
        density_selected_variant="m2_diag",
    )

    assert report["density_selected_variant"] == "m2_diag"
    assert report["variant"] == "m2_low_rank"
    assert report["crossfit_passed"] is True
    assert report["coefficient"] > 0.0
    assert set(report["candidate_reports"]) == {"m2_low_rank", "m2_diag"}


def test_fit_with_subject_ids_runs_val_early_stop():
    """端到端：fit(subject_ids=...) 产生 val 曲线且 last_val_subjects 被记录。"""
    X, y, s = make_subject_data(n_subjects=8, trials_per_subject=8)
    adapter = make_adapter(
        val_subject_frac=0.25,
        trainer_kwargs=dict(epochs=3, early_stop_patience=2),
    )
    adapter.fit(X, y, subject_ids=s)
    assert adapter.last_val_subjects == 2
    assert len(adapter.last_history["val_losses"]) >= 1
    logits = adapter.predict_logit(X[:4])
    assert logits.shape == (4,)
    assert np.isfinite(logits).all()


def test_fit_without_subject_ids_keeps_old_path():
    """不传 subject_ids（旧调用方）：无验证、行为与旧版一致。"""
    X, y, s = make_subject_data(n_subjects=8, trials_per_subject=8)
    adapter = make_adapter()
    adapter.fit(X, y)
    assert adapter.last_val_subjects is None
    assert adapter.last_history["val_losses"] == []


def test_evaluate_passes_subject_ids_only_to_supporting_models():
    """evaluate()：声明 fit_accepts_subject_ids 的模型收到 subject_ids，其余走旧契约。"""
    from baselines.evaluate import evaluate, loso_folds
    from tests.test_evaluate import make_event_timeline

    X, y, s = make_subject_data(n_subjects=6, trials_per_subject=9)
    y = (np.arange(len(y)) % 3 == 0).astype("int64")  # 两类均衡
    digits = np.tile(np.arange(1, 10), len(y) // 9 + 1)[: len(y)]
    true_digits = {int(sub): 1 for sub in np.unique(s)}

    calls = {}

    class SpyWithSubjects:
        fit_accepts_subject_ids = True

        def fit(self, X_, y_, subject_ids=None):
            calls["with_subjects"] = (len(X_), len(subject_ids))
            self.calibration_logits_ = np.asarray([-1.0, 1.0])
            self.calibration_labels_ = np.asarray([0, 1])
            self.calibration_source_ = "subject_disjoint_validation"
            return self

        def predict_logit(self, X_):
            return np.zeros(len(X_))

    class SpyPlain:
        def fit(self, X_, y_):
            calls["plain"] = (len(X_), len(y_))
            return self

        def predict_logit(self, X_):
            return np.zeros(len(X_))

    folds = loso_folds(s)
    evaluate(
        SpyWithSubjects(),
        X,
        y,
        digits,
        s,
        true_digits,
        folds,
        event_timeline=make_event_timeline(digits, s),
        n_jobs=1,
    )
    assert "with_subjects" in calls and calls["with_subjects"][0] == calls["with_subjects"][1]
    evaluate(
        SpyPlain(),
        X,
        y,
        digits,
        s,
        true_digits,
        folds,
        event_timeline=make_event_timeline(digits, s),
        n_jobs=1,
    )
    assert "plain" in calls


# ---------------- encoder BN 消融 + σ 界透传 ----------------


def test_encoder_norm_bn_forward_and_backward():
    torch.manual_seed(0)
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        encoder_type="tcn",
        encoder_depth=2,
        encoder_norm="bn",
    )
    X = torch.randn(4, 3, T)
    out = model(X)
    assert out.heads.logit_target.shape == (4, 1)
    assert not torch.isnan(out.heads.logit_target).any()
    # BN 生效校验：blocks 内是 BatchNorm1d
    assert isinstance(model.encoder.blocks[0].ln, torch.nn.BatchNorm1d)
    loss = out.heads.logit_target.sum()
    loss.backward()
    assert model.encoder.blocks[0].depthwise.weight.grad is not None
    assert not torch.isnan(model.encoder.blocks[0].depthwise.weight.grad).any()


def test_encoder_norm_invalid_raises():
    from models.encoder import Stage2Encoder

    try:
        Stage2Encoder(encoder_type="tcn", depth=1, norm="gn")
    except ValueError:
        return
    raise AssertionError("norm='gn' 应抛 ValueError")


def test_sigma_bounds_forwarded_to_component_window():
    model = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        sigma_bounds=((20.0, 50.0), (20.0, 80.0), (20.0, 150.0)),
    )
    assert model.component_window.sigma_hi[2].item() == 150.0
    assert model.component_window.sigma_lo[2].item() == 20.0


def test_glm_default_model_budget_within_80k():
    """Fail-closed default PCW model stays within the 80k ceiling."""
    torch.manual_seed(0)
    model = N2P3Net(n_channels=3, channel_names=("Fz", "Cz", "Pz"))
    assert model.num_parameters() <= 80000


def test_glm_mini_presets_budget():
    """Capacity presets scale both PCW and the independent likelihood path."""
    torch.manual_seed(0)
    mini = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        d_model=32,
        filters_per_scale=4,
        temporal_kernels=(33, 65),
        encoder_depth=1,
        component_decoder=False,
        use_innovation_likelihood=True,
        innovation_d_model=16,
    )
    mini_a = N2P3Net(
        n_channels=3,
        channel_names=("Fz", "Cz", "Pz"),
        d_model=16,
        filters_per_scale=2,
        temporal_kernels=(65,),
        encoder_depth=0,
        component_decoder=False,
        use_innovation_likelihood=True,
        innovation_d_model=8,
    )
    # The strict-past likelihood encoder remains parameter-disjoint from PCW.
    assert mini.num_parameters() <= 16000
    assert mini_a.num_parameters() <= 6000
    assert mini_a.num_parameters() < mini.num_parameters()
    X = torch.randn(2, 3, T)
    assert mini(X).heads.logit_target.shape == (2, 1)
    assert mini_a(X).heads.logit_target.shape == (2, 1)


def test_glm_noref_forward():
    """GLM 前端修复：use_rereference=False 时 reference 层不存在（鼻参考数据默认）。"""
    torch.manual_seed(0)
    model = N2P3Net(n_channels=3, channel_names=("Fz", "Cz", "Pz"), use_rereference=False)
    assert model.reference is None
    X = torch.randn(2, 3, T)
    out = model(X)
    assert not torch.isnan(out.heads.logit_target).any()
