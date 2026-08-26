from __future__ import annotations

import pytest
import torch

from models.heads import MultiTaskHeads

D = 64


def _heads(**kwargs) -> MultiTaskHeads:
    torch.manual_seed(0)
    return MultiTaskHeads(**kwargs)


def _components(batch_size: int = 4) -> torch.Tensor:
    return torch.randn(batch_size, 3, D)


def test_heads_emit_pcw_constrained_logits_and_interpretable_outputs() -> None:
    output = _heads()(_components())
    assert output.logit_target.shape == output.logit_pcw.shape == (4, 1)
    assert output.logit_early.shape == output.amplitude.shape == (4, 1)
    assert torch.equal(output.logit_target, output.logit_pcw)
    assert torch.allclose(output.p_target, torch.sigmoid(output.logit_target))
    assert torch.isfinite(output.logit_target).all()


def test_heads_reject_non_pcw_component_shape() -> None:
    with pytest.raises(ValueError):
        _heads()(torch.randn(4, 5, D))


def test_component_views_remain_explicit() -> None:
    heads = _heads().eval()
    components = torch.zeros(2, 3, D)
    base = heads(components)

    n2 = components.clone()
    n2[:, 0] = 1.0
    changed_n2 = heads(n2)
    assert not torch.allclose(base.logit_target, changed_n2.logit_target)
    assert not torch.allclose(base.logit_early, changed_n2.logit_early)
    assert torch.allclose(base.amplitude, changed_n2.amplitude)

    p3a = components.clone()
    p3a[:, 1] = 1.0
    changed_p3a = heads(p3a)
    assert not torch.allclose(base.logit_target, changed_p3a.logit_target)
    assert torch.allclose(base.logit_early, changed_p3a.logit_early)
    assert torch.allclose(base.amplitude, changed_p3a.amplitude)

    p3b = components.clone()
    p3b[:, 2] = 1.0
    changed_p3b = heads(p3b)
    assert not torch.allclose(base.logit_target, changed_p3b.logit_target)
    assert torch.allclose(base.logit_early, changed_p3b.logit_early)
    assert not torch.allclose(base.amplitude, changed_p3b.amplitude)


def test_all_heads_backpropagate_to_pcw_components() -> None:
    components = _components(2).requires_grad_()
    output = _heads()(components)
    (output.logit_target.sum() + output.logit_early.sum() + output.amplitude.sum()).backward()
    assert components.grad is not None
    assert torch.isfinite(components.grad).all()


def test_z2_auxiliary_head_pools_and_emits_logit() -> None:
    from models.heads import Z2AuxiliaryHead

    for pool in ("global_pool", "maxmean", "attention"):
        torch.manual_seed(0)
        head = Z2AuxiliaryHead(d_model=16, pool=pool)
        z2 = torch.randn(3, 7, 16)
        logit = head(z2)
        assert logit.shape == (3, 1)
        assert torch.isfinite(logit).all()
        logit.sum().backward()


def test_z2_auxiliary_head_rejects_bad_contract() -> None:
    from models.heads import Z2AuxiliaryHead

    with pytest.raises(ValueError):
        Z2AuxiliaryHead(pool="flatten")
    with pytest.raises(ValueError):
        Z2AuxiliaryHead(d_model=0)
    head = Z2AuxiliaryHead(d_model=8, pool="global_pool")
    with pytest.raises(ValueError):
        head(torch.randn(2, 3, 7))
