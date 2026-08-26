from __future__ import annotations

import pytest
import torch

from data.channel import STANDARD_CHANNELS
from models.multidataset import (
    MULTIMONTAGE_CHECKPOINT_SCHEMA,
    MontageBranchSpec,
    MultiMontageN2P3Net,
    load_multimontage_checkpoint,
    save_multimontage_checkpoint,
)


def _model(
    order: tuple[str, ...] = ("gtn", "dry8"),
    *,
    canonical_noise_variance: float = 0.05,
) -> MultiMontageN2P3Net:
    specs = {
        "gtn": MontageBranchSpec(
            ("Fz", "Cz", "Pz"),
            coordinate_registration={"source": "device_montage", "units": "m"},
        ),
        "dry8": MontageBranchSpec(
            STANDARD_CHANNELS,
            coordinate_registration={"source": "average_head_template", "units": "m"},
        ),
    }
    return MultiMontageN2P3Net(
        {name: specs[name] for name in order},
        canonical_channel_names=STANDARD_CHANNELS,
        model_kwargs={
            "n_time": 256,
            "d_model": 16,
            "temporal_kernels": (13,),
            "filters_per_scale": 2,
            "encoder_depth": 1,
            "component_decoder": False,
            "dataset_adapter_rank": 4,
            "shared_private": True,
            "private_dim": 8,
            "canonical_noise_variance": canonical_noise_variance,
        },
    )


def test_multimontage_accepts_different_channel_counts_with_shared_backbone() -> None:
    model = _model()
    gtn = model("gtn", torch.randn(2, 3, 256))
    dry = model("dry8", torch.randn(2, 8, 256))

    assert gtn.heads.logit_target.shape == dry.heads.logit_target.shape == (2, 1)
    assert gtn.canonical_covariance.shape == dry.canonical_covariance.shape == (2, 8, 8)
    assert model.branch("gtn").encoder is model.branch("dry8").encoder
    assert (
        model.branch("gtn").tokenizer.temporal_convs
        is model.branch("dry8").tokenizer.temporal_convs
    )
    assert model.branch("gtn").reference is not model.branch("dry8").reference


def test_multimontage_keeps_gtn_uncertainty_higher_than_full_canonical_layout() -> None:
    model = _model()
    gtn = model("gtn", torch.zeros(1, 3, 256))
    dry = model("dry8", torch.zeros(1, 8, 256))
    gtn_variance = torch.diagonal(gtn.canonical_covariance, dim1=-2, dim2=-1)
    dry_variance = torch.diagonal(dry.canonical_covariance, dim1=-2, dim2=-1)
    assert gtn_variance.mean() > 5.0 * dry_variance.mean()


def test_multimontage_shared_parameters_receive_gradients_from_each_domain() -> None:
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for domain, channels in (("gtn", 3), ("dry8", 8)):
        optimizer.zero_grad(set_to_none=True)
        output = model(domain, torch.randn(2, channels, 256))
        output.heads.logit_target.sum().backward()
        gradient = model.branch("gtn").tokenizer.temporal_convs[0].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()


def test_multimontage_checkpoint_binds_domain_and_coordinate_vocabulary(tmp_path) -> None:
    model = _model()
    path = save_multimontage_checkpoint(tmp_path / "model.pt", model)
    restored = _model()
    payload = load_multimontage_checkpoint(path, restored)
    assert payload["schema"] == MULTIMONTAGE_CHECKPOINT_SCHEMA
    assert payload["model_contract"]["domain_vocabulary"] == ["gtn", "dry8"]
    assert payload["model_contract"]["canonical_kernels"]["gtn"]["coordinate_units"] == "m"
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])

    with pytest.raises(ValueError, match="domain vocabulary/order"):
        load_multimontage_checkpoint(path, _model(("dry8", "gtn")))

    with pytest.raises(ValueError, match="canonical_kernels"):
        load_multimontage_checkpoint(path, _model(canonical_noise_variance=0.1))


def test_multimontage_loader_rejects_unversioned_state_dict(tmp_path) -> None:
    path = tmp_path / "raw.pt"
    torch.save(_model().state_dict(), path)
    with pytest.raises(ValueError, match="raw state_dict"):
        load_multimontage_checkpoint(path, _model())
