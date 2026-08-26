"""Joint Neural-RIDE model for heterogeneous, variable-size EEG montages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.n2p3net import N2P3Net, N2P3NetOutput

MULTIMONTAGE_CHECKPOINT_SCHEMA = "n2p3net_multimontage_checkpoint/1"


@dataclass(frozen=True)
class MontageBranchSpec:
    channel_names: tuple[str, ...]
    channel_positions_m: tuple[tuple[float, float, float], ...] | None = None
    coordinate_registration: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Every montage branch requires unique channel names.")
        if self.channel_positions_m is not None and len(self.channel_positions_m) != len(
            self.channel_names
        ):
            raise ValueError("Montage branch positions must align with channel_names.")


class MultiMontageN2P3Net(nn.Module):
    """Dataset-specific sensor paths around one shared canonical task backbone.

    Batches remain montage-homogeneous because their channel dimensions differ.
    Training code should alternate batches and pass the corresponding branch
    name. Parameters of the temporal bank, canonical spatial mixer, sequence
    encoder, PCW, shared/private split, and classifier are the same Python
    objects in every branch; references and sensor-space decoders stay private.
    """

    _SHARED_MODEL_MODULES = (
        "encoder",
        "dataset_adapter",
        "shared_private_encoder",
        "component_window",
        "heads",
        "repetition_evidence",
    )
    _SHARED_TOKENIZER_MODULES = (
        "chn_proj",
        "sub_proj",
        "temporal_convs",
        "post_bns",
        "pointwise",
        "spatial_priors",
        "coord_mods",
        "uncertainty_proj",
    )

    def __init__(
        self,
        montages: Mapping[str, MontageBranchSpec],
        *,
        canonical_channel_names: Sequence[str],
        model_kwargs: Mapping[str, object],
    ) -> None:
        super().__init__()
        if len(montages) < 2:
            raise ValueError("MultiMontageN2P3Net requires at least two dataset montages.")
        if not canonical_channel_names:
            raise ValueError("A non-empty canonical sensor set is required.")
        kwargs = dict(model_kwargs)
        if "n_channels" in kwargs or "channel_names" in kwargs:
            raise ValueError(
                "n_channels/channel_names belong to MontageBranchSpec, not model_kwargs."
            )
        configured_domains = kwargs.get("n_domains")
        if configured_domains is None:
            kwargs["n_domains"] = len(montages)
        elif int(configured_domains) != len(montages):
            raise ValueError("n_domains must equal the number of montage branches.")
        kwargs["canonical_channel_names"] = tuple(canonical_channel_names)

        self.domain_names = tuple(montages)
        self.domain_index = {name: index for index, name in enumerate(self.domain_names)}
        self.montage_specs = dict(montages)
        self.canonical_channel_names = tuple(str(name) for name in canonical_channel_names)
        branches: dict[str, N2P3Net] = {}
        for name, spec in montages.items():
            branch_kwargs = dict(kwargs)
            branch_kwargs.update(
                n_channels=len(spec.channel_names),
                channel_names=spec.channel_names,
                channel_positions_m=spec.channel_positions_m,
            )
            branches[name] = N2P3Net(**branch_kwargs)
        self.branches = nn.ModuleDict(branches)
        self._tie_shared_backbone()

    def _tie_shared_backbone(self) -> None:
        anchor = self.branches[self.domain_names[0]]
        for name in self.domain_names[1:]:
            branch = self.branches[name]
            for attribute in self._SHARED_MODEL_MODULES:
                setattr(branch, attribute, getattr(anchor, attribute))
            for attribute in self._SHARED_TOKENIZER_MODULES:
                setattr(branch.tokenizer, attribute, getattr(anchor.tokenizer, attribute))

    def branch(self, domain: str) -> N2P3Net:
        if domain not in self.branches:
            raise KeyError(f"Unknown domain {domain!r}; available={self.domain_names}.")
        return self.branches[domain]

    def forward(
        self,
        domain: str,
        X: torch.Tensor,
        E_chn: torch.Tensor | None = None,
        E_sub: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> N2P3NetOutput:
        model = self.branch(domain)
        domain_id = torch.full(
            (X.shape[0],),
            self.domain_index[domain],
            device=X.device,
            dtype=torch.long,
        )
        return model(
            X,
            E_chn=E_chn,
            E_sub=E_sub,
            channel_mask=channel_mask,
            domain_id=domain_id,
            **kwargs,
        )

    def checkpoint_contract(self) -> dict[str, Any]:
        """Return all non-tensor semantics required to interpret model rows."""

        montage_contract = {
            name: {
                "channel_names": list(spec.channel_names),
                "channel_positions_m": (
                    [list(row) for row in spec.channel_positions_m]
                    if spec.channel_positions_m is not None
                    else None
                ),
                "coordinate_registration": (
                    dict(spec.coordinate_registration)
                    if spec.coordinate_registration is not None
                    else None
                ),
            }
            for name, spec in self.montage_specs.items()
        }
        kernel_contract = {
            name: self.branch(name).tokenizer.canonical_projector.kernel_spec()
            for name in self.domain_names
        }
        return {
            "domain_vocabulary": list(self.domain_names),
            "canonical_channel_names": list(self.canonical_channel_names),
            "montages": montage_contract,
            "canonical_kernels": kernel_contract,
        }


def save_multimontage_checkpoint(
    path: str | Path,
    model: MultiMontageN2P3Net,
) -> Path:
    """Save weights together with the exact domain and coordinate vocabulary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": MULTIMONTAGE_CHECKPOINT_SCHEMA,
            "model_state_dict": model.state_dict(),
            "model_contract": model.checkpoint_contract(),
        },
        destination,
    )
    return destination


def load_multimontage_checkpoint(
    path: str | Path,
    model: MultiMontageN2P3Net,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only when every row-indexed model semantic exactly matches."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != MULTIMONTAGE_CHECKPOINT_SCHEMA:
        raise ValueError(
            "Checkpoint is not a versioned MultiMontage checkpoint; raw state_dict loading "
            "cannot verify the domain vocabulary."
        )
    saved_contract = payload.get("model_contract")
    expected_contract = model.checkpoint_contract()
    if not isinstance(saved_contract, dict):
        raise ValueError("MultiMontage checkpoint lacks model_contract metadata.")
    if saved_contract.get("domain_vocabulary") != expected_contract["domain_vocabulary"]:
        raise ValueError(
            "Checkpoint domain vocabulary/order does not match the model: "
            f"saved={saved_contract.get('domain_vocabulary')}, "
            f"expected={expected_contract['domain_vocabulary']}."
        )
    for field in ("canonical_channel_names", "montages", "canonical_kernels"):
        if saved_contract.get(field) != expected_contract[field]:
            raise ValueError(f"Checkpoint model contract mismatch in {field}.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("MultiMontage checkpoint lacks model_state_dict.")
    model.load_state_dict(state_dict)
    return payload
