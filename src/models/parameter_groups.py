"""Complete, disjoint parameter ownership declared by the model structure."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import torch
from torch import nn


def parameter_groups(
    model: nn.Module, modules: Mapping[str, Iterable[nn.Module]]
) -> dict[str, tuple[nn.Parameter, ...]]:
    """Resolve structural groups, refusing overlapping or unowned parameters.

    Module owners enumerate their actual modules; callers never infer ownership
    from parameter-name prefixes. Empty groups are permitted for architectures
    without a particular extension.
    """
    groups = {}
    owned: set[int] = set()
    for name, members in modules.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter groups require nonempty names")
        values = tuple(p for module in members for p in module.parameters())
        identities = {id(p) for p in values}
        if len(identities) != len(values) or owned & identities:
            raise ValueError(f"parameter ownership overlaps in group {name!r}")
        owned.update(identities)
        groups[name] = values
    if owned != {id(p) for p in model.parameters()}:
        raise ValueError("parameter groups must own every model parameter exactly once")
    return groups


class SourceParameterPrior:
    """A fixed source-centered metric, averaged by tensor and structural role.

    Weight scales use source RMS squared plus a declared fan-in reference
    variance. Bias, normalization affine and gates use unit scales. This is a
    parameter-coordinate prior; it does not infer initialization history or
    claim invariance to equivalent network parameterizations.
    """

    def __init__(self, model: nn.Module) -> None:
        ownership = {
            id(parameter): role
            for role, parameters in model.adaptation_parameter_groups().items()
            for parameter in parameters
        }
        self.groups: dict[str, list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]]] = {}
        self.scales: dict[str, float] = {}
        digest = hashlib.sha256()
        for module_name, module in model.named_modules():
            for local_name, parameter in module.named_parameters(recurse=False):
                name = f"{module_name}.{local_name}" if module_name else local_name
                reference = parameter.detach().clone()
                ordinary_weight = local_name == "weight" and isinstance(
                    module, (nn.Linear, nn.Conv1d, nn.Conv2d)
                )
                if ordinary_weight:
                    fan_in = parameter[0].numel()
                    scale = reference.float().square().mean() + 1.0 / (3.0 * fan_in)
                    kind = "weight"
                else:
                    scale = reference.new_ones((), dtype=torch.float32)
                    kind = local_name
                role = f"{ownership[id(parameter)]}.{kind}"
                self.groups.setdefault(role, []).append((parameter, reference, scale))
                self.scales[name] = float(scale.cpu())
                digest.update(name.encode())
                digest.update(str(tuple(reference.shape)).encode())
                digest.update(reference.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.center_sha256 = digest.hexdigest()
        # The metric is frozen. Packing it once avoids dozens of tiny GPU
        # reductions for every update while retaining each tensor's role weight.
        self._parameters = []
        centers, coefficients = [], []
        for members in self.groups.values():
            for parameter, reference, scale in members:
                self._parameters.append(parameter)
                centers.append(reference.float().reshape(-1))
                coefficients.append((1.0 / (2.0 * len(members) * parameter.numel() * scale))
                                    .expand(parameter.numel()))
        self._center = torch.cat(centers)
        self._coefficient = torch.cat(coefficients)

    def loss(self) -> torch.Tensor:
        values = torch.cat([parameter.float().reshape(-1) for parameter in self._parameters])
        return ((values - self._center).square() * self._coefficient).sum()

    def record(self) -> dict[str, object]:
        return {
            "schema": "n2p3_source_parameter_prior/1",
            "center_sha256": self.center_sha256,
            "scale_policy": "source_weight_rms2_plus_fan_in_variance;other_parameters_unit",
            "reduction": "mean_per_tensor_then_mean_per_structural_role_then_sum",
            "scale_squared": dict(self.scales),
            "roles": {name: len(values) for name, values in self.groups.items()},
        }
