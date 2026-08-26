"""Explicit model adapters for P300 Probe and Erase interventions.

The core audit never guesses how a model exposes an intermediate activation.
An adapter must declare the layer binding, the score head, and how an edited
audit vector is lifted back to the native tensor.  This is deliberately more
verbose than a duck-typed ``model.forward`` helper: silent intervention at the
wrong tensor is worse than a clear integration error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import AuditInputError

ArrayTransform = Callable[[np.ndarray], np.ndarray]
TorchInputFn = Callable[[np.ndarray, np.ndarray, Any], Any]


@dataclass(frozen=True)
class LayerIntervention:
    """A fixed transformation applied to one declared audit-layer vector."""

    layer: str
    transform: ArrayTransform
    label: str = "custom"


class P300ModelAdapter(ABC):
    """Model-independent interface required by Probe and Erase."""

    @property
    @abstractmethod
    def layer_names(self) -> tuple[str, ...]:
        """Ordered layer names exposed to the audit."""

    @abstractmethod
    def collect_activations(
        self,
        epochs: np.ndarray,
        *,
        layers: Sequence[str] | None = None,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        """Return ``(N,D)`` audit activations for each requested layer."""

    @abstractmethod
    def predict_scores(
        self,
        epochs: np.ndarray,
        *,
        intervention: LayerIntervention | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Return one target-vs-non-target logit/score per trial."""

    def validate_intervention(self, intervention: LayerIntervention) -> None:
        if intervention.layer not in self.layer_names:
            raise AuditInputError(
                f"Unknown intervention layer {intervention.layer!r}; "
                f"available layers: {self.layer_names}."
            )


@dataclass(frozen=True)
class ArrayP300Adapter(P300ModelAdapter):
    """Deterministic adapter for cached activations and synthetic tests.

    ``score_from_activations`` is called with a copied activation mapping, so
    an intervention cannot mutate the cache shared by another feature.
    """

    activation_cache: Mapping[str, np.ndarray]
    score_from_activations: Callable[[Mapping[str, np.ndarray]], np.ndarray]

    def __post_init__(self) -> None:
        if not self.activation_cache:
            raise AuditInputError("activation_cache must contain at least one layer.")
        counts = set()
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.activation_cache.items():
            array = np.asarray(value, dtype=float)
            if array.ndim != 2:
                raise AuditInputError(
                    f"cached layer {name!r} must have shape (N,D), got {array.shape}."
                )
            if not np.all(np.isfinite(array)):
                raise AuditInputError(f"cached layer {name!r} contains NaN or infinite values.")
            counts.add(array.shape[0])
            normalized[str(name)] = array
        if len(counts) != 1:
            raise AuditInputError("all cached layers must have the same row count.")
        object.__setattr__(self, "activation_cache", normalized)

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.activation_cache.keys())

    def collect_activations(
        self,
        epochs: np.ndarray,
        *,
        layers: Sequence[str] | None = None,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        del batch_size
        array = np.asarray(epochs)
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
            raise AuditInputError("epochs must have numeric shape (N,C,T).")
        if not np.all(np.isfinite(array)):
            raise AuditInputError("epochs contains NaN or infinite values.")
        n = array.shape[0]
        if any(value.shape[0] != n for value in self.activation_cache.values()):
            raise AuditInputError("X row count does not match cached activation row count.")
        requested = self.layer_names if layers is None else tuple(layers)
        for layer in requested:
            self.validate_intervention(LayerIntervention(layer, lambda x: x))
        return {layer: self.activation_cache[layer].copy() for layer in requested}

    def predict_scores(
        self,
        epochs: np.ndarray,
        *,
        intervention: LayerIntervention | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        del batch_size
        array = np.asarray(epochs)
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
            raise AuditInputError("epochs must have numeric shape (N,C,T).")
        if not np.all(np.isfinite(array)):
            raise AuditInputError("epochs contains NaN or infinite values.")
        n = array.shape[0]
        if any(value.shape[0] != n for value in self.activation_cache.values()):
            raise AuditInputError("X row count does not match cached activation row count.")
        activations = {name: value.copy() for name, value in self.activation_cache.items()}
        if intervention is not None:
            self.validate_intervention(intervention)
            edited = np.asarray(
                intervention.transform(activations[intervention.layer].copy()), dtype=float
            )
            if edited.shape != activations[intervention.layer].shape:
                raise AuditInputError(
                    f"intervention {intervention.label!r} changed shape from "
                    f"{activations[intervention.layer].shape} to {edited.shape}."
                )
            if not np.all(np.isfinite(edited)):
                raise AuditInputError(
                    f"intervention {intervention.label!r} produced non-finite values."
                )
            activations[intervention.layer] = edited
        score = np.asarray(self.score_from_activations(activations), dtype=float).reshape(-1)
        if score.shape[0] != n or not np.all(np.isfinite(score)):
            raise AuditInputError("score_from_activations must return finite shape (N,) scores.")
        return score


@dataclass(frozen=True)
class TorchLayerBinding:
    """A declared PyTorch module output binding.

    ``read`` selects a tensor from a module output.  ``write`` puts an edited
    tensor back into that output; it is required for tuple/dataclass outputs.
    For rank > 2 tensors, ``reduce`` and ``lift`` default to mean-pooling and
    broadcasting the edit delta across non-feature axes.  Integrations should
    replace these defaults when that approximation is not scientifically
    justified.
    """

    name: str
    module: Any
    read: Callable[[Any], Any] = lambda output: output
    write: Callable[[Any, Any], Any] = lambda output, tensor: tensor
    reduce: Callable[[Any], Any] | None = None
    lift: Callable[[Any, Any, Any], Any] | None = None

    def audit_tensor(self, native: Any) -> Any:
        tensor = self.read(native)
        if not hasattr(tensor, "ndim") or tensor.ndim < 2:
            raise AuditInputError(f"layer {self.name!r} binding must expose a batched tensor.")
        if self.reduce is not None:
            audit_tensor = self.reduce(tensor)
            if not hasattr(audit_tensor, "ndim") or audit_tensor.ndim != 2:
                raise AuditInputError(
                    f"layer {self.name!r} reduce must return a rank-2 (B,D) tensor."
                )
            return audit_tensor
        if tensor.ndim == 2:
            return tensor
        return tensor.mean(dim=tuple(range(1, tensor.ndim - 1)))

    def replace_with_audit(self, native: Any, edited_audit: Any) -> Any:
        tensor = self.read(native)
        if self.lift is not None:
            lifted = self.lift(tensor, self.audit_tensor(native), edited_audit)
        elif tensor.ndim == 2:
            lifted = edited_audit
        elif self.reduce is not None:
            raise AuditInputError(
                f"layer {self.name!r} uses a custom reduce and must declare a matching lift."
            )
        else:
            delta = edited_audit - self.audit_tensor(native)
            shape = (delta.shape[0],) + (1,) * (tensor.ndim - 2) + (delta.shape[-1],)
            lifted = tensor + delta.reshape(shape)
        return self.write(native, lifted)


class TorchP300Adapter(P300ModelAdapter):
    """PyTorch adapter with explicit hooks and score extraction.

    Parameters
    ----------
    model:
        An already constructed model.  The adapter switches it to ``eval``
        while auditing and restores the previous training flag afterward.
    bindings:
        Mapping from audit layer name to a ``TorchLayerBinding``.  Each binding
        must be a distinct module object or deliberately share one name; the
        latter is rejected to prevent ambiguous hook placement.
    score_fn:
        Converts the model's final output to a tensor of target logits.
    input_fn:
        Optional function ``(epochs, batch_indices, device) -> model_input``.
        ``batch_indices`` are global row indices into the original audit array;
        this is the integration point for per-trial metadata or multiple model
        inputs and avoids relying on an implicit batch cursor.
    """

    def __init__(
        self,
        model: Any,
        bindings: Mapping[str, TorchLayerBinding],
        score_fn: Callable[[Any], Any],
        *,
        device: Any = None,
        input_fn: TorchInputFn | None = None,
    ) -> None:
        if not bindings:
            raise AuditInputError("TorchP300Adapter requires at least one layer binding.")
        modules = [binding.module for binding in bindings.values()]
        if len({id(module) for module in modules}) != len(modules):
            raise AuditInputError("Each adapter layer binding must refer to a distinct module.")
        self.model = model
        self.bindings = dict(bindings)
        self.score_fn = score_fn
        self.input_fn = input_fn or self._default_input
        self.device = device

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.bindings.keys())

    def _default_input(self, epochs: np.ndarray, batch_indices: np.ndarray, device: Any) -> Any:
        import torch

        del batch_indices
        return torch.as_tensor(epochs, dtype=torch.float32, device=device)

    def _resolve_device(self) -> Any:
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _run_batches(
        self,
        epochs: np.ndarray,
        *,
        batch_size: int,
        layers: Sequence[str] | None,
        intervention: LayerIntervention | None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        import torch

        array = np.asarray(epochs)
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
            raise AuditInputError("X must have numeric shape (N,C,T).")
        if not np.all(np.isfinite(array)):
            raise AuditInputError("X contains NaN or infinite values.")
        requested = self.layer_names if layers is None else tuple(layers)
        for layer in requested:
            if layer not in self.bindings:
                raise AuditInputError(f"Unknown layer {layer!r}; available {self.layer_names}.")
        if intervention is not None:
            self.validate_intervention(intervention)
            if intervention.layer not in requested:
                requested = tuple(dict.fromkeys((*requested, intervention.layer)))

        capture: dict[str, list[np.ndarray]] = {layer: [] for layer in requested}
        handles = []
        previous_training = bool(self.model.training)
        self.model.eval()
        device = self._resolve_device()

        try:
            for name in requested:
                binding = self.bindings[name]

                def hook(
                    module: Any,
                    args: tuple[Any, ...],
                    output: Any,
                    binding: TorchLayerBinding = binding,
                    name: str = name,
                ) -> Any:
                    del module, args
                    audit_tensor = binding.audit_tensor(output)
                    if intervention is not None and name == intervention.layer:
                        edited_np = np.asarray(
                            intervention.transform(audit_tensor.detach().cpu().numpy()),
                            dtype=np.float32,
                        )
                        if edited_np.shape != tuple(audit_tensor.shape):
                            raise AuditInputError(
                                f"intervention {intervention.label!r} changed layer {name!r} shape "
                                f"from {tuple(audit_tensor.shape)} to {edited_np.shape}."
                            )
                        edited = torch.as_tensor(
                            edited_np, dtype=audit_tensor.dtype, device=audit_tensor.device
                        )
                        output = binding.replace_with_audit(output, edited)
                    if intervention is None or name != intervention.layer:
                        capture[name].append(audit_tensor.detach().cpu().numpy())
                    return output

                handles.append(binding.module.register_forward_hook(hook))

            scores: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, array.shape[0], max(1, int(batch_size))):
                    batch = array[start : start + batch_size]
                    batch_indices = np.arange(start, start + batch.shape[0], dtype=int)
                    model_input = self.input_fn(batch, batch_indices, device)
                    output = self.model(model_input)
                    score_tensor = self.score_fn(output)
                    if not torch.is_tensor(score_tensor):
                        raise AuditInputError("score_fn must return a torch.Tensor.")
                    score = score_tensor.detach().float().cpu().numpy().reshape(-1)
                    if score.shape[0] != batch.shape[0]:
                        raise AuditInputError("score_fn must return one score per input row.")
                    scores.append(score)
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(previous_training)

        return (
            {name: np.concatenate(chunks, axis=0) for name, chunks in capture.items() if chunks},
            np.concatenate(scores),
        )

    def collect_activations(
        self,
        epochs: np.ndarray,
        *,
        layers: Sequence[str] | None = None,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        activations, _ = self._run_batches(
            epochs, batch_size=batch_size, layers=layers, intervention=None
        )
        for name, value in activations.items():
            if value.ndim != 2 or not np.all(np.isfinite(value)):
                raise AuditInputError(
                    f"layer {name!r} did not produce finite shape (N,D) activations."
                )
        return activations

    def predict_scores(
        self,
        epochs: np.ndarray,
        *,
        intervention: LayerIntervention | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        _, scores = self._run_batches(
            epochs, batch_size=batch_size, layers=(), intervention=intervention
        )
        assert scores is not None
        if not np.all(np.isfinite(scores)):
            raise AuditInputError("model score contains NaN or infinite values.")
        return scores
