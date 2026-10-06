"""Immutable response-prior programs and calibration information compilation.

The g prior uses every authorized calibration design row, with an independent
diffuse intercept/trend removed per record. Its AR0 reference and precision
root are fixed before Burg initialization and every subsequent noise objective.
No prediction signal, candidate or fitted noise enters this compiler.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from research.execution import UnidentifiableFitError
from transfer.noise_model import (
    array_digest,
    freeze_metadata,
    frozen_array,
    metadata_digest,
    require_identifiers,
    require_sha256,
)

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_SCHEMA = "n2p3_compiled_kernel_prior/1"
_G_CONTRACT = {
    "reference_noise": "AR0_unit_innovation", "information_support": "all_authorized_C",
    "nuisance": "per_record_diffuse_affine", "rank_policy": "require_full_rank",
}


def _positive(value, name):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw) or raw <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return float(value)


def _matrix(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real matrix")
    return np.asarray(raw, dtype=np.float64)


@dataclass(frozen=True)
class KernelPriorSpec:
    kind: str
    precision: float | None = None
    g: float | None = None
    reference_noise: str | None = None
    information_support: str | None = None
    nuisance: str | None = None
    rank_policy: str | None = None
    mean_center: str = "zero"

    def __post_init__(self):
        if self.mean_center != "zero":
            raise ValueError("this kernel-prior slice requires the declared zero mean")
        if self.kind == "isotropic_ridge":
            if self.g is not None or any(getattr(self, name) is not None for name in _G_CONTRACT):
                raise ValueError("ridge precision and calibration g parameters are mutually exclusive")
            object.__setattr__(self, "precision", _positive(self.precision, "ridge precision"))
        elif self.kind == "calibration_g":
            if self.precision is not None:
                raise ValueError("ridge precision and calibration g parameters are mutually exclusive")
            object.__setattr__(self, "g", _positive(self.g, "g variance multiplier"))
            if any(getattr(self, name) != expected for name, expected in _G_CONTRACT.items()):
                raise ValueError("calibration g requires the fixed AR0/all-C/full-rank contract")
        else:
            raise ValueError("unsupported kernel prior kind")

    @classmethod
    def isotropic_ridge(cls, precision):
        return cls("isotropic_ridge", precision=precision)

    @classmethod
    def calibration_g(cls, g):
        return cls("calibration_g", g=g, **_G_CONTRACT)

    def record(self):
        values = {"kind": self.kind, "mean_center": self.mean_center}
        if self.kind == "isotropic_ridge":
            values["precision"] = self.precision
        else:
            values.update(g=self.g, **_G_CONTRACT)
        return values

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        expected = ({"kind", "precision", "mean_center"} if values.get("kind") == "isotropic_ridge"
                    else {"kind", "g", "mean_center", *_G_CONTRACT})
        if set(values) != expected:
            raise ValueError("kernel prior record must contain only its declared kind parameters")
        return cls(**values)


def resolve_kernel_prior(kernel_prior, ridge):
    """Normalize the external choice once; no g-to-ridge conversion exists."""
    if kernel_prior is not None:
        if not isinstance(kernel_prior, KernelPriorSpec):
            raise ValueError("kernel_prior must be a validated KernelPriorSpec")
        if ridge is not None:
            raise ValueError("explicit ridge and kernel_prior are mutually exclusive")
        return kernel_prior
    return KernelPriorSpec.isotropic_ridge(1. if ridge is None else ridge)


class KernelPriorRankError(UnidentifiableFitError):
    """A valid requested prior has unidentifiable calibration directions."""

    def __init__(self, spec, rank_ledger, provenance):
        self._evidence = freeze_metadata({"kind": "kernel_prior_rank_failure", "prior_spec": spec.record(),
                                         "rank_ledger": rank_ledger, "provenance": provenance})
        super().__init__(f"calibration g prior requires full rank: {rank_ledger['rank']}/{rank_ledger['requested_dimensions']}; "
                         f"threshold={rank_ledger['absolute_threshold']:.9g}")

    @property
    def rank_ledger(self):
        return self._evidence["rank_ledger"]

    def record(self):
        return json.loads(json.dumps(self._evidence))


def _rank_ledger(root, design_root, rows, count):
    singular = np.linalg.svd(root, compute_uv=False)
    original_singular = np.linalg.svd(design_root, compute_uv=False)
    width = root.shape[1]
    relative = float(np.finfo(np.float64).eps*max(rows, width))
    original_scale = float(original_singular[0]) if len(original_singular) else 0.
    residual_scale = float(singular[0]) if len(singular) else 0.
    # Projection error is measured against its input, not against an almost
    # zero residual that may consist entirely of floating-point roundoff.
    resolution_scale = max(original_scale, residual_scale)
    threshold = relative*resolution_scale
    rank = int(np.sum(singular > threshold))
    return {"requested_dimensions": width, "rank": rank, "observed_rows": rows,
            "nuisance_dimensions": 2*count, "residual_rows": rows-2*count,
            "singular_values": singular.tolist(), "relative_threshold": relative,
            "original_design_singular_values": original_singular.tolist(),
            "original_design_scale": original_scale, "residual_design_scale": residual_scale,
            "roundoff_reference_scale": resolution_scale,
            "absolute_threshold": threshold, "criterion": "float64_original_design_projection_resolution",
            "condition_number": float(singular[0]/singular[-1]) if rank == width else None,
            "statistical_identifiability_claimed": False}


@dataclass(frozen=True)
class CompiledKernelPrior:
    spec: KernelPriorSpec
    precision_root: np.ndarray
    reference_root: np.ndarray | None
    provenance: dict
    design_root: np.ndarray | None = None
    _integrity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.spec, KernelPriorSpec):
            raise ValueError("compiled kernel prior requires its validated specification")
        root = _matrix(self.precision_root, "kernel prior precision root")
        width = len(root)
        if width == 0 or root.shape != (width, width) or np.any(np.diag(root) == 0) or np.any(np.tril(root, -1)):
            raise ValueError("kernel prior precision root must be full-rank upper triangular")
        provenance = json.loads(json.dumps(self.provenance, allow_nan=False))
        if provenance.get("implementation_sha256") != IMPLEMENTATION_SHA256:
            raise ValueError("kernel prior compiler source differs from its executed identity")
        reference = None
        design_root = None
        if self.spec.kind == "isotropic_ridge":
            if self.reference_root is not None or self.design_root is not None or not np.array_equal(root, np.sqrt(self.spec.precision)*np.eye(width)):
                raise ValueError("isotropic prior precision root differs from its declared precision")
        else:
            reference = _matrix(self.reference_root, "calibration reference root")
            design_root = _matrix(self.design_root, "original calibration design root")
            if (reference.shape != root.shape or design_root.shape != root.shape
                    or np.any(np.tril(reference, -1)) or np.any(np.tril(design_root, -1))):
                raise ValueError("calibration reference root must match the kernel coordinates")
            if not np.allclose(root, reference/np.sqrt(self.spec.g), rtol=1e-13, atol=0):
                raise ValueError("calibration g precision root differs from its actual reference and g")
            records = provenance.get("records", [])
            if (provenance.get("fit_scope") not in {"target_calibration", "source_training"}
                    or provenance.get("method") != "all_C_AR0_per_record_diffuse_incremental_QR" or not records):
                raise ValueError("calibration prior lacks its complete authorized design provenance")
            require_identifiers([row["record_id"] for row in records], "prior design records")
            for row in records:
                if (set(row) != {"record_id", "n_samples", "read_sample_support", "design_sha256"}
                        or type(row["n_samples"]) is not int or row["n_samples"] < 3
                        or row["read_sample_support"] != [0, row["n_samples"]]):
                    raise ValueError("calibration prior design ledger differs from observed record support")
                require_sha256(row["design_sha256"], "prior calibration design")
            if provenance.get("reference_design_sha256") != metadata_digest(records):
                raise ValueError("calibration prior reference design identity changed")
            _validate_reference_binding(records, provenance["binding"])
            actual = _rank_ledger(reference, design_root, sum(row["n_samples"] for row in records), len(records))
            recorded = provenance.get("rank_ledger", {})
            float_fields = {"singular_values", "relative_threshold", "absolute_threshold", "condition_number",
                            "original_design_singular_values", "original_design_scale", "residual_design_scale",
                            "roundoff_reference_scale"}
            if (set(recorded) != set(actual) or actual["rank"] != width
                    or any(recorded[key] != value for key, value in actual.items() if key not in float_fields)
                    or any(not np.allclose(recorded[key], actual[key], rtol=2e-13, atol=0) for key in float_fields)):
                raise ValueError("calibration prior rank ledger differs from its actual information root")
        object.__setattr__(self, "precision_root", frozen_array(root))
        object.__setattr__(self, "reference_root", None if reference is None else frozen_array(reference))
        object.__setattr__(self, "design_root", None if design_root is None else frozen_array(design_root))
        object.__setattr__(self, "provenance", freeze_metadata(provenance))
        object.__setattr__(self, "_integrity", metadata_digest(self._record()))

    @property
    def row_covariance(self):
        inverse = solve_triangular(self.precision_root, np.eye(len(self.precision_root)))
        return frozen_array(inverse@inverse.T)

    @property
    def rank_ledger(self):
        return self.provenance.get("rank_ledger")

    def _record(self):
        return {"schema": _SCHEMA, "spec": self.spec.record(), "precision_root": self.precision_root.tolist(),
                "reference_root": None if self.reference_root is None else self.reference_root.tolist(),
                "design_root": None if self.design_root is None else self.design_root.tolist(),
                "provenance": self.provenance, "compute_dtype": np.dtype(np.float64).str,
                "state_sha256": array_digest(self.precision_root, *( () if self.reference_root is None else (self.reference_root, self.design_root)))}

    def validate_binding(self, binding):
        """The reference design ledger and model C share one authoritative binding."""
        if metadata_digest(self.provenance["binding"]) != metadata_digest(binding):
            raise ValueError("compiled kernel prior differs from the actual C/basis/input binding")
        if self.spec.kind == "calibration_g":
            _validate_reference_binding(self.provenance["records"], binding)

    def digest(self):
        if metadata_digest(self._record()) != self._integrity:
            raise ValueError("frozen kernel prior changed")
        return self._integrity

    def record(self):
        result = json.loads(json.dumps(self._record(), allow_nan=False))
        result["artifact_sha256"] = self.digest()
        return result

    @classmethod
    def from_record(cls, record):
        values = json.loads(json.dumps(record, allow_nan=False))
        expected = values.pop("artifact_sha256", None)
        if values.get("schema") != _SCHEMA or metadata_digest(values) != expected or values.get("compute_dtype") != np.dtype(np.float64).str:
            raise ValueError("compiled kernel prior record changed or has an unsupported identity")
        result = cls(KernelPriorSpec.from_record(values["spec"]), values["precision_root"],
                     values["reference_root"], values["provenance"], design_root=values["design_root"])
        if metadata_digest(result._record()) != metadata_digest(values):
            raise ValueError("compiled prior numerical state differs from its saved record")
        return result


def _validate_reference_binding(records, binding):
    actual = binding.get("calibration_inputs")
    if actual is not None:
        expected = [{key: row[key] for key in ("record_id", "n_samples", "read_sample_support", "design_sha256")}
                    for row in actual]
        if metadata_digest(list(records)) != metadata_digest(expected):
            raise ValueError("prior reference design records differ from actual authorized C identity/support/design")


def compile_kernel_prior(spec, designs, *, record_ids, fit_scope, binding):
    """Compile one all-C program; no held inputs or noise-eta update is accepted."""
    if not isinstance(spec, KernelPriorSpec):
        raise ValueError("prior compilation requires a validated specification")
    arrays = tuple(_matrix(value, "calibration design") for value in designs)
    ids = tuple(record_ids)
    require_identifiers(ids, "prior design records")
    if not arrays or len(arrays) != len(ids) or len({d.shape[1] for d in arrays}) != 1 or any(len(d) < 3 for d in arrays):
        raise ValueError("prior designs require independent records with matched kernel coordinates")
    width = arrays[0].shape[1]
    provenance = {"implementation_sha256": IMPLEMENTATION_SHA256, "binding": binding}
    if spec.kind == "isotropic_ridge":
        provenance.update(method="fixed_isotropic_coefficient_precision", fit_scope="fixed_prior")
        return CompiledKernelPrior(spec, np.sqrt(spec.precision)*np.eye(width), None, provenance)
    if fit_scope not in {"target_calibration", "source_training"}:
        raise ValueError("calibration information prior requires authorized source/C fit scope")
    root = np.empty((0, width))
    design_root = np.empty((0, width))
    records = []
    for identity, design in zip(ids, arrays, strict=True):
        drift = np.column_stack((np.ones(len(design)), np.linspace(-1., 1., len(design))))
        q = np.linalg.qr(drift, mode="reduced")[0]
        projected = design-q@(q.T@design)
        root = np.linalg.qr(np.vstack((root, projected)), mode="r")
        design_root = np.linalg.qr(np.vstack((design_root, design)), mode="r")
        records.append({"record_id": identity, "n_samples": len(design), "read_sample_support": [0, len(design)],
                        "design_sha256": array_digest(design)})
    ledger = _rank_ledger(root, design_root, sum(len(d) for d in arrays), len(arrays))
    provenance.update(method="all_C_AR0_per_record_diffuse_incremental_QR", fit_scope=fit_scope,
                      records=records, rank_ledger=ledger, reference_design_sha256=metadata_digest(records))
    if ledger["rank"] != width:
        raise KernelPriorRankError(spec, ledger, provenance)
    return CompiledKernelPrior(spec, root/np.sqrt(spec.g), root, provenance, design_root=design_root)
