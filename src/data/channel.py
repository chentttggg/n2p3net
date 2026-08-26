"""Physical channel-layout resolution and deterministic coordinate embeddings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

HEAD_RADIUS: float = 0.1
DEFAULT_N_FREQS: int = 8
DEFAULT_MONTAGE: str = "standard_1005"
STANDARD_CHANNELS: tuple[str, ...] = ("Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz")

_LEGACY_1020_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_REFERENCE_SUFFIX = re.compile(
    r"(?:[-_.\s]+(?:REF(?:ERENCE)?|AVG|AVERAGE|A[12]|M[12]|LE|RE|EAR[12]))+$",
    flags=re.IGNORECASE,
)
_EEG_PREFIX = re.compile(r"^(?:EEG|CHANNEL|CHAN)\s*[:#_-]?\s*", flags=re.IGNORECASE)
_FIDUCIAL_ALIASES = {
    "NAS": "nasion",
    "NASION": "nasion",
    "LPA": "lpa",
    "LEFT": "lpa",
    "RPA": "rpa",
    "RIGHT": "rpa",
}


@dataclass(frozen=True)
class CoordinateRegistrationSpec:
    """Auditable rigid registration from one sensor frame into MNE head coordinates."""

    source: str
    input_frame: str
    method: str
    transform_to_head: tuple[tuple[float, float, float, float], ...]
    output_frame: str = "head"
    units: str = "m"
    fiducials_used: tuple[str, ...] = ()
    icp_iterations: int = 0
    icp_rmse_m: float | None = None
    spherical_fallback: bool = False

    def __post_init__(self) -> None:
        transform = np.asarray(self.transform_to_head, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("transform_to_head must be a finite 4x4 matrix.")
        if self.output_frame != "head" or self.units != "m":
            raise ValueError("Registered EEG coordinates must use the MNE head frame in metres.")
        if self.icp_iterations < 0:
            raise ValueError("icp_iterations cannot be negative.")
        if self.icp_rmse_m is not None and (
            not np.isfinite(self.icp_rmse_m) or self.icp_rmse_m < 0.0
        ):
            raise ValueError("icp_rmse_m must be finite and non-negative.")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _transform_tuple(transform: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in transform)


def _as_position_array(
    positions: Sequence[Sequence[float]] | np.ndarray,
    *,
    expected_rows: int | None = None,
    name: str = "positions_m",
) -> np.ndarray:
    try:
        value = np.asarray(positions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric (C,3) coordinate sequence.") from exc
    expected = (expected_rows, 3) if expected_rows is not None else None
    if value.ndim != 2 or value.shape[1] != 3 or (expected is not None and value.shape != expected):
        requirement = expected if expected is not None else "(C,3)"
        raise ValueError(f"{name} must be {requirement}, got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite coordinates.")
    return value


def _canonical_fiducials(
    fiducials_m: Mapping[str, Sequence[float]] | None,
) -> dict[str, np.ndarray]:
    if fiducials_m is None:
        return {}
    output: dict[str, np.ndarray] = {}
    for name, position in fiducials_m.items():
        key = _FIDUCIAL_ALIASES.get(str(name).strip().upper())
        if key is None:
            continue
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Fiducial {name!r} must be a finite 3-vector in metres.")
        output[key] = value
    return output


def fiducial_head_transform(fiducials_m: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Return the rigid native-to-head transform defined by LPA/RPA/Nasion."""

    fiducials = _canonical_fiducials(fiducials_m)
    missing = {"lpa", "rpa", "nasion"} - set(fiducials)
    if missing:
        raise ValueError(
            f"Fiducial registration requires LPA/RPA/Nasion; missing {sorted(missing)}."
        )
    lpa, rpa, nasion = fiducials["lpa"], fiducials["rpa"], fiducials["nasion"]
    x_axis = rpa - lpa
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-6:
        raise ValueError("LPA and RPA are coincident; the head frame is undefined.")
    x_axis /= x_norm
    # MNE/Neuromag head origin: orthogonal projection of the nasion onto
    # the LPA-RPA axis. The auricular midpoint is only correct for symmetric
    # digitizations.
    origin = lpa + np.dot(nasion - lpa, x_axis) * x_axis
    nasion_direction = nasion - origin
    y_norm = np.linalg.norm(nasion_direction)
    if y_norm <= 1e-6:
        raise ValueError("LPA/RPA/Nasion are collinear; the head frame is undefined.")
    y_axis = nasion_direction / y_norm
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.stack((x_axis, y_axis, z_axis), axis=0)
    transform = _identity_transform()
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    return transform


def apply_rigid_transform(positions_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    positions = _as_position_array(positions_m)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Rigid transform must be a finite 4x4 matrix.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("Coordinate registration transform must be a proper rigid transform.")
    return positions @ rotation.T + matrix[:3, 3]


def _kabsch_transform(source_m: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    source = _as_position_array(source_m, name="ICP source")
    target = _as_position_array(target_m, expected_rows=len(source), name="ICP target")
    if len(source) < 3:
        raise ValueError("Rigid registration requires at least three point correspondences.")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    transform = _identity_transform()
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def rigid_icp(
    source_m: Sequence[Sequence[float]] | np.ndarray,
    target_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_iterations: int = 25,
    tolerance_m: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Rigid point-to-point ICP; returns registered points and cumulative transform."""

    source = _as_position_array(source_m, name="ICP source")
    target = _as_position_array(target_m, name="ICP target")
    if len(source) < 3 or len(target) < 3:
        raise ValueError("ICP requires at least three source and target points.")
    if max_iterations < 1 or tolerance_m <= 0.0:
        raise ValueError("ICP max_iterations and tolerance_m must be positive.")
    current = source.copy()
    cumulative = _identity_transform()
    previous_rmse = float("inf")
    for iteration in range(1, max_iterations + 1):
        distance = np.linalg.norm(current[:, None, :] - target[None, :, :], axis=-1)
        matched = target[np.argmin(distance, axis=1)]
        delta = _kabsch_transform(current, matched)
        current = apply_rigid_transform(current, delta)
        cumulative = delta @ cumulative
        rmse = float(np.sqrt(np.mean(np.sum((current - matched) ** 2, axis=1))))
        if abs(previous_rmse - rmse) <= tolerance_m:
            return current, cumulative, iteration, rmse
        previous_rmse = rmse
    return current, cumulative, max_iterations, previous_rmse


def register_sensor_positions(
    positions_m: Sequence[Sequence[float]] | np.ndarray,
    *,
    source: str,
    coordinate_frame: str = "head",
    fiducials_m: Mapping[str, Sequence[float]] | None = None,
    icp_target_m: Sequence[Sequence[float]] | np.ndarray | None = None,
    allow_spherical_fallback: bool = False,
) -> tuple[np.ndarray, CoordinateRegistrationSpec]:
    """Register sensor coordinates to head space without changing physical scale."""

    positions = _as_position_array(positions_m)
    input_frame = str(coordinate_frame).lower()
    fiducials = _canonical_fiducials(fiducials_m)
    spherical_fallback = False
    if input_frame == "head":
        transform = _identity_transform()
        registered = positions.copy()
        method = "identity_head"
        used: tuple[str, ...] = ()
    elif {"lpa", "rpa", "nasion"}.issubset(fiducials):
        transform = fiducial_head_transform(fiducials)
        registered = apply_rigid_transform(positions, transform)
        method = "fiducial_rigid"
        used = ("lpa", "rpa", "nasion")
    elif allow_spherical_fallback:
        radii = np.linalg.norm(positions, axis=1, keepdims=True)
        if np.any(radii <= 1e-12):
            raise ValueError("Spherical fallback cannot project a coordinate at the origin.")
        registered = positions / radii * HEAD_RADIUS
        transform = _identity_transform()
        method = "explicit_unit_sphere_fallback"
        used = ()
        spherical_fallback = True
    else:
        raise ValueError(
            f"Coordinates in frame {coordinate_frame!r} need LPA/RPA/Nasion registration. "
            "Unit-sphere fallback is disabled unless explicitly requested."
        )

    icp_iterations = 0
    icp_rmse_m = None
    if icp_target_m is not None:
        registered, correction, icp_iterations, icp_rmse_m = rigid_icp(registered, icp_target_m)
        transform = correction @ transform
        method += "+rigid_icp"

    spec = CoordinateRegistrationSpec(
        source=str(source),
        input_frame=input_frame,
        method=method,
        transform_to_head=_transform_tuple(transform),
        fiducials_used=used,
        icp_iterations=icp_iterations,
        icp_rmse_m=icp_rmse_m,
        spherical_fallback=spherical_fallback,
    )
    return registered, spec


def canonical_channel_name(
    name: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Normalize an EEG channel label without inventing a physical location."""

    value = _EEG_PREFIX.sub("", str(name).strip())
    value = _REFERENCE_SUFFIX.sub("", value).strip().upper()
    if not value:
        raise ValueError(f"Invalid empty channel name derived from {name!r}.")
    alias_map = {**_LEGACY_1020_ALIASES}
    if aliases:
        alias_map.update(
            {
                canonical_channel_name(key): canonical_channel_name(target)
                for key, target in aliases.items()
            }
        )
    return alias_map.get(value, value)


def _canonical_unique(
    ch_names: Sequence[str], aliases: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    names = tuple(canonical_channel_name(name, aliases=aliases) for name in ch_names)
    if len(set(names)) != len(names):
        raise ValueError(f"Channel names are not unique after normalization: {names}.")
    return names


def load_montage(montage: str | Path | mne.channels.DigMontage) -> mne.channels.DigMontage:
    """Load an MNE standard montage or a custom montage file."""

    if isinstance(montage, mne.channels.DigMontage):
        return montage
    path = Path(montage)
    if path.exists():
        return mne.channels.read_custom_montage(path)
    try:
        return mne.channels.make_standard_montage(str(montage), head_size="auto")
    except ValueError as exc:
        available = ", ".join(mne.channels.get_builtin_montages())
        raise ValueError(
            f"Unknown montage {montage!r}. Provide a supported MNE montage name or a "
            f"custom montage file. Built-ins: {available}."
        ) from exc


def montage_positions(
    montage: str | Path | mne.channels.DigMontage,
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Return canonical channel positions in metres from one montage."""

    loaded = load_montage(montage)
    positions: dict[str, np.ndarray] = {}
    for name, position in loaded.get_positions()["ch_pos"].items():
        key = canonical_channel_name(name, aliases=aliases)
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            continue
        if key in positions and not np.allclose(positions[key], value, atol=1e-7):
            raise ValueError(f"Montage contains conflicting positions for channel {key!r}.")
        positions[key] = value
    return positions


def positions_from_raw(
    raw: mne.io.BaseRaw,
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract finite EEG sensor positions embedded in an MNE Raw object."""

    montage = raw.get_montage()
    if montage is None:
        return {}
    return montage_positions(montage, aliases=aliases)


@dataclass(frozen=True)
class ChannelLayout:
    """Resolved physical sensor layout in a common head coordinate frame."""

    names: tuple[str, ...]
    positions_m: np.ndarray
    position_mask: np.ndarray
    source: str
    registration: CoordinateRegistrationSpec

    def __post_init__(self) -> None:
        if self.positions_m.shape != (len(self.names), 3):
            raise ValueError("ChannelLayout.positions_m must be (C,3).")
        if self.position_mask.shape != (len(self.names),):
            raise ValueError("ChannelLayout.position_mask must be (C,).")


def _montage_registration_inputs(
    montage: str | Path | mne.channels.DigMontage,
    *,
    aliases: Mapping[str, str] | None,
) -> tuple[dict[str, np.ndarray], str, dict[str, np.ndarray]]:
    values = load_montage(montage).get_positions()
    positions: dict[str, np.ndarray] = {}
    for name, position in values["ch_pos"].items():
        key = canonical_channel_name(name, aliases=aliases)
        value = np.asarray(position, dtype=np.float64)
        if value.shape == (3,) and np.isfinite(value).all():
            positions[key] = value
    fiducials = {
        key: np.asarray(values[key], dtype=np.float64)
        for key in ("nasion", "lpa", "rpa")
        if values.get(key) is not None
    }
    return positions, str(values.get("coord_frame", "unknown")), fiducials


def resolve_channel_layout(
    ch_names: Sequence[str],
    *,
    positions_m: Mapping[str, Sequence[float]]
    | Sequence[Sequence[float]]
    | np.ndarray
    | None = None,
    montage: str | Path | mne.channels.DigMontage | None = DEFAULT_MONTAGE,
    aliases: Mapping[str, str] | None = None,
    allow_missing: bool = False,
    coordinate_frame: str = "head",
    fiducials_m: Mapping[str, Sequence[float]] | None = None,
    coordinate_source: str | None = None,
    icp_target_m: Sequence[Sequence[float]] | np.ndarray | None = None,
    allow_spherical_fallback: bool = False,
) -> ChannelLayout:
    """Resolve sensors in a registered, metre-scale MNE head frame."""

    names = _canonical_unique(ch_names, aliases)
    resolved = np.zeros((len(names), 3), dtype=np.float64)
    mask = np.zeros(len(names), dtype=bool)
    sources: list[str] = []
    registration: CoordinateRegistrationSpec | None = None

    if positions_m is not None:
        if isinstance(positions_m, Mapping):
            by_name = {
                canonical_channel_name(name, aliases=aliases): np.asarray(value, dtype=np.float64)
                for name, value in positions_m.items()
            }
            explicit = np.zeros_like(resolved)
            for index, name in enumerate(names):
                value = by_name.get(name)
                if value is not None and value.shape == (3,) and np.isfinite(value).all():
                    explicit[index] = value
        else:
            explicit = _as_position_array(
                positions_m,
                expected_rows=len(names),
                name="Explicit channel positions",
            )
        valid = np.isfinite(explicit).all(axis=1) & (np.linalg.norm(explicit, axis=1) > 0)
        if valid.any():
            icp_target = None
            if icp_target_m is not None:
                icp_target = _as_position_array(
                    icp_target_m,
                    expected_rows=len(names),
                    name="icp_target_m",
                )[valid]
            registered, registration = register_sensor_positions(
                explicit[valid],
                source=coordinate_source or "explicit_coordinates",
                coordinate_frame=coordinate_frame,
                fiducials_m=fiducials_m,
                icp_target_m=icp_target,
                allow_spherical_fallback=allow_spherical_fallback,
            )
            resolved[valid] = registered
            mask[valid] = True
        sources.append("explicit")

    if montage is not None and not mask.all():
        standard, montage_frame, montage_fiducials = _montage_registration_inputs(
            montage, aliases=aliases
        )
        available_indices = [
            index for index, name in enumerate(names) if not mask[index] and name in standard
        ]
        available_names = [names[index] for index in available_indices]
        montage_source = coordinate_source
        if montage_source is None:
            montage_source = (
                "average_head_template"
                if isinstance(montage, str) and montage in {"standard_1005", "standard_1020"}
                else "device_montage"
            )
        if available_indices:
            montage_positions_m = np.asarray(
                [standard[name] for name in available_names], dtype=np.float64
            )
            registered, montage_registration = register_sensor_positions(
                montage_positions_m,
                source=montage_source,
                coordinate_frame=montage_frame,
                fiducials_m=montage_fiducials,
                icp_target_m=icp_target_m,
                allow_spherical_fallback=allow_spherical_fallback,
            )
            for index, value in zip(available_indices, registered, strict=True):
                resolved[index] = value
                mask[index] = True
            if registration is None:
                registration = montage_registration
            else:
                registration = CoordinateRegistrationSpec(
                    source=f"{registration.source}+{montage_registration.source}",
                    input_frame="mixed",
                    method=f"{registration.method}+{montage_registration.method}",
                    transform_to_head=_transform_tuple(_identity_transform()),
                    spherical_fallback=(
                        registration.spherical_fallback or montage_registration.spherical_fallback
                    ),
                )
        sources.append(str(montage))

    if not allow_missing and not mask.all():
        missing = [name for name, valid in zip(names, mask, strict=True) if not valid]
        raise ValueError(
            f"No physical coordinates for channels {missing}. Supply an embedded/custom montage "
            "or an explicit positions_m mapping; channel-name embeddings are not allowed."
        )

    if registration is None:
        registration = CoordinateRegistrationSpec(
            source="unresolved",
            input_frame="head",
            method="no_observed_coordinates",
            transform_to_head=_transform_tuple(_identity_transform()),
        )
    return ChannelLayout(
        names=names,
        positions_m=resolved.astype(np.float32),
        position_mask=mask,
        source="+".join(sources) if sources else "unresolved",
        registration=registration,
    )


@dataclass(frozen=True)
class ChannelIdentity:
    embedding: np.ndarray
    coords: np.ndarray
    mask: np.ndarray
    names: tuple[str, ...]
    layout_source: str
    registration: CoordinateRegistrationSpec

    @property
    def n_channels(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embedding.shape[1])


def channel_coords(
    ch_names: Sequence[str],
    *,
    positions_m: Mapping[str, Sequence[float]]
    | Sequence[Sequence[float]]
    | np.ndarray
    | None = None,
    montage: str | Path | mne.channels.DigMontage | None = DEFAULT_MONTAGE,
    aliases: Mapping[str, str] | None = None,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    layout = resolve_channel_layout(
        ch_names,
        positions_m=positions_m,
        montage=montage,
        aliases=aliases,
        allow_missing=allow_missing,
    )
    return layout.positions_m.astype(np.float64), layout.position_mask.copy()


def standard_coords() -> np.ndarray:
    coords, _ = channel_coords(STANDARD_CHANNELS)
    return coords


def sinusoidal_encode_1d(c: np.ndarray, n_freqs: int = DEFAULT_N_FREQS) -> np.ndarray:
    values = np.asarray(c, dtype=np.float64)
    frequencies = np.pi * (2.0 ** np.arange(n_freqs, dtype=np.float64))
    angles = values[:, None] * frequencies[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


def sinusoidal_embedding(coords: np.ndarray, n_freqs: int = DEFAULT_N_FREQS) -> np.ndarray:
    """Encode metre-scale head coordinates with a band-limited Fourier basis."""

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (C,3), got {coords.shape}.")
    scaled = coords / HEAD_RADIUS
    frequencies = np.pi * np.geomspace(0.5, 8.0, n_freqs, dtype=np.float64)
    encoded = []
    for axis in range(3):
        angles = scaled[:, axis, None] * frequencies[None, :]
        encoded.append(np.concatenate([np.sin(angles), np.cos(angles)], axis=1))
    return np.concatenate(encoded, axis=1).astype(np.float32)


def build_channel_identity(
    ch_names: Sequence[str] | None = None,
    channel_mask: Sequence[bool] | np.ndarray | None = None,
    *,
    positions_m: Mapping[str, Sequence[float]]
    | Sequence[Sequence[float]]
    | np.ndarray
    | None = None,
    montage: str | Path | mne.channels.DigMontage | None = DEFAULT_MONTAGE,
    aliases: Mapping[str, str] | None = None,
    n_freqs: int = DEFAULT_N_FREQS,
    allow_missing_positions: bool = True,
) -> ChannelIdentity:
    names = tuple(ch_names) if ch_names is not None else STANDARD_CHANNELS
    layout = resolve_channel_layout(
        names,
        positions_m=positions_m,
        montage=montage,
        aliases=aliases,
        allow_missing=allow_missing_positions,
    )
    mask = layout.position_mask.copy()
    if channel_mask is not None:
        observed = np.asarray(channel_mask, dtype=bool)
        if observed.shape != mask.shape:
            raise ValueError(f"channel_mask must be {mask.shape}, got {observed.shape}.")
        mask &= observed
    coords = layout.positions_m.astype(np.float64)
    embedding = sinusoidal_embedding(coords, n_freqs)
    embedding[~mask] = 0.0
    return ChannelIdentity(
        embedding=embedding,
        coords=coords.astype(np.float32),
        mask=mask,
        names=layout.names,
        layout_source=layout.source,
        registration=layout.registration,
    )
