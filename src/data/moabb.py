"""Generic MOABB P300 adapter producing the universal EpochDataset contract."""

from __future__ import annotations

import inspect
import os
import socket
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, MethodType

import mne
import moabb
import numpy as np
import pandas as pd

from data.channel import DEFAULT_MONTAGE, build_channel_identity
from data.dataset import annotation_events_and_labels, build_subject
from data.epochs import (
    P300_PERFORMANCE_PREPROCESSING,
    EpochDataset,
    PreprocessingSpec,
    concatenate_epoch_datasets,
)
from data.events import observed_only_timeline, selection_group_id
from data.preprocess import apply_trial_baseline
from data.raw_artifacts import RawArtifactAttestation

ATTESTED_MOABB_RESOLVER_SCHEMA = "n2p3_attested_moabb_resolver/1"
ATTESTED_MOABB_OFFLINE_GUARD_SCHEMA = "python_socket_offline_guard/1"
VALIDATED_MOABB_VERSION = "1.6.1"
MOABB_DOWNLOAD_PROVIDER_ENV = "MOABB_DOWNLOAD_PROVIDER"

CANDIDATE_METADATA_ALIASES = (
    "candidate_id",
    "stimulus_candidate_id",
    "command_id",
)
TARGET_CANDIDATE_METADATA_ALIASES = (
    "target_candidate_id",
    "selected_candidate_id",
    "intended_candidate_id",
)
REPETITION_METADATA_ALIASES = (
    "repetition_index",
    "repetition_id",
    "sequence_index",
)

_MOABB_IO_SCOPE_LOCK = threading.RLock()


class _AttestedMoabbNetworkAccessError(RuntimeError):
    """Raised when a third-party parser attempts network I/O."""


@dataclass
class _AttestedMoabbDataPathBinding:
    paths_by_subject: dict[int, tuple[Path, ...]]
    calls: list[int] = field(default_factory=list)
    parser_call_start: int | None = None

    def begin_parser_phase(self) -> None:
        self.parser_call_start = len(self.calls)

    def parser_calls(self) -> tuple[int, ...]:
        if self.parser_call_start is None:
            raise RuntimeError("Attested MOABB parser phase was not started.")
        return tuple(self.calls[self.parser_call_start :])

    def assert_parser_consumed(self, subjects: Sequence[int]) -> None:
        requested = set(subjects)
        consumed = set(self.parser_calls())
        if consumed != requested:
            raise RuntimeError(
                "MOABB parser did not consume the complete attested data_path mapping: "
                f"requested={sorted(requested)}, consumed={sorted(consumed)}."
            )


@dataclass(frozen=True)
class _MoabbRuntimeResolver:
    declared_nemar_id: str | None
    requested_download_provider: str
    strict_base_dataset_contract: bool
    subjects: tuple[int, ...]

    def record(self) -> dict[str, object]:
        return {
            "schema": ATTESTED_MOABB_RESOLVER_SCHEMA,
            "moabb_version": VALIDATED_MOABB_VERSION,
            "declared_nemar_id": self.declared_nemar_id,
            "requested_download_provider": self.requested_download_provider,
            "effective_download_provider": "upstream",
            "prefetch_bypassed": self.strict_base_dataset_contract,
            "upstream_data_path_bypassed": True,
            "parser_data_path_verified": self.strict_base_dataset_contract,
            "subjects": list(self.subjects),
            "offline_guard": ATTESTED_MOABB_OFFLINE_GUARD_SCHEMA,
        }


@dataclass(frozen=True)
class _MoabbLoaderPathResolution:
    subject: int
    reported_path: Path
    resolved_loader_path: Path
    appendmat_resolution: bool

    def record(self, artifact_root: Path) -> dict[str, object]:
        def display(path: Path) -> str:
            absolute = path.resolve(strict=False)
            try:
                return absolute.relative_to(artifact_root).as_posix()
            except ValueError:
                return str(absolute)

        return {
            "subject": self.subject,
            "reported_path": display(self.reported_path),
            "resolved_loader_path": display(self.resolved_loader_path),
            "appendmat_resolution": self.appendmat_resolution,
        }


def _code_references_name(code: CodeType, name: str) -> bool:
    return name in code.co_names or any(
        isinstance(constant, CodeType) and _code_references_name(constant, name)
        for constant in code.co_consts
    )


def _call_graph_references_name(
    function: Callable[..., object],
    name: str,
    *,
    visited: set[int] | None = None,
) -> bool:
    """Follow same-module Python helpers and inspect their referenced names."""

    function = getattr(function, "__func__", function)
    code = getattr(function, "__code__", None)
    globals_by_name = getattr(function, "__globals__", None)
    if code is None or not isinstance(globals_by_name, dict):
        return False
    seen = set() if visited is None else visited
    identity = id(function)
    if identity in seen:
        return False
    seen.add(identity)
    if _code_references_name(code, name):
        return True
    module_name = getattr(function, "__module__", None)
    for referenced_name in code.co_names:
        candidate = globals_by_name.get(referenced_name)
        if (
            inspect.isfunction(candidate)
            and getattr(candidate, "__module__", None) == module_name
            and _call_graph_references_name(candidate, name, visited=seen)
        ):
            return True
    return False


def _assert_validated_moabb_runtime(dataset: object) -> bool:
    """Fail closed if the inspected MOABB 1.6.1 dispatch contract has changed."""

    from moabb.datasets.base import BaseDataset

    installed_version = str(moabb.__version__)
    if installed_version != VALIDATED_MOABB_VERSION:
        raise RuntimeError(
            "Attested MOABB loading is validated only for "
            f"moabb=={VALIDATED_MOABB_VERSION}; installed {installed_version}."
        )
    required_base_names = {
        "_prefetch_nemar_sourcedata",
        "_get_selected_subject_data",
    }
    if not all(
        _code_references_name(BaseDataset.get_data.__code__, name) for name in required_base_names
    ):
        raise RuntimeError("MOABB BaseDataset.get_data dispatch contract has changed.")
    if "_get_single_subject_data_using_cache" not in (
        BaseDataset._get_selected_subject_data.__code__.co_names
    ):
        raise RuntimeError("MOABB selected-subject dispatch contract has changed.")
    cache_loader_names = set(BaseDataset._get_single_subject_data_using_cache.__code__.co_names)
    if not {"_sourcedata_store", "_get_single_subject_data"} <= cache_loader_names:
        raise RuntimeError("MOABB single-subject loader dispatch contract has changed.")

    if not isinstance(dataset, BaseDataset):
        raise RuntimeError("Attested MOABB loading requires a BaseDataset instance.")
    if getattr(type(dataset), "get_data", None) is not BaseDataset.get_data:
        raise RuntimeError("Attested MOABB loading does not accept an overridden get_data method.")
    parser = getattr(dataset, "_get_single_subject_data", None)
    if not callable(parser) or not _call_graph_references_name(parser, "data_path"):
        raise RuntimeError(
            "MOABB single-subject parser no longer dispatches through the instance data_path."
        )
    return True


def _blocked_network_io(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise _AttestedMoabbNetworkAccessError(
        "Network I/O is forbidden while parsing attested MOABB snapshots."
    )


@contextmanager
def _scoped_attested_moabb_io(
    dataset: object,
    subjects: Sequence[int],
    binding: _AttestedMoabbDataPathBinding,
) -> Iterator[_MoabbRuntimeResolver]:
    """Use MOABB's upstream mode locally while blocking every outbound socket."""

    from moabb.utils import get_download_provider

    strict_contract = _assert_validated_moabb_runtime(dataset)
    declared_nemar_id = getattr(dataset, "nemar_id", None)
    if declared_nemar_id is not None:
        declared_nemar_id = str(declared_nemar_id)
    requested_provider = str(get_download_provider())
    resolver = _MoabbRuntimeResolver(
        declared_nemar_id=declared_nemar_id,
        requested_download_provider=requested_provider,
        strict_base_dataset_contract=strict_contract,
        subjects=tuple(subjects),
    )
    socket_targets = [
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket.socket, "send"),
        (socket.socket, "sendall"),
        (socket.socket, "sendto"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    ]
    if hasattr(socket.socket, "sendmsg"):
        socket_targets.append((socket.socket, "sendmsg"))

    with _MOABB_IO_SCOPE_LOCK:
        missing = object()
        previous_provider: object = os.environ.get(MOABB_DOWNLOAD_PROVIDER_ENV, missing)
        originals: list[tuple[object, str, object]] = []
        try:
            os.environ[MOABB_DOWNLOAD_PROVIDER_ENV] = "upstream"
            if get_download_provider() != "upstream":
                raise RuntimeError("MOABB ignored the scoped upstream provider override.")
            for owner, attribute in socket_targets:
                original = getattr(owner, attribute)
                originals.append((owner, attribute, original))
                setattr(owner, attribute, _blocked_network_io)
            binding.begin_parser_phase()
            yield resolver
            if strict_contract:
                binding.assert_parser_consumed(subjects)
        finally:
            for owner, attribute, original in reversed(originals):
                setattr(owner, attribute, original)
            if previous_provider is missing:
                os.environ.pop(MOABB_DOWNLOAD_PROVIDER_ENV, None)
            else:
                os.environ[MOABB_DOWNLOAD_PROVIDER_ENV] = str(previous_provider)


def _resolve_reported_moabb_path(value: str | Path, *, subject: int) -> _MoabbLoaderPathResolution:
    reported = Path(value)
    appended_mat = Path(f"{reported}.mat") if reported.suffix == "" else None
    if reported.exists():
        if appended_mat is not None and appended_mat.exists():
            raise ValueError(
                "MOABB reported an ambiguous extensionless loader path because both the "
                f"reported path and its .mat form exist: {reported}."
            )
        resolved = reported.resolve(strict=True)
        appendmat_resolution = False
    elif appended_mat is not None:
        if not appended_mat.is_file():
            raise FileNotFoundError(
                f"MOABB extensionless loader path and its .mat form are both missing: {reported}."
            )
        resolved = appended_mat.resolve(strict=True)
        appendmat_resolution = True
    else:
        raise FileNotFoundError(f"MOABB reported loader path is missing: {reported}.")
    return _MoabbLoaderPathResolution(
        subject=subject,
        reported_path=reported,
        resolved_loader_path=resolved,
        appendmat_resolution=appendmat_resolution,
    )


def _reported_moabb_loader_paths(
    dataset: object, subjects: Sequence[int]
) -> tuple[_MoabbLoaderPathResolution, ...]:
    data_path = getattr(dataset, "data_path", None)
    if not callable(data_path):
        raise TypeError("An attested MOABB dataset must expose the BaseDataset.data_path API.")
    reported: list[_MoabbLoaderPathResolution] = []
    for subject in subjects:
        values = data_path(
            subject,
            path=None,
            force_update=False,
            update_path=False,
            verbose=False,
        )
        if isinstance(values, (str, Path)):
            values = [values]
        if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
            raise TypeError("MOABB data_path must return a sequence of physical paths.")
        reported.extend(_resolve_reported_moabb_path(value, subject=subject) for value in values)
    if not reported:
        raise ValueError("MOABB data_path returned no physical paths for selected subjects.")
    return tuple(reported)


def _install_attested_data_path(
    dataset: object,
    paths_by_subject: dict[int, tuple[Path, ...]],
) -> _AttestedMoabbDataPathBinding:
    """Bind one dataset instance to verified local files without upstream I/O."""

    binding = _AttestedMoabbDataPathBinding(paths_by_subject=paths_by_subject)

    def data_path(
        _dataset: object,
        subject: int,
        path: str | Path | None = None,
        force_update: bool = False,
        update_path: bool | None = None,
        verbose: object = None,
    ) -> list[str]:
        del verbose
        if path is not None or force_update or update_path not in {None, False}:
            raise ValueError("Attested MOABB data_path is read-only and accepts no overrides.")
        try:
            paths = binding.paths_by_subject[subject]
        except KeyError as exc:
            raise ValueError(f"No attested loader mapping exists for subject {subject}.") from exc
        binding.calls.append(subject)
        return [str(value) for value in paths]

    dataset.data_path = MethodType(data_path, dataset)  # type: ignore[attr-defined]
    return binding


def _available_mne_source_paths(instances: Sequence[object]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for instance in instances:
        filenames = getattr(instance, "filenames", None)
        if filenames is None:
            raw = getattr(instance, "_raw", None)
            filenames = getattr(raw, "filenames", None)
        if filenames:
            paths.extend(Path(value) for value in filenames if value is not None)
    return tuple(paths)


def _optional_metadata_column(
    metadata: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    integer: bool = False,
) -> np.ndarray | None:
    present = [name for name in aliases if name in metadata]
    if not present:
        return None
    reference = metadata[present[0]].to_numpy()
    for name in present[1:]:
        if not np.array_equal(reference.astype(str), metadata[name].to_numpy().astype(str)):
            raise ValueError(f"Conflicting metadata aliases {present}.")
    if integer:
        if not np.issubdtype(reference.dtype, np.integer) or np.issubdtype(
            reference.dtype, np.bool_
        ):
            raise ValueError(f"Metadata column {present[0]!r} must contain integers.")
        return reference.astype(np.int64, copy=False)
    values = reference.astype(str)
    if np.any(np.char.strip(values) == ""):
        raise ValueError(f"Metadata column {present[0]!r} contains empty identifiers.")
    return values


def resolve_moabb_dataset(dataset_class: str):
    """Resolve any installed MOABB dataset class by public class name."""

    import moabb.datasets

    dataset_type = getattr(moabb.datasets, dataset_class, None)
    if dataset_type is None or not isinstance(dataset_type, type):
        available = sorted(
            name
            for name, value in vars(moabb.datasets).items()
            if isinstance(value, type) and not name.startswith("_")
        )
        raise ValueError(
            f"Unknown MOABB dataset class {dataset_class!r}. Installed public classes include "
            f"{available}."
        )
    return dataset_type()


def encode_binary_labels(values: np.ndarray, *, target_label: str = "Target") -> np.ndarray:
    labels = np.asarray(values)
    if np.issubdtype(labels.dtype, np.number):
        unique = set(np.unique(labels).tolist())
        if unique <= {0, 1}:
            return labels.astype(np.int64)
    normalized = np.char.lower(np.char.strip(labels.astype(str)))
    target = target_label.strip().lower()
    unique = set(normalized.tolist())
    if target not in unique or len(unique) != 2:
        raise ValueError(
            f"Binary P300 labels must contain target={target_label!r} and one non-target class; "
            f"got {sorted(unique)}."
        )
    return (normalized == target).astype(np.int64)


def prepare_moabb_p300(
    dataset_class: str,
    *,
    raw_artifact_attestation: RawArtifactAttestation,
    subjects: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    montage: str = DEFAULT_MONTAGE,
    preprocessing: PreprocessingSpec = P300_PERFORMANCE_PREPROCESSING,
    target_label: str = "Target",
) -> EpochDataset:
    """Process an attested MOABB P300 dataset without dataset-specific branches."""

    from moabb.paradigms import P300

    preprocessing.validate()
    if not isinstance(raw_artifact_attestation, RawArtifactAttestation):
        raise TypeError("raw_artifact_attestation must be a verified RawArtifactAttestation.")
    raw_artifact_attestation.assert_dataset_class(dataset_class)
    if preprocessing.reject_threshold_v is not None:
        raise ValueError(
            "Fixed absolute-voltage artifact rejection is retired; set reject_threshold_v=None "
            "and use fold-local artifact QC during evaluation."
        )
    dataset = resolve_moabb_dataset(dataset_class)
    acquisition = getattr(getattr(dataset, "metadata", None), "acquisition", None)
    source_reference = str(getattr(acquisition, "reference", None) or "unspecified")
    source_sample_rate = getattr(acquisition, "sampling_rate", None)
    if source_reference == "unspecified" or source_sample_rate is None:
        raise ValueError(
            f"MOABB dataset {dataset_class} lacks acquisition reference/sample-rate metadata."
        )
    selected_subjects = list(subjects) if subjects is not None else list(dataset.subject_list)
    if not selected_subjects:
        raise ValueError("At least one MOABB subject must be selected.")
    if len(set(selected_subjects)) != len(selected_subjects):
        raise ValueError("MOABB subject selection must not contain duplicates.")
    unknown_subjects = set(selected_subjects) - set(dataset.subject_list)
    if unknown_subjects:
        raise ValueError(f"Unknown subjects for {dataset_class}: {sorted(unknown_subjects)}.")
    materialization = raw_artifact_attestation.materialize_moabb_loaders(selected_subjects)
    attested_data_path = _install_attested_data_path(
        dataset,
        dict(materialization.paths_by_subject),
    )
    loader_path_resolutions = _reported_moabb_loader_paths(dataset, selected_subjects)
    reported_loader_paths = tuple(item.resolved_loader_path for item in loader_path_resolutions)
    materialization.verify_loader_paths(reported_loader_paths)
    if preprocessing.filter_phase == "forward":
        event_id = getattr(dataset, "event_id", None)
        if not isinstance(event_id, dict) or target_label not in event_id:
            raise ValueError(
                f"MOABB dataset {dataset_class} does not expose target event {target_label!r}."
            )
        label_map = {
            str(description): int(str(description) == target_label) for description in event_id
        }
        with _scoped_attested_moabb_io(
            dataset,
            selected_subjects,
            attested_data_path,
        ) as runtime_resolver:
            runs = dataset.get_data(subjects=selected_subjects)
        raw_instances = [
            raw
            for subject_runs in runs.values()
            for session_runs in subject_runs.values()
            for raw in session_runs.values()
        ]
        loader_paths = reported_loader_paths + _available_mne_source_paths(raw_instances)
        materialization.verify_loader_paths(loader_paths)
        raw_artifact_provenance = materialization.attestation.provenance_record()
        raw_artifact_provenance["raw_artifact_loader_path_resolutions"] = [
            item.record(Path(materialization.expected_mne_data_root))
            for item in loader_path_resolutions
        ]
        raw_artifact_provenance["moabb_attested_resolver"] = runtime_resolver.record()
        datasets: list[EpochDataset] = []
        for subject in selected_subjects:
            for session_id, session_runs in runs[subject].items():
                for run_id, raw in session_runs.items():
                    events, labels = annotation_events_and_labels(raw, label_map)
                    record = build_subject(
                        raw,
                        events,
                        labels,
                        subject_id=str(subject),
                        metadata={
                            "dataset_id": dataset_class,
                            "session": str(session_id),
                            "run": str(run_id),
                            "selection_id": f"{session_id}:{run_id}",
                            "reference": source_reference,
                        },
                        sfreq=preprocessing.sfreq,
                        l_freq=preprocessing.l_freq,
                        h_freq=preprocessing.h_freq,
                        tmin=preprocessing.tmin_ms / 1000.0,
                        tmax=preprocessing.tmax_ms / 1000.0,
                        n_times=preprocessing.n_times,
                        reject_threshold=preprocessing.reject_threshold_v,
                        baseline=None,
                        baseline_mode=preprocessing.baseline_mode,
                        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
                        trial_reference_center=preprocessing.trial_reference_center,
                        trial_reference_scale=preprocessing.trial_reference_scale,
                        filter_method=preprocessing.filter_method,
                        filter_order=preprocessing.filter_order,
                        filter_phase=preprocessing.filter_phase,
                        causal_iir_initial_state=preprocessing.causal_iir_initial_state,
                        resample_domain=preprocessing.resample_domain,
                        resample_method=preprocessing.resample_method,
                        resample_npad=preprocessing.resample_npad,
                        resample_window=preprocessing.resample_window,
                        resample_pad=preprocessing.resample_pad,
                        channels=None if channels is None else list(channels),
                        montage=montage,
                    )
                    n_epochs = record.n_epochs
                    datasets.append(
                        EpochDataset(
                            name=dataset_class,
                            X=np.asarray(record.data, dtype=np.float32),
                            y=np.asarray(record.labels, dtype=np.int64),
                            subject_ids=np.repeat(str(subject), n_epochs),
                            channel_names=record.channel_names,
                            channel_positions_m=record.channel_positions_m,
                            channel_mask=record.channel_mask,
                            preprocessing=preprocessing,
                            event_timeline=record.event_timeline,
                            metadata=pd.DataFrame(
                                {
                                    "subject": np.repeat(str(subject), n_epochs),
                                    "session": np.repeat(str(session_id), n_epochs),
                                    "run": np.repeat(str(run_id), n_epochs),
                                }
                            ),
                            provenance={
                                "source": "moabb_raw_causal_p300",
                                "dataset_class": dataset_class,
                                **raw_artifact_provenance,
                                "source_reference": source_reference,
                                "source_sample_rate_hz": float(raw.info["sfreq"]),
                                "model_input_sample_rate_hz": preprocessing.sfreq,
                                "signal_unit": preprocessing.signal_unit,
                                "filter_phase": "forward",
                            },
                        )
                    )
        return concatenate_epoch_datasets(
            datasets,
            name=dataset_class,
            provenance={
                "source": "moabb_raw_causal_p300",
                "dataset_class": dataset_class,
                **raw_artifact_provenance,
                "subjects": selected_subjects,
                "source_reference": source_reference,
                "source_sample_rate_hz": float(source_sample_rate),
                "model_input_sample_rate_hz": preprocessing.sfreq,
                "moabb_version": moabb.__version__,
                "mne_version": mne.__version__,
                "signal_unit": preprocessing.signal_unit,
            },
        )
    paradigm = P300(
        fmin=preprocessing.l_freq,
        fmax=preprocessing.h_freq,
        tmin=preprocessing.tmin_ms / 1000.0,
        tmax=preprocessing.tmax_ms / 1000.0,
        baseline=None,
        resample=None,
        channels=None if channels is None else list(channels),
    )
    with _scoped_attested_moabb_io(
        dataset,
        selected_subjects,
        attested_data_path,
    ) as runtime_resolver:
        epochs, raw_labels, metadata = paradigm.get_data(
            dataset=dataset,
            subjects=selected_subjects,
            return_epochs=True,
        )
    loader_paths = reported_loader_paths + _available_mne_source_paths([epochs])
    materialization.verify_loader_paths(loader_paths)
    raw_artifact_provenance = materialization.attestation.provenance_record()
    raw_artifact_provenance["raw_artifact_loader_path_resolutions"] = [
        item.record(Path(materialization.expected_mne_data_root))
        for item in loader_path_resolutions
    ]
    raw_artifact_provenance["moabb_attested_resolver"] = runtime_resolver.record()
    expected_tmin = preprocessing.tmin_ms / 1000.0
    if not np.isclose(float(epochs.times[0]), expected_tmin, atol=0.5 / preprocessing.sfreq):
        raise ValueError(
            "MOABB epoch origin does not match the declared stimulus-relative time contract: "
            f"{float(epochs.times[0]):g}s vs {expected_tmin:g}s."
        )
    source_epoch_sfreq = float(epochs.info["sfreq"])
    source_epoch_events = np.asarray(epochs.events, dtype=np.int64).copy()
    if not np.isclose(source_epoch_sfreq, preprocessing.sfreq):
        epochs.resample(preprocessing.sfreq, verbose=False)
    X = epochs.get_data()
    raw_labels = np.asarray(raw_labels)
    if len(X) != len(raw_labels) or len(metadata) != len(X):
        raise ValueError("MOABB epochs, labels, and metadata are not row-aligned.")
    if "subject" not in metadata:
        raise ValueError("MOABB metadata does not contain a subject column required for LOSO.")
    requested_subject_keys = {str(subject) for subject in selected_subjects}
    returned_subject_keys = set(metadata["subject"].astype(str))
    missing_before_rejection = requested_subject_keys - returned_subject_keys
    if missing_before_rejection:
        raise ValueError(
            f"MOABB returned no epochs for selected subjects {sorted(missing_before_rejection)}."
        )
    if X.shape[2] < preprocessing.n_times:
        raise ValueError(
            f"MOABB returned {X.shape[2]} time samples; {preprocessing.n_times} are required."
        )
    X = X[:, :, : preprocessing.n_times].astype(np.float32, copy=False)
    X = apply_trial_baseline(
        X,
        sfreq=preprocessing.sfreq,
        tmin=preprocessing.tmin_ms / 1000.0,
        baseline_mode=preprocessing.baseline_mode,
        trial_reference_window_ms=preprocessing.trial_reference_window_ms,
        trial_reference_center=preprocessing.trial_reference_center,
        trial_reference_scale=preprocessing.trial_reference_scale,
    )
    keep = np.ones(len(X), dtype=bool)
    metadata = metadata.reset_index(drop=True).copy()
    channel_names = tuple(str(name) for name in epochs.ch_names)
    identity = build_channel_identity(
        channel_names,
        channel_mask=np.ones(len(channel_names), dtype=bool),
        montage=montage,
        allow_missing_positions=False,
    )
    metadata["acquisition_time_s"] = source_epoch_events[keep, 0].astype(float) / source_epoch_sfreq
    subjects_array = metadata["subject"].astype(str).to_numpy()
    sessions = (
        metadata["session"].astype(str).to_numpy()
        if "session" in metadata
        else [""] * len(metadata)
    )
    runs = metadata["run"].astype(str).to_numpy() if "run" in metadata else [""] * len(metadata)
    selections = (
        metadata["selection_id"].astype(str).to_numpy()
        if "selection_id" in metadata
        else subjects_array
    )
    groups = np.asarray(
        [
            selection_group_id(dataset_class, subject, session, run, selection)
            for subject, session, run, selection in zip(
                subjects_array, sessions, runs, selections, strict=True
            )
        ]
    )
    timeline = observed_only_timeline(
        dataset_id=dataset_class,
        subject_ids=subjects_array,
        stimulus_ids=source_epoch_events[keep, 2],
        onset_times_s=metadata["acquisition_time_s"].to_numpy(dtype=float),
        group_ids=groups,
        selection_ids=np.asarray(selections, dtype=str),
        session_ids=np.asarray(sessions, dtype=str),
        run_ids=np.asarray(runs, dtype=str),
        candidate_ids=_optional_metadata_column(metadata, CANDIDATE_METADATA_ALIASES),
        target_candidate_ids=_optional_metadata_column(metadata, TARGET_CANDIDATE_METADATA_ALIASES),
        repetition_indices=_optional_metadata_column(
            metadata, REPETITION_METADATA_ALIASES, integer=True
        ),
        online_causal=False,
        timing_source=(
            "moabb_post_resample_epoch_events;observed_epochs_only;preprocessing_not_online_causal"
        ),
    )
    result = EpochDataset(
        name=dataset_class,
        X=X,
        y=encode_binary_labels(raw_labels, target_label=target_label),
        subject_ids=subjects_array,
        channel_names=identity.names,
        channel_positions_m=identity.coords,
        channel_mask=identity.mask,
        preprocessing=preprocessing,
        event_timeline=timeline,
        metadata=metadata,
        provenance={
            "source": "moabb_p300",
            "dataset_class": dataset_class,
            **raw_artifact_provenance,
            "subjects": selected_subjects,
            "moabb_version": moabb.__version__,
            "mne_version": mne.__version__,
            "source_sample_rate_hz": source_sample_rate,
            "model_input_sample_rate_hz": preprocessing.sfreq,
            "source_reference": source_reference,
            "source_ground": getattr(acquisition, "ground", None),
            "event_time_basis": "post_resample_epoch_event_samples",
            "grouping_fidelity": "subject_session_run_only; source blocks are not exposed",
            "signal_unit": preprocessing.signal_unit,
            "filter": {
                "method": preprocessing.filter_method,
                "order": preprocessing.filter_order,
                "phase": preprocessing.filter_phase,
            },
            "resample_domain": preprocessing.resample_domain,
            "resample": {
                "method": preprocessing.resample_method,
                "npad": preprocessing.resample_npad,
                "window": preprocessing.resample_window,
                "pad": preprocessing.resample_pad,
            },
            "montage": montage,
            "coordinate_registration": identity.registration.record(),
            "right_endpoint": "exclusive_crop",
            "epoch_baseline": {
                "mode": preprocessing.baseline_mode,
                "window_ms": (
                    list(preprocessing.trial_reference_window_ms)
                    if preprocessing.trial_reference_window_ms is not None
                    else [preprocessing.tmin_ms, 0.0]
                ),
                "center": preprocessing.trial_reference_center,
                "scale": (
                    "std"
                    if preprocessing.baseline_mode == "trial"
                    else preprocessing.trial_reference_scale
                ),
            },
            "artifact_rejection": {"method": "fold_local_ptp_cv", "n_before": int(len(keep))},
        },
    )
    result.validate(require_labels=True)
    return result
