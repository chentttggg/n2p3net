"""Development/confirmatory evaluation protocol and seed-level uncertainty."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_CONFIRMATORY_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
_DECISION_METRIC_PATTERN = re.compile(
    r"(?:"
    r"all_(?:sum|mean|llr|chain_llr)"
    r"|(?:exact|prefix_minK)_(?:sum|mean|llr)@[1-9][0-9]*"
    r"|prefix_minK_chain_llr@[1-9][0-9]*"
    r"|flash_(?:sum|mean|llr)@[1-9][0-9]*"
    r"|time_(?:sum|mean|llr)@(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)s"
    r")"
)


def canonical_sha256(payload: Mapping) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_decision_metric_name(name: str) -> str:
    """Validate an explicit acquisition-budget and aggregation metric name."""

    if name.startswith(("sum@", "mean@", "llr@", "chain_llr@")):
        raise ValueError(
            "Ambiguous legacy @K metrics are forbidden; use exact_, prefix_minK_, "
            "flash_, time_, or all_ semantics."
        )
    if _DECISION_METRIC_PATTERN.fullmatch(name) is None:
        raise ValueError(f"Unsupported decision metric {name!r}.")
    if name.startswith("time_"):
        seconds = float(name.rsplit("@", maxsplit=1)[1].removesuffix("s"))
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("time metric budget must be positive and finite.")
    return name


def frozen_truth_sha256(dataset: str, truth_by_unit: Mapping[object, object]) -> str:
    """Hash the ordered eligible unit/truth universe independently of preprocessing."""

    normalized: dict[str, object] = {}
    for raw_unit, truth in truth_by_unit.items():
        unit = str(raw_unit)
        if not unit or unit in normalized:
            raise ValueError("Frozen truth units must be non-empty and unique after normalization.")
        normalized[unit] = truth.item() if isinstance(truth, np.generic) else truth
    if not normalized:
        raise ValueError("Frozen truth universe cannot be empty.")
    return canonical_sha256(
        {
            "dataset": str(dataset),
            "units": [[unit, normalized[unit]] for unit in sorted(normalized)],
        }
    )


def source_tree_sha256(root: str | Path) -> str:
    """Hash executable project sources/configs, including uncommitted content."""

    root_path = Path(root).resolve()
    paths = [
        *sorted((root_path / "src").rglob("*.py")),
        *sorted((root_path / "experiments").rglob("*.py")),
        root_path / "pyproject.toml",
        root_path / "requirements.txt",
        root_path / "experiments" / "protocols" / "gtn_confirmatory_cohort_v1.json",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen source input is missing: {path}.")
        relative = path.relative_to(root_path).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def runtime_environment_sha256() -> str:
    """Bind confirmatory seeds to one Python/dependency runtime."""

    packages = ("numpy", "torch", "mne", "scikit-learn", "braindecode", "moabb")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return canonical_sha256(
        {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": versions,
        }
    )


def external_assets_sha256(assets: Mapping[str, str | Path | None]) -> str:
    """Hash optional checkpoints/calibration files by role, independent of mtime."""

    content_hashes: dict[str, str | None] = {}
    for role, raw_path in sorted(assets.items()):
        if raw_path is None:
            content_hashes[str(role)] = None
            continue
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Frozen external asset is missing: {path}.")
        content_hashes[str(role)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return canonical_sha256(content_hashes)


def validate_eligibility_manifest(
    path: str | Path,
    *,
    dataset: str,
    truth_by_unit: Mapping[object, object],
) -> dict:
    """Fail closed unless a source-level frozen cohort matches the loaded cache truth."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "n2p3net_eligibility_manifest/1":
        raise ValueError(f"Unsupported eligibility manifest schema in {manifest_path}.")
    if payload.get("dataset") != dataset:
        raise ValueError(f"Eligibility manifest dataset does not match {dataset!r}.")
    expected_count = int(payload.get("n_evaluation_units", -1))
    actual_count = len(truth_by_unit)
    actual_hash = frozen_truth_sha256(dataset, truth_by_unit)
    if expected_count != actual_count or payload.get("truth_universe_sha256") != actual_hash:
        raise ValueError(
            "Loaded cache truth differs from the independently frozen eligibility universe: "
            f"count={actual_count}/{expected_count}, sha256={actual_hash}/"
            f"{payload.get('truth_universe_sha256')}."
        )
    exposed = tuple(str(unit) for unit in payload.get("development_exposed_units", []))
    if len(exposed) != len(set(exposed)) or not set(exposed).issubset(
        {str(unit) for unit in truth_by_unit}
    ):
        raise ValueError("Eligibility manifest has invalid development_exposed_units.")
    normalized_truth = {str(unit): truth for unit, truth in truth_by_unit.items()}
    confirmatory_truth = {
        unit: truth for unit, truth in normalized_truth.items() if unit not in set(exposed)
    }
    confirmatory_dataset = payload.get("confirmatory_dataset_id")
    if exposed and (
        payload.get("n_confirmatory_units") != len(confirmatory_truth)
        or not isinstance(confirmatory_dataset, str)
        or payload.get("confirmatory_truth_sha256")
        != frozen_truth_sha256(confirmatory_dataset, confirmatory_truth)
    ):
        raise ValueError("Eligibility manifest confirmatory subset identity is invalid.")
    return payload


def confirmatory_units_from_manifest(
    manifest: Mapping,
    truth_by_unit: Mapping[object, object],
) -> tuple[str, ...]:
    """Return the unexposed confirmatory units after manifest validation."""

    if manifest.get("confirmatory_status") != "available_unexposed_cohort":
        raise ValueError(
            "This cohort has no statistically unexposed confirmatory units; run it only as "
            "locked development/replication evidence and use a new external cohort for SOTA claims."
        )
    exposed = {str(unit) for unit in manifest.get("development_exposed_units", [])}
    units = tuple(sorted(str(unit) for unit in truth_by_unit if str(unit) not in exposed))
    if len(units) != int(manifest.get("n_confirmatory_units", -1)):
        raise ValueError("Confirmatory unit count differs from its frozen manifest.")
    return units


def parse_seeds(value: str | Sequence[int] | None, *, mode: str) -> tuple[int, ...]:
    if value is None:
        seeds = DEFAULT_CONFIRMATORY_SEEDS if mode == "confirmatory" else (0,)
    elif isinstance(value, str):
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        seeds = tuple(int(seed) for seed in value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed list must be non-empty and unique.")
    return seeds


@dataclass(frozen=True)
class EvaluationProtocol:
    mode: str
    seeds: tuple[int, ...]
    primary_metric: str
    subject_limit: int | None
    benchmark: bool = False

    def validate(self) -> EvaluationProtocol:
        if self.mode not in ("development", "confirmatory"):
            raise ValueError("mode must be development or confirmatory.")
        validate_decision_metric_name(self.primary_metric)
        if self.mode == "confirmatory":
            if len(self.seeds) < 5:
                raise ValueError("confirmatory evaluation requires at least five unique seeds.")
            if self.subject_limit is not None:
                raise ValueError("confirmatory evaluation must use the complete locked cohort.")
            if self.benchmark:
                raise ValueError("benchmark mode cannot be marked confirmatory.")
        return self


def reserve_confirmatory_lock(
    lock_dir: Path,
    *,
    confirmatory_id: str,
    dataset_sha256: str,
    protocol: EvaluationProtocol,
    frozen_config: Mapping,
) -> Path:
    """Atomically pre-register a one-use confirmatory analysis identifier."""

    protocol.validate()
    if protocol.mode != "confirmatory":
        raise ValueError("locks are only used for confirmatory evaluation.")
    if not confirmatory_id.strip():
        raise ValueError("confirmatory_id is required.")
    payload = {
        "schema": "n2p3net_confirmatory_lock/1",
        "confirmatory_id": confirmatory_id,
        "dataset_sha256": dataset_sha256,
        "protocol": asdict(protocol),
        "frozen_config": dict(frozen_config),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in confirmatory_id)
    path = lock_dir / f"{safe_id}.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except FileExistsError as exc:
        raise RuntimeError(
            f"confirmatory id {confirmatory_id!r} is already locked at {path}; "
            "do not reuse final results for model selection."
        ) from exc
    return path


def validate_confirmatory_lock(
    path: str | Path,
    *,
    dataset_sha256: str,
    protocol_sha256: str,
    primary_metric: str,
    seed: int,
    runner: str,
    model: str,
) -> tuple[dict, str]:
    """Verify a child run against the immutable lock created by the wrapper."""

    lock_path = Path(path)
    raw = lock_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema") != "n2p3net_confirmatory_lock/1":
        raise ValueError(f"Unsupported confirmatory lock schema in {lock_path}.")
    stored_manifest_hash = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stored_manifest_hash != canonical_sha256(unsigned):
        raise ValueError(f"Confirmatory lock {lock_path} failed its integrity check.")
    protocol = payload.get("protocol", {})
    frozen = payload.get("frozen_config", {})
    expected = {
        "dataset_sha256": dataset_sha256,
        "primary_metric": primary_metric,
        "runner": runner,
        "model": model,
    }
    actual = {
        "dataset_sha256": payload.get("dataset_sha256"),
        "primary_metric": protocol.get("primary_metric"),
        "runner": frozen.get("runner"),
        "model": frozen.get("model"),
    }
    if actual != expected:
        raise ValueError(f"Confirmatory lock identity mismatch: expected={expected}, actual={actual}.")
    if protocol.get("mode") != "confirmatory" or int(seed) not in protocol.get("seeds", []):
        raise ValueError("Confirmatory lock does not authorize this mode/seed.")
    if canonical_sha256(frozen) != protocol_sha256:
        raise ValueError("protocol_sha256 does not match the lock's frozen configuration.")
    confirmatory_id = payload.get("confirmatory_id")
    if not isinstance(confirmatory_id, str) or not confirmatory_id:
        raise ValueError("Confirmatory lock lacks a valid confirmatory_id.")
    return payload, hashlib.sha256(raw).hexdigest()


def claim_confirmatory_seed(
    lock_path: str | Path,
    *,
    seed: int,
    run_identity: Mapping,
) -> Path:
    """Atomically consume one seed slot so a confirmatory result cannot be rerun."""

    path = Path(lock_path)
    claims = path.parent / f"{path.stem}.claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_path = claims / f"seed_{int(seed)}.json"
    payload = {
        "schema": "n2p3net_confirmatory_seed_claim/1",
        "lock_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": int(seed),
        "run_identity": dict(run_identity),
    }
    try:
        with claim_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Confirmatory seed {seed} was already consumed for lock {path}."
        ) from exc
    return claim_path


@dataclass(frozen=True)
class SeedUncertainty:
    mean_hit_rate: float
    seed_std: float
    hierarchical_ci_low: float
    hierarchical_ci_high: float
    n_seeds: int
    n_subjects: int
    n_bootstrap: int
    mean_coverage: float = 1.0


def aggregate_seed_subject_hits(
    hits_by_seed: Mapping[int, Mapping[object, float]],
    *,
    subject_universe: Sequence[object] | None = None,
    availability_by_seed: Mapping[int, Mapping[object, bool]] | None = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 2026,
) -> SeedUncertainty:
    """Aggregate crossed seed/subject outcomes with a hierarchical bootstrap."""

    if len(hits_by_seed) < 1:
        raise ValueError("at least one seed result is required.")
    seeds = tuple(sorted(hits_by_seed))
    expected = (
        {str(subject) for subject in subject_universe}
        if subject_universe is not None
        else {str(subject) for subject in hits_by_seed[seeds[0]]}
    )
    if not expected or (subject_universe is not None and len(expected) != len(subject_universe)):
        raise ValueError("The frozen subject universe must be non-empty and unique.")
    normalized_hits: dict[int, dict[str, float]] = {}
    for seed in seeds:
        values = {str(subject): float(hit) for subject, hit in hits_by_seed[seed].items()}
        if set(values) != expected:
            raise ValueError(
                f"Seed {seed} subject ids differ from the frozen universe; "
                f"missing={sorted(expected - set(values))[:5]}, "
                f"unknown={sorted(set(values) - expected)[:5]}."
            )
        normalized_hits[seed] = values
    ordered_subjects = tuple(sorted(expected, key=str))
    matrix = np.asarray(
        [[normalized_hits[seed][subject] for subject in ordered_subjects] for seed in seeds],
        dtype=float,
    )
    if not np.isfinite(matrix).all() or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("seed/subject ITT hit matrix must contain finite values in [0,1].")
    availability = np.ones_like(matrix)
    if availability_by_seed is not None:
        if set(availability_by_seed) != set(seeds):
            raise ValueError("Availability seed ids must exactly match hit seed ids.")
        for row, seed in enumerate(seeds):
            values = {
                str(subject): bool(value) for subject, value in availability_by_seed[seed].items()
            }
            if set(values) != expected:
                raise ValueError(f"Seed {seed} availability ids differ from the frozen universe.")
            availability[row] = [float(values[subject]) for subject in ordered_subjects]
            if np.any((availability[row] == 0.0) & (matrix[row] != 0.0)):
                raise ValueError("Unavailable evaluation units must contribute ITT hit=0.")
    seed_means = matrix.mean(axis=1)
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        seed_idx = rng.integers(0, len(seeds), size=len(seeds))
        subject_idx = rng.integers(0, len(ordered_subjects), size=len(ordered_subjects))
        bootstrap[index] = matrix[np.ix_(seed_idx, subject_idx)].mean()
    return SeedUncertainty(
        mean_hit_rate=float(matrix.mean()),
        seed_std=float(seed_means.std(ddof=1)) if len(seeds) > 1 else 0.0,
        hierarchical_ci_low=float(np.quantile(bootstrap, 0.025)),
        hierarchical_ci_high=float(np.quantile(bootstrap, 0.975)),
        n_seeds=len(seeds),
        n_subjects=len(ordered_subjects),
        n_bootstrap=int(n_bootstrap),
        mean_coverage=float(availability.mean()),
    )
