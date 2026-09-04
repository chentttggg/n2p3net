"""Fit AJD congruence whitening (ACW) on explicit per-domain feature matrices.

Each ``--source`` points to an ``.npz`` containing a finite two-dimensional
``features`` array (or another key selected with ``--feature-key``).  Raw EEG
``EpochDataset`` caches are deliberately not treated as feature matrices here;
callers must first apply and record an explicit domain-specific feature lift.
This is NOT the paper GALIA (Altindis et al. 2023); it is the local shared-Q
AJD contract whitening variant, used only for auxiliary coordinates/decoders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.domain_alignment import AcwConfig, AjdCongruenceWhitening  # noqa: E402


def _source_spec(value: str) -> tuple[str, Path]:
    domain, separator, path = value.partition("=")
    if not separator or not domain.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--source must use DOMAIN=features.npz")
    return domain.strip(), Path(path.strip())


def _load_features(path: Path, key: str) -> np.ndarray:
    resolved = path.resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"{resolved} lacks feature array {key!r}.")
        values = np.asarray(archive[key], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"{resolved}:{key} must be a two-dimensional feature matrix; "
            f"raw epoch tensors require an explicit feature lift first, got {values.shape}."
        )
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source_spec, required=True)
    parser.add_argument("--feature-key", default="features")
    parser.add_argument("--method", choices=("ajd_pham", "rjd", "uwedge"), default="ajd_pham")
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--aligned-output",
        type=Path,
        default=None,
        help="Optional .npz containing concatenated aligned_features and domain_ids.",
    )
    parser.add_argument(
        "--target-calibration",
        type=Path,
        default=None,
        help="Optional feature .npz for target calibration fast alignment.",
    )
    parser.add_argument(
        "--target-domain",
        default="target",
        help="Domain label used by the fast-aligned target transform.",
    )
    parser.add_argument(
        "--fast-fit-scope",
        choices=("target_calibration", "source_weighted"),
        default="target_calibration",
    )
    parser.add_argument(
        "--fast-output",
        type=Path,
        default=None,
        help="Optional .npz containing fast-aligned target features.",
    )

    args = parser.parse_args(argv)
    if len(args.source) < 2:
        parser.error("ACW requires at least two --source values")
    domains_seen = [domain for domain, _ in args.source]
    if len(set(domains_seen)) != len(domains_seen):
        parser.error("source domains must be unique")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end with .json")
    if args.aligned_output is not None and args.aligned_output.suffix.lower() != ".npz":
        parser.error("--aligned-output must end with .npz")
    domains = {domain: _load_features(path, args.feature_key) for domain, path in args.source}
    adapter = AjdCongruenceWhitening(
        AcwConfig(
            method=args.method,
            regularization=args.regularization,
            tolerance=args.tolerance,
            max_iter=args.max_iter,
        )
    )
    state = adapter.fit(domains)
    if (args.target_calibration is None) != (args.fast_output is None):
        parser.error("--target-calibration and --fast-output must be supplied together")
    if args.target_calibration is not None and args.fast_output is not None:
        if args.fast_output.suffix.lower() != ".npz":
            parser.error("--fast-output must end with .npz")
        target = _load_features(args.target_calibration, args.feature_key)
        fast = state.fast_alignment(
            target,
            domain_id=args.target_domain,
            fit_scope=args.fast_fit_scope,
            regularization=args.regularization,
        )
        fast_output = args.fast_output.resolve(strict=False)
        if fast_output.exists():
            raise FileExistsError(f"fast-aligned output already exists: {fast_output}")
        aligned_target = state.transform_fast(target, fast)
        fast_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fast_output,
            schema=np.asarray("n2p3_aligned_feature_matrix/1"),
            aligned_features=aligned_target.astype(np.float32),
            domain_ids=np.repeat(args.target_domain, len(target)),
            acw_state_digest=np.asarray(state.state_digest),
            fast_alignment_state_digest=np.asarray(fast.record()["state_digest"]),
        )
        fast_record = fast.record()
    else:
        fast_record = None

    output = args.output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"ACW output already exists: {output}")
    aligned_output = (
        None if args.aligned_output is None else args.aligned_output.resolve(strict=False)
    )
    if aligned_output is not None and aligned_output.exists():
        raise FileExistsError(f"aligned ACW output already exists: {aligned_output}")
    record = {
        "schema": "n2p3_acw_fit/1",
        "state": state.record(),
        "sources": {
            domain: {
                "path": str(path.resolve(strict=True)),
                "rows": int(len(domains[domain])),
                "feature_dim": int(domains[domain].shape[1]),
            }
            for domain, path in args.source
        },
        "fast_alignment": fast_record,
        "feature_key": args.feature_key,
        "fit_scope": "source_only",
        "raw_epoch_input": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if aligned_output is not None:
        aligned = np.concatenate(
            [state.transform(domains[domain], domain) for domain, _ in args.source], axis=0
        )
        domain_ids = np.concatenate(
            [np.repeat(domain, len(domains[domain])) for domain, _ in args.source]
        )
        aligned_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            aligned_output,
            schema=np.asarray("n2p3_aligned_feature_matrix/1"),
            aligned_features=aligned.astype(np.float32),
            domain_ids=domain_ids,
            acw_state_digest=np.asarray(state.state_digest),
        )
    print(f"[acw] state={output} feature_dim={state.feature_dim} domains={len(state.domain_order)}")


if __name__ == "__main__":
    main()
