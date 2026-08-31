"""Audit finite-input temporal receptive fields for the compact N2P3-Net trunk.

This is an analysis tool, not a model configuration entry point. It reports
structural support relative to the cached model tensor. Continuous IIR filtering,
FFT resampling, and baseline subtraction have separate source-domain support.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.n2p3net import (  # noqa: E402
    BROAD_REFERENCE_N2P3_ARCHITECTURE,
    TUNED_FULL_UNFOLD_ARCHITECTURE,
    stacked_temporal_receptive_field_samples,
)

AUDIT_SCHEMA = "n2p3net_receptive_field_audit/1"


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    temporal_kernel: int
    pool_size: int
    branch_kernels: tuple[int, ...]
    readout_pool_size: int


def same_padded_dependencies(dependencies: np.ndarray, kernel_size: int) -> np.ndarray:
    """Propagate a boolean dependency matrix through odd, stride-1 same convolution."""

    dependencies = np.asarray(dependencies, dtype=bool)
    if dependencies.ndim != 2 or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("dependencies must be 2-D and kernel_size must be positive and odd.")
    half = kernel_size // 2
    return np.stack(
        [
            dependencies[max(0, index - half) : min(len(dependencies), index + half + 1)].any(
                axis=0
            )
            for index in range(len(dependencies))
        ]
    )


def valid_pool_dependencies(
    dependencies: np.ndarray,
    kernel_size: int,
    *,
    stride: int | None = None,
) -> np.ndarray:
    """Propagate dependencies through a no-padding pooling layer."""

    dependencies = np.asarray(dependencies, dtype=bool)
    stride = kernel_size if stride is None else int(stride)
    if dependencies.ndim != 2 or kernel_size < 1 or stride < 1:
        raise ValueError("dependencies must be 2-D and pool geometry must be positive.")
    n_outputs = (len(dependencies) - kernel_size) // stride + 1
    if n_outputs < 1:
        raise ValueError("pool kernel is wider than its input.")
    return np.stack(
        [
            dependencies[index * stride : index * stride + kernel_size].any(axis=0)
            for index in range(n_outputs)
        ]
    )


def branch_dependencies(
    n_times: int,
    *,
    temporal_kernel: int,
    pool_size: int,
    branch_kernel: int,
) -> np.ndarray:
    """Return ``(T_feature,T_input)`` cached-tensor support for one MST branch."""

    source = np.eye(n_times, dtype=bool)
    after_temporal = same_padded_dependencies(source, temporal_kernel)
    after_pool = valid_pool_dependencies(after_temporal, pool_size)
    return same_padded_dependencies(after_pool, branch_kernel)


def feature_centers_ms(
    n_features: int,
    *,
    sample_rate_hz: float,
    tmin_ms: float,
    pool_size: int,
) -> np.ndarray:
    """Return input-domain centers for post-ST-pool feature coordinates."""

    return tmin_ms + (
        np.arange(n_features, dtype=float) * pool_size + (pool_size - 1) / 2.0
    ) * (1000.0 / sample_rate_hz)


def _support_record(
    support: np.ndarray,
    *,
    sample_rate_hz: float,
    tmin_ms: float,
) -> dict[str, int | float | bool | None]:
    indices = np.flatnonzero(np.asarray(support, dtype=bool))
    if not len(indices):
        return {
            "start_sample": None,
            "end_sample": None,
            "n_samples": 0,
            "start_ms": None,
            "end_ms": None,
            "whole_cached_epoch": False,
        }
    start = int(indices[0])
    end = int(indices[-1])
    return {
        "start_sample": start,
        "end_sample": end,
        "n_samples": int(len(indices)),
        "start_ms": float(tmin_ms + start * 1000.0 / sample_rate_hz),
        "end_ms": float(tmin_ms + end * 1000.0 / sample_rate_hz),
        "whole_cached_epoch": bool(len(indices) == len(support)),
    }


def _rows_for_mask(
    dependencies: np.ndarray,
    mask: np.ndarray,
    *,
    sample_rate_hz: float,
    tmin_ms: float,
) -> dict[str, int | float | bool | None]:
    return _support_record(
        dependencies[np.asarray(mask, dtype=bool)].any(axis=0),
        sample_rate_hz=sample_rate_hz,
        tmin_ms=tmin_ms,
    )


def summarize_architecture(
    spec: ArchitectureSpec,
    *,
    n_times: int,
    sample_rate_hz: float,
    tmin_ms: float,
    reference_window_ms: tuple[float, float] = (-200.0, 0.0),
    evidence_window_ms: tuple[float, float] = (250.0, 600.0),
    latency_offsets_ms: tuple[float, ...] = (-100.0, -50.0, 0.0, 50.0, 100.0),
) -> tuple[dict[str, object], dict[int, np.ndarray]]:
    """Summarize branch, readout-bin, and LMBC structural support."""

    branch_summaries: dict[str, object] = {}
    dependency_matrices: dict[int, np.ndarray] = {}
    for branch_kernel in spec.branch_kernels:
        dependencies = branch_dependencies(
            n_times,
            temporal_kernel=spec.temporal_kernel,
            pool_size=spec.pool_size,
            branch_kernel=branch_kernel,
        )
        dependency_matrices[branch_kernel] = dependencies
        centers = feature_centers_ms(
            len(dependencies),
            sample_rate_hz=sample_rate_hz,
            tmin_ms=tmin_ms,
            pool_size=spec.pool_size,
        )
        theoretical = stacked_temporal_receptive_field_samples(
            spec.temporal_kernel,
            spec.pool_size,
            branch_kernel,
        )
        feature_records = []
        for index, support in enumerate(dependencies):
            feature_records.append(
                {
                    "output_index": index,
                    "center_ms": float(centers[index]),
                    **_support_record(
                        support,
                        sample_rate_hz=sample_rate_hz,
                        tmin_ms=tmin_ms,
                    ),
                }
            )

        readout_dependencies = valid_pool_dependencies(
            dependencies,
            spec.readout_pool_size,
        )
        readout_centers = np.asarray(
            [
                centers[
                    index
                    * spec.readout_pool_size : (index + 1) * spec.readout_pool_size
                ].mean()
                for index in range(len(readout_dependencies))
            ]
        )
        readout_records = [
            {
                "output_index": index,
                "center_ms": float(readout_centers[index]),
                **_support_record(
                    support,
                    sample_rate_hz=sample_rate_hz,
                    tmin_ms=tmin_ms,
                ),
            }
            for index, support in enumerate(readout_dependencies)
        ]

        reference_mask = (centers >= reference_window_ms[0]) & (
            centers < reference_window_ms[1]
        )
        reference_support = dependencies[reference_mask].any(axis=0)
        latency_candidates = []
        for offset in latency_offsets_ms:
            candidate_mask = (centers >= evidence_window_ms[0] + offset) & (
                centers < evidence_window_ms[1] + offset
            )
            candidate_support = dependencies[candidate_mask].any(axis=0)
            latency_candidates.append(
                {
                    "offset_ms": float(offset),
                    "feature_indices": np.flatnonzero(candidate_mask).astype(int).tolist(),
                    "candidate": _support_record(
                        candidate_support,
                        sample_rate_hz=sample_rate_hz,
                        tmin_ms=tmin_ms,
                    ),
                    "contrast": _support_record(
                        reference_support | candidate_support,
                        sample_rate_hz=sample_rate_hz,
                        tmin_ms=tmin_ms,
                    ),
                }
            )

        finite_widths = dependencies.sum(axis=1)
        branch_summaries[str(branch_kernel)] = {
            "branch_kernel": branch_kernel,
            "theoretical_rf_samples": theoretical,
            "theoretical_rf_span_ms": (theoretical - 1) * 1000.0 / sample_rate_hz,
            "finite_support_min_samples": int(finite_widths.min()),
            "finite_support_mean_samples": float(finite_widths.mean()),
            "finite_support_median_samples": float(np.median(finite_widths)),
            "finite_support_max_samples": int(finite_widths.max()),
            "mean_support_fraction_of_theoretical": float(finite_widths.mean() / theoretical),
            "positions_touching_left_padding": np.flatnonzero(dependencies[:, 0])
            .astype(int)
            .tolist(),
            "positions_touching_right_padding": np.flatnonzero(dependencies[:, -1])
            .astype(int)
            .tolist(),
            "positions_covering_whole_cached_epoch": np.flatnonzero(
                finite_widths == n_times
            )
            .astype(int)
            .tolist(),
            "full_theoretical_support_positions": np.flatnonzero(
                finite_widths == theoretical
            ).astype(int).tolist(),
            "feature_supports": feature_records,
            "ms_flatten_theoretical_rf_samples": theoretical
            + (spec.readout_pool_size - 1) * spec.pool_size,
            "ms_flatten_supports": readout_records,
            "final_logit_union": _support_record(
                dependencies.any(axis=0),
                sample_rate_hz=sample_rate_hz,
                tmin_ms=tmin_ms,
            ),
            "lmbc": {
                "reference_feature_indices": np.flatnonzero(reference_mask).astype(int).tolist(),
                "reference": _support_record(
                    reference_support,
                    sample_rate_hz=sample_rate_hz,
                    tmin_ms=tmin_ms,
                ),
                "candidates": latency_candidates,
            },
        }

    return (
        {
            "spec": asdict(spec),
            "feature_centers_ms": feature_centers_ms(
                next(iter(dependency_matrices.values())).shape[0],
                sample_rate_hz=sample_rate_hz,
                tmin_ms=tmin_ms,
                pool_size=spec.pool_size,
            ).tolist(),
            "branches": branch_summaries,
        },
        dependency_matrices,
    )


def _write_csv(path: Path, summaries: dict[str, object]) -> None:
    fields = (
        "architecture",
        "stage",
        "temporal_kernel",
        "branch_kernel",
        "output_index",
        "center_ms",
        "theoretical_rf_samples",
        "start_sample",
        "end_sample",
        "n_samples",
        "start_ms",
        "end_ms",
        "whole_cached_epoch",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for architecture_name, architecture in summaries.items():
            spec = architecture["spec"]
            for branch in architecture["branches"].values():
                common = {
                    "architecture": architecture_name,
                    "temporal_kernel": spec["temporal_kernel"],
                    "branch_kernel": branch["branch_kernel"],
                }
                for stage, theoretical_key, records_key in (
                    ("branch_feature", "theoretical_rf_samples", "feature_supports"),
                    (
                        "ms_flatten_bin",
                        "ms_flatten_theoretical_rf_samples",
                        "ms_flatten_supports",
                    ),
                ):
                    for record in branch[records_key]:
                        writer.writerow(
                            {
                                **common,
                                "stage": stage,
                                "theoretical_rf_samples": branch[theoretical_key],
                                **record,
                            }
                        )


def _write_figure(
    path: Path,
    matrices: dict[str, dict[int, np.ndarray]],
    summaries: dict[str, object],
    *,
    sample_rate_hz: float,
    tmin_ms: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    names = list(matrices)
    kernels = list(next(iter(matrices.values())))
    fig, axes = plt.subplots(
        len(names),
        len(kernels),
        figsize=(10.0, 7.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(len(names), len(kernels))
    raw_times = tmin_ms + np.arange(next(iter(next(iter(matrices.values())).values())).shape[1]) * (
        1000.0 / sample_rate_hz
    )
    cmap = ListedColormap(["#F7F7F7", "#0072B2"])
    for row, name in enumerate(names):
        centers = np.asarray(summaries[name]["feature_centers_ms"])
        for column, kernel in enumerate(kernels):
            ax = axes[row, column]
            matrix = matrices[name][kernel].astype(float)
            ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=(raw_times[0], raw_times[-1], centers[0], centers[-1]),
            )
            ax.contour(raw_times, centers, matrix, levels=[0.5], colors="black", linewidths=0.45)
            for marker, linestyle in ((0.0, "-"), (250.0, "--"), (600.0, "--")):
                ax.axvline(marker, color="#D55E00", linestyle=linestyle, linewidth=0.9)
            branch = summaries[name]["branches"][str(kernel)]
            ax.set_title(
                f"{name}, branch k={kernel}\n"
                f"theoretical span={branch['theoretical_rf_span_ms']:.1f} ms"
            )
            ax.set_xlabel("Cached input time (ms)")
            ax.set_ylabel("Feature center (ms)")
    fig.suptitle("Finite cached-tensor support of N2P3-Net temporal branches")
    fig.savefig(path, dpi=220, facecolor="white")
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)

    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )

    from PIL import Image

    with Image.open(path) as raster:
        raster.convert("RGB").save(path, dpi=(220, 220))


def fft_resampling_impulse_diagnostic() -> dict[str, int | float]:
    """Record the global-support counterexample for the executable FFT resampler."""

    from mne.filter import resample

    source = np.zeros(513, dtype=np.float64)
    source[len(source) // 2] = 1.0
    output = resample(
        source,
        down=4.0,
        method="fft",
        npad="auto",
        window="auto",
        pad="edge",
        verbose=False,
    )
    return {
        "source_samples": len(source),
        "output_samples": len(output),
        "nonzero_output_samples": int(np.count_nonzero(output)),
        "output_samples_above_1e_6": int(np.count_nonzero(np.abs(output) > 1e-6)),
        "max_abs_output": float(np.max(np.abs(output))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "json": output_dir / "receptive_field_support.json",
        "csv": output_dir / "receptive_field_support.csv",
        "png": output_dir / "receptive_field_support.png",
    }
    expected = [*outputs.values(), outputs["png"].with_suffix(".svg")]
    existing = [path for path in expected if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    specs = (
        ArchitectureSpec(
            "K65",
            BROAD_REFERENCE_N2P3_ARCHITECTURE.temporal_kernel_size,
            BROAD_REFERENCE_N2P3_ARCHITECTURE.st_pool_size,
            BROAD_REFERENCE_N2P3_ARCHITECTURE.mst_kernel_sizes,
            BROAD_REFERENCE_N2P3_ARCHITECTURE.mst_pool_size,
        ),
        ArchitectureSpec(
            "K35",
            TUNED_FULL_UNFOLD_ARCHITECTURE.temporal_kernel_size,
            TUNED_FULL_UNFOLD_ARCHITECTURE.st_pool_size,
            TUNED_FULL_UNFOLD_ARCHITECTURE.mst_kernel_sizes,
            TUNED_FULL_UNFOLD_ARCHITECTURE.mst_pool_size,
        ),
    )
    summaries: dict[str, object] = {}
    matrices: dict[str, dict[int, np.ndarray]] = {}
    for spec in specs:
        summary, architecture_matrices = summarize_architecture(
            spec,
            n_times=128,
            sample_rate_hz=128.0,
            tmin_ms=-200.0,
        )
        summaries[spec.name] = summary
        matrices[spec.name] = architecture_matrices

    payload = {
        "schema": AUDIT_SCHEMA,
        "scope": "structural support relative to the cached 128-point model tensor",
        "not_in_scope": [
            "continuous IIR source-domain support",
            "epoch-domain FFT resampling support",
            "baseline-reference coupling",
            "trained effective receptive field",
        ],
        "input": {
            "n_times": 128,
            "sample_rate_hz": 128.0,
            "tmin_ms": -200.0,
            "right_endpoint": "exclusive",
        },
        "preprocessing_diagnostics": {
            "fft_resampling_impulse": fft_resampling_impulse_diagnostic(),
            "mean_only_baseline_cache_indices": list(range(26)),
            "mean_only_baseline_dependency": (
                "Every baseline-corrected cache sample subtracts the mean of indices 0..25."
            ),
            "continuous_iir_theoretical_support": "infinite",
        },
        "architectures": summaries,
        "figure_alt": (
            "Four heatmaps compare K65 and K35 branch support. K65's long branch reaches the "
            "entire cached epoch at central feature positions; K35 narrows both branches but its "
            "long branch remains broad. Orange vertical lines mark stimulus onset and 250/600 ms."
        ),
    }
    outputs["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(outputs["csv"], summaries)
    _write_figure(
        outputs["png"],
        matrices,
        summaries,
        sample_rate_hz=128.0,
        tmin_ms=-200.0,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
