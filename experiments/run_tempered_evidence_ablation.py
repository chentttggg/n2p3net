"""Compare count-tempered all-evidence aggregation on frozen GTN ledgers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.decision import count_tempered_evidence_scores  # noqa: E402

LEDGER_PATTERN = re.compile(r"^k(?P<kernel>\d+)_seed(?P<seed>\d+)_blk(?P<block>\d+)\.jsonl\.gz$")
SCHEMA = "n2p3_tempered_evidence_ablation/2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unique_prediction(scores: np.ndarray, candidates: tuple[str, ...]) -> str | None:
    maximum = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, maximum, rtol=1e-12, atol=1e-12))
    return candidates[int(tied[0])] if len(tied) == 1 else None


def _validate_manifest(path: str | Path) -> tuple[dict[str, object], str]:
    manifest = _read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest must use schema {SCHEMA!r}")
    kernels = tuple(int(value) for value in manifest.get("kernels", ()))
    seeds = tuple(int(value) for value in manifest.get("seeds", ()))
    blocks = tuple(int(value) for value in manifest.get("blocks", ()))
    betas = tuple(float(value) for value in manifest.get("count_powers", ()))
    if kernels != (35, 65) or seeds != (20260828, 20260829, 20260830):
        raise ValueError("manifest kernels/seeds differ from the registered comparison")
    if blocks != (0, 1, 2, 3) or betas != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("manifest blocks/count powers differ from the registered comparison")
    if float(manifest.get("primary_count_power", -1.0)) != 0.5:
        raise ValueError("manifest primary_count_power must be 0.5")
    learned = manifest.get("learned_aggregator")
    if not isinstance(learned, dict) or learned.get("objective") != "candidate_listwise_ce":
        raise ValueError("manifest lacks the registered learned aggregator")
    if int(learned.get("epochs", 0)) < 1 or float(learned.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("learned aggregator training settings are invalid")
    return manifest, sha256_file(path)


def _block_subjects(directory: str | Path, blocks: tuple[int, ...]) -> dict[int, tuple[str, ...]]:
    output: dict[int, tuple[str, ...]] = {}
    for block in blocks:
        decoded = _read_json(Path(directory) / f"block_{block}.json")
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError(f"block_{block}.json must be a JSON string list")
        if len(decoded) != len(set(decoded)):
            raise ValueError(f"block_{block}.json contains duplicate subjects")
        output[block] = tuple(decoded)
    union = [subject for block in blocks for subject in output[block]]
    if len(union) != 245 or len(set(union)) != 245:
        raise ValueError("block manifests must be disjoint and cover 245 subjects")
    return output


def load_ledger_evidence(
    ledger_dir: str | Path,
    *,
    kernels: tuple[int, ...],
    seeds: tuple[int, ...],
    blocks: tuple[int, ...],
    subjects_by_block: dict[int, tuple[str, ...]],
) -> tuple[
    dict[tuple[int, int, str], dict[str, object]],
    tuple[str, ...],
    dict[str, int],
]:
    """Stream ledgers into per-arm subject/candidate sums and counts."""

    ledger_dir = Path(ledger_dir)
    expected = {
        (kernel, seed, block) for kernel in kernels for seed in seeds for block in blocks
    }
    paths: dict[tuple[int, int, int], Path] = {}
    for path in ledger_dir.glob("*.jsonl.gz"):
        match = LEDGER_PATTERN.match(path.name)
        if match is None:
            continue
        key = tuple(int(match.group(name)) for name in ("kernel", "seed", "block"))
        if key in expected:
            paths[key] = path
    if set(paths) != expected:
        raise ValueError(f"ledger set mismatch: missing={sorted(expected - set(paths))}")

    evidence: dict[tuple[int, int, str], dict[str, object]] = {}
    reference_counts: dict[str, dict[str, int]] | None = None
    subject_blocks: dict[str, int] = {}
    candidate_vocabulary: tuple[str, ...] | None = None

    for kernel, seed, block in sorted(expected):
        local: dict[str, dict[str, object]] = {}
        with gzip.open(paths[(kernel, seed, block)], "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    int(row["kernel"]) != kernel
                    or int(row["seed"]) != seed
                    or int(row["block"]) != block
                ):
                    raise ValueError("ledger row arm identity mismatch")
                subject = str(row["subject"])
                candidate = str(row["candidate"])
                target = str(row["target"])
                llr = float(row["llr_score"])
                if not np.isfinite(llr):
                    raise ValueError("ledger contains non-finite LLR")
                record = local.setdefault(
                    subject,
                    {
                        "truth": target,
                        "sums": defaultdict(float),
                        "counts": defaultdict(int),
                        "values": defaultdict(list),
                        "occurrences": defaultdict(list),
                    },
                )
                if record["truth"] != target:
                    raise ValueError(f"subject {subject!r} has mixed target labels")
                record["sums"][candidate] += llr
                record["counts"][candidate] += 1
                record["values"][candidate].append(llr)
                record["occurrences"][candidate].append(
                    int(row["available_occurrence_index"])
                )

        if set(local) != set(subjects_by_block[block]):
            raise ValueError(f"ledger subjects do not equal frozen block {block}")
        for subject, record in local.items():
            candidates = tuple(sorted(record["counts"]))
            if candidate_vocabulary is None:
                candidate_vocabulary = candidates
            if candidates != candidate_vocabulary or len(candidates) != 9:
                raise ValueError(f"subject {subject!r} lacks the fixed nine-candidate vocabulary")
            if record["truth"] not in candidates:
                raise ValueError(f"subject {subject!r} target is outside the candidate vocabulary")
            normalized = {
                "block": block,
                "truth": record["truth"],
                "sums": {candidate: float(record["sums"][candidate]) for candidate in candidates},
                "counts": {candidate: int(record["counts"][candidate]) for candidate in candidates},
                "values": {
                    candidate: tuple(float(value) for value in record["values"][candidate])
                    for candidate in candidates
                },
                "occurrences": {
                    candidate: tuple(
                        int(value) for value in record["occurrences"][candidate]
                    )
                    for candidate in candidates
                },
            }
            evidence[(kernel, seed, subject)] = normalized
            previous_block = subject_blocks.setdefault(subject, block)
            if previous_block != block:
                raise ValueError(f"subject {subject!r} moved between blocks")

    for kernel in kernels:
        for seed in seeds:
            current_counts = {
                subject: record["counts"]
                for (current_kernel, current_seed, subject), record in evidence.items()
                if current_kernel == kernel and current_seed == seed
            }
            if len(current_counts) != 245:
                raise ValueError(f"K{kernel}/seed{seed} does not contain all 245 subjects")
            if reference_counts is None:
                reference_counts = current_counts
            elif current_counts != reference_counts:
                raise ValueError("candidate occurrence counts differ across kernel/seed arms")

    assert candidate_vocabulary is not None and reference_counts is not None
    return evidence, candidate_vocabulary, subject_blocks


def score_subject(
    record: dict[str, object],
    candidates: tuple[str, ...],
    *,
    count_power: float,
) -> tuple[str | None, np.ndarray]:
    sums = np.asarray([record["sums"][candidate] for candidate in candidates], dtype=float)
    counts = np.asarray([record["counts"][candidate] for candidate in candidates], dtype=float)
    scores = count_tempered_evidence_scores(
        sums,
        counts,
        counts,
        count_power=count_power,
    )
    return _unique_prediction(scores, candidates), scores


class LearnedTemperedEvidence(nn.Module):
    """Shared four-parameter all-evidence decision model."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.weight_abs = nn.Parameter(torch.tensor(0.0))
        self.weight_occurrence = nn.Parameter(torch.tensor(0.0))
        self.raw_gain = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        occurrence: torch.Tensor,
        *,
        abs_center: torch.Tensor,
        abs_scale: torch.Tensor,
        score_scale: torch.Tensor,
    ) -> torch.Tensor:
        normalized_abs = (torch.log1p(values.abs()) - abs_center) / abs_scale
        weight_logits = (
            self.weight_abs * normalized_abs
            + self.weight_occurrence * (occurrence - 0.5)
        )
        weights = (F.softplus(weight_logits) + 1e-4) * mask
        weight_sum = weights.sum(dim=-1)
        squared_weight_sum = (weights * weights).sum(dim=-1)
        weighted_mean = (weights * (values / score_scale)).sum(dim=-1) / weight_sum.clamp_min(
            1e-8
        )
        effective_count = weight_sum.square() / squared_weight_sum.clamp_min(1e-8)
        beta = torch.sigmoid(self.raw_beta)
        gain = F.softplus(self.raw_gain) + 1e-4
        return gain * weighted_mean * effective_count.clamp_min(1.0).pow(beta)

    def parameters_record(self) -> dict[str, float]:
        return {
            "count_power": float(torch.sigmoid(self.raw_beta).detach().cpu()),
            "weight_abs": float(self.weight_abs.detach().cpu()),
            "weight_occurrence": float(self.weight_occurrence.detach().cpu()),
            "gain": float(F.softplus(self.raw_gain).detach().cpu() + 1e-4),
        }


def build_learned_tensors(
    evidence: dict[tuple[int, int, str], dict[str, object]],
    *,
    kernel: int,
    seeds: tuple[int, ...],
    subjects: tuple[str, ...],
    candidates: tuple[str, ...],
    subject_blocks: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    max_count = max(
        int(record["counts"][candidate])
        for (record_kernel, _, _), record in evidence.items()
        if record_kernel == kernel
        for candidate in candidates
    )
    n_rows = len(subjects) * len(seeds)
    values = np.zeros((n_rows, len(candidates), max_count), dtype=np.float32)
    occurrence = np.zeros_like(values)
    mask = np.zeros_like(values)
    targets = np.empty(n_rows, dtype=np.int64)
    row_subjects: list[str] = []
    row_seeds = np.empty(n_rows, dtype=np.int64)
    row_blocks = np.empty(n_rows, dtype=np.int64)
    row = 0
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    for subject in subjects:
        for seed in seeds:
            record = evidence[(kernel, seed, subject)]
            targets[row] = candidate_index[str(record["truth"])]
            row_subjects.append(subject)
            row_seeds[row] = seed
            row_blocks[row] = subject_blocks[subject]
            for candidate, candidate_col in candidate_index.items():
                candidate_values = np.asarray(record["values"][candidate], dtype=np.float32)
                candidate_occurrences = np.asarray(
                    record["occurrences"][candidate], dtype=np.float32
                )
                count = len(candidate_values)
                values[row, candidate_col, :count] = candidate_values
                mask[row, candidate_col, :count] = 1.0
                denominator = max(float(candidate_occurrences.max()), 1.0)
                occurrence[row, candidate_col, :count] = (
                    candidate_occurrences / denominator
                )
            row += 1
    return {
        "values": torch.as_tensor(values, device=device),
        "mask": torch.as_tensor(mask, device=device),
        "occurrence": torch.as_tensor(occurrence, device=device),
        "targets": torch.as_tensor(targets, device=device),
        "subjects": tuple(row_subjects),
        "seeds": row_seeds,
        "blocks": row_blocks,
    }


def fit_learned_tempered_evidence(
    tensors: dict[str, object],
    *,
    train_blocks: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[LearnedTemperedEvidence, dict[str, object]]:
    values = tensors["values"]
    mask = tensors["mask"]
    occurrence = tensors["occurrence"]
    targets = tensors["targets"]
    assert isinstance(values, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(occurrence, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    blocks = np.asarray(tensors["blocks"], dtype=np.int64)
    train_rows = torch.as_tensor(np.isin(blocks, train_blocks), device=values.device)
    train_values = values[train_rows]
    train_mask = mask[train_rows]
    train_occurrence = occurrence[train_rows]
    train_targets = targets[train_rows]
    observed_abs = train_values.abs()[train_mask.bool()]
    observed_log_abs = torch.log1p(observed_abs)
    abs_center = observed_log_abs.mean().detach()
    abs_scale = observed_log_abs.std(unbiased=False).clamp_min(1e-6).detach()
    score_scale = observed_abs.square().mean().sqrt().clamp_min(1e-6).detach()

    torch.manual_seed(20260831)
    model = LearnedTemperedEvidence().to(values.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            train_values,
            train_mask,
            train_occurrence,
            abs_center=abs_center,
            abs_scale=abs_scale,
            score_scale=score_scale,
        )
        loss = F.cross_entropy(logits, train_targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, {
        "train_rows": int(train_rows.sum().item()),
        "epochs": epochs,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "abs_center": float(abs_center.cpu()),
        "abs_scale": float(abs_scale.cpu()),
        "score_scale": float(score_scale.cpu()),
        "parameters": model.parameters_record(),
    }


def evaluate_learned_crossfit(
    evidence: dict[tuple[int, int, str], dict[str, object]],
    *,
    kernels: tuple[int, ...],
    seeds: tuple[int, ...],
    blocks: tuple[int, ...],
    subjects: tuple[str, ...],
    candidates: tuple[str, ...],
    subject_blocks: dict[str, int],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, object], dict[tuple[int, str], float]]:
    output: dict[str, object] = {}
    subject_correctness: dict[tuple[int, str], float] = {}
    for kernel in kernels:
        tensors = build_learned_tensors(
            evidence,
            kernel=kernel,
            seeds=seeds,
            subjects=subjects,
            candidates=candidates,
            subject_blocks=subject_blocks,
            device=device,
        )
        values = tensors["values"]
        mask = tensors["mask"]
        occurrence = tensors["occurrence"]
        targets = tensors["targets"]
        assert isinstance(values, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(occurrence, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        row_blocks = np.asarray(tensors["blocks"], dtype=np.int64)
        row_subjects = np.asarray(tensors["subjects"])
        folds = {}
        kernel_subject_values: dict[str, list[float]] = defaultdict(list)
        for heldout in blocks:
            train_blocks = tuple(block for block in blocks if block != heldout)
            model, training = fit_learned_tempered_evidence(
                tensors,
                train_blocks=train_blocks,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            test_rows_np = row_blocks == heldout
            test_rows = torch.as_tensor(test_rows_np, device=values.device)
            with torch.no_grad():
                logits = model(
                    values[test_rows],
                    mask[test_rows],
                    occurrence[test_rows],
                    abs_center=torch.tensor(training["abs_center"], device=values.device),
                    abs_scale=torch.tensor(training["abs_scale"], device=values.device),
                    score_scale=torch.tensor(training["score_scale"], device=values.device),
                )
                predictions = logits.argmax(dim=1)
                correct = predictions.eq(targets[test_rows]).float().cpu().numpy()
            test_subjects = row_subjects[test_rows_np]
            for subject in np.unique(test_subjects):
                kernel_subject_values[str(subject)].extend(
                    correct[test_subjects == subject].astype(float).tolist()
                )
            folds[str(heldout)] = {
                "test_rows": int(test_rows_np.sum()),
                "test_subjects": int(len(np.unique(test_subjects))),
                "operational_hit": float(correct.mean()),
                "training": training,
            }
        for subject in subjects:
            values_for_subject = kernel_subject_values[subject]
            if len(values_for_subject) != len(seeds):
                raise ValueError("learned cross-fit did not produce one result per EEG seed")
            subject_correctness[(kernel, subject)] = float(np.mean(values_for_subject))
        output[str(kernel)] = {
            "operational_hit_seed_mean": float(
                np.mean([subject_correctness[(kernel, subject)] for subject in subjects])
            ),
            "folds": folds,
        }
    return output, subject_correctness


def _paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> list[float]:
    output = np.empty(iterations, dtype=float)
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        indices = rng.integers(0, len(differences), size=(take, len(differences)))
        output[offset : offset + take] = differences[indices].mean(axis=1)
        offset += take
    return [float(value) for value in np.quantile(output, [0.025, 0.975])]


def _sign_flip_p(
    differences: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))
    exceed = 0
    offset = 0
    while offset < iterations:
        take = min(4000, iterations - offset)
        signs = rng.integers(0, 2, size=(take, len(differences)), dtype=np.int8) * 2 - 1
        exceed += int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= observed))
        offset += take
    return float((exceed + 1) / (iterations + 1))


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def analyze(args: argparse.Namespace) -> None:
    manifest, manifest_sha = _validate_manifest(args.manifest)
    kernels = tuple(int(value) for value in manifest["kernels"])
    seeds = tuple(int(value) for value in manifest["seeds"])
    blocks = tuple(int(value) for value in manifest["blocks"])
    betas = tuple(float(value) for value in manifest["count_powers"])
    primary_beta = float(manifest["primary_count_power"])
    iterations = int(manifest["bootstrap_iterations"])
    subjects_by_block = _block_subjects(args.block_manifest_dir, blocks)
    evidence, candidates, subject_blocks = load_ledger_evidence(
        args.ledger_dir,
        kernels=kernels,
        seeds=seeds,
        blocks=blocks,
        subjects_by_block=subjects_by_block,
    )
    subjects = tuple(sorted(subject_blocks))

    correctness: dict[tuple[int, int, float, str], float] = {}
    count_max_prediction: dict[tuple[int, int, float, str], float] = {}
    per_seed = []
    for kernel in kernels:
        for seed in seeds:
            for beta in betas:
                hits = 0
                max_count_predictions = 0
                for subject in subjects:
                    record = evidence[(kernel, seed, subject)]
                    predicted, _ = score_subject(record, candidates, count_power=beta)
                    hit = float(predicted is not None and predicted == record["truth"])
                    correctness[(kernel, seed, beta, subject)] = hit
                    counts = record["counts"]
                    maximum_count = max(counts.values())
                    favors_max = float(
                        predicted is not None and counts[predicted] == maximum_count
                    )
                    count_max_prediction[(kernel, seed, beta, subject)] = favors_max
                    hits += int(hit)
                    max_count_predictions += int(favors_max)
                per_seed.append(
                    {
                        "kernel": kernel,
                        "seed": seed,
                        "count_power": beta,
                        "hits": hits,
                        "operational_hit": hits / len(subjects),
                        "predicted_max_count_fraction": max_count_predictions / len(subjects),
                    }
                )

    subject_seed_mean: dict[tuple[int, float, str], float] = {}
    metrics: dict[str, dict[str, object]] = {}
    for kernel in kernels:
        kernel_metrics: dict[str, object] = {}
        for beta in betas:
            values = []
            max_count_values = []
            for subject in subjects:
                seed_mean = float(
                    np.mean([correctness[(kernel, seed, beta, subject)] for seed in seeds])
                )
                subject_seed_mean[(kernel, beta, subject)] = seed_mean
                values.append(seed_mean)
                max_count_values.append(
                    np.mean(
                        [count_max_prediction[(kernel, seed, beta, subject)] for seed in seeds]
                    )
                )
            kernel_metrics[str(beta)] = {
                "operational_hit_seed_mean": float(np.mean(values)),
                "predicted_max_count_fraction_seed_mean": float(np.mean(max_count_values)),
                "per_seed": {
                    str(seed): next(
                        row["operational_hit"]
                        for row in per_seed
                        if row["kernel"] == kernel
                        and row["seed"] == seed
                        and row["count_power"] == beta
                    )
                    for seed in seeds
                },
            }
        metrics[str(kernel)] = kernel_metrics

    # Four-fold subject-block cross-fit is exploratory. The primary beta remains fixed at 0.5.
    priority = (0.5, 0.25, 0.75, 0.0, 1.0)
    crossfit: dict[str, object] = {}
    for kernel in kernels:
        total_correct = 0.0
        selections = {}
        for heldout in blocks:
            development_subjects = [s for s in subjects if subject_blocks[s] != heldout]
            scores = {
                beta: float(
                    np.mean(
                        [subject_seed_mean[(kernel, beta, subject)] for subject in development_subjects]
                    )
                )
                for beta in betas
            }
            best_value = max(scores.values())
            selected = next(beta for beta in priority if np.isclose(scores[beta], best_value))
            heldout_subjects = [s for s in subjects if subject_blocks[s] == heldout]
            hits = float(
                sum(subject_seed_mean[(kernel, selected, subject)] for subject in heldout_subjects)
            )
            total_correct += hits
            selections[str(heldout)] = {
                "selected_count_power": selected,
                "development_hit": scores[selected],
                "heldout_subjects": len(heldout_subjects),
                "heldout_hits_seed_mean": hits,
                "heldout_operational_hit": hits / len(heldout_subjects),
            }
        crossfit[str(kernel)] = {
            "operational_hit": total_correct / len(subjects),
            "folds": selections,
        }

    learned_config = manifest["learned_aggregator"]
    assert isinstance(learned_config, dict)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    learned_crossfit, learned_subject_correctness = evaluate_learned_crossfit(
        evidence,
        kernels=kernels,
        seeds=seeds,
        blocks=blocks,
        subjects=subjects,
        candidates=candidates,
        subject_blocks=subject_blocks,
        device=device,
        epochs=int(learned_config["epochs"]),
        learning_rate=float(learned_config["learning_rate"]),
        weight_decay=float(learned_config["weight_decay"]),
    )

    contrasts: dict[str, dict[str, object]] = {}
    p_values: dict[str, float] = {}
    planned = [(kernel, primary_beta, baseline) for kernel in kernels for baseline in (0.0, 1.0)]
    for index, (kernel, left, right) in enumerate(planned):
        name = f"K{kernel}:beta{left}-beta{right}"
        differences = np.asarray(
            [
                subject_seed_mean[(kernel, left, subject)]
                - subject_seed_mean[(kernel, right, subject)]
                for subject in subjects
            ],
            dtype=float,
        )
        seed_deltas = {
            str(seed): float(
                np.mean(
                    [
                        correctness[(kernel, seed, left, subject)]
                        - correctness[(kernel, seed, right, subject)]
                        for subject in subjects
                    ]
                )
            )
            for seed in seeds
        }
        p_value = _sign_flip_p(
            differences,
            iterations=iterations,
            rng=np.random.default_rng(int(manifest["analysis_seed"]) + index * 10),
        )
        p_values[name] = p_value
        contrasts[name] = {
            "kernel": kernel,
            "left_count_power": left,
            "right_count_power": right,
            "operational_delta": float(differences.mean()),
            "paired_subject_bootstrap_ci95": _paired_bootstrap_ci(
                differences,
                iterations=iterations,
                rng=np.random.default_rng(int(manifest["analysis_seed"]) + index * 10 + 1),
            ),
            "paired_sign_flip_p": p_value,
            "delta_by_seed": seed_deltas,
        }
    adjusted = _holm_adjust(p_values)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value

    learned_contrasts: dict[str, dict[str, object]] = {}
    learned_p_values: dict[str, float] = {}
    comparison_index = 0
    for kernel in kernels:
        for baseline in (0.0, 0.5, 1.0):
            name = f"K{kernel}:learned-beta{baseline}"
            differences = np.asarray(
                [
                    learned_subject_correctness[(kernel, subject)]
                    - subject_seed_mean[(kernel, baseline, subject)]
                    for subject in subjects
                ],
                dtype=float,
            )
            p_value = _sign_flip_p(
                differences,
                iterations=iterations,
                rng=np.random.default_rng(
                    int(manifest["analysis_seed"]) + 100 + comparison_index * 10
                ),
            )
            learned_p_values[name] = p_value
            learned_contrasts[name] = {
                "kernel": kernel,
                "baseline_count_power": baseline,
                "operational_delta": float(differences.mean()),
                "paired_subject_bootstrap_ci95": _paired_bootstrap_ci(
                    differences,
                    iterations=iterations,
                    rng=np.random.default_rng(
                        int(manifest["analysis_seed"]) + 101 + comparison_index * 10
                    ),
                ),
                "paired_sign_flip_p": p_value,
            }
            comparison_index += 1
    learned_adjusted = _holm_adjust(learned_p_values)
    for name, value in learned_adjusted.items():
        learned_contrasts[name]["holm_adjusted_p"] = value

    reference = _read_json(args.reference_analysis)
    if not isinstance(reference, dict) or not isinstance(reference.get("kernels"), dict):
        raise ValueError("reference analysis lacks kernel metrics")
    reproduction = {}
    for kernel in kernels:
        expected = float(reference["kernels"][str(kernel)]["raw_all_operational_hit_seed_mean"])
        observed = float(metrics[str(kernel)]["1.0"]["operational_hit_seed_mean"])
        if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"beta=1 failed to reproduce K{kernel} raw-all: {observed} != {expected}"
            )
        reproduction[str(kernel)] = {"expected": expected, "observed": observed}

    result = {
        "schema": "n2p3_tempered_evidence_ablation_result/2",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "ledger_directory": str(Path(args.ledger_dir).resolve()),
        "requested_subjects": len(subjects),
        "candidate_vocabulary": list(candidates),
        "formula": "weighted_mean * effective_count ** count_power",
        "primary_count_power": primary_beta,
        "metrics": metrics,
        "per_seed_metrics": per_seed,
        "planned_contrasts": contrasts,
        "crossfit_exploratory": crossfit,
        "learned_crossfit": learned_crossfit,
        "learned_contrasts": learned_contrasts,
        "learned_device": str(device),
        "raw_sum_reproduction": reproduction,
        "inference_scope": (
            "paired subject inference conditional on the frozen three seeds, checkpoints, "
            "GTN cohort, and v4 trial ledgers"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "metrics": metrics,
                "crossfit": crossfit,
                "learned_crossfit": learned_crossfit,
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", required=True)
    parser.add_argument("--block-manifest-dir", required=True)
    parser.add_argument("--reference-analysis", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
