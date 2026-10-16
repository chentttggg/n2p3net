"""Participant-paired development contrasts with fixed seed and study scopes."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from numbers import Real

import numpy as np

from stats.hierarchical import paired_interval

ANALYSIS_SEED = 20260907
BOOTSTRAP_ITERATIONS = 10000
INFERENCE_SCOPE = "conditional_on_fitted_sources_and_fixed_development_studies"


def standard_training_contrasts(stage: str) -> dict[str, dict[str, float]]:
    if stage == "P2":
        return {"full_minus_scratch": {"p2_full_fine": 1, "p2_scratch": -1},
                "full_minus_lda": {"p2_full_fine": 1, "p2_waveform_lda": -1}}
    if stage == "P1":
        contrasts = {
            "spatial": {"p1_s1t0": .5, "p1_s1t1": .5, "p1_s0t0": -.5, "p1_s0t1": -.5},
            "temporal": {"p1_s0t1": .5, "p1_s1t1": .5, "p1_s0t0": -.5, "p1_s1t0": -.5},
            "spatial_x_temporal": {"p1_s1t1": 1, "p1_s0t0": 1, "p1_s1t0": -1, "p1_s0t1": -1},
            "full_minus_scratch": {"p1_full_fine": 1, "p1_scratch": -1},
            "full_minus_lda": {"p1_full_fine": 1, "p1_waveform_lda": -1},
            "affine_minus_frozen_head": {"p1_affine_head": 1, "p1_s0t0": -1},
            "residual_minus_head": {"p1_residual_head": 1, "p1_s0t0": -1},
        }
        for arm in ("p1_s0t0", "p1_s1t0", "p1_s0t1", "p1_s1t1", "p1_affine_head", "p1_residual_head"):
            contrasts[f"{arm}_minus_full"] = {arm: 1, "p1_full_fine": -1}
            contrasts[f"{arm}_minus_scratch"] = {arm: 1, "p1_scratch": -1}
        return contrasts
    if stage == "ES":
        contrasts = {}
        for treatment in ("full", "head", "scratch"):
            for left, right in (("full_trajectory", "fixed"), ("patience", "fixed"), ("patience", "full_trajectory")):
                contrasts[f"{treatment}_{left}_minus_{right}"] = {f"es_{treatment}_{left}": 1, f"es_{treatment}_{right}": -1}
        for mode in ("fixed", "full_trajectory", "patience"):
            contrasts[f"{mode}_full_minus_scratch"] = {f"es_full_{mode}": 1, f"es_scratch_{mode}": -1}
            contrasts[f"{mode}_full_minus_lda"] = {f"es_full_{mode}": 1, "es_waveform_lda": -1}
        return contrasts
    if stage == "compact":
        return {
            "full_minus_scratch": {"compact_full_fine": 1, "compact_scratch": -1},
            "full_minus_lda": {"compact_full_fine": 1, "compact_waveform_lda": -1},
            "source_statistics_minus_full": {"compact_source_statistics": 1, "compact_full_fine": -1},
            "frozen_bn_buffers_minus_full": {"compact_frozen_bn_buffers": 1, "compact_full_fine": -1},
            "head_only_minus_full": {"compact_head_only": 1, "compact_full_fine": -1},
        }
    if stage == "E1":
        arms = {(i, b, w): f"e1_i{i}b{b}w{w}" for i, b, w in product((0, 1), repeat=3)}
        output = {}
        for name, axes in (("I", (0,)), ("B", (1,)), ("W", (2,)),
                           ("IxB", (0, 1)), ("BxW", (1, 2)), ("IxW", (0, 2))):
            output[name] = {arm: float(np.prod([2 * cell[axis] - 1 for axis in axes])) / (2 ** (3 - len(axes)))
                            for cell, arm in arms.items()}
        output.update({
            "online_minus_scratch_no_dropout": {"e1_online_bn": 1, "e1_scratch_no_dropout": -1},
            "dropout_minus_scratch_dropout": {"e1_dropout": 1, "e1_scratch_dropout": -1},
            "dropout_bridge": {"e1_dropout": 1, "e1_online_bn": -1},
            "bn_affine_b0": {"e1_affine_b0": 1, "e1_i1b0w1": -1},
            "bn_affine_b1": {"e1_affine_b1": 1, "e1_i1b1w1": -1},
        })
        return output
    if stage == "E3":
        return {
            "five_minus_no_transfer": {"five_candidates": 1, "no_transfer": -1},
            "five_minus_transfer_only": {"five_candidates": 1, "transfer_only": -1},
            "five_minus_legacy": {"five_candidates": 1, "legacy_product": -1},
            "transfer_only_minus_no_transfer": {"transfer_only": 1, "no_transfer": -1},
        }
    if stage == "E0":
        return {
            "full_fine_minus_scratch": {"full_fine": 1, "target_only": -1},
            "full_fine_minus_lda": {"full_fine": 1, "shrinkage_lda": -1},
        }
    raise ValueError("this stage needs an explicit contrast declaration")


def standard_s2_contrasts(source_arms: Sequence[str] | Mapping[str, Sequence[str]]) -> dict:
    """Use declared source-arm identities, retaining each single-source result."""
    if isinstance(source_arms, Mapping):
        merged = {}
        for study, arms in source_arms.items():
            for name, weights in standard_s2_contrasts(arms).items():
                merged.setdefault(name, {"by_study": {}})["by_study"][study] = weights
        return merged
    output = {}
    for dose in ("full", "dose"):
        prefix = f"S2_{dose}_"
        selected = tuple(arm for arm in source_arms if arm.startswith(prefix + "single_"))
        multi = prefix + "multi"
        if multi not in source_arms:
            continue
        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError("S2 contrasts require exactly three explicitly identified single-source arms")
        output[f"{dose}_multi_minus_single_mean"] = {multi: 1.0, **{arm: -1.0 / 3 for arm in selected}}
        for arm in selected:
            output[f"{dose}_multi_minus_{arm}"] = {multi: 1.0, arm: -1.0}
    if not output:
        raise ValueError("no complete declared S2 multi/single source comparison exists")
    return output


def _declared_values(values: Sequence, name: str) -> tuple:
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be a nonempty unique declaration")
    return result


def resolve_seed_pairs(source_seeds, target_seeds, *, seed_pairs=None, mode="crossed") -> tuple:
    """Resolve the planned replicate relation, never infer it from completed rows."""
    source = _declared_values(source_seeds, "source_seeds")
    target = _declared_values(target_seeds, "target_seeds")
    if any(type(value) is not int for value in (*source, *target)):
        raise ValueError("seed declarations require integers")
    if mode not in {"paired", "crossed"}:
        raise ValueError("seed design must be paired or crossed")
    if seed_pairs is None:
        if mode == "paired" and set(source) != set(target):
            raise ValueError("paired seed design requires matching identities or explicit seed pairs")
        pairs = tuple((seed, seed) for seed in source) if mode == "paired" else tuple(product(source, target))
    else:
        try:
            if isinstance(seed_pairs, (str, bytes, Mapping)):
                raise TypeError("seed pairs must be a sequence of pairs")
            records = tuple(seed_pairs)
            if any(isinstance(pair, (str, bytes, Mapping)) for pair in records):
                raise TypeError("each seed pair must be a sequence")
            pairs = tuple(tuple(pair) for pair in records)
        except (TypeError, ValueError) as error:
            raise ValueError("explicit seed pairs require sequences of two integer seeds") from error
    if (not pairs or any(len(pair) != 2 or any(type(v) is not int for v in pair) for pair in pairs)
            or len(set(pairs)) != len(pairs)
            or {pair[0] for pair in pairs} != set(source)
            or {pair[1] for pair in pairs} != set(target)):
        raise ValueError("explicit seed pairs must be unique integer pairs covering the declared seeds")
    return pairs


def _valid_auc(row: Mapping | None) -> bool:
    value = None if row is None else row.get("auc")
    return bool(row is not None and row.get("status") == "completed"
                and isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value) and 0 <= value <= 1)


def _participant_rows(unit_rows: list[dict], seeds: tuple[tuple, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in unit_rows:
        grouped[(row["study"], row["participant_key"])].append(row)
    result = []
    for (study, participant), rows in sorted(grouped.items()):
        lookup = {(row["source_seed"], row["target_seed"]): row for row in rows}
        complete = [lookup[pair] for pair in seeds if pair in lookup and lookup[pair]["status"] == "complete_pair"]
        missing = [{"source_seed": source, "target_seed": target, "reason":
                    lookup[(source, target)].get("failure", "missing_seed_pair")
                    if (source, target) in lookup else "missing_seed_pair"}
                   for source, target in seeds if (source, target) not in lookup
                   or lookup[(source, target)]["status"] != "complete_pair"]
        result.append({
            "study": study, "participant_key": participant,
            "status": "complete_pair" if not missing else "incomplete_seed_grid",
            "difference": float(np.mean([row["difference"] for row in complete])) if not missing else None,
            "requested_seed_pairs": len(seeds), "failed_or_missing_seed_pairs": len(missing),
            "seed_failures": missing,
        })
    return result


def _summary(rows: list[dict], studies: tuple[str, ...], *, iterations: int, seed: int,
             with_intervals: bool = True, contrast_bounds: Mapping[str, float | None] | None = None) -> dict:
    complete = [row for row in rows if row["status"] == "complete_pair"]
    by_study = {}
    samples = {}
    for study in studies:
        requested = [row for row in rows if row["study"] == study]
        values = np.asarray([row["difference"] for row in complete if row["study"] == study], dtype=float)
        if len(values):
            samples[study] = values
        interval = None
        if with_intervals and len(values) > 1:
            interval = paired_interval(values, inference_scope="conditional_frozen_models",
                                       iterations=iterations, seed=seed).record()
        by_study[study] = {
            "n_requested_participants": len(requested), "n_complete_participants": len(values),
            "mean_difference": float(values.mean()) if len(values) else None,
            "negative_fraction": float(np.mean(values < 0)) if len(values) else None,
            "decline_over_0_01_fraction": float(np.mean(values < -0.01)) if len(values) else None,
            "p10": float(np.quantile(values, 0.1)) if len(values) else None,
            "worst": float(values.min()) if len(values) else None,
            "replacement_decline_gate_passed": bool(values.mean() >= -0.01) if len(values) else None,
            "paired_interval": interval,
        }
        if contrast_bounds is not None:
            bound = contrast_bounds.get(study)
            missing = len(requested) - len(values)
            by_study[study].update(
                contrast_bound=bound,
                missing_result_bounds=(
                    [(float(values.sum()) - missing * bound) / len(requested),
                     (float(values.sum()) + missing * bound) / len(requested)]
                    if len(requested) and bound is not None else None),
            )
    overall_interval = None
    if with_intervals and any(len(values) > 1 for values in samples.values()):
        rng = np.random.default_rng(seed)
        draws = np.zeros(iterations)
        for values in samples.values():
            for offset in range(0, iterations, 1000):
                count = min(1000, iterations - offset)
                draws[offset:offset + count] += values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
        draws /= len(samples)
        overall_interval = np.quantile(draws, [0.025, 0.975]).tolist()
    values = np.asarray([row["difference"] for row in complete], dtype=float)
    complete_studies = len(samples) == len(studies)
    partial_mean = float(np.mean([value.mean() for value in samples.values()])) if samples else None
    partial_negative = float(np.mean([np.mean(value < 0) for value in samples.values()])) if samples else None
    partial_decline = float(np.mean([np.mean(value < -0.01) for value in samples.values()])) if samples else None
    result = {
        "n_requested_participants": len(rows), "n_complete_participants": len(complete),
        "requested_seed_pairs": sum(row["requested_seed_pairs"] for row in rows),
        "failed_or_missing_seed_pairs": sum(row["failed_or_missing_seed_pairs"] for row in rows),
        "mean_difference": partial_mean if complete_studies else None,
        "expected_studies": list(studies), "estimable_studies": list(samples), "n_estimable_studies": len(samples),
        "missing_expected_studies": [study for study in studies if study not in samples],
        "full_study_estimate_available": complete_studies,
        "negative_fraction_study_equal": partial_negative if complete_studies else None,
        "decline_over_0_01_fraction_study_equal": partial_decline if complete_studies else None,
        "partial_observed_studies": {
            "studies": list(samples), "mean_difference": partial_mean,
            "negative_fraction_study_equal": partial_negative,
            "decline_over_0_01_fraction_study_equal": partial_decline,
            "bootstrap_interval": overall_interval,
            "estimand": "equal_weight_mean_over_only_studies_with_complete_participants",
        },
        "pooled_participant_p10": float(np.quantile(values, 0.1)) if len(values) else None,
        "worst_participant_difference": float(values.min()) if len(values) else None,
        "by_study": by_study, "participant_rows": rows,
        "bootstrap": {"interval": overall_interval if complete_studies else None, "confidence_level": 0.95, "iterations": iterations,
                      "seed": seed, "sampling_unit": "participant_within_fixed_study", "scope": INFERENCE_SCOPE,
                      "limitation": "does_not_cover_new_studies_source_resampling_or_training_procedure_uncertainty"},
        "failure_policy": "complete_pairs_with_all_declared_seeds_only_failures_retained_in_denominator",
    }
    if contrast_bounds is not None:
        bounds = [by_study[study]["missing_result_bounds"] for study in studies]
        result["missing_result_bounds"] = {
            "interval": np.mean(bounds, axis=0).tolist() if all(bound is not None for bound in bounds) else None,
            "is_confidence_interval": False,
            "weighting": "equal_declared_study_weights_after_within_study_requested_participant_bounds",
            "population": "known_actual_K_requested_and_observed_participants",
            "assumption": "each_missing_zero_sum_AUC_contrast_lies_in_plus_minus_sum_positive_weights",
        }
    return result


def _contrast_report(
    lookup: dict, weights: Mapping[str, float], *, studies: tuple[str, ...],
    seeds: tuple[tuple, ...], standard_budgets: tuple[int, ...], iterations: int, seed: int,
    requested_scopes: Sequence[Mapping] = (),
) -> dict:
    study_weights = weights.get("by_study")

    def coefficients(study):
        return study_weights.get(study, {}) if study_weights is not None else weights

    bounds = {study: float(sum(value for value in coefficients(study).values() if value > 0))
              if coefficients(study) else None for study in studies}

    def summarize(rows, **options):
        return _summary(rows, studies, iterations=iterations, seed=seed, contrast_bounds=bounds, **options)

    relevant = {key[:-1] for key in lookup if coefficients(key[0])}
    relevant.update((row["study"], row["participant_key"], row["actual_K"], *pair)
                    for row in requested_scopes if coefficients(row["study"]) for pair in seeds)
    units = []
    for study, participant, budget, source_seed, target_seed in sorted(relevant):
        current_weights = coefficients(study)
        rows = {arm: lookup.get((study, participant, budget, source_seed, target_seed, arm)) for arm in current_weights}
        present = [row for row in rows.values() if row is not None]
        signatures = {(row["calibration_rows_sha256"], row["test_rows_sha256"]) for row in present}
        if present and len(signatures) != 1:
            raise ValueError("contrast arms must have matched calibration and final-test row digests")
        failed = [arm for arm, row in rows.items() if not _valid_auc(row)]
        units.append({
            "study": study, "participant_key": participant, "K": budget,
            "source_seed": source_seed, "target_seed": target_seed,
            "status": "incomplete_pair" if failed else "complete_pair",
            "difference": None if failed else float(sum(current_weights[arm] * row["auc"] for arm, row in rows.items())),
            "failure": {"failed_or_missing_arms": failed} if failed else None,
        })
    actual_budgets = tuple(sorted({row["K"] for row in units}))
    by_budget = {budget: _participant_rows([row for row in units if row["K"] == budget], seeds)
                 for budget in actual_budgets}
    exact = {str(budget): summarize(by_budget.get(budget, []))
             for budget in standard_budgets}
    fallback = {str(budget): summarize(by_budget[budget])
                for budget in actual_budgets if budget not in standard_budgets}
    policy_rows = defaultdict(list)
    for budget, rows in by_budget.items():
        for row in rows:
            policy_rows[(row["study"], row["participant_key"])].append((budget, row))
    policy = []
    for (study, participant), rows in sorted(policy_rows.items()):
        legal = all(row["status"] == "complete_pair" for _, row in rows)
        policy.append({
            "study": study, "participant_key": participant, "status": "complete_pair" if legal else "incomplete_budget_policy",
            "actual_Ks": [budget for budget, _ in rows],
            "difference": float(np.mean([row["difference"] for _, row in rows])) if legal else None,
            "requested_seed_pairs": sum(row["requested_seed_pairs"] for _, row in rows),
            "failed_or_missing_seed_pairs": sum(row["failed_or_missing_seed_pairs"] for _, row in rows),
        })
    seed_panels = []
    for source_seed, target_seed in seeds:
        for budget in actual_budgets:
            rows = _participant_rows(
                [row for row in units if row["K"] == budget], ((source_seed, target_seed),),
            )
            # Seed panels are stability descriptions; their units remain participants.
            panel = summarize(rows, with_intervals=False)
            seed_panels.append({"source_seed": source_seed, "target_seed": target_seed, "K": budget,
                                "mean_difference": panel["mean_difference"],
                                "n_requested_participants": panel["n_requested_participants"],
                                "n_complete_participants": panel["n_complete_participants"],
                                "failed_or_missing_seed_pairs": panel["failed_or_missing_seed_pairs"]})
    sets = [{(row["study"], row["participant_key"]) for row in by_budget.get(budget, [])
             if row["status"] == "complete_pair"} for budget in standard_budgets]
    common = set.intersection(*sets) if sets else set()
    common_rows = {str(budget): [row for row in by_budget.get(budget, [])
                                if (row["study"], row["participant_key"]) in common] for budget in standard_budgets}
    persistent = []
    for study, participant in sorted(common):
        differences = {str(budget): next(row["difference"] for row in common_rows[str(budget)]
                                        if row["study"] == study and row["participant_key"] == participant)
                       for budget in standard_budgets}
        persistent.append({"study": study, "participant_key": participant,
                           "negative_at_all_standard_K": all(value < 0 for value in differences.values()),
                           "differences": differences})
    seed_ranges = {}
    for budget in actual_budgets:
        finite = [row["mean_difference"] for row in seed_panels if row["K"] == budget and row["mean_difference"] is not None]
        seed_ranges[str(budget)] = [min(finite), max(finite)] if finite else None
    marginal_seeds = {"source_seed": [], "target_seed": []}
    for axis, index in (("source_seed", 0), ("target_seed", 1)):
        for value in sorted({pair[index] for pair in seeds}):
            subset = tuple(pair for pair in seeds if pair[index] == value)
            for budget in actual_budgets:
                rows = _participant_rows([row for row in units if row["K"] == budget], subset)
                panel = summarize(rows, with_intervals=False)
                marginal_seeds[axis].append({axis: value, "K": budget, "mean_difference": panel["mean_difference"],
                                             "n_complete_participants": panel["n_complete_participants"],
                                             "failed_or_missing_seed_pairs": panel["failed_or_missing_seed_pairs"]})
    return {
        "weights": dict(weights), "exact_K": exact, "fallback_actual_K": fallback,
        "budget_policy": {**summarize(policy),
                          "estimand": "participant_mean_over_unique_available_actual_positive_K_including_fallback"},
        "per_seed": seed_panels,
        "seed_stability_range_by_K": seed_ranges, "seed_marginals": marginal_seeds,
        "common_standard_K_cohort": {"n_participants": len(common), "participant_rows": persistent,
                                     "by_K": {budget: summarize(rows)
                                              for budget, rows in common_rows.items()}},
        "unit_failures": [row for row in units if row["status"] != "complete_pair"],
    }


def analyze_transfer_contrasts(
    cells: Sequence[Mapping], contrasts: Mapping[str, Mapping[str, float]], *,
    expected_studies: Sequence[str], source_seeds: Sequence[int], target_seeds: Sequence[int],
    standard_budgets: Sequence[int] = (200, 800, 2000), iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = ANALYSIS_SEED, arm_field: str = "arm", group_fields: Sequence[str] = (),
    requested_participants: Sequence[Mapping] = (),
    evidence_scope: str = "public_development",
    seed_pairs: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Pair actual K and row digests, average declared seeds, then bootstrap people."""
    if evidence_scope not in {"public_development", "software_correctness"}:
        raise ValueError("analysis evidence scope must be public_development or software_correctness")
    studies = _declared_values(expected_studies, "expected_studies")
    source_seeds = _declared_values(source_seeds, "source_seeds")
    target_seeds = _declared_values(target_seeds, "target_seeds")
    budgets = _declared_values(standard_budgets, "standard_budgets")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (*source_seeds, *target_seeds, seed)):
        raise ValueError("source seeds, target seeds and analysis seed must be explicit integers")
    seed_pairs = resolve_seed_pairs(source_seeds, target_seeds, seed_pairs=seed_pairs)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in budgets):
        raise ValueError("standard budgets must be positive integer actual-K panels")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1000:
        raise ValueError("bootstrap iterations must be an integer >= 1000")
    if not contrasts:
        raise ValueError("an explicit nonempty contrast declaration is required")
    for name, weights in contrasts.items():
        rows = weights.get("by_study", {"all": weights})
        if "by_study" in weights and (not rows or set(rows) - set(studies)):
            raise ValueError("study-specific contrasts must use declared study identities")
        for coefficients in rows.values():
            if not name or not coefficients or not np.isfinite(list(coefficients.values())).all() or not np.isclose(sum(coefficients.values()), 0):
                raise ValueError("contrast weights must be finite, nonempty and sum to zero")
    grouped = defaultdict(dict)
    participant_studies = {}
    for input_row in cells:
        row = dict(input_row)
        required = {"study", "participant_key", "K", "source_seed", "target_seed", "status",
                    "calibration_rows_sha256", "test_rows_sha256", arm_field, *group_fields}
        if required - row.keys():
            raise ValueError(f"analysis row missing explicit contract fields: {sorted(required - row.keys())}")
        if isinstance(row["K"], bool) or not isinstance(row["K"], int) or row["K"] < 0:
            raise ValueError("actual K must be a nonnegative integer")
        if row["K"] == 0:
            continue
        if row["study"] not in studies or row["source_seed"] not in source_seeds or row["target_seed"] not in target_seeds:
            raise ValueError("row study or seed is outside the predeclared analysis scope")
        if (row["source_seed"], row["target_seed"]) not in seed_pairs:
            raise ValueError("row seed pair is outside the predeclared analysis scope")
        if not isinstance(row["participant_key"], str) or not row["participant_key"].strip():
            raise ValueError("participant_key must be the explicit canonical participant identity")
        previous = participant_studies.setdefault(row["participant_key"], row["study"])
        if previous != row["study"]:
            raise ValueError("the same canonical participant in multiple studies needs a shared-cluster analysis")
        for field in ("calibration_rows_sha256", "test_rows_sha256"):
            if not isinstance(row[field], str) or len(row[field]) != 64:
                raise ValueError("matched row digests must be explicit SHA256 values")
        if any(row[field] is None for field in group_fields):
            raise ValueError("analysis group fields must have explicitly known identities")
        grouping = tuple(row[field] for field in group_fields)
        key = tuple(row[field] for field in ("study", "participant_key", "K", "source_seed", "target_seed", arm_field))
        if key in grouped[grouping]:
            raise ValueError("duplicate participant/K/source-seed/target-seed/arm outcome")
        grouped[grouping][key] = row
    requests, scoped_requests = defaultdict(list), defaultdict(list)
    unresolved_group_requests = []
    for original in requested_participants:
        row = dict(original)
        if row.get("study") not in studies or not isinstance(row.get("participant_key"), str) or not row["participant_key"].strip():
            raise ValueError("requested participant denominators require explicit study and canonical participant_key")
        previous = participant_studies.setdefault(row["participant_key"], row["study"])
        if previous != row["study"]:
            raise ValueError("the same canonical participant in multiple studies needs a shared-cluster analysis")
        actual = row.get("actual_K")
        if actual is not None and (type(actual) is not int or actual < 0):
            raise ValueError("requested actual_K must be a nonnegative integer or absent")
        requests[(row["study"], row["participant_key"])].append(row)
        missing_group = [field for field in group_fields if row.get(field) is None]
        if missing_group:
            unresolved_group_requests.append({**row, "missing_group_fields": missing_group})
        elif actual is not None and actual > 0:
            grouping = tuple(row[field] for field in group_fields)
            scoped_requests[grouping].append(row)
            grouped.setdefault(grouping, {})
    for lookup in grouped.values():
        for study, participant, *_ in lookup:
            requests.setdefault((study, participant), [])
    outer_denominator = {
        "n_requested_participants": len(requests),
        "n_protocol_excluded_participants": sum(any(item.get("status") == "protocol_excluded" for item in rows)
                                                for rows in requests.values()),
        "participant_rows": [{"study": study, "participant_key": participant, "request_records": rows}
                             for (study, participant), rows in sorted(requests.items())],
        "source": "explicit_requests_plus_observed_canonical_participants" if requested_participants else "observed_rows_only",
        "actual_K_invented_for_excluded_participants": False,
        "unresolved_group_requests": unresolved_group_requests,
        "group_scope_policy": "missing_group_fields_retained_in_outer_denominator_without_panel_assignment",
    }
    if not grouped and requests and not group_fields:
        grouped[()] = {}
    groups = []
    for grouping, lookup in sorted(grouped.items()):
        group_requests = {key: [row for row in records if all(row.get(field) == value
                            for field, value in zip(group_fields, grouping, strict=True))]
                          for key, records in requests.items()}
        observed_people = {key[:2] for key in lookup}
        group_requests = {key: records for key, records in group_requests.items() if records or key in observed_people}
        reports = {name: _contrast_report(
                           lookup, weights, studies=studies, seeds=seed_pairs,
                           standard_budgets=budgets, iterations=iterations, seed=seed,
                           requested_scopes=scoped_requests[grouping],
                       ) for name, weights in contrasts.items()}
        for report in reports.values():
            for budget, panel in report["exact_K"].items():
                declared = {key for key, rows in group_requests.items() if any(
                    row.get("actual_K") == int(budget) or int(budget) in (row.get("requested_Ks") or ()) for row in rows
                )}
                unknown = {key for key, rows in group_requests.items() if not rows or all(
                    row.get("requested_Ks") is None and row.get("actual_K") is None for row in rows)}
                reachable = {key[:2] for key in lookup if key[2] == int(budget)}
                panel.update(n_outer_requested_participants=len(requests),
                             n_group_scoped_requested_participants=len(group_requests),
                             n_declared_requested_at_K=len(declared), n_unknown_requested_K_participants=len(unknown),
                             requested_without_actual_K_result=[{"study": study, "participant_key": participant}
                                                                for study, participant in sorted(declared - reachable)])
        arm_descriptives = {}
        declared_arms = {arm for weights in contrasts.values()
                         for coefficients in weights.get("by_study", {"all": weights}).values()
                         for arm in coefficients}
        panel_budgets = sorted({key[2] for key in lookup} | {row["actual_K"] for row in scoped_requests[grouping]})
        for arm in sorted({key[-1] for key in lookup} | declared_arms):
            panels = {}
            for budget in panel_budgets:
                relevant = {key[:-1] for key in lookup if key[2] == budget}
                relevant.update((row["study"], row["participant_key"], budget, *pair)
                                for row in scoped_requests[grouping] if row["actual_K"] == budget for pair in seed_pairs)
                units = []
                for key in sorted(relevant):
                    row = lookup.get((*key, arm))
                    valid = _valid_auc(row)
                    units.append({"study": key[0], "participant_key": key[1], "source_seed": key[3], "target_seed": key[4],
                                  "status": "complete_pair" if valid else "incomplete_pair",
                                  "difference": float(row["auc"]) if valid else None,
                                  "failure": None if valid else "missing_failed_or_invalid_auc"})
                summary = _summary(_participant_rows(units, seed_pairs), studies, iterations=iterations,
                                   seed=seed, with_intervals=False)
                panels[str(budget)] = {"mean_auc": summary["mean_difference"],
                                      "n_requested_participants": summary["n_requested_participants"],
                                      "n_complete_participants": summary["n_complete_participants"],
                                      "failed_or_missing_seed_pairs": summary["failed_or_missing_seed_pairs"],
                                      "participant_rows": summary["participant_rows"],
                                      "partial_observed_studies": {
                                          "studies": summary["partial_observed_studies"]["studies"],
                                          "mean_auc": summary["partial_observed_studies"]["mean_difference"],
                                          "estimand": "equal_weight_mean_AUC_over_only_studies_with_complete_participants"},
                                      "by_study": {study: {"mean_auc": item["mean_difference"],
                                                           "n_requested_participants": item["n_requested_participants"],
                                                           "n_complete_participants": item["n_complete_participants"]}
                                                   for study, item in summary["by_study"].items()}}
            arm_descriptives[arm] = panels
        costs = {}
        for arm in arm_descriptives:
            rows = [row for key, row in lookup.items() if key[-1] == arm]
            times = [float(row["elapsed_seconds"]) for row in rows if row.get("elapsed_seconds") is not None]
            costs[arm] = {"requested_cells": len(rows), "completed_cells": sum(row["status"] == "completed" for row in rows),
                          "failed_cells": sum(row["status"] != "completed" for row in rows),
                          "timed_cells": len(times), "total_elapsed_seconds": sum(times) if times else None,
                          "calibration_fallback_cells": sum(row.get("selection", {}).get("fallback") is not None for row in rows)}
            selections = [row["epoch_selection_result"] for row in rows
                          if row["status"] == "completed" and "epoch_selection_result" in row]
            if selections:
                eligible = [row for row in selections if row["selection_evidence"]]
                costs[arm]["epoch_selection"] = {
                    "completed_cells": len(selections), "selection_evidence_cells": len(eligible),
                    "fallback_cells": sum(row["fallback"] is not None for row in selections),
                    "selected_epochs_median": float(np.median([row["selected_epochs"] for row in selections])),
                    "inner_epochs_median": float(np.median([row["inner_epochs_run"] for row in selections])),
                    "stopped_early_fraction_among_eligible": float(np.mean([row["stopped_early"] for row in eligible])) if eligible else None,
                    **{field: sum(row[field] for row in selections) for field in (
                        "total_optimizer_updates", "physical_optimizer_updates", "total_trial_draws", "physical_trial_draws",
                        "inner_wall_seconds", "refit_wall_seconds", "total_wall_seconds")},
                    "cost_includes_selection_and_full_refit": True,
                    "physical_cost_depends_on_exact_fit_cache_hits": True,
                }
        usage = {}
        for budget in sorted({key[2] for key in lookup if key[-1] == "five_candidates"}):
            units = []
            for key, row in lookup.items():
                if key[-1] != "five_candidates" or key[2] != budget:
                    continue
                initialization = row.get("selection", {}).get("refit_recipe", {}).get("initialization")
                legal = row["status"] == "completed" and initialization in {"source", "scratch", "waveform_lda"}
                units.append({"study": key[0], "participant_key": key[1], "source_seed": key[3], "target_seed": key[4],
                              "status": "complete_pair" if legal else "incomplete_pair",
                              "difference": float(initialization != "source") if legal else None})
            summary = _summary(_participant_rows(units, seed_pairs), studies, iterations=iterations, seed=seed, with_intervals=False)
            usage[str(budget)] = {"abandoned_transfer_rate_study_equal": summary["mean_difference"],
                                 "n_requested_participants": summary["n_requested_participants"],
                                 "n_complete_participants": summary["n_complete_participants"]}
        groups.append({"group": dict(zip(group_fields, grouping, strict=True)), "contrasts": reports,
                       "arm_descriptives": arm_descriptives, "program_costs": costs, "selector_usage": usage})
    report = {
        "schema": "n2p3_training_chain_analysis/1", "evidence_scope": evidence_scope,
        "inference_scope": INFERENCE_SCOPE, "expected_studies": list(studies),
        "declared_source_seeds": list(source_seeds), "declared_target_seeds": list(target_seeds),
        "declared_seed_pairs": [list(pair) for pair in seed_pairs],
        "standard_budgets": list(budgets), "arm_field": arm_field, "group_fields": list(group_fields),
        "bootstrap_iterations": iterations, "analysis_seed": seed, "groups": groups,
        "outer_request_denominator": outer_denominator,
        "confirmation_claim": False,
        "diagnostic_decline_threshold_auc": 0.01,
        "threshold_is_confirmatory_sesoi": False,
    }
    if evidence_scope == "software_correctness":
        scope = "software_correctness_only_no_population_inference"
        report["inference_scope"] = scope

        def mark_intervals(value):
            if isinstance(value, dict):
                if value.get("scope") == INFERENCE_SCOPE:
                    value["scope"] = scope
                for child in value.values():
                    mark_intervals(child)
            elif isinstance(value, list):
                for child in value:
                    mark_intervals(child)

        mark_intervals(report)
    return report


def analyze_source_objective_contrasts(
    cells: Sequence[Mapping], *, source_arms: Sequence[str], reference_source_arm="P2_source_ce",
    group_fields: Sequence[str] = (), **analysis_options,
) -> dict:
    """Pair P2 source effects and adaptation interactions, retaining shared controls.

    The JSON pair is an injective encoding of two declared factor levels, not
    an inferred method name. All statistics reuse the participant-paired engine.
    """
    sources = _declared_values(source_arms, "source_arms")
    if any(not isinstance(arm, str) or not arm.strip() for arm in sources) or reference_source_arm not in sources:
        raise ValueError("source objective comparisons require an explicit reference source arm")
    if "source_arm" in group_fields or "arm" in group_fields:
        raise ValueError("source and target factors cannot also be analysis panel groups")

    def level(source, target):
        return json.dumps([source, target], separators=(",", ":"))

    rows, levels = [], {}
    for original in cells:
        if original.get("source_arm") not in sources:
            raise ValueError("outcome source arm is outside the declared source design")
        key = level(original["source_arm"], original["arm"])
        rows.append({**original, "source_target_arm": key})
        levels[key] = {"source_arm": original["source_arm"], "target_arm": original["arm"]}
    contrasts = {}
    for source in sources:
        if source == reference_source_arm:
            continue
        full = {level(source, "p2_full_fine"): 1., level(reference_source_arm, "p2_full_fine"): -1.}
        contrasts[f"{source}_minus_{reference_source_arm}"] = full
        for control in ("p2_scratch", "p2_waveform_lda"):
            contrasts[f"{source}_minus_{reference_source_arm}_relative_to_{control}"] = {
                **full, level(source, control): -1., level(reference_source_arm, control): 1.}
    if not contrasts:
        return {"declared_source_arms": list(sources), "reference_source_arm": reference_source_arm,
                "groups": [], "status": "no_declared_source_contrast"}
    result = analyze_transfer_contrasts(rows, contrasts, arm_field="source_target_arm",
                                       group_fields=group_fields, **analysis_options)
    result.update(declared_source_arms=list(sources), reference_source_arm=reference_source_arm,
                  factor_levels=levels, shared_control_policy="same_physical_fit_is_not_an_independent_replicate")
    return result
