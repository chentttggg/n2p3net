from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from stats.transfer import analyze_transfer_contrasts, standard_training_contrasts
from tests import PROJECT_ROOT
from tests.support.io import read_json, write_json

SOURCE_SEEDS = (1, 2)
TARGET_SEEDS = (10, 20)
STUDIES = ("A", "B")
TRANSFER_CONTRAST = {"transfer": {"source": 1.0, "scratch": -1.0}}


def _cells():
    rows = []
    for study, person, delta in (("A", "p1", 0.1), ("A", "p2", 0.3), ("B", "p3", -0.2)):
        for source_seed in SOURCE_SEEDS:
            for target_seed in TARGET_SEEDS:
                for arm, value in (("source", 0.5 + delta), ("scratch", 0.5)):
                    rows.append(dict(
                        study=study, participant_key=f"authority:{person}", K=200,
                        source_seed=source_seed, target_seed=target_seed, arm=arm, auc=value,
                        status="completed", calibration_rows_sha256="a" * 64,
                        test_rows_sha256="b" * 64,
                    ))
    return rows


def _analyze(cells, contrasts=TRANSFER_CONTRAST, **options):
    return analyze_transfer_contrasts(
        cells, contrasts, expected_studies=STUDIES,
        source_seeds=SOURCE_SEEDS, target_seeds=TARGET_SEEDS, iterations=1000, **options,
    )


def test_seed_average_precedes_participant_and_study_equal_summary():
    result = _analyze(_cells())
    panel = result["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["mean_difference"] == pytest.approx(0.0)
    assert panel["n_complete_participants"] == 3
    assert panel["requested_seed_pairs"] == 12
    assert panel["by_study"]["A"]["mean_difference"] == pytest.approx(0.2)
    assert panel["by_study"]["B"]["mean_difference"] == pytest.approx(-0.2)
    assert panel["negative_fraction_study_equal"] == 0.5
    assert panel["bootstrap"]["sampling_unit"] == "participant_within_fixed_study"


def test_one_failed_seed_keeps_participant_failure_in_denominator():
    cells = _cells()
    cells[0].update(status="fit_failed", auc=None)
    panel = _analyze(cells)["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["n_requested_participants"] == 3
    assert panel["n_complete_participants"] == 2
    assert panel["failed_or_missing_seed_pairs"] == 1
    assert panel["participant_rows"][0]["status"] == "incomplete_seed_grid"
    assert panel["mean_difference"] == pytest.approx(0.05)


def test_declared_complete_flow_seed_pairs_do_not_invent_crossed_runs():
    pairs = [(1, 10), (2, 20)]
    cells = [row for row in _cells() if (row["source_seed"], row["target_seed"]) in pairs]
    report = _analyze(cells, seed_pairs=pairs)
    panel = report["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert report["declared_seed_pairs"] == [[1, 10], [2, 20]]
    assert panel["n_complete_participants"] == 3
    assert panel["requested_seed_pairs"] == 6
    assert panel["mean_difference"] == pytest.approx(0.0)
    cells[0].update(status="fit_failed", auc=None)
    panel = _analyze(cells, seed_pairs=pairs)["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["n_complete_participants"] == 2
    assert panel["failed_or_missing_seed_pairs"] == 1
    with pytest.raises(ValueError, match="seed pair"):
        _analyze(_cells(), seed_pairs=pairs)
    with pytest.raises(ValueError, match="seed pairs"):
        _analyze(cells, seed_pairs=[(1, 10)])


def test_protocol_excluded_person_stays_requested_without_inventing_actual_k():
    requests = [{"study": study, "participant_key": f"authority:{person}", "requested_Ks": [200, 800, 2000]}
                for study, person in (("A", "p1"), ("A", "p2"), ("B", "p3"))]
    requests.append({"study": "B", "participant_key": "authority:p4", "requested_Ks": [200, 800, 2000],
                     "status": "protocol_excluded", "reason": "all positive K lack a legal test"})
    report = _analyze(_cells(), requested_participants=requests)
    assert report["outer_request_denominator"]["n_requested_participants"] == 4
    assert report["outer_request_denominator"]["n_protocol_excluded_participants"] == 1
    panel = report["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["n_outer_requested_participants"] == 4
    assert panel["n_requested_participants"] == 3
    assert panel["requested_without_actual_K_result"] == [{"study": "B", "participant_key": "authority:p4"}]


def test_positive_mean_decline_gate_does_not_claim_universal_replacement():
    cells = _cells()
    for row in cells:
        if row["arm"] == "source" and row["participant_key"] == "authority:p1":
            row["auc"] = 0.3
    panel = _analyze(cells)["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]["by_study"]["A"]
    assert panel["mean_difference"] == pytest.approx(0.05)
    assert panel["negative_fraction"] == 0.5
    assert panel["replacement_decline_gate_passed"] is True
    assert "supports_universal_replacement" not in panel


def test_pair_digest_disagreement_and_duplicate_rows_are_rejected():
    cells = _cells()
    cells[0]["test_rows_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="matched"):
        _analyze(cells)
    cells = _cells()
    with pytest.raises(ValueError, match="duplicate"):
        _analyze(cells + [copy.deepcopy(cells[0])])


def test_e1_factorial_weights_match_difference_in_differences():
    contrasts = standard_training_contrasts("E1")
    cells = _cells()
    factorial = []
    for row in cells:
        if row["arm"] != "source":
            continue
        for i in (0, 1):
            for b in (0, 1):
                for w in (0, 1):
                    factorial.append({**row, "arm": f"e1_i{i}b{b}w{w}",
                                      "auc": 0.4 + 0.1 * i + 0.04 * b + 0.02 * w + 0.08 * i * b})
    output = _analyze(
        factorial, {name: contrasts[name] for name in ("I", "B", "W", "IxB")}
    )["groups"][0]["contrasts"]
    expected = {"I": 0.14, "B": 0.08, "W": 0.02, "IxB": 0.08}
    for name, value in expected.items():
        assert output[name]["exact_K"]["200"]["mean_difference"] == pytest.approx(value)


def test_compact_contrasts_pair_single_ablations_without_requiring_factorial_arms():
    values = {
        "compact_waveform_lda": 0.6, "compact_scratch": 0.5, "compact_full_fine": 0.7,
        "compact_source_statistics": 0.65, "compact_frozen_bn_buffers": 0.68,
        "compact_head_only": 0.62,
    }
    rows = [{**row, "arm": arm, "auc": value}
            for row in _cells() if row["arm"] == "source" for arm, value in values.items()]
    rows[0].update(status="fit_failed", auc=None)
    output = _analyze(rows, standard_training_contrasts("compact"))["groups"][0]["contrasts"]
    expected = {
        "full_minus_scratch": (0.2, 3), "full_minus_lda": (0.1, 2),
        "source_statistics_minus_full": (-0.05, 3),
        "frozen_bn_buffers_minus_full": (-0.02, 3), "head_only_minus_full": (-0.08, 3),
    }
    for name, (difference, complete_participants) in expected.items():
        panel = output[name]["exact_K"]["200"]
        assert panel["mean_difference"] == pytest.approx(difference)
        assert panel["n_requested_participants"] == 3
        assert panel["n_complete_participants"] == complete_participants


def test_budget_policy_keeps_fallback_actual_k_out_of_standard_k():
    rows = _cells()
    fallback = [{**row, "K": 1600, "auc": 0.5 if row["arm"] == "scratch" else row["auc"] + 0.1} for row in rows]
    output = _analyze(rows + fallback)["groups"][0]["contrasts"]["transfer"]
    assert output["exact_K"]["2000"]["mean_difference"] is None
    assert output["fallback_actual_K"]["1600"]["mean_difference"] == pytest.approx(0.1)
    assert output["budget_policy"]["mean_difference"] == pytest.approx(0.05)
    assert output["budget_policy"]["n_complete_participants"] == 3
    assert output["budget_policy"]["participant_rows"][0]["actual_Ks"] == [200, 1600]


def test_s2_study_specific_sources_keep_target_recipe_groups_separate():
    from stats.transfer import standard_s2_contrasts

    source_arms = {
        "A": ("S2_full_multi", "S2_full_single_X", "S2_full_single_Y", "S2_full_single_Z"),
        "B": ("S2_full_multi", "S2_full_single_W", "S2_full_single_Y", "S2_full_single_Z"),
    }
    rows = []
    for row in _cells():
        if row["arm"] != "source":
            continue
        for source_arm in source_arms[row["study"]]:
            rows.append({**row, "arm": "frozen_target_recipe", "source_arm": source_arm,
                         "auc": 0.7 if source_arm == "S2_full_multi" else 0.6})
    result = _analyze(
        rows, standard_s2_contrasts(source_arms), arm_field="source_arm", group_fields=("arm",)
    )["groups"][0]
    assert result["group"] == {"arm": "frozen_target_recipe"}
    assert result["contrasts"]["full_multi_minus_single_mean"]["exact_K"]["200"]["mean_difference"] == pytest.approx(0.1)


@pytest.fixture
def analysis_cli(tmp_path):
    record = tmp_path / "record.json"
    contrasts = tmp_path / "contrasts.json"
    output = tmp_path / "analysis.json"
    write_json(contrasts, TRANSFER_CONTRAST)
    argv = [
        "--record", str(record), "--stage", "E3", "--studies", *STUDIES,
        "--source-seeds", *map(str, SOURCE_SEEDS), "--target-seeds", *map(str, TARGET_SEEDS),
        "--contrast-json", str(contrasts), "--bootstrap-iterations", "1000",
        "--output", str(output),
    ]
    return record, output, argv


def test_analysis_cli_from_outside_repository_preserves_record_hashes(tmp_path, analysis_cli):
    record, output, argv = analysis_cli
    write_json(record, {"cells": _cells(), "schema": "test_stage/1", "evidence_scope": "software_correctness"})
    process = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "analyze_training_chain.py"),
         *argv, "--allow-software-correctness"], cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    report = read_json(output)
    assert report["analysis_seed"] == 20260907
    assert len(report["source_records"][0]["sha256"]) == 64
    assert report["confirmation_claim"] is False


def test_analysis_cli_rejects_software_records_unless_explicitly_kept_software(analysis_cli):
    from experiments.analyze_training_chain import main

    record, output, argv = analysis_cli
    cells = _cells()
    cells[0]["evidence_scope"] = "software_correctness"
    write_json(record, {"cells": cells, "evidence_scope": "development"})
    with pytest.raises(SystemExit):
        main(argv)
    assert not output.exists()
    main([*argv, "--allow-software-correctness"])
    report = read_json(output)
    assert report["evidence_scope"] == "software_correctness"
    assert report["inference_scope"] == "software_correctness_only_no_population_inference"
    panel = report["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["bootstrap"]["scope"] == report["inference_scope"]


def test_p2_cli_uses_declared_pairs_and_keeps_source_factor_and_evidence_type(tmp_path):
    from experiments.analyze_training_chain import main

    pairs = [(1, 10), (2, 20)]
    sources = ["P2_source_ce", "P2_source_mmd"]
    cells = []
    for original in _cells():
        if original["arm"] != "source" or (original["source_seed"], original["target_seed"]) not in pairs:
            continue
        for source in sources:
            for arm, auc in (("p2_full_fine", .7 + .04 * (source == sources[1])),
                             ("p2_scratch", .5), ("p2_waveform_lda", .6)):
                cells.append({**original, "source_arm": source, "arm": arm, "auc": auc})
    record, output = tmp_path / "record.json", tmp_path / "analysis.json"
    write_json(record, dict(cells=cells, evidence_scope="software_correctness",
        analysis_seed_pairs=pairs, analysis_source_arms=sources,
        task_manifest={"scopes": {"one": {"study": "A", "participant_key": "authority:p1", "K": 200}}}))
    arguments = ["--record", str(record), "--stage", "P2", "--studies", *STUDIES,
                 "--source-seeds", "1", "2", "--target-seeds", "10", "20",
                 "--bootstrap-iterations", "1000", "--allow-software-correctness", "--output", str(output)]
    main(arguments)
    result = read_json(output)
    assert result["declared_seed_pairs"] == [list(p) for p in pairs]
    assert result["source_records"][0]["evidence_scope"] == "software_correctness"
    assert {g["group"]["source_arm"] for g in result["groups"]} == set(sources)
    cross = result["source_objective_comparisons"]["groups"][0]["contrasts"]
    for contrast in cross.values():
        panel = contrast["exact_K"]["200"]
        assert panel["mean_difference"] == pytest.approx(.04)
        assert panel["n_complete_participants"] == 3
    # A CLI declaration cannot quietly overwrite the execution design.
    with pytest.raises(ValueError, match="seed.*declaration"):
        main(arguments[:-1] + [str(tmp_path / "mismatch.json"), "--seed-pair", "1", "20",
                               "--seed-pair", "2", "10"])


def test_p2_missing_source_is_a_failed_comparison_not_an_inferred_smaller_design():
    from stats.transfer import analyze_source_objective_contrasts

    cells = [{**row, "source_arm": "P2_source_ce", "arm": arm, "auc": value}
             for row in _cells() if row["arm"] == "source"
             for arm, value in (("p2_full_fine", .7), ("p2_scratch", .5), ("p2_waveform_lda", .6))]
    report = analyze_source_objective_contrasts(cells, source_arms=("P2_source_ce", "P2_source_mmd"),
        expected_studies=STUDIES, source_seeds=SOURCE_SEEDS, target_seeds=TARGET_SEEDS, iterations=1000)
    for contrast in report["groups"][0]["contrasts"].values():
        panel = contrast["exact_K"]["200"]
        assert panel["n_complete_participants"] == 0 and panel["n_requested_participants"] == 3
        assert panel["mean_difference"] is None


def test_missing_expected_study_has_no_full_study_mean_or_interval():
    rows = [row for row in _cells() if row["study"] == "A"]
    panel = _analyze(rows)["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["mean_difference"] is None
    assert panel["bootstrap"]["interval"] is None
    assert panel["negative_fraction_study_equal"] is None
    assert panel["partial_observed_studies"]["studies"] == ["A"]
    assert panel["partial_observed_studies"]["mean_difference"] == pytest.approx(.2)
    assert panel["missing_result_bounds"]["interval"] is None


def test_missing_result_bounds_use_contrast_mass_and_equal_study_weights():
    weights = {"interaction": {"a": 1., "b": 1., "c": -1., "d": -1.}}
    cells = [{**row, "arm": arm, "auc": row["auc"] if arm == "a" else .5}
             for row in _cells() if row["arm"] == "source" for arm in weights["interaction"]]
    missing = {"study": "A", "participant_key": "authority:p4", "actual_K": 200}
    panel = _analyze(cells, contrasts=weights, requested_participants=[missing, missing])["groups"][0]["contrasts"]["interaction"]["exact_K"]["200"]
    assert panel["n_requested_participants"] == 4 and panel["n_complete_participants"] == 3
    assert panel["failed_or_missing_seed_pairs"] == 4
    assert panel["by_study"]["A"]["missing_result_bounds"] == pytest.approx([-1.6 / 3, 2.4 / 3])
    assert panel["by_study"]["A"]["contrast_bound"] == 2
    assert panel["missing_result_bounds"]["interval"] == pytest.approx([(-1.6 / 3 - .2) / 2, (2.4 / 3 - .2) / 2])
    assert panel["missing_result_bounds"]["is_confidence_interval"] is False
    assert panel["mean_difference"] == pytest.approx(0.)


@pytest.mark.parametrize("invalid_auc", [float("nan"), float("inf"), -.01, 1.01])
def test_invalid_auc_is_visible_in_arm_descriptives_and_does_not_enter_summary(invalid_auc):
    import json

    cells = _cells()
    next(row for row in cells if row["study"] == "B" and row["arm"] == "source")["auc"] = invalid_auc
    report = _analyze(cells)
    arm = report["groups"][0]["arm_descriptives"]["source"]["200"]
    assert arm["mean_auc"] is None
    assert arm["n_complete_participants"] == 2
    assert arm["failed_or_missing_seed_pairs"] == 1
    assert arm["by_study"]["B"]["mean_auc"] is None
    json.dumps(report, allow_nan=False)


def test_known_actual_k_requests_create_missing_groups_without_guessing_unknown_groups():
    cells = [{**row, "source_arm": "mmd"} for row in _cells()]
    requests = [
        {"study": "A", "participant_key": "authority:missing", "actual_K": 200, "source_arm": "ce"},
        {"study": "B", "participant_key": "authority:unknown", "actual_K": 200},
        {"study": "B", "participant_key": "authority:excluded", "requested_Ks": [200],
         "status": "protocol_excluded", "source_arm": "ce"},
    ]
    report = _analyze(cells, requested_participants=requests, group_fields=("source_arm",))
    groups = {group["group"]["source_arm"]: group for group in report["groups"]}
    missing = groups["ce"]["contrasts"]["transfer"]["exact_K"]["200"]
    assert missing["n_requested_participants"] == 1 and missing["n_complete_participants"] == 0
    assert missing["by_study"]["A"]["missing_result_bounds"] == [-1., 1.]
    assert missing["mean_difference"] is None and missing["missing_result_bounds"]["interval"] is None
    assert groups["mmd"]["contrasts"]["transfer"]["exact_K"]["200"]["n_requested_participants"] == 3
    unresolved = report["outer_request_denominator"]["unresolved_group_requests"]
    assert len(unresolved) == 1 and unresolved[0]["participant_key"] == "authority:unknown"
    assert unresolved[0]["missing_group_fields"] == ["source_arm"]


def test_all_requested_actual_k_results_can_be_missing_without_losing_denominator():
    requests = [{"study": study, "participant_key": f"authority:{study}", "actual_K": 200} for study in STUDIES]
    panel = _analyze([], requested_participants=requests)["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["n_requested_participants"] == 2 and panel["n_complete_participants"] == 0
    assert panel["requested_seed_pairs"] == panel["failed_or_missing_seed_pairs"] == 8
    assert panel["mean_difference"] is None and panel["missing_result_bounds"]["interval"] == [-1., 1.]


@pytest.mark.parametrize("pairs", [4, "1,10", [1, 2], ["1,10", (2, 20)], [{1: 10}, (2, 20)]])
def test_malformed_seed_pair_containers_raise_contract_value_error(pairs):
    from stats.transfer import resolve_seed_pairs

    with pytest.raises(ValueError, match="seed pairs"):
        resolve_seed_pairs(SOURCE_SEEDS, TARGET_SEEDS, seed_pairs=pairs)


def test_missing_fallback_request_uses_its_actual_k_and_not_requested_standard_k():
    request = {"study": "A", "participant_key": "authority:p4", "actual_K": 1600, "requested_Ks": [2000]}
    report = _analyze(_cells(), requested_participants=[request])["groups"][0]["contrasts"]["transfer"]
    assert report["exact_K"]["2000"]["n_requested_participants"] == 0
    fallback = report["fallback_actual_K"]["1600"]
    assert fallback["n_requested_participants"] == 1 and fallback["n_complete_participants"] == 0
    assert fallback["by_study"]["A"]["missing_result_bounds"] == [-1., 1.]
    assert report["budget_policy"]["n_requested_participants"] == 4


def test_identification_bounds_respect_study_specific_contrast_mass():
    contrast = {"transfer": {"by_study": {"A": {"source": 2., "scratch": -2.},
                                            "B": {"source": 1., "scratch": -1.}}}}
    requests = [{"study": study, "participant_key": f"authority:{study}", "actual_K": 200} for study in STUDIES]
    panel = _analyze([], contrasts=contrast, requested_participants=requests)["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["by_study"]["A"]["missing_result_bounds"] == [-2., 2.]
    assert panel["by_study"]["B"]["missing_result_bounds"] == [-1., 1.]
    assert panel["missing_result_bounds"]["interval"] == [-1.5, 1.5]


def test_study_specific_contrast_does_not_redefine_fixed_study_estimand():
    contrast = {"only_A": {"by_study": {"A": {"source": 1., "scratch": -1.}}}}
    panel = _analyze(_cells(), contrasts=contrast)["groups"][0]["contrasts"]["only_A"]["exact_K"]["200"]
    assert panel["mean_difference"] is None and panel["bootstrap"]["interval"] is None
    assert panel["partial_observed_studies"]["studies"] == ["A"]
    assert panel["partial_observed_studies"]["mean_difference"] == pytest.approx(.2)
