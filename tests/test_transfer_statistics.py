from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stats.transfer import analyze_transfer_contrasts, standard_training_contrasts


def _cells():
    rows = []
    for study, person, delta in (("A", "p1", 0.1), ("A", "p2", 0.3), ("B", "p3", -0.2)):
        for source_seed in (1, 2):
            for target_seed in (10, 20):
                for arm, value in (("source", 0.5 + delta), ("scratch", 0.5)):
                    rows.append(dict(
                        study=study, participant_key=f"authority:{person}", K=200,
                        source_seed=source_seed, target_seed=target_seed, arm=arm, auc=value,
                        status="completed", calibration_rows_sha256="a" * 64,
                        test_rows_sha256="b" * 64,
                    ))
    return rows


def _analyze(cells):
    return analyze_transfer_contrasts(
        cells, {"transfer": {"source": 1.0, "scratch": -1.0}}, expected_studies=("A", "B"),
        source_seeds=(1, 2), target_seeds=(10, 20), iterations=1000,
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


def test_protocol_excluded_person_stays_requested_without_inventing_actual_k():
    requests = [{"study": study, "participant_key": f"authority:{person}", "requested_Ks": [200, 800, 2000]}
                for study, person in (("A", "p1"), ("A", "p2"), ("B", "p3"))]
    requests.append({"study": "B", "participant_key": "authority:p4", "requested_Ks": [200, 800, 2000],
                     "status": "protocol_excluded", "reason": "all positive K lack a legal test"})
    report = analyze_transfer_contrasts(
        _cells(), {"transfer": {"source": 1, "scratch": -1}}, expected_studies=("A", "B"),
        source_seeds=(1, 2), target_seeds=(10, 20), requested_participants=requests, iterations=1000,
    )
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
    output = analyze_transfer_contrasts(
        factorial, {name: contrasts[name] for name in ("I", "B", "W", "IxB")},
        expected_studies=("A", "B"), source_seeds=(1, 2), target_seeds=(10, 20), iterations=1000,
    )["groups"][0]["contrasts"]
    expected = {"I": 0.14, "B": 0.08, "W": 0.02, "IxB": 0.08}
    for name, value in expected.items():
        assert output[name]["exact_K"]["200"]["mean_difference"] == pytest.approx(value)


def test_compact_contrasts_pair_single_ablations_without_requiring_factorial_arms():
    from transfer.recipes import compact_target_recipes

    values = dict(zip((recipe.name for recipe in compact_target_recipes()),
                      (0.6, 0.5, 0.7, 0.65, 0.68, 0.62), strict=True))
    rows = [{**row, "arm": arm, "auc": value}
            for row in _cells() if row["arm"] == "source" for arm, value in values.items()]
    rows[0].update(status="fit_failed", auc=None)
    output = analyze_transfer_contrasts(
        rows, standard_training_contrasts("compact"), expected_studies=("A", "B"),
        source_seeds=(1, 2), target_seeds=(10, 20), iterations=1000,
    )["groups"][0]["contrasts"]
    for name, difference in {"full_minus_scratch": 0.2, "full_minus_lda": 0.1,
                             "source_statistics_minus_full": -0.05,
                             "frozen_bn_buffers_minus_full": -0.02,
                             "head_only_minus_full": -0.08}.items():
        panel = output[name]["exact_K"]["200"]
        assert panel["mean_difference"] == pytest.approx(difference)
        assert panel["n_requested_participants"] == 3
        assert panel["n_complete_participants"] == (2 if name == "full_minus_lda" else 3)


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
    result = analyze_transfer_contrasts(
        rows, standard_s2_contrasts(source_arms), expected_studies=("A", "B"),
        source_seeds=(1, 2), target_seeds=(10, 20), arm_field="source_arm", group_fields=("arm",), iterations=1000,
    )["groups"][0]
    assert result["group"] == {"arm": "frozen_target_recipe"}
    assert result["contrasts"]["full_multi_minus_single_mean"]["exact_K"]["200"]["mean_difference"] == pytest.approx(0.1)


def test_analysis_cli_from_outside_repository_preserves_record_hashes(tmp_path):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"cells": _cells(), "schema": "test_stage/1", "evidence_scope": "software_correctness"}), encoding="utf-8")
    contrasts = tmp_path / "contrasts.json"
    contrasts.write_text(json.dumps({"transfer": {"source": 1, "scratch": -1}}), encoding="utf-8")
    output = tmp_path / "analysis.json"
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "analyze_training_chain.py"),
         "--record", str(record), "--stage", "E3", "--studies", "A", "B", "--source-seeds", "1", "2",
         "--target-seeds", "10", "20", "--contrast-json", str(contrasts), "--bootstrap-iterations", "1000",
         "--output", str(output), "--allow-software-correctness"], cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(output.read_text())
    assert report["analysis_seed"] == 20260907
    assert len(report["source_records"][0]["sha256"]) == 64
    assert report["confirmation_claim"] is False


def test_analysis_cli_rejects_software_records_unless_explicitly_kept_software(tmp_path):
    from experiments.analyze_training_chain import main

    record = tmp_path / "record.json"
    cells = _cells()
    cells[0]["evidence_scope"] = "software_correctness"
    record.write_text(json.dumps({"cells": cells, "evidence_scope": "development"}), encoding="utf-8")
    contrasts = tmp_path / "contrasts.json"
    contrasts.write_text(json.dumps({"transfer": {"source": 1, "scratch": -1}}), encoding="utf-8")
    output = tmp_path / "analysis.json"
    argv = ["--record", str(record), "--stage", "E3", "--studies", "A", "B", "--source-seeds", "1", "2",
            "--target-seeds", "10", "20", "--contrast-json", str(contrasts), "--bootstrap-iterations", "1000",
            "--output", str(output)]
    with pytest.raises(SystemExit):
        main(argv)
    assert not output.exists()
    main([*argv, "--allow-software-correctness"])
    report = json.loads(output.read_text())
    assert report["evidence_scope"] == "software_correctness"
    assert report["inference_scope"] == "software_correctness_only_no_population_inference"
    panel = report["groups"][0]["contrasts"]["transfer"]["exact_K"]["200"]
    assert panel["bootstrap"]["scope"] == report["inference_scope"]
