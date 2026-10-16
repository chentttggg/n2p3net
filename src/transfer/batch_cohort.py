"""Predeclared B0 participant and actual-budget coverage, including exclusions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from research.contracts import semantic_sha256
from transfer.binary import matched_split_digests

SCHEMA = "n2p3_target_batch_cohort/1"


def _cases(manifest: Mapping, study: str) -> list[dict]:
    cases = []
    for scope in manifest["scopes"].values():
        if scope["K"] == 0:
            continue
        split = scope["inner_split"]
        digests = matched_split_digests(np.asarray(scope["inner_training_rows"], dtype=int),
                                        np.asarray(scope["inner_validation_rows"], dtype=int))
        cases.append({"study": study, "participant_key": scope["participant_key"], "K": scope["K"],
                      "eligible": split["legal"], "reason": split.get("reason"),
                      "calibration_rows_sha256": digests["calibration_rows_sha256"],
                      "validation_rows_sha256": digests["test_rows_sha256"]})
    return cases


def declare_batch_cohort(preflights: Sequence[Mapping], design: Mapping[str, Sequence[str]]) -> dict:
    """Bind complete production preflights to an independently declared roster."""
    requested = {(study, person) for study, people in design.items() for person in people}
    if not requested or len(requested) != sum(len(people) for people in design.values()):
        raise ValueError("batch cohort design requires unique nonempty study/participant identities")
    observed, cases, excluded = set(), {}, []
    for record in preflights:
        manifest = record.get("task_manifest", record)
        if manifest.get("stage") != "B0" or manifest.get("dry_run_fits_data") is not False:
            raise ValueError("batch cohort requires B0 no-fit preflight manifests")
        study = record["target_study"]
        for scope in manifest["scopes"].values():
            observed.add((study, scope["participant_key"]))
        for item in manifest.get("protocol_exclusions", []):
            observed.add((study, item["participant_key"]))
            excluded.append({"study": study, **item})
        for case in _cases(manifest, study):
            key = (study, case["participant_key"], case["K"])
            if key in cases:
                raise ValueError("duplicate B0 participant/actual K preflight")
            cases[key] = case
    if observed != requested:
        raise ValueError("complete predeclared participant coverage is required for every study")
    record = {"schema": SCHEMA, "expected_studies": sorted(design),
              "participant_keys_by_study": {study: sorted(people) for study, people in design.items()},
              "requested_participants": len(requested), "cases": [cases[key] for key in sorted(cases)],
              "protocol_exclusions": excluded,
              "eligible_actual_K_cases": sum(case["eligible"] for case in cases.values()),
              "preflight_records_sha256": [semantic_sha256(item) for item in preflights]}
    return {**record, "digest": semantic_sha256(record)}


def validate_batch_cohort(cohort: Mapping) -> dict:
    if not isinstance(cohort, Mapping) or cohort.get("schema") != SCHEMA:
        raise ValueError("B0 requires a versioned predeclared batch cohort")
    record = {key: value for key, value in cohort.items() if key != "digest"}
    if semantic_sha256(record) != cohort.get("digest"):
        raise ValueError("batch cohort digest disagrees with its declared scope")
    return dict(cohort)


def verify_batch_cohort_manifest(manifest: Mapping, study: str, cohort: Mapping) -> None:
    """Allow distributed subject shards, each carrying every declared actual K."""
    cohort = validate_batch_cohort(cohort)
    requested = set(cohort["participant_keys_by_study"].get(study, []))
    people = {scope["participant_key"] for scope in manifest["scopes"].values()}
    people.update(item["participant_key"] for item in manifest.get("protocol_exclusions", []))
    if not people <= requested:
        raise ValueError("B0 execution contains participants outside the declared cohort")
    actual = {(row["participant_key"], row["K"]): row for row in _cases(manifest, study)}
    expected = {(row["participant_key"], row["K"]): row for row in cohort["cases"]
                if row["study"] == study and row["participant_key"] in people}
    if actual != expected:
        raise ValueError("B0 execution changed the predeclared actual K, inner rows or eligibility")


def verify_batch_cohort_results(rows: Sequence[Mapping], cohort: Mapping) -> None:
    cohort = validate_batch_cohort(cohort)
    cases = {(case["study"], case["participant_key"], case["K"]): case
             for case in cohort["cases"] if case["eligible"]}
    expected = {(*key, batch, arm) for key in cases for batch in (32, 128, 512)
                for arm in ("source", "scratch")}
    observed = set()
    for row in rows:
        if row["budget_mode"] != "epochs":
            continue
        case_key = (row["study"], row["participant_key"], row["K"])
        key = (*case_key, row["batch_size"], row["arm"])
        if key in observed or key not in expected:
            raise ValueError("duplicate or unrequested B0 cohort result")
        observed.add(key)
        if any(row[field] != cases[case_key][field]
               for field in ("calibration_rows_sha256", "validation_rows_sha256")):
            raise ValueError("B0 result rows differ from the predeclared cohort")
    if observed != expected:
        raise ValueError("complete predeclared B0 cohort results are required, including failed cells")


def freeze_batch_cohort_rule(rows: list[dict], cohort: Mapping) -> dict:
    """Produce a usable shared rule only after full declared coverage is checked."""
    from transfer.target_experiments import freeze_target_batch

    cohort = validate_batch_cohort(cohort)
    verify_batch_cohort_results(rows, cohort)
    rule = freeze_target_batch(rows)
    rule.update(batch_cohort_digest=cohort["digest"], cohort_validation="predeclared_complete",
                requested_participants=cohort["requested_participants"], expected_studies=cohort["expected_studies"])
    return rule
