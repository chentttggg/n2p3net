"""Final admission checks for the bounded B12 public runner.

The source and target adapters already own their physical data checks.  This
module owns the small cross-stage check that is only possible after a
participant result has been assembled: every requested K/arm/seed cell is
represented, unavailable and failed cells remain in the denominator, and the
source and target producer identities are still attached to the result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from research.contracts import semantic_sha256

_CELL_STATUSES = frozenset({"completed", "failed", "protocol_unavailable"})


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"B12 {field} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"B12 {field} must be a nonempty string")
    return value


def _declared_pairs(plan: Mapping, report: Mapping) -> tuple[tuple[int, int], ...]:
    pairs = plan.get("seed_pairs", report.get("seed_pairs"))
    if not pairs and plan.get("source_seed_catalog") is None:
        targets = report.get("target_seeds", (0,))
        if isinstance(targets, Sequence) and targets:
            pairs = [(0, target) for target in targets]
    if not isinstance(pairs, Sequence) or not pairs:
        raise ValueError("B12 execution requires an explicit source/target seed pair grid")
    result = []
    for pair in pairs:
        if not isinstance(pair, Sequence) or len(pair) != 2 or any(
            type(seed) is not int or seed < 0 for seed in pair
        ):
            raise ValueError("B12 seed pairs must contain two nonnegative integer seeds")
        result.append((pair[0], pair[1]))
    if len(set(result)) != len(result):
        raise ValueError("B12 seed pair grid contains duplicate source/target identities")
    return tuple(result)


def validate_b12_execution_manifest(report: Mapping, plan: Mapping) -> dict:
    """Validate the final source-to-target ledger and return its binding.

    This is deliberately a manifest check, not a second scientific evaluator.
    Physical whole-study exclusion remains in ``_nn_anchor`` and the ensemble
    fitter, where both checkpoint and target dataset membership are available.
    """

    if not isinstance(report, Mapping) or report.get("schema") != "n2p3_erp_comparator_result/2":
        raise ValueError("B12 admission requires an ERP comparator result")
    if not isinstance(plan, Mapping) or plan.get("schema") != "n2p3_erp_comparator_plan/1":
        raise ValueError("B12 admission requires an ERP comparator plan")
    arms = plan.get("arms")
    subjects = plan.get("subjects")
    budgets = plan.get("budgets")
    if not isinstance(arms, Sequence) or not arms or not isinstance(subjects, Sequence) or not subjects:
        raise ValueError("B12 plan must declare nonempty subject and arm rosters")
    if not isinstance(budgets, Sequence) or not budgets or any(type(value) is not int or value < 0 for value in budgets):
        raise ValueError("B12 plan must declare nonnegative requested K values")
    arm_names = tuple(_require_text(arm.get("name"), field="arm name") for arm in arms)
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("B12 arm roster contains duplicate producer names")
    subject_names = tuple(_require_text(subject, field="subject") for subject in subjects)
    if len(set(subject_names)) != len(subject_names):
        raise ValueError("B12 subject roster contains duplicate identities")
    pairs = _declared_pairs(plan, report)

    cells = report.get("cells")
    requests = report.get("requested_participants")
    if not isinstance(cells, Sequence) or not isinstance(requests, Sequence):
        raise ValueError("B12 result must preserve both cells and requested_participants")
    expected = {
        (subject, arm, budget, source, target)
        for subject in subject_names
        for arm in arm_names
        for budget in budgets
        for source, target in pairs
    }
    observed = set()
    referenced_indices = set()
    requests_by_cell = {}
    for request in requests:
        if not isinstance(request, Mapping):
            raise ValueError("B12 requested ledger contains a non-record")
        index = request.get("cell_index")
        if type(index) is not int or not 0 <= index < len(cells):
            raise ValueError("B12 requested ledger references no actual cell")
        referenced_indices.add(index)
        cell = cells[index]
        if not isinstance(cell, Mapping):
            raise ValueError("B12 requested ledger references a non-record cell")
        subject = request.get("subject", cell.get("subject"))
        key = (subject, request.get("arm"), request.get("requested_K"), request.get("source_seed"), request.get("target_seed"))
        if key not in expected or key in observed:
            raise ValueError("B12 requested ledger is incomplete or duplicates a scientific cell")
        if tuple(cell.get(name) for name in ("subject", "arm", "source_seed", "target_seed")) != (key[0], key[1], key[3], key[4]):
            raise ValueError("B12 requested ledger differs from its resolved cell")
        requested_ks = cell.get("requested_Ks", ())
        if request["requested_K"] not in requested_ks:
            raise ValueError("B12 requested K is absent from its resolved K alias ledger")
        actual_k = cell.get("actual_K")
        if actual_k is not None and (type(actual_k) is not int or actual_k < 0):
            raise ValueError("B12 actual K is not a nonnegative integer")
        if request.get("actual_K") != actual_k:
            raise ValueError("B12 actual K differs from its resolved cell")
        requests_by_cell.setdefault(index, []).append(request["requested_K"])
        observed.add(key)
    if observed != expected:
        raise ValueError("B12 requested ledger does not cover the complete K/arm/seed roster")

    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError("B12 cells contain a non-record")
        if sorted(cell.get("requested_Ks", ())) != sorted(requests_by_cell.get(index, ())):
            raise ValueError("B12 cell's requested K aliases differ from its complete request ledger")
        if cell.get("status") not in _CELL_STATUSES:
            raise ValueError("B12 cell has no terminal success/failure/unavailable status")
        if cell["status"] != "completed" and not _require_text(cell.get("error"), field="cell failure"):
            raise ValueError("B12 failed or unavailable cell lacks a failure reason")
    if referenced_indices != set(range(len(cells))):
        raise ValueError("B12 cells contain an unrequested or omitted resolved cell")

    coverage = report.get("requested_budget_coverage")
    if (not isinstance(coverage, Sequence) or len(coverage) != len(budgets)
            or any(not isinstance(row, Mapping) for row in coverage)
            or {row.get("requested_K") for row in coverage} != set(budgets)):
        raise ValueError("B12 result lacks one requested coverage row per K")
    for row in coverage:
        requested = sum(key[2] == row["requested_K"] for key in expected)
        counts = tuple(row.get(name) for name in ("requested_cells", "completed_cells", "protocol_unavailable_cells", "failed_cells"))
        if counts[0] != requested or any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("B12 requested coverage changes its denominator")
        if sum(counts[1:]) != counts[0]:
            raise ValueError("B12 requested coverage drops a completed, unavailable or failed cell")
        actual_counts = tuple(sum(request["requested_K"] == row["requested_K"]
            and cells[request["cell_index"]]["status"] == status for request in requests)
            for status in ("completed", "protocol_unavailable", "failed"))
        if counts[1:] != actual_counts:
            raise ValueError("B12 requested coverage disagrees with actual terminal cell statuses")

    authority = report.get("participant_authority")
    if not isinstance(authority, Mapping) or set(authority) - set(subject_names):
        raise ValueError("B12 target producer authority does not cover the fixed subject roster")
    unresolved_subjects = set(subject_names) - set(authority)
    if unresolved_subjects and any(cell["status"] == "completed" and cell["subject"] in unresolved_subjects for cell in cells):
        raise ValueError("B12 completed cell lacks its target producer identity")
    for subject in authority:
        receipt = authority[subject]
        if not isinstance(receipt, Mapping) or receipt.get("subject") != subject:
            raise ValueError("B12 target producer authority has an inconsistent subject identity")
        _require_text(receipt.get("study"), field="target study identity")
        _require_text(receipt.get("participant_key"), field="target participant identity")
        if not isinstance(receipt.get("identity_record"), Mapping):
            raise ValueError("B12 target producer authority lacks its identity record")

    input_manifests = report.get("input_manifests")
    if not isinstance(input_manifests, Mapping) or not input_manifests:
        raise ValueError("B12 target producer manifest is missing")
    target_producers = {
        _require_text(path, field="target producer path"): _sha256(digest, field="target producer")
        for path, digest in input_manifests.items()
    }

    source_catalog = report.get("source_seed_catalog")
    source_producers = []
    unresolved_sources = []
    needs_source = any(arm.get("family") in {"full_fine", "scratch"} for arm in arms)
    if plan.get("source_seed_catalog") is not None and needs_source:
        declared = {}
        for entry in plan["source_seed_catalog"]:
            key = (entry.get("outer_target_study"), entry.get("source_seed"))
            if (key in declared or not isinstance(key[0], str)
                    or type(key[1]) is not int or key[1] not in {source for source, _ in pairs}):
                raise ValueError("B12 declared source catalog has invalid or duplicate study/seed keys")
            digest = entry.get("sha256", entry.get("checkpoint_sha256"))
            _sha256(digest, field="declared source checkpoint")
            if not isinstance(entry.get("procedure"), Mapping):
                raise ValueError("B12 declared source catalog lacks its procedure")
            declared[key] = (digest, entry["procedure"])
        if isinstance(source_catalog, Mapping):
            source_catalog = source_catalog.get("entries")
        if source_catalog is None:
            source_catalog = []
        if not isinstance(source_catalog, Sequence):
            raise ValueError("B12 public source pipeline lacks admitted source producer records")
        studies = {authority[subject]["study"] for subject in authority}
        expected_sources = {(study, source) for study in studies for source, _ in pairs}
        required_sources = {(authority[cell["subject"]]["study"], cell["source_seed"])
            for cell in cells if cell["status"] == "completed" and cell["subject"] in authority
            and cell["arm"] in {arm["name"] for arm in arms if arm.get("family") in {"full_fine", "scratch"}}}
        seen_sources = set()
        for entry in source_catalog:
            if not isinstance(entry, Mapping):
                raise ValueError("B12 source producer catalog contains a non-record")
            key = (entry.get("outer_target_study"), entry.get("source_seed"))
            if key in seen_sources:
                raise ValueError("B12 source producer catalog duplicates a study/seed key")
            if key not in declared or (not unresolved_subjects and key not in expected_sources):
                raise ValueError("B12 source producer catalog contains an undeclared study/seed")
            seen_sources.add(key)
            _sha256(entry.get("checkpoint_sha256"), field="source checkpoint producer")
            if not isinstance(entry.get("procedure"), Mapping):
                raise ValueError("B12 source producer lacks its executed procedure")
            if (entry["checkpoint_sha256"] != declared[key][0]
                    or semantic_sha256(entry["procedure"]) != semantic_sha256(declared[key][1])):
                raise ValueError("B12 source checkpoint or procedure differs from the declared producer")
            if entry.get("producer_digest") is not None:
                _sha256(entry["producer_digest"], field="source producer")
            if plan.get("evidence_scope") in {"public_development", "public_calibration_development"}:
                producer = entry.get("producer")
                if not isinstance(producer, Mapping) or entry.get("producer_digest") != semantic_sha256(producer):
                    raise ValueError("B12 public source producer is missing or differs from its digest")
                expected_selection = "per_source_seed" if entry["procedure"]["kind"] == "selection_refit" else "not_applicable"
                if (producer.get("source_seed") != key[1]
                        or producer.get("kind") != entry["procedure"]["kind"]
                        or producer.get("selection") != expected_selection
                        or not isinstance(producer.get("source_studies"), Sequence)
                        or not producer["source_studies"]
                        or key[0] in producer["source_studies"]
                        or semantic_sha256(entry.get("source_studies")) != semantic_sha256(producer["source_studies"])):
                    raise ValueError("B12 public source producer contradicts its seed, procedure or study exclusion")
            source_producers.append({"outer_target_study": key[0], "source_seed": key[1],
                                    "checkpoint_sha256": entry["checkpoint_sha256"],
                                    "producer_digest": entry.get("producer_digest"),
                                    "procedure": entry["procedure"]})
        if required_sources - seen_sources:
            raise ValueError("B12 completed target cells lack their required source producer")
        # Missing producers of failed actors stay explicitly unresolved. A
        # failed peer cannot weaken source authority for any completed cell.
        unresolved_sources = sorted((expected_sources if not unresolved_subjects else set(declared)) - seen_sources)

    status = ("unresolved_target_authority" if unresolved_subjects else
              "unresolved_source_producers" if unresolved_sources else "bound")
    return {"schema": "n2p3_b12_execution_binding/1", "status": status,
            "unresolved_subjects": sorted(unresolved_subjects),
            "unresolved_source_producers": [{"outer_target_study": study, "source_seed": seed}
                                             for study, seed in unresolved_sources],
            "plan_digest": semantic_sha256(plan),
            "report_digest": semantic_sha256({key: value for key, value in report.items()
                                               if key != "b12_execution_binding"}), "arm_roster": list(arm_names),
            "subjects": list(subject_names), "requested_Ks": list(budgets),
            "seed_pairs": [list(pair) for pair in pairs], "source_producers": source_producers,
            "target_producers": target_producers,
            "failure_accounting": {str(row["requested_K"]): {key: row[key] for key in (
                "requested_cells", "completed_cells", "protocol_unavailable_cells", "failed_cells")}
                                   for row in coverage}}
