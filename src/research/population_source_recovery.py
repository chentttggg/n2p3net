"""Externally accepted partial execution -> named reuse of original person bytes.

This owner grants no original-reader or live data-factor fit authority. Fresh
missing people and all fold fits keep their original numerical owners.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ACCEPTANCE_SCHEMA = "independent_partial_population_source_execution_acceptance/1"
RECIPE_SCHEMA = "full_original_population_source_recipe/3"
_LIVE, _RESTORED = WeakKeyDictionary(), WeakKeyDictionary()


def _bytes(reference):
    payload = Path(reference["path"]).read_bytes()
    if len(payload) != reference["bytes"] or hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("an accepted partial-execution dependency changed")
    return payload


def _read(reference):
    return json.loads(_bytes(reference))


def validate_recovery_recipe(recipe, *, trusted_acceptance_reference):
    """A new lineage recipe preserves every original scientific field."""
    if (recipe["schema"] != RECIPE_SCHEMA
            or recipe["partial_execution_acceptance"] != trusted_acceptance_reference):
        raise ValueError("the caller-pinned partial acceptance and named recovery recipe are required")
    original = _read(recipe["original_recipe"])
    remaining = {key: value for key, value in recipe.items()
                 if key not in {"schema", "original_recipe", "partial_execution_acceptance"}}
    if (original["schema"] != "full_original_population_source_recipe/2"
            or remaining != {key: value for key, value in original.items() if key != "schema"}):
        raise ValueError("person recovery must preserve the original complete scientific recipe")
    return original


def _accepted(reference, matrix_reference, original_recipe_reference):
    record = _read(reference)
    matrix, recipe = _read(matrix_reference), _read(original_recipe_reference)
    if (record["schema"] != ACCEPTANCE_SCHEMA or record["accepted"] is not True
            or record["independent_source_review"] is not True or not record["reviewer"]
            or not record["evidence"] or record["source_matrix"] != matrix_reference
            or record["original_recipe"] != original_recipe_reference
            or record["recovery_scope"] != "reuse_original_same_fit_person_products_only"
            or record["grants_live_source_fit_authority"] is not False
            or recipe["schema"] != "full_original_population_source_recipe/2"
            or recipe["source_matrix"] != matrix_reference
            or recipe["source_product_kind"] != "same_fit_full_response_data_factor_v2"
            or recipe["programme_seed"] != matrix["programme_seed"]):
        raise ValueError("an independently accepted exact original partial programme is required")
    for evidence in record["evidence"]:
        _bytes(evidence)
    manifest = _read(record["original_source_manifest"])
    checkpoint = _read(record["original_checkpoint"])
    launch = _read(record["original_launch"])
    retired = _read(record["retirement"])
    if (manifest["schema"] != "public_population_effective_source_manifest/3"
            or manifest["recipe"] != original_recipe_reference
            or checkpoint["recipe"] != original_recipe_reference
            or checkpoint["source_manifest"] != record["original_source_manifest"]
            or {"pid": checkpoint["pid"], "create_time": checkpoint["process_create_time"]} != record["original_process"]
            or launch["manifest"] != record["original_source_manifest"]
            or retired["schema"] != "root_incomplete_population_source_retirement/1"
            or {"pid": retired["original_actor"]["pid"], "create_time": retired["original_actor"]["created"]} != record["original_process"]
            or retired["original_launch"] != record["original_launch"]
            or retired["source_manifest"] != record["original_source_manifest"]
            or retired["source_recipe"] != original_recipe_reference
            or retired["original_checkpoint"] != record["original_checkpoint"]
            or retired["owner_tree_retired"] is not True
            or retired["all_original_births_absent"] is not True
            or retired["remaining"] != []
            or record["original_process"] not in [
                {"pid": row["pid"], "create_time": row["created"]} for row in retired["original_owned_processes"]]
            or not retired["checked_utc"]
            or any(retired["checkpoint_snapshot"][key] != record["original_checkpoint"][key] for key in ("bytes", "sha256"))):
        raise ValueError("original recipe/launch/checkpoint/process retirement lineage differs")
    # The independent partial checker owns the full retirement evidence audit,
    # including its long telemetry. Consume the pinned actual small witnesses.
    for key in ("checkpoint_snapshot", "stop_requested", "supervision_launch", "supervision_terminal"):
        _bytes(retired[key])
    requested = [row["source_person_id"] for row in matrix["person_programmes"]]
    completed = [row["participant_key"] for row in record["completed_people"]]
    pending = [key for key in requested if key not in completed]
    observed_complete = {key for key, row in checkpoint["persons"].items() if row["status"] == "complete"}
    if (len(requested) != 13 or len(set(requested)) != 13 or len(matrix["fold_programmes"]) != 13
            or record["requested_participant_keys"] != requested
            or not 0 < len(completed) <= len(requested) or len(set(completed)) != len(completed)
            or not set(completed) <= set(requested) or record["pending_participant_keys"] != pending
            or set(checkpoint["persons"]) != set(requested) or observed_complete != set(completed)
            or checkpoint["source_requested"] != 13 or checkpoint["fold_requested"] != 13
            or matrix["pending_new_gaussian_score_cells"] != 4914):
        raise ValueError("partial recovery must partition the original complete thirteen-person programme")
    people = {row["source_person_id"]: row for row in matrix["person_programmes"]}
    for row in record["completed_people"]:
        event = checkpoint["persons"][row["participant_key"]]
        publication = _read(row["publication"])
        person, expected = publication["source_person"], people[row["participant_key"]]
        if (event["publication"] != row["publication"] or event["shared_likelihood"] != row["shared_likelihood"]
                or publication["schema"] != "same_fit_source_person_likelihood_publication/2"
                or person["participant_key"] != row["participant_key"]
                or person["source_freeze"] != matrix["source_freeze"]
                or person["source_declaration"] != expected["declaration"]
                or set(person["source_record_ids"]) != set(expected["selected_source_records"])
                or person["source_labels"] != expected["selected_label_count"]
                or person["future_records_attached"] is not False
                or publication["source_product"] != row["source_product"]
                or publication["shared_likelihood"] != row["shared_likelihood"]
                or publication["noise_fit_programmes"] != 1
                or publication["fixed_noise_response_compression_passes"] != 1
                or publication["auxiliary_physical_posterior_fits"] != 0
                or publication["individual_posterior_used_as_measurement"] is not False):
            raise ValueError("an accepted complete original person differs from its actual checkpoint and matrix")
    return record


@dataclass(frozen=True, eq=False)
class AcceptedPersonRecovery:
    acceptance_reference: dict
    record: dict


def open_accepted_person_recovery(trusted_acceptance_reference, *, expected_matrix_reference,
                                 original_recipe_reference):
    record = _accepted(trusted_acceptance_reference, expected_matrix_reference, original_recipe_reference)
    value = AcceptedPersonRecovery(freeze_metadata(trusted_acceptance_reference), freeze_metadata(record))
    _LIVE[value] = (value.acceptance_reference, value.record)
    return value


def require_accepted_person_recovery(value):
    original = _LIVE.get(value) if type(value) is AcceptedPersonRecovery else None
    if original is None or value.acceptance_reference is not original[0] or value.record is not original[1]:
        raise TypeError("the original external-acceptance person-recovery issuer is required")
    return value


def _import_member(source, destination, reference):
    content = _bytes({"path": str(source), "bytes": reference["bytes"], "sha256": reference["sha256"]})
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError("a recovered compact member conflicts with the destination store")
        return "existing_identical_bytes"
    try:
        os.link(source, destination)
        mode = "hardlink_original_bytes"
    except OSError:
        with destination.open("xb") as stream:
            stream.write(content)
        mode = "copied_original_compact_bytes"
    _bytes({"path": str(destination), "bytes": reference["bytes"], "sha256": reference["sha256"]})
    return mode


def restore_accepted_person_products(value, *, shared_store):
    """Cold-verify and import original compact bytes; never mint live fit rights."""
    from research.source_data_factor import load_source_data_factor
    from research.source_likelihood_export import SharedLikelihoodReference, SharedLikelihoodStore

    value = require_accepted_person_recovery(value)
    result, restored, imports = [], {}, []
    for row in value.record["completed_people"]:
        publication = _read(row["publication"])
        old_store = SharedLikelihoodStore(row["shared_store_directory"])
        cold = load_source_data_factor(row["source_product"], shared_store=old_store)
        compact = SharedLikelihoodReference(**row["shared_likelihood"])
        if (metadata_digest(cold.anchor["source_person"]) != metadata_digest(publication["source_person"])
                or cold.product["shared_likelihood"] != publication["shared_likelihood"]
                or cold.likelihood.metadata != publication["source_export_metadata"]
                or cold.moments.record() != publication["stationary_moments"]
                or cold.product["same_fit_block_count"] != publication["same_fit_block_count"]):
            raise ValueError("accepted original product/publication/baseline/moments relations differ")
        source = old_store.directory / compact.manifest_name
        header = _read({"path": str(source), "bytes": compact.bytes, "sha256": compact.sha256})
        array = header["arrays"]
        modes = [_import_member(old_store.directory / array["file"], shared_store.directory / array["file"], array),
                 _import_member(source, shared_store.directory / compact.manifest_name,
                                {"bytes": compact.bytes, "sha256": compact.sha256})]
        shared_store.get(compact)
        restored[row["publication"]["sha256"]] = freeze_metadata(publication)
        result.append((compact, plain_metadata(row["publication"])))
        imports.append({"participant_key": row["participant_key"], "member_modes": modes})
    _RESTORED[value] = restored, imports
    return tuple(result)


def require_recovered_person_publication(value, reference):
    value = require_accepted_person_recovery(value)
    restored = _RESTORED.get(value)
    original = None if restored is None else restored[0].get(reference["sha256"])
    expected = next((row["publication"] for row in value.record["completed_people"]
                     if row["publication"] == reference), None)
    if expected is None or original is None or metadata_digest(_read(reference)) != metadata_digest(original):
        raise TypeError("the unchanged independently accepted and cold-restored person publication is required")
    return original


def execution_recovery_lineage(value, *, matrix_reference, persons):
    value = require_accepted_person_recovery(value)
    if value.record["source_matrix"] != matrix_reference:
        raise ValueError("the recovery belongs to another source matrix")
    reused = {row["participant_key"]: row for row in value.record["completed_people"]}
    actual = {row["participant_key"]: row for row in persons}
    if (len(actual) != len(persons) or set(actual) != set(value.record["requested_participant_keys"])
            or set(actual) - set(reused) != set(value.record["pending_participant_keys"])):
        raise ValueError("complete recovery must retain every original person and fit exactly the missing people")
    for key, row in reused.items():
        require_recovered_person_publication(value, actual[key]["publication"])
        if actual[key]["publication"] != row["publication"] or actual[key]["source_product"] != row["source_product"]:
            raise ValueError("recovery relabeled an original accepted person product")
    return {"schema": "accepted_source_person_recovery_lineage/1",
        "partial_execution_acceptance": plain_metadata(value.acceptance_reference),
        **{key: plain_metadata(value.record[key]) for key in (
            "original_recipe", "original_source_manifest", "original_checkpoint", "original_launch",
            "original_process", "retirement", "requested_participant_keys", "pending_participant_keys")},
        "reused_person_publications": [plain_metadata(row["publication"]) for row in value.record["completed_people"]],
        "newly_fitted_participant_keys": plain_metadata(value.record["pending_participant_keys"]),
        "original_person_producers_preserved": True, "recovery_grants_live_fit_authority": False,
        "producer_sha256": IMPLEMENTATION_SHA256}


def validate_execution_recovery(record):
    """Cold execution validation consumes lineage without issuing a live handle."""
    lineage = record["person_recovery"]
    partial = _accepted(lineage["partial_execution_acceptance"], record["source_matrix"], lineage["original_recipe"])
    expected = {row["participant_key"]: row for row in partial["completed_people"]}
    persons = {row["participant_key"]: row for row in record["persons"]}
    if (lineage["schema"] != "accepted_source_person_recovery_lineage/1"
            or lineage["newly_fitted_participant_keys"] != partial["pending_participant_keys"]
            or lineage["reused_person_publications"] != [row["publication"] for row in partial["completed_people"]]
            or lineage["original_person_producers_preserved"] is not True
            or lineage["recovery_grants_live_fit_authority"] is not False
            or any(lineage[key] != partial[key] for key in (
                "original_recipe", "original_source_manifest", "original_checkpoint", "original_launch",
                "original_process", "retirement", "requested_participant_keys", "pending_participant_keys"))
            or set(persons) != set(partial["requested_participant_keys"])
            or len(persons) != len(record["persons"])):
        raise ValueError("the complete recovered execution changed its accepted original lineage")
    for key, row in expected.items():
        if any(persons[key][field] != row[field] for field in ("publication", "source_product", "shared_likelihood")):
            raise ValueError("a recovered execution replaced an original accepted person reference")
    recipes = [_read(reference) for reference in record["recipes"]]
    recovery_recipes = [recipe for recipe in recipes if recipe.get("schema") == RECIPE_SCHEMA]
    if len(recovery_recipes) != 1 or recovery_recipes[0]["original_recipe"] != lineage["original_recipe"]:
        raise ValueError("a recovered complete execution requires its one named recovery recipe")
    validate_recovery_recipe(recovery_recipes[0], trusted_acceptance_reference=lineage["partial_execution_acceptance"])
