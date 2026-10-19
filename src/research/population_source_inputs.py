"""Project source metadata locations, preserving every scientific field.

Only recipe/matrix/freeze/declaration JSON is decoded. Raw inputs, dictionaries,
marker receipts and historical evidence remain opaque, SHA-bound leaf inputs.
Existing original readers consume the new metadata without a second IO owner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from data.identity import parse_origin_subject_key
from research.execution import publish_exclusive

RECIPE_SCHEMA = "full_original_population_source_recipe/1"
MATRIX_SCHEMA = "public_population_source_execution_matrix_candidate/1"
FREEZE_SCHEMA = "n2p3_H_LOSO31_population_source_freeze_candidate/1"
LOCATION_SCHEMA = "n2p3_population_source_locations/1"


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _reference(path, content):
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _bound_reference(reference):
    if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
            or type(reference.get("bytes")) is not int or reference["bytes"] < 0
            or not isinstance(reference.get("sha256"), str)
            or re.fullmatch("[0-9a-f]{64}", reference["sha256"]) is None):
        raise ValueError("input locations retain an explicit path, byte count and SHA256")
    return reference


def _read_metadata(reference):
    _bound_reference(reference)
    if Path(reference["path"]).suffix.lower() != ".json":
        raise ValueError("only the named recipe/matrix/freeze/declaration JSON can be decoded")
    content = Path(reference["path"]).read_bytes()
    if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError("the original metadata differs from its caller-pinned bytes")
    return json.loads(content), content


def _absolute_locator(value):
    if not isinstance(value, str) or not value:
        raise ValueError("input location requires an explicit absolute filesystem path")
    if re.match(r"^[A-Za-z]:[\\/]", value):
        path = PureWindowsPath(value)
    elif value.startswith("/") and not value.startswith("//") and "\\" not in value:
        path = PurePosixPath(value)
    else:
        raise ValueError("unsupported input location; use an absolute drive or POSIX filesystem path")
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("input location cannot be relative or contain parent traversal")
    return path


class SourceLocationProjection:
    """Explicit disjoint root mappings; they move locators, never identities."""

    def __init__(self, record, *, original_metadata=()):
        if record.get("schema") != LOCATION_SCHEMA:
            raise ValueError("the explicit source location-map schema is required")
        self.metadata_root = _absolute_locator(record["metadata_root"])
        self.rules = tuple((_absolute_locator(row["original_root"]), _absolute_locator(row["actual_root"]))
            for row in record["roots"])
        if not self.rules or len({old for old, _new in self.rules}) != len(self.rules):
            raise ValueError("source roots must be explicit and unique")
        if any(type(new) is not type(self.metadata_root) for _old, new in self.rules):
            raise ValueError("projected inputs and metadata use one declared filesystem flavour")
        self.assets, self.destinations = {}, {}
        self.metadata_destinations = set()
        self.original_metadata = {_absolute_locator(reference["path"]) for reference in original_metadata}

    def _destination(self, original):
        path = _absolute_locator(original)
        selected = []
        for old, actual in self.rules:
            if type(path) is not type(old):
                continue
            try:
                suffix = path.relative_to(old)
            except ValueError:
                continue
            if not suffix.parts:
                raise ValueError("a root rule cannot rename an individual input file")
            selected.append(actual.joinpath(*suffix.parts))
        if len(selected) != 1:
            raise ValueError("each consumed input needs exactly one explicit root mapping: " + original)
        result = selected[0]
        if result.name != path.name:
            raise ValueError("location projection must preserve the original reader filename")
        return result

    def leaf(self, reference, *, role, participant_key=None, record_id=None):
        _bound_reference(reference)
        old = reference["path"]
        actual = self._destination(old)
        if actual in self.metadata_destinations:
            raise ValueError("an original input cannot replace derived metadata")
        previous = self.assets.get(old)
        identity = {"bytes": reference["bytes"], "sha256": reference["sha256"]}
        if previous is not None and previous["identity"] != identity:
            raise ValueError("one original input location has conflicting byte identities")
        alias = self.destinations.get(actual)
        if alias is not None and alias != old:
            raise ValueError("two original inputs cannot alias one projected location")
        self.destinations[actual] = old
        if previous is None:
            previous = {"original_path": old, "actual_path": str(actual), "identity": identity,
                "bindings": [], "payload_opened_by_projector": False, "cloud_copy_verified": False}
            self.assets[old] = previous
        binding = {"role": role, "participant_key": participant_key, "record_id": record_id}
        if binding not in previous["bindings"]:
            previous["bindings"].append(binding)
        return {**reference, "path": str(actual)}

    def metadata_reference(self, relative, content):
        path = self.metadata_root.joinpath(*PurePosixPath(relative).parts)
        if path in self.original_metadata:
            raise ValueError("derived metadata requires a new location distinct from the original source metadata")
        if path in self.destinations:
            raise ValueError("derived metadata cannot replace a projected original input")
        self.metadata_destinations.add(path)
        return _reference(path, content)


def _validate_preparation_locations(output, original, generated):
    protected = {Path(reference["path"]).resolve() for reference, _content in original.values()}
    targets = [output / "original" / name for name in original]
    targets.extend(output / "projected" / name for name in generated)
    targets.append(output / "PROJECTION.json")
    if output.resolve() in protected or any(path.resolve() in protected for path in targets):
        raise ValueError("metadata preparation output cannot occupy an original source metadata location")


def _load_graph(recipe_reference):
    recipe, recipe_bytes = _read_metadata(recipe_reference)
    if recipe["schema"] != RECIPE_SCHEMA:
        raise ValueError("the original full source recipe is required")
    matrix, matrix_bytes = _read_metadata(recipe["source_matrix"])
    if matrix["schema"] != MATRIX_SCHEMA:
        raise ValueError("the original source matrix is required")
    freeze, freeze_bytes = _read_metadata(matrix["source_freeze"])
    if freeze["schema"] != FREEZE_SCHEMA:
        raise ValueError("the original source freeze is required")
    declarations, original = {}, {
        "SOURCE_RECIPE.json": (recipe_reference, recipe_bytes),
        "SOURCE_MATRIX.json": (recipe["source_matrix"], matrix_bytes),
        "SOURCE_FREEZE_CANDIDATE.json": (matrix["source_freeze"], freeze_bytes)}
    for person in freeze["persons"]:
        key = person["participant_key"]
        if key in declarations:
            raise ValueError("source metadata repeats a participant")
        declaration, content = _read_metadata(person["declaration"])
        _dataset, subject = parse_origin_subject_key(key)
        name = f"declarations/participant_{subject}.json"
        declarations[key] = declaration
        original[name] = person["declaration"], content
    return recipe, matrix, freeze, declarations, original


def _validate_graph(recipe, matrix, freeze, declarations):
    people = {row["participant_key"]: row for row in freeze["persons"]}
    requested_people = {row["source_person_id"]: row for row in matrix["person_programmes"]}
    scope = recipe["execution_scope"]
    if (len(people) != len(freeze["persons"]) or len(requested_people) != len(matrix["person_programmes"])
            or set(people) != set(requested_people) or set(people) != set(declarations)
            or len(people) != scope["people"] or scope["people_per_fold"] != len(people) - 1
            or recipe["programme_seed"] != matrix["programme_seed"]
            or recipe["programme_seed"] != freeze["programme_seed"]
            or matrix["geometry"] != freeze["geometry"]):
        raise ValueError("source recipe, geometry, seed and participant identities differ")
    if sorted(freeze["inputs"]["participants"], key=lambda row: row["path"]) != sorted(
            [row["declaration"] for row in people.values()], key=lambda row: row["path"]):
        raise ValueError("the original participant-input references differ from the source people")
    ledger = freeze["source_record_ledger"]
    for row in ledger:
        dataset, subject = parse_origin_subject_key(row["participant_key"])
        if (row["participant_key"] not in people or dataset != "Huebner2017"
                or type(row["session"]) is not int or row["session"] not in (0, 1)
                or type(row["run"]) is not int or type(row["requested_labels"]) is not int
                or row["requested_labels"] < 1
                or row["record_id"] != f"Huebner2017:{subject}:session:{row['session']}:run:{row['run']}"):
            raise ValueError("the complete source ledger has a foreign participant/session/run identity")
    common = [row for row in ledger if row["common_source_admission_candidate"]]
    excluded = [row for row in ledger if not row["common_source_admission_candidate"]]
    if (len({row["record_id"] for row in ledger}) != len(ledger)
            or len(ledger) != matrix["requested_source_records"]
            or len(common) != matrix["common_source_records"]
            or sum(row["requested_labels"] for row in common) != matrix["common_source_labels"]
            or excluded != matrix["not_admitted_source_records"]
            or matrix["pending_new_gaussian_score_cells"] != scope["new_gaussian_score_cells"]
            or matrix["completed_historical_baseline_score_cells"] != scope["completed_historical_score_cells"]):
        raise ValueError("source requested/common/excluded/label or score denominators differ")
    selected = {}
    for key, person in people.items():
        dataset, subject = parse_origin_subject_key(key)
        declaration, planned = declarations[key], requested_people[key]
        if (dataset != "Huebner2017" or declaration["dataset_id"] != dataset
                or declaration["source_subject_id"] != subject
                or declaration["source_format"] != "original_vmrk_plus_NEMAR_BDF"
                or planned["declaration"] != person["declaration"]):
            raise ValueError("a source declaration changed its original participant or decoder identity")
        by_record = {}
        for row in declaration["record_sources"]:
            if (str(row["subject"]) != subject or type(row["block"]) is not int
                    or type(row["run"]) is not int):
                raise ValueError("an original declaration row has a different participant/session/run identity")
            record_id = f"Huebner2017:{subject}:session:{row['block']}:run:{row['run']}"
            if record_id in by_record:
                raise ValueError("source declaration repeats an acquisition identity")
            by_record[record_id] = row
        rows = [row for row in common if row["participant_key"] == key]
        all_rows = [row for row in ledger if row["participant_key"] == key]
        record_ids = [row["record_id"] for row in rows]
        if (record_ids != planned["selected_source_records"] or record_ids != person["common_source_record_ids"]
                or [row["record_id"] for row in all_rows] != person["requested_source_record_ids"]
                or person["requested_source_record_ids"] != planned["requested_source_records"]
                or [row["original_BDF_reference"]["sha256"] for row in rows] != planned["selected_eeg_sha256s"]
                or sum(row["requested_labels"] for row in rows) != planned["selected_label_count"]
                or planned["selected_label_count"] != person["common_metadata_labels"]):
            raise ValueError("one source person's complete acquisition/label ledger changed")
        for row in rows:
            record = by_record.get(row["record_id"])
            if (record is None or row["session"] not in (0, 1)
                    or row["record_id"] != f"Huebner2017:{subject}:session:{row['session']}:run:{row['run']}"
                    or record.get("bdf") != row["original_BDF_reference"]):
                raise ValueError("source declaration and original frozen BDF identity differ")
            _validate_record_locations(record)
        selected[key] = {row["record_id"] for row in rows}
    folds = {row["fold_id"]: row for row in freeze["folds"]}
    planned_folds = {row["fold_id"]: row for row in matrix["fold_programmes"]}
    if (len(folds) != len(freeze["folds"]) or len(planned_folds) != len(matrix["fold_programmes"])
            or set(folds) != set(planned_folds) or len(folds) != scope["folds"]
            or len(folds) != len(people)
            or {row["target_participant_key"] for row in folds.values()} != set(people)):
        raise ValueError("the complete source fold roster changed")
    for key, fold in folds.items():
        planned = planned_folds[key]
        sources = fold["source_participant_keys"]
        target = fold["target_participant_key"]
        if (target not in people or set(sources) != set(people) - {target}
                or len(sources) != len(people) - 1 or planned["target_participant_key"] != target
                or planned["source_person_references"] != sources or planned["source_person_count"] != len(sources)
                or planned["source_labels"] != sum(requested_people[p]["selected_label_count"] for p in sources)
                or any(planned[field] != fold[field] for field in ("target_C_K", "future_session_id", "future_requested_characters"))):
            raise ValueError("a fold changed its exact other-person source scope or future request")
    return selected


def _validate_record_locations(record):
    subject, session, run = str(record["subject"]), record["block"], record["run"]
    stem = f"sub-{subject}_ses-{session}_task-p300_run-{run}"
    bdf = _absolute_locator(_bound_reference(record["bdf"])["path"])
    expected = {"bdf": stem + "_eeg.bdf", "channels": stem + "_channels.tsv",
        "events": stem + "_events.tsv", "eeg": stem + "_eeg.json"}
    for role, name in expected.items():
        path = _absolute_locator(_bound_reference(record[role])["path"])
        if path.name != name or path.parent != bdf.parent:
            raise ValueError("input projection cannot hide a foreign physical acquisition filename or parent")
    marker = record["marker"]
    if "archive_path" in marker:
        raise ValueError("this source projection consumes the admitted extracted-marker format")
    _bound_reference(marker)
    _bound_reference(marker["original_member_receipt"])
    member = marker["original_member"]
    member_pattern = rf"subject{int(subject):02d}/(?:block_without_optical/)?matrixSpeller_Block{session + 1}_Run{run + 1}\.vmrk"
    if re.fullmatch(member_pattern, member) is None:
        raise ValueError("the original marker member belongs to another participant/session/run")
    if "source_header" in marker:
        _bound_reference(marker["source_header"])


def _project_record(record, projection, *, participant_key, record_id):
    result = copy.deepcopy(record)
    for role in ("bdf", "channels", "events", "eeg"):
        result[role] = projection.leaf(record[role], role=role,
            participant_key=participant_key, record_id=record_id)
    marker = result["marker"] = projection.leaf(record["marker"], role="marker",
        participant_key=participant_key, record_id=record_id)
    for role in ("original_member_receipt", "source_header"):
        if role in record["marker"]:
            marker[role] = projection.leaf(record["marker"][role], role="marker." + role,
                participant_key=participant_key, record_id=record_id)
    _validate_record_locations(result)
    return result


def inspect_population_source_inputs(recipe_reference):
    recipe, matrix, freeze, declarations, _original = _load_graph(recipe_reference)
    selected = _validate_graph(recipe, matrix, freeze, declarations)
    deferred = []
    for key, declaration in declarations.items():
        for row in declaration["record_sources"]:
            record_id = f"Huebner2017:{row['subject']}:session:{row['block']}:run:{row['run']}"
            if record_id not in selected[key]:
                deferred.append({"participant_key": key, "record_id": record_id,
                    "scope": "future" if row["block"] == 2 else "not_common_source", "payload_opened": False})
    return {"people": len(declarations), "folds": len(freeze["folds"]),
        "requested_records": len(freeze["source_record_ledger"]), "common_records": matrix["common_source_records"],
        "excluded_records": len(matrix["not_admitted_source_records"]), "labels": matrix["common_source_labels"],
        "new_score_cells": matrix["pending_new_gaussian_score_cells"],
        "old_baseline_cells": matrix["completed_historical_baseline_score_cells"],
        "geometry": freeze["geometry"], "programme_seed": freeze["programme_seed"],
        "deferred_acquisitions": deferred, "opaque_payloads_opened": 0}


def project_population_source_inputs(recipe_reference, location_map, output_directory):
    """Create a new immutable metadata graph; never inspect/copy opaque inputs."""
    recipe, matrix, freeze, declarations, original = _load_graph(recipe_reference)
    selected = _validate_graph(recipe, matrix, freeze, declarations)
    projection = SourceLocationProjection(location_map,
        original_metadata=[reference for reference, _content in original.values()])
    generated, declaration_refs, replacements = {}, {}, {}
    for key, declaration in declarations.items():
        subject = declaration["source_subject_id"]
        result = copy.deepcopy(declaration)
        for index, row in enumerate(declaration["record_sources"]):
            record_id = f"Huebner2017:{subject}:session:{row['block']}:run:{row['run']}"
            if record_id in selected[key]:
                result["record_sources"][index] = _project_record(row, projection,
                    participant_key=key, record_id=record_id)
                replacements[record_id] = result["record_sources"][index]["bdf"]
        name = f"declarations/participant_{subject}.json"
        generated[name] = _encode(result)
        declaration_refs[key] = projection.metadata_reference(name, generated[name])
    new_freeze = copy.deepcopy(freeze)
    for row in new_freeze["persons"]:
        row["declaration"] = declaration_refs[row["participant_key"]]
    old_people = {row["declaration"]["path"]: row["participant_key"] for row in freeze["persons"]}
    new_freeze["inputs"]["participants"] = [declaration_refs[old_people[row["path"]]]
        for row in freeze["inputs"]["participants"]]
    for row in new_freeze["source_record_ledger"]:
        if row["common_source_admission_candidate"]:
            row["original_BDF_reference"] = replacements[row["record_id"]]
    name = "SOURCE_FREEZE_CANDIDATE.json"
    generated[name] = _encode(new_freeze)
    new_matrix = copy.deepcopy(matrix)
    new_matrix["source_freeze"] = projection.metadata_reference(name, generated[name])
    for row in new_matrix["person_programmes"]:
        row["declaration"] = declaration_refs[row["source_person_id"]]
    name = "SOURCE_MATRIX.json"
    generated[name] = _encode(new_matrix)
    new_recipe = copy.deepcopy(recipe)
    new_recipe["source_matrix"] = projection.metadata_reference(name, generated[name])
    for role in ("dictionary", "native_dictionary", "scale_policy"):
        new_recipe[role] = projection.leaf(recipe[role], role=role)
    name = "SOURCE_RECIPE.json"
    generated[name] = _encode(new_recipe)
    new_recipe_reference = projection.metadata_reference(name, generated[name])
    # Reuse the original cross-document checks on the actual projected bytes.
    projected_declarations = {key: json.loads(generated[f"declarations/participant_{value['source_subject_id']}.json"])
        for key, value in declarations.items()}
    _validate_graph(new_recipe, new_matrix, new_freeze, projected_declarations)
    output = Path(output_directory)
    _validate_preparation_locations(output, original, generated)
    output.mkdir(parents=True, exist_ok=False)
    retained = []
    for name, (reference, content) in original.items():
        path = output / "original" / name
        publish_exclusive(path, lambda stream, raw=content: stream.write(raw))
        retained.append({"original": reference, "retained": _reference(path.resolve(), content)})
    projected = []
    for name, content in generated.items():
        path = output / "projected" / name
        publish_exclusive(path, lambda stream, raw=content: stream.write(raw))
        projected.append({"relative": name, "materialized": _reference(path.resolve(), content),
            "cloud_destination": projection.metadata_reference(name, content)})
    receipt = {"schema": "population_source_input_location_projection/1", "status": "metadata_prepared_not_cloud_executed",
        "original_recipe": recipe_reference, "projected_recipe": new_recipe_reference,
        "location_map": location_map, "original_documents": retained, "projected_documents": projected,
        "opaque_inputs": sorted(projection.assets.values(), key=lambda row: row["original_path"]),
        "science_fields_preserved": True, "original_metadata_bytes_preserved": True,
        "held_and_noncommon_rows_retained_unmapped": True,
        "original_marker_receipt_bytes_preserved": True, "opaque_payloads_opened": 0,
        "fits": 0, "training_authority_granted": False,
        "source_contract": {"people": len(declarations), "folds": len(freeze["folds"]),
            "requested_records": len(freeze["source_record_ledger"]), "common_records": matrix["common_source_records"],
            "excluded_records": len(matrix["not_admitted_source_records"]), "labels": matrix["common_source_labels"],
            "new_score_cells": matrix["pending_new_gaussian_score_cells"],
            "old_baseline_cells": matrix["completed_historical_baseline_score_cells"]}}
    content = _encode(receipt)
    publish_exclusive(output / "PROJECTION.json", lambda stream: stream.write(content))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--locations", required=True, type=Path)
    parser.add_argument("--locations-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    recipe_bytes = args.recipe.read_bytes()
    location_bytes = args.locations.read_bytes()
    if (hashlib.sha256(recipe_bytes).hexdigest() != args.recipe_sha256
            or hashlib.sha256(location_bytes).hexdigest() != args.locations_sha256):
        raise ValueError("the caller-pinned original recipe or explicit location map changed")
    result = project_population_source_inputs(_reference(args.recipe.resolve(), recipe_bytes),
        json.loads(location_bytes), args.output)
    print(json.dumps({"status": result["status"], "projected_recipe": result["projected_recipe"],
        "source_contract": result["source_contract"], "opaque_payloads_opened": 0, "fits": 0}, indent=2))


if __name__ == "__main__":
    main()
