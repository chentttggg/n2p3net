"""Small explicit input contracts for statistics-only fixtures."""
from research.comparison import INPUT_IDENTITY_SCHEMA
from research.contracts import semantic_sha256


def comparison_identity(*, person="fixture", fit_count=200):
    def scope(name, count):
        return {"n_rows": count, **{field: semantic_sha256([person, name, field, count])
                for field in ("evidence_sha256", "values_sha256", "labels_sha256", "observed_sha256", "rows_sha256")}}
    record = {"schema": INPUT_IDENTITY_SCHEMA, "fit": scope("fit", fit_count),
              "score": scope("score", 20), "raw_bytes_bound": False,
              "operator": {"preprocessing": {"baseline_mode": "none"}, "spatial": {},
                           "raw_support": [], "execution": None},
              "decision_semantics": {"event_contract": None, "candidate_task_contract": None,
                  "candidate_vocabulary": None, "session_schema": None, "timeline_complete": True,
                  "requested_schedule_sha256": semantic_sha256([person, "schedule"])},
              "fit_context": {"timeline_sha256": semantic_sha256([person, "timeline"]),
                  "metadata_sha256": semantic_sha256([person, "metadata"]), "qc_features_sha256": None}}
    return {**record, "digest": semantic_sha256(record)}
