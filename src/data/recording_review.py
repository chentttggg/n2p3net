"""Content-bound recording exclusions authorised by the data owner."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files


class RecordingExcludedError(ValueError):
    def __init__(self, session_id, scheduled_trials, review):
        self.session_id = session_id
        self.scheduled_trials = scheduled_trials
        self.review = dict(review)
        super().__init__(f"Recording {session_id} excluded: {review['reason']}")


def recording_reviews():
    record = json.loads(files("data").joinpath("recording_reviews.json").read_text(encoding="utf-8"))
    if record.get("schema") != "brainsync_recording_reviews/1":
        raise ValueError("unsupported recording review registry")
    return {entry["raw_sha256"]: entry for entry in record["recordings"]}


def require_admitted_recording(bids, session_id):
    reviews = recording_reviews()
    if not reviews:
        return
    binding = bids.content_record()["files"].get("recording")
    if binding is None:
        with bids.raw_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        digest = binding["sha256"]
    review = reviews.get(digest)
    if review is not None and review["status"] == "excluded":
        raise RecordingExcludedError(session_id, len(bids.stimuli), review)
