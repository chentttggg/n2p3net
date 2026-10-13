"""Original prefix input factories shared by storage and dispatcher tests."""
import hashlib
from pathlib import Path

from research.prefix_replay_inputs import export_prepared_views
from tests.support.independent_prefix import independent_prefix_case


def prefix_input_request(views, spec, profiles):
    import transfer.ensemble_views as owner

    prefix = views.calibration[max(views.calibration)]
    return {"schema": "n2p3_prefix_replay_input_request/1", "preparation_task_id": "prepare-original-inputs",
        "original_source_sha256": prefix.target_contract["source_sha256"],
        "producer_source_sha256": hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
        "participant_key": prefix.target_contract["participant_key"],
        "source_relative_path": prefix.target_contract["source_relative_path"],
        "requested_Ks": views.ledger["requested_Ks"], "actual_Ks": sorted(views.calibration),
        "profiles": [profile.record() for profile in profiles], "authorization_preprocessing": spec.record(),
        "observation_response_seconds": 1.2, "pre_roll_seconds": .2}


def prepared_prefix_inputs(root):
    _, _, _, spec, experts, views = independent_prefix_case(root)
    profiles = (experts[1].profile,)
    request = prefix_input_request(views, spec, profiles)
    descriptor = export_prepared_views(views, profiles, root / "prepared", request)
    return views, profiles, request, descriptor
