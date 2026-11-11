import json

import numpy as np
import pytest

from data.domain import adapt_observed_montage
from data.eog_preprocessing import EOGPreprocessingPolicy
from data.epochs import load_epoch_dataset, save_epoch_dataset
from data.source_records import SIX, file_sha256
from experiments import merge_consistent_source_pools as pooling
from tests.train.test_prepare_multidomain_source import _source_dataset


def test_real_pool_writer_keeps_source_policy_and_matches_early_late_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(pooling, "STUDIES", ("A", "B"))
    queue = {"completed": {}}
    for pipeline in ("basic", "QC"):
        for index, study in enumerate(("A", "B")):
            data = _source_dataset(source_id=study, subject="01", offset=float(index))
            data.X[:, 0] = 0.  # Known measured Cz is zero in this physical fixture.
            if pipeline == "QC":
                data.provenance["source_reference"] = "measured_channel:CZ"
            data = adapt_observed_montage(data, SIX, reference_channel="Cz" if pipeline == "basic" else None,
                preserve_reference_label=pipeline == "QC")
            data.provenance["artifact_preprocessing_policy"] = EOGPreprocessingPolicy(
                pipeline + "_eog_ols_v1", pipeline == "QC", "record_unavailable", continuous_qc=pipeline == "QC").record()
            path = save_epoch_dataset(tmp_path / f"{pipeline}_{study}.npz", data)
            admission = tmp_path / f"{pipeline}_{study}_admission.json"
            admission.write_text(json.dumps({"fixture": True, "records": 1}))
            queue["completed"][pipeline + ":" + study] = {"cache": str(path), "cache_sha256": file_sha256(path),
                "preparation": str(admission), "records": 1}
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue))
    binding = tmp_path / "binding.json"
    arguments = ["--queue", str(path), "--output", str(tmp_path / "pools"), "--binding", str(binding),
                 "--memory-gib", ".128"]
    pooling.main(arguments)
    result = json.loads(binding.read_text())
    assert result["status"] == "completed"
    basic, qc = (load_epoch_dataset(result["pools"][p]["cache"]) for p in ("basic", "QC"))
    np.testing.assert_array_equal(basic.X, qc.X)
    assert basic.n_epochs == qc.n_epochs == 4
    assert basic.provenance["artifact_preprocessing_policy"]["enabled"] is False
    assert qc.provenance["artifact_preprocessing_policy"]["enabled"] is True
    # The same publication can be re-admitted, while changed parent identities cannot.
    pooling.main(arguments)
    queue["completed"]["basic:A"]["cache_sha256"] = "f"*64
    path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="parent identities changed"):
        pooling.main(arguments)
