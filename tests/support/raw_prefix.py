"""Real BDF containers with deterministic synthetic EEG for raw-prefix tests."""

import json

import numpy as np

from data.contract import BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT
from data.epochs import preprocessing_spec_from_contract
from data.source_records import file_sha256


def write_bdf(source, seed=508):
    import edfio

    samples = np.random.default_rng(seed).normal(0, 4, (3, 26 * 250))
    edfio.Bdf([edfio.BdfSignal(values, sampling_frequency=250, label=name,
        physical_dimension="uV", physical_range=(-100, 100))
        for name, values in zip(("Fz", "Cz", "Pz"), samples, strict=True)]).write(source)
    return source


def write_real_record(tmp_path):
    source = write_bdf(tmp_path / "person_eeg.bdf")
    (tmp_path / "person_channels.tsv").write_text(
        "name\ttype\tstatus\nFz\tEEG\tgood\nCz\tEEG\tgood\nPz\tEEG\tgood\n", encoding="utf-8")
    (tmp_path / "person_events.tsv").write_text("sample\tonset\ttrial_type\n" + "".join(
        f"{(i+2)*250}\t{i+2}\t{'Target' if i % 2 else 'NonTarget'}\n" for i in range(20)), encoding="utf-8")
    item = dict(study="Huebner2017", subject="1", path=source.name, sha256=file_sha256(source),
                sample_rate_hz=250., reference="native common", session="s1", run="r1")
    spec = preprocessing_spec_from_contract(BRAIN_SYNC_BIDIRECTIONAL_DATA_CONTRACT)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "n2p3_source_records/1", "study": item["study"],
        "remote_root": str(tmp_path), "records": [item]}), encoding="utf-8")
    return item, spec, manifest

