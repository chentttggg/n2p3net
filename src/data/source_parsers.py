"""Format adapters for verified continuous public ERP records.

Readers preserve physical sample positions and binary truth. They never filter,
train, download, or silently split continuous recordings into new runs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np


def _huebner_sidecars(source):
    if not str(source).endswith("_eeg.bdf"):
        raise ValueError("Huebner source must use the explicit _eeg.bdf filename")
    base = str(source).removesuffix("_eeg.bdf")
    return {role: Path(base + f"_{role}.tsv") for role in ("channels", "events")}


def _read_existing_record(item, path):
    study, subject = item["study"], item["subject"]
    if study == "GTN":
        from data.gtn import read_gtn_experiment

        gtn = read_gtn_experiment(path)
        if gtn.subject_id != subject:
            raise ValueError("GTN internal identity differs from verified manifest")
        return gtn.raw, gtn.events, (gtn.events[:, 2] == gtn.thought_number).astype(int)
    if study in {"BI2012", "BI2013a", "BI2014a", "BI2015a"}:
        from moabb import datasets

        dataset = getattr(datasets, study)()
        dataset.data_path = lambda *_args, **_kwargs: [str(path)]
        sessions = dataset._get_single_subject_data(int(subject))
        raws = [raw for runs in sessions.values() for raw in runs.values()]
        if len(raws) != 1:
            raise ValueError("one declared source record must parse into one continuous raw")
        raw = raws[0]
        events = mne.find_events(raw, shortest_event=1, verbose=False)
        events = events[np.isin(events[:, 2], list(dataset.event_id.values()))]
        return raw, events, (events[:, 2] == dataset.event_id["Target"]).astype(int)
    if study == "EPFLP300":
        from moabb.datasets import EPFLP300

        raw = EPFLP300()._get_single_run_data(str(path))
    elif study == "Chailloux2020":
        from moabb.datasets import Chailloux2020

        raw = Chailloux2020._load_run(path)
    elif study == "Speier2017":
        from moabb.datasets.speier2017 import Speier2017

        raw = Speier2017._load_bci2000(path)
    elif study == "BNCI2014-008":
        from data.bnci2014_008_candidate import (
            bnci2014_008_raw,
            read_bnci2014_008_candidate_record,
        )

        with Path(path).open("rb") as handle:
            record = read_bnci2014_008_candidate_record(handle, source_relative_path=Path(path).name,
                                                       source_sha256=item["sha256"])
        if str(item["subject"]) != record.subject_id or item["session"] != "0" or item["run"] != "0":
            raise ValueError("BNCI manifest identity must match its official MAT subject and recording")
        return (*bnci2014_008_raw(record), record)
    elif study == "Huebner2017":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
        sidecars = _huebner_sidecars(path)
        with sidecars["channels"].open(encoding="utf-8-sig") as stream:
            channels = list(csv.DictReader(stream, delimiter="\t"))
        if [c["name"] for c in channels] != raw.ch_names:
            raise ValueError("Huebner raw and sidecar channel order differs")
        if any(c["status"] not in {"good", "bad"} for c in channels):
            raise ValueError("Huebner channels require explicit good/bad status")
        raw.set_channel_types({c["name"]: c["type"].lower() for c in channels})
        raw.info["bads"] = [c["name"] for c in channels if c["status"] == "bad"]
        with sidecars["events"].open(encoding="utf-8-sig") as stream:
            rows = [r for r in csv.DictReader(stream, delimiter="\t") if r["trial_type"] in {"Target", "NonTarget"}]
        samples = np.array([int(r["sample"]) for r in rows])
        onsets = np.array([float(r["onset"]) for r in rows])
        if not np.allclose(samples / raw.info["sfreq"], onsets, atol=.5/raw.info["sfreq"], rtol=0):
            raise ValueError("Huebner sample/onset mismatch")
        labels = np.array([r["trial_type"] == "Target" for r in rows], int)
        return raw, np.column_stack([samples, np.zeros(len(samples), int), labels + 1]), labels
    else:
        raise ValueError(f"unsupported explicit source parser: {study}")
    events = mne.find_events(raw, shortest_event=1, verbose=False)
    if not set(events[:, 2]) <= {1, 2}:
        raise ValueError("source parser has unexpected binary stimulus codes")
    return raw, events, (events[:, 2] == 2).astype(int)


def source_companions(item, source):
    """Identify all bytes actually consumed by multi-file format adapters."""
    if item["study"] == "Huebner2017":
        return _huebner_sidecars(source)
    if item["study"] == "GTN":
        nix_files = list(source.parent.glob("*.nix"))
        if len(nix_files) != 1 or nix_files[0].resolve() != source.resolve():
            raise ValueError("GTN manifest must identify the only parsed NIX file")
        companions = {"metadata:" + p.name: p for p in sorted((source.parent / "Data").glob("*.txt"))}
        if not companions:
            raise ValueError("GTN source lacks participant metadata files")
        return companions
    if item["study"] == "Chailloux2020":
        # BrainVision names its binary and marker files inside the header.
        import configparser

        text = source.read_text(encoding="utf-8-sig")
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(text[text.index("[Common Infos]"):])
        companions = {}
        for role, field in (("signal", "DataFile"), ("markers", "MarkerFile")):
            name = config["Common Infos"][field]
            if Path(name).name != name or "\\" in name or "/" in name:
                raise ValueError("BrainVision companions must be explicit sibling files")
            companions[role] = source.parent / name
        return companions
    return {}


def _read_cattan(item, path):
    """Preserve the original continuous MAT, including gaps between blocks."""
    from scipy.io import loadmat

    data = loadmat(path)["data"]
    if data.ndim != 2 or data.shape[1] != 22:
        raise ValueError("Cattan continuous MAT requires its original 22 columns")
    expected = f"subject_{int(item['subject']):02d}_{item['session']}.mat"
    if Path(path).name != expected or item["session"] not in {"PC", "VR"}:
        raise ValueError("Cattan subject/condition must match the official MAT filename")
    stop = item.get("valid_samples_exclusive_end", len(data))
    if type(stop) is not int or not 0 < stop <= len(data):
        raise ValueError("invalid declared Cattan sample boundary")
    # A known malformed final sample may be excluded explicitly. Never discard
    # a finite tail, an event, or a larger interval under this format repair.
    if stop != len(data) and (stop != len(data) - 1 or np.isfinite(data[-1]).all()
                              or np.any(data[-1, 18:20] != 0)):
        raise ValueError("Cattan repair only permits one incomplete event-free final sample")
    data = data[:stop]
    if not np.isfinite(data).all() or not np.isin(data[:, 18:20], (0, 1)).all():
        raise ValueError("Cattan signal/markers are nonfinite or nonbinary")
    if np.any(data[:, 18] + data[:, 19] > 1):
        raise ValueError("Cattan target and nontarget markers overlap")
    channels = ["Fp1", "Fp2", "FC5", "Fz", "FC6", "T7", "Cz", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
    raw = mne.io.RawArray(data[:, 1:17].T * 1e-6, mne.create_info(channels, 512, "eeg"), verbose=False)
    raw.set_montage("standard_1020")
    samples = np.flatnonzero(data[:, 18] + data[:, 19])
    labels = data[samples, 18].astype(int)
    return raw, np.column_stack([samples, np.zeros(len(samples), int), labels + 1]), labels


def _read_brainform(item, path):
    """Read verified calibration markers from original CSV, retaining Pxx IDs."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        header = [stream.readline().strip() for _ in range(5)]
    if "#samplingrate: 250" not in header:
        raise ValueError("BrainForm CSV requires its original 250 Hz header")
    people = [part.removeprefix("EXP_") for part in Path(item.get("member", item["path"])).parts
              if part.startswith("EXP_P")]
    if people != [item["subject"]] or Path(path).name.split("__")[0] != item["run"]:
        raise ValueError("BrainForm physical participant/run differs from manifest")
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] != 14 or not np.isfinite(data[:, 8:]).all():
        raise ValueError("BrainForm requires 8 EEG columns and 6 finite marker columns")
    onsets = np.flatnonzero(data[:, 8] == 1)
    markers = data[onsets, 8:]
    if not len(onsets):
        raise ValueError("BrainForm record has no calibration stimuli")
    resets = np.flatnonzero(np.diff(markers[:, 4]) < 0) + 1
    count = min(int(resets[0]) if len(resets) else len(onsets), 600)
    codes = markers[:, 1]
    expected = np.arange(count)
    if (not np.array_equal(np.flatnonzero(data[:, 9] != 0), onsets)
            or not np.isin(markers[:, 2:4], (0, 1)).all()
            or not np.all(markers[:, 2] + markers[:, 3] == 1)
            or not np.array_equal(codes[:count], expected % 10 + 1)
            or not np.array_equal(markers[:count, 4], expected // 10 + 1)
            or not np.array_equal(markers[:count, 2], codes[:count] == 1)
            or np.any(markers[count:, 2])
            or not np.array_equal(markers[:, 5], codes == 10)):
        raise ValueError("BrainForm calibration truth/sequence differs from verified marker protocol")
    channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    excluded = item.get("excluded_channels", [])
    if (not isinstance(excluded, list) or len(set(excluded)) != len(excluded)
            or not set(excluded) <= set(channels) or "Cz" in excluded
            or (excluded and not item.get("channel_exclusion_reason"))):
        raise ValueError("BrainForm channel exclusions require named nonreference EEG and evidence")
    # Audited published BIDS startup trim, with original sample coordinates.
    crop = 1245
    if len(data) <= crop:
        raise ValueError("BrainForm recording ends before its published startup trim")
    raw = mne.io.RawArray(data[crop:, :8].T * 1e-6, mne.create_info(channels, 250, "eeg"),
                          first_samp=crop, verbose=False)
    raw.set_montage("standard_1005")
    raw.info["bads"] = excluded
    samples, labels = onsets[:count], markers[:count, 2].astype(int)
    return raw, np.column_stack([samples, np.zeros(count, int), labels + 1]), labels


SOURCE_READERS = {study: _read_existing_record for study in (
    "GTN", "BI2012", "BI2013a", "BI2014a", "BI2015a", "EPFLP300", "Chailloux2020", "Speier2017", "BNCI2014-008", "Huebner2017")}
SOURCE_READERS.update(BrainForm=_read_brainform, Cattan2019_VR=_read_cattan)


def read_source_record(item, path):
    """Return continuous raw, original events, binary labels and optional membership."""
    if item["study"] not in SOURCE_READERS:
        raise ValueError(f"unsupported explicit source parser: {item['study']}; require a verified continuous-format adapter")
    return SOURCE_READERS[item["study"]](item, path)


def validate_source_manifest(manifest):
    """Validate one study's full record declaration before any preparation starts."""
    if manifest.get("schema") != "n2p3_source_records/1":
        raise ValueError("unsupported raw source records manifest")
    study = manifest.get("study")
    if study not in SOURCE_READERS:
        raise ValueError(f"unsupported explicit source parser: {study}; download completion is not parser admission")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("source manifest requires explicit records")
    identities = set()
    for item in records:
        if item.get("study") != study:
            raise ValueError("source manifest mixes physical study identities")
        if any(item.get(key) is None for key in ("subject", "session", "run")):
            raise ValueError("source manifest requires physical participant/session/run identities")
        identity = tuple(str(item.get(key, "")) for key in ("subject", "session", "run"))
        if not all(identity) or identity in identities:
            raise ValueError("source manifest requires unique participant/session/run records")
        identities.add(identity)
        if not isinstance(item.get("reference"), str) or not item["reference"].strip():
            raise ValueError("source manifest requires an explicit reference declaration")
        path = Path(item["path"])
        if path.is_absolute() or path.drive or ".." in path.parts:
            raise ValueError("source raw paths must remain inside the manifest root")
    return records

