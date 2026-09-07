# Mainsah S2 metadata and conversion admission

Date: 2026-09-07. Decision: **the reference and EDF/BDF integrity blockers are resolved**.
Mainsah S2 is admitted at the metadata/integrity layer and remains a sealed external target.
No target model was trained, predicted, scored, or selected during this audit.
This does not authorize an external evaluation before freezing the development recipe and target protocol.

## Acquisition reference

The university-hosted 2017 thesis by Marissa R. Jones, *Faces, Locations, and Tools: A Proposed
Two-Stimulus P300 Brain Computer Interface*, provides direct acquisition evidence:

- PDF page 27 / printed page 26, Experiment 2: "Channels were referenced to the right mastoid
  and grounded to the left mastoid."
- PDF pages 27-28 / printed pages 26-27: 32 channels, 256 Hz, two 16-channel g.USBamp amplifiers,
  acquisition bandpass 0.5-30 Hz; eight channels selected for historical BCI operation.
- PDF page 42 / printed page 41, Experiment 3: 24 participants and "Data acquisition and processing
  as described in Experiment 2 was also used for Experiment 3."
- PDF pages 42-43 / printed pages 41-42 identify House and Tool calibration, followed by copy spelling.

Sources:

1. [ETSU university thesis landing page](https://dc.etsu.edu/etd/3268/).
2. [University PDF archived on 2024-04-14](https://web.archive.org/web/20240414104610id_/https://dc.etsu.edu/cgi/viewcontent.cgi?article=4706&context=etd).
   The live PDF is currently blocked by the publisher platform; the archived PDF was read.
3. [PhysioNet dataset description](https://physionet.org/content/bigp3bci/1.0.0/) links Study S2
   to Jones and Sellers (2019), DOI [10.1088/1741-2552/aaff22](https://doi.org/10.1088/1741-2552/aaff22).
   Its S2 inventory matches the thesis Experiment 3's 24 participants, one session, House/Tool protocol.
   The 2019 journal full text was not obtained; acquisition evidence is from the university thesis.

Resolved source reference is **right mastoid** and source ground is **left mastoid**.
These facts live in a reviewed metadata overlay, not a rewrite of original `n/a` sidecars.
The thesis specifies tin electrodes whereas original EDF transducer headers specify AgAgCl;
both statements are retained as a material provenance discrepancy. This discrepancy does not
change the documented reference circuit or the verified sample mapping.

## Complete file verification

The old audit overlooked EDFs already present in the NEMAR `sourcedata` directory.
The complete local original collection exists there; a separate empty MOABB cache is irrelevant.

| Check | Result |
|---|---|
| Original EDF file hashes | 288/288 match both NEMAR provenance and live PhysioNet SHA256SUMS |
| Inventory | 24 participants, 12 files each, three per Train/Test x House/Tool cell |
| Channels | All 32 original EEG channels map to all 32 BDF channels; all product seven are measured |
| Waveform check | Every retained sample of every EEG channel, all files, within half a BDF quantization step |
| Largest absolute waveform difference | 1.1718036182226457e-11 V, about 0.00001172 uV |
| Event check | All 207360 event samples and target/non-target labels exactly match original state channels |
| Target events | 17280 total |
| Original / BDF length | 42760 / 42752 samples for every file; eight tail samples removed |
| Original / BDF rate | 256.0000766323896 / 256 Hz |
| Smallest last-event-to-end margin | 1576 samples, 6.15625 s, preserving a 1.2 s post-event epoch |
| Original filter header | HP:0.5Hz LP:30Hz NHP:58Hz NLP:62Hz, all 32 channels and 288 files |

The previously uncertain conversion is now empirically validated over the entire distribution as
sample-preserving tail cropping plus BDF quantization. Its exact historical converter commit remains
unknown and is not claimed to have been reproduced. Integrity no longer depends on that hypothesis:
the complete retained arrays and event labels were directly compared with hash-verified originals.
The audit rejects a gain change even at perfect correlation, a channel swap, a one-sample event shift,
and a target/non-target flip. Three counterexample tests pass.

Derivative 58/62 sidecar cutoff values describe notch edges, not the acquisition bandpass.
When importing BDF, use actual BDF sample indices and 256 Hz consistently. Preserve the original
sampling rate and acquisition filters as provenance. Additional 0.1-30 Hz filtering cannot undo
the acquired 0.5 Hz high-pass or recover removed low frequencies.

## Roles and sealed evaluation boundary

The full ledger maps each BDF run to its original path and binds both file hashes. Lexicographic
run assignment is used only to propose a match; exact waveform/event verification establishes it.
All 288 matches pass, including non-contiguous original record numbers ending in 04.

| BIDS runs | Verified original role |
|---|---|
| 0-2 | Test / House |
| 3-5 | Test / Tool |
| 6-8 | Train / House |
| 9-11 | Train / Tool |

Use the ledger, never numeric BIDS order, to select calibration and subsequent test records.
The thesis establishes two calibration phases followed by copy spelling. Anonymized dates do not
recover each participant's House-versus-Tool order; never invent cross-condition chronology.
House calibration instructs counting; Tool calibration instructs imagining use of the graspable
object. Testing uses a matrix with both stimulus categories and condition-specific attended targets.
`Test/House` and `Test/Tool` therefore do not mean that every presented image in the file belongs
to one category. These task differences require declared condition handling and separate reporting.

The eventual fixed-budget protocol must declare how K is selected from original Train records,
which conditions contribute to K, and the common original Test endpoint. Keep the entire target
out of development fitting and recipe selection. This public speller's binary/character results
cannot replace the final BrainSync nine-choice confirmation.

## Reproduction and evidence

```powershell
.venv/Scripts/python.exe experiments/runs/four_dataset_20260907/fetch_mainsah_methods.py
.venv/Scripts/python.exe experiments/audit_mainsah_s2.py --root mne_data/mainsah2025_s2_nemar_v1.0.3 --output experiments/runs/four_dataset_20260907/mainsah_integrity.json --upstream-manifest experiments/runs/four_dataset_20260907/mainsah_sources/SHA256SUMS.txt --jobs 4
.venv/Scripts/python.exe -m pytest tests/test_mainsah_audit.py -q
```

The audit is CPU-only, bounded to four independent file readers, and reads raw files without mutation.
It neither imports model code nor reads model predictions/performance artifacts.

| Artifact | SHA256 |
|---|---|
| Archived thesis PDF | ff3e62c3964238ca87317f1212523c8363f61512071e0a792423316f76a99222 |
| Methods excerpts JSON | 165fcea7f95559f0b300356dd7ba390587e12bd9f47f3d1c7b34c1a69c958170 |
| Official SHA256SUMS | 75ce052ae8626a73b43887c994c4c0d17e5b0d775ad3083f759af20028e32fbb |
| NEMAR provenance JSON | d40b3f0cebf9a131bbe5d465d982f62530d854ff8e6ff75c883c7dc92a399f43 |
| Complete verification ledger | b2ee59f150f9fbc085ed10272b2854cc2d4a9d718f22e90a793906f285a437f9 |

Artifacts are under `experiments/runs/four_dataset_20260907/mainsah_sources/` and
`experiments/runs/four_dataset_20260907/mainsah_integrity.json`.
The machine-readable corrected metadata is `doc/evidence/mainsah_s2_metadata_20260907.json`.
