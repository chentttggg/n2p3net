# BI2014a Candidate-v2 Cache Evidence

The local raw source contains all 64 public BI2014a subject CSV/MAT pairs under
Zenodo record `3266223`. The causal candidate cache was rebuilt from the CSV
event columns after restoring decision boundaries from raw codes 100/104 and
target-pair changes.

Cache contract:

- shape: `61013 x 16 x 128`;
- preprocessing: `p300_single_subject_causal_v2`;
- filter: 2-30 Hz forward IIR with `steady_state_first_sample`;
- cache SHA-256: `252237a8ebae8e5edfe1bdf3ec7bcc28fd2adb815fa0c60323e666a372fb75f5`;
- byte size: `474972098`;
- source reference: right earlobe;
- candidate task: 6x6 row/column speller.

The `.npz` cache and 8.56GB raw source are intentionally not tracked in Git.
The adjacent record, candidate audit, and four-block manifest are sufficient to
verify an externally supplied cache, not to recreate the raw data themselves.

Rebuild with the project environment:

```powershell
.\.venv\Scripts\python.exe experiments\prepare_bi2014a_candidate.py `
  --root mne_data\MNE-braininvaders2014a-data\zenodo\3266223 `
  --output experiments\cache\bi2014a_candidate_causal_v2.npz
```

The four target blocks contain 16 subjects each and were balanced only by
unlabeled epoch count. They are used to train target-block-excluded supervised
source checkpoints; they are not performance-selected folds.

## Cross-Decision Result

The matched experiment ran from Git commit `fc7c8f2` with 12 source
checkpoints and 13 model/normalization arms across three seeds. Compact analysis,
checkpoint metadata and SHA indexes are in `cross_decision/`; the `.pt` files
and 156 full result JSON files remain external.

- requested decisions: 1416;
- eligible at two repetitions: 964 (68.1%);
- zero-shot/source-stats subject-macro hit@2: 19.41%;
- classifier-fine/shrinkage: 19.58%, no reliable paired gain;
- full-fine/shrinkage: 19.89%, no reliable paired gain;
- target-prefix normalization reduced every head;
- linear scratch/shrinkage reduced hit by 4.22 percentage points versus zero-shot.

The compact archive downloaded from the cloud had SHA-256
`9643ff74fca209434e3c3c17cefa6987cb4e84bc8d280f721b98f457f316a8e0`.
BI2014a is a 6x6 character mechanism benchmark, not 9-choice BrainSync
confirmation.
