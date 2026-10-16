# Public ERP Source Candidate

`bnci_cz6_seed20260906_A01.pt` is an existing N2P3 supervised source checkpoint
from the public BNCI2014-008 development study. It is a research initializer,
not a validated BrainSync personal model or a claim of decoding accuracy.

Original file:
`D:/BCI/n2p3-net/experiments/runs/p300_mainline_20260906/retrieved/p300_mainline_20260906/bnci6cz_loso_s20260906_A01.pt`.

The physical input is six measured-Cz difference channels:
FZ, P3, PZ, P4, PO8, OZ; the raw acquisition remains seven EEG plus EOG.
The backend explicitly applies the chosen Cz reference before personal
calibration. Channel order, preprocessing, source identity and reference are
checked by the existing checkpoint loader. An incompatible source does not
silently fall back to LDA.

Training uses source seed 20260906 with public A01 excluded. BrainSync personal
calibration and later test records were not used to choose this source file.
Its original training metadata and weights are preserved in the checkpoint.
