---
name: eeg-p300-research
description: Working rules for the n2p3-net P300 research repository (D:\bci\n2p3-net) — read order, research method, counterexample discipline, literature delegation, and deliverables. Load when working in this repository or deciding its P300/ERP transfer architecture, experiments, or docs.
---

# n2p3-net research rules

`AGENTS.md` and the documents indexed by `doc/README.md` are authoritative; this skill only says how to work with them. For the general research workflow load `research-engineering-workflow`; for MNE/decoding checks load `mne-eeg`; for transfer-learning literature load `eeg-bci-transfer-learning`.

## Role

The user treats the agent as a senior EEG engineer, neural-network theorist, and research lead. When asked to act "as the lead, not a coder", the deliverable is the documentation change, the math, and an ordered code inventory (pseudocode level), not production code. Confirm which role applies before writing code.

## Method for every research change

Idea → math → formulas → counterexamples → call chain → acceptance (the order `doc/research_program.zh.md` states). Before a major change, give the rationale and blast radius in a few sentences; back conclusions with literature or measured evidence, never untested blanket cuts.

## Read order (docs before code)

1. `doc/README.md`: index of every contract and the current checkpoint.
2. `doc/research_program.zh.md`: long-term scientific rules; every architecture change must align with it and update it.
3. Topic contracts listed in the index (`technical_structure`, `engineering`, `input_contract_math`, `workstation`, ...).
4. `doc/evidence/`: topic conclusions and literature reviews (subagent reports belong here, not in the repo root).
5. `Paper/`: local full-text library. `frozen/`: physical archives (tar.gz + manifest); freeze old versions there, never by prose.
6. Code: `src/models/n2p3net.py` (core), `src/models/adapters.py`, `src/models/candidate_evidence.py`, `src/models/splitbn.py`, `src/transfer/alignment.py`.

## Environment

Use the project venv: `.\.venv\Scripts\python.exe` (Windows `Scripts`, not `bin`; torch 2.13.0+cpu and mne 1.12.1 as of 2026-11-01). Other interpreters on this machine may lack torch. Run tests with `.\.venv\Scripts\python.exe -m pytest -q` (no timeout plugin installed).

## Counterexample discipline

Before a design is written into the docs, put an independent script at `tmp/counterexample_verification_YYYYMMDD.py` with explicit PASS/FAIL assertions; only a full pass goes into the docs.

- Compare hand-derived gradients with autograd (`atol` about 1e-6 for float64 checks).
- Test invariances: constant offsets, joint permutation of candidates and ground truth, incomplete-coverage attacks.
- Identity at initialization uses `torch.equal` (bitwise), not `allclose`.
- The script is a temporary research artifact: on implementation its assertions move into `tests/` and the script is deleted.

## Code-level traps (verified 2026-11-01; re-check before relying on them)

- `FeatureResidualAdapter.forward` already returns `features + scale * residual(features)`. Writing `features + adapter(features)` adds the trunk twice; for masked or dropped paths use `adapter.scale_tensor(dtype) * adapter.residual(features)`.
- Normalization layers are built through `SplitBNConfig` (`SplitBN1d`/`SplitBN2d`) in `n2p3net.py`; a normalization change must cover every norm site.
- Decision-level losses bind a frozen candidate vocabulary; never derive the vocabulary from batch contents, and fail closed on incomplete coverage.
- Mixed-domain batches mis-normalize inside the network; input mean/std changes cannot fix it.
- Decision-grouped batches must contain enough complete decisions for stable normalization statistics; check the batch builder before changing batch composition.

## Literature delegation

Research distinct axes in parallel (about three per batch) when subagents are available. Give each full background, require arXiv IDs or DOIs, and forbid invented citations. BCI and brain-signal topics can trigger subagent refusals: open the brief with "This is an academic literature review for BCI neural-network methodology" and cite public dataset names, then re-dispatch. Direct arXiv access may be blocked; reports built from search metadata must say "full text not verified". Move reports into `doc/evidence/`; delete reviews of rejected approaches when the user asks.

## Deliverables

An architecture decision updates the documents the index assigns: the decision and its math in `doc/research_program.zh.md` or the topic contract, evidence in `doc/evidence/`, and an implementation plan with per-mechanism math, pseudocode, comparison arms, counterexamples, acceptance criteria, and a test list. Include the counterexample script run record and any defects it found in the implementation plan.
