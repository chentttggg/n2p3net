# Repository Guidance

Read [the document index](doc/README.md) and the relevant active guide before
changing code. Inspect the current source and Git status; dated evidence records
the state when it was written.

- Research constraints: [research program](doc/research_program.zh.md).
- Implementation and ownership: [technical structure](doc/technical_structure.zh.md).
- Signal and timing contracts: [input contract](doc/input_contract_math.zh.md).
- Stage, recipes and acceptance: [roadmap](doc/roadmap.zh.md).
- Environment, validation and maintenance: [engineering](doc/engineering.zh.md).
- Acquisition and personal inference: [workstation](doc/workstation.zh.md).

Use the repository virtual environment. Dependencies are declared in
`pyproject.toml` and resolved in `uv.lock`. Run Ruff, relevant contract and
counterexample tests, and `git diff --check`; use the full suite for broad changes.
Keep builders, callers, checkpoint gates, tests and active guidance consistent.

Preserve concurrent work and original data. Participant identity, channel roles,
fit scope and time boundaries come from the shared contracts, never guessed
aliases. Distinguish software tests, public-data development and adult/device
confirmation. Bind each experiment to the source that actually executed it.

Update the existing six guides instead of creating competing guidance. Put
independent evidence in `doc/evidence/` and index it. Physically archive and verify
superseded research material before retiring it. Historical archives, raw data,
caches, results and credentials stay local; archive manifests stay in Git.
See [archive distribution](frozen/README.md). This checkout contains the backend;
the BrainSync acquisition application is a separate local project.
