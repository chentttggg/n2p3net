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

Before every cloud training launch or resume, apply the
`performance-optimization` and `get-available-resources` skills and follow the
[cloud throughput procedure](doc/engineering.zh.md#云端训练性能优先).
Default to resource-aware parallel execution across preparation, source fits,
target calibration and evaluation. The user prioritizes measured training
throughput over low code complexity: implement the scheduling, caching and
parallelism needed to remove bottlenecks. Preserve scientific contracts and
completed artifacts; verify actual utilization and progress after launch.

When the user requests training, execute the full agreed matrix unless they
explicitly limit the run. Perform necessary tests and short probes autonomously,
then continue directly to full training without another approval round. Do not
stop at a pilot, dry-run, execution plan, or an unfavorable pilot result.

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
