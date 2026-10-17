# Repository Guidance

Start with [doc/README.md](doc/README.md), then read the relevant guide and actual
source, callers and Git changes. The index routes rules, evidence and checkpoints
to their authoritative owners.

- Scientific rules and final nine-choice task: [research program](doc/research_program.zh.md).
- Module ownership and integration: [technical structure](doc/technical_structure.zh.md).
- Signal, identity and timing: [input contract](doc/input_contract_math.zh.md).
- Research dependencies: [roadmap](doc/roadmap.zh.md).
- Environment, verification and delivery: [engineering](doc/engineering.zh.md).
- Acquisition and personal inference: [workstation](doc/workstation.zh.md).

Use the repository virtual environment. Preserve concurrent work and original
scientific artifacts; bind comparisons to actual inputs and executed source.
Review affected modules and run checks that distinguish the changed behavior.

Before cloud launch or resume, apply `performance-optimization` and
`get-available-resources` and follow the
[cloud throughput procedure](doc/engineering.zh.md#云端训练性能优先).

Delivery includes [old-build retirement](doc/engineering.zh.md#交付与旧构建清理).
After confirming the replacement, delete obsolete build outputs with cmd tools;
handle failed batches one explicit target at a time and record actual outcomes.

This checkout contains the backend. BrainSync acquisition is a separate local
project. Archive distribution follows [frozen/README.md](frozen/README.md).
