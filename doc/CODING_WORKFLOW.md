# Coding Workflow

The development order is data contract, split protocol, metrics, classical
baselines, compact neural candidates, then transfer experiments. A model cannot
be added before the same test harness can score it.

Each module declares input shape, dtype, metadata dependencies, fitted state,
and error behavior. Tests are required for shape validation, train-only fitting,
group-disjoint splits, metric edge cases, and 9-choice aggregation. Model tests
also assert CPU forward/backward execution and parameter count.

Every experiment writes its immutable config, source manifest hash, outer-fold
predictions, aggregate metrics, latency/memory profile, and package versions.
Changes to the data contract or protocol require a new experiment version; old
results are never compared across an undocumented contract change.
