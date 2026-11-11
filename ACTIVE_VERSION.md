# Active version: three-model mixture experts

This branch is the current `main` working-tree consolidation with the three-model candidate mixture enabled as the project version: `NN source`, `waveform LDA`, and `xDAWN ERP`.

Primary entrypoint: `experiments/run_candidate_ensemble_study.py`.
Core implementation: `src/transfer/ensemble.py`, `src/transfer/ensemble_fit.py`, `src/transfer/ensemble_views.py`.

The branch contains source, tests, configuration, and integrated documentation only. Data, cloud inputs, caches, result payloads, model files, environments, Git history, and historical frozen manifests are excluded.
