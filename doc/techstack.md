# Technology Stack

- Python 3.14 via the repository `.venv` only.
- NumPy/SciPy/MNE for format adapters, continuous-record filtering, epochs,
  and provenance. Preprocessing parameters are explicit Python dataclass/CLI
  contract values.
- scikit-learn and pyRiemann for linear and xDAWN-RG baselines.
- PyTorch for compact CNNs; CUDA/XPU/CPU selection follows
  `device-portability.md`.
- argparse plus attested JSON manifests for experiment configuration, pytest
  for contract tests, and ruff for static checks. Hydra is not a dependency.

Torch is installed from the device-specific index before the `train` extra.
`pyproject.toml` is the single dependency authority; `requirements.txt` is a
convenience installation entry point for baseline development.
