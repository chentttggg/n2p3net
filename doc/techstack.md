# Technology Stack

- Python 3.14 via the repository `.venv` only.
- NumPy/SciPy/MNE for format adapters, continuous-record filtering, epochs,
  and provenance. Preprocessing parameters are explicit Hydra config values.
- scikit-learn and pyRiemann for linear and xDAWN-RG baselines.
- PyTorch for compact CNNs; CUDA/XPU/CPU selection follows
  `device-portability.md`.
- Hydra for immutable experiment configurations, pytest for contract tests,
  and ruff for static checks.

Torch is installed from the device-specific index before the `train` extra.
`pyproject.toml` is the single dependency authority; `requirements.txt` is a
convenience installation entry point for baseline development.
