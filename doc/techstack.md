# Technology Stack

- Python 3.11 or newer via the repository `.venv` only.
- NumPy/SciPy/MNE for format adapters, continuous-record filtering, epochs,
  and provenance. Preprocessing parameters are explicit Python dataclass/CLI
  contract values.
- scikit-learn and pyRiemann for linear and xDAWN-RG baselines.
- PyTorch for compact CNNs; CUDA/XPU/CPU selection follows
  `device-portability.md`.
- argparse plus attested JSON manifests for experiment configuration, pytest
  for contract tests, and ruff for static checks. Hydra is not a dependency.

`pyproject.toml` is the single dependency authority. The validated RTX 5090
training environment is Linux x86_64 with Python 3.11--3.13, Torch 2.8.0,
Torchaudio 2.8.0, and the CUDA 12.8 runtime dependencies published in the
official Torch wheel metadata. Torchaudio is explicit in this branch because
its compiled extension must match Torch exactly. Python 3.14 and non-Linux or
non-x86_64 platforms retain a `torch>=2.4` development/test branch; successful
installation there is not RTX 5090 training evidence. `requirements.txt` is a
convenience installation entry point for baseline development.
