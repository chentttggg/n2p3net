# N2P3-Net

Performance-first oddball P300 decoding research framework. The current scope
is a validated common data contract and evaluation foundation before committing
to one model family.

## Environment

Use the repository-local Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Install `.[baselines]`, `.[signals]`, and device-specific PyTorch plus
`.[train]` only when beginning those phases. The research contract is in
`doc/constitution.md`, `doc/blueprint.md`, and `doc/roadmap.md`.
