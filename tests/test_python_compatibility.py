from __future__ import annotations

import ast
from pathlib import Path


def test_active_python_tree_parses_with_declared_minimum_version() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = tuple(
        path
        for directory in ("src", "experiments", "tests")
        for path in sorted((root / directory).rglob("*.py"))
    )

    assert paths
    for path in paths:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path.relative_to(root)),
            feature_version=(3, 11),
        )
