from __future__ import annotations

import ast

from tests import PROJECT_ROOT


def test_active_python_tree_parses_with_declared_minimum_version() -> None:
    root = PROJECT_ROOT
    # Experiment entrypoints live at the root; runs/ contains historical snapshots.
    paths = (
        root / "conftest.py",
        *(root / "src").rglob("*.py"),
        *(root / "experiments").glob("*.py"),
        *(root / "tests").rglob("*.py"),
    )

    for path in paths:
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path.relative_to(root)),
            feature_version=(3, 11),
        )
