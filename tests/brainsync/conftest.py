"""Availability of the separate acquisition application's test runtime."""

import pytest

from tests import PROJECT_ROOT


@pytest.fixture(scope="session")
def frontend_runtime():
    root = PROJECT_ROOT.parent / "BrainSync_Tool_newest" / "BrainSync_Tool"
    executable = root / ".venv" / "Scripts" / "python.exe"
    if not executable.is_file():
        pytest.skip("Separate BrainSync frontend runtime is unavailable")
    return root, executable
