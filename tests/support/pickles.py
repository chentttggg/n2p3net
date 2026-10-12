"""Pickled objects that must never run while audited artifacts are decoded."""

EXECUTED = []


def _record_execution():
    EXECUTED.append(True)
    return {}


class ExecutesOnLoad:
    """Unpickling this object calls code; weights-only loading must refuse it."""

    def __reduce__(self):
        return (_record_execution, ())
