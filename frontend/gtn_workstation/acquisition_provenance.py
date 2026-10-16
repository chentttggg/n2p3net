"""Source and SDK bytes observed at process import and session setup."""

import hashlib
import importlib.metadata
import platform
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


_ROOT = Path(__file__).resolve().parent
_PROCESS_SOURCE = {path.name: _sha256(path) for path in sorted(_ROOT.glob("*.py"))}


def acquisition_runtime_record(sdk=None):
    record = {"schema": "brainsync_acquisition_runtime/1", "python": platform.python_version(),
              "source_files_sha256": dict(_PROCESS_SOURCE), "source_capture": "module_import",
              "sdk_runtime_version": None, "sdk_distribution_versions": {}, "sdk_files_sha256": {}}
    if sdk is None:
        return record
    record["sdk_runtime_version"] = str(getattr(sdk, "__version__", "unknown"))
    package = str(getattr(sdk, "__name__", "")).split(".")[0]
    for name in importlib.metadata.packages_distributions().get(package, []):
        record["sdk_distribution_versions"][name] = importlib.metadata.version(name)
    filename = getattr(sdk, "__file__", None)
    if filename and Path(filename).is_file():
        entry = Path(filename).resolve()
        files = {entry, *entry.parent.rglob("*.pyd"), *entry.parent.rglob("*.dll")}
        record["sdk_files_sha256"] = {path.relative_to(entry.parent).as_posix(): _sha256(path)
                                      for path in sorted(files)}
        record["sdk_file_capture"] = "session_setup_package_binaries_not_device_firmware"
    return record
