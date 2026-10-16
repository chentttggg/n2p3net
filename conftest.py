"""在数值库导入前设置测试线程预算，并隔离 MNE 配置。"""

import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

_FAKE_HOME = Path(__file__).resolve().parent / ".mne_home"
_FAKE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["_MNE_FAKE_HOME_DIR"] = str(_FAKE_HOME)
