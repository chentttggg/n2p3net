"""pytest 根配置：在 import mne 之前重定向其配置目录。

MNE 在 import 时会读写 ``~/.mne/mne-python.json``；在沙箱/只读 HOME 环境下会抛
``PermissionError``。``_MNE_FAKE_HOME_DIR`` 是 MNE 官方提供的测试钩子（见
mne/utils/config.py 的 ``_get_extra_data_path``），将其指向项目内可写目录即可绕开。

注意：本文件必须放在项目根（pytest rootdir），且**不得**在此 import mne，以保证环境
变量在测试文件 ``import mne`` 之前已生效。
"""

import os
from pathlib import Path

_FAKE_HOME = Path(__file__).resolve().parent / ".mne_home"
_FAKE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["_MNE_FAKE_HOME_DIR"] = str(_FAKE_HOME)
