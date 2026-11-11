# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('D:/BCI/BrainSync_Tool_newest/BrainSync_Tool/resources', 'resources')]
binaries = [('D:/BCI/BrainSync_Tool_newest/BrainSync_Tool/libusb-1.0.dll', '.'), ('D:/BCI/BrainSync_Release_20260920/_internal/icuuc.dll', '.')]
hiddenimports = []
tmp_ret = collect_all('brainsync_sdk')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shiboken6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['D:/BCI/BrainSync_Tool_newest/BrainSync_Tool/run_gtn_workstation.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick', 'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DExtras', 'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BrainSync',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BrainSync',
)
