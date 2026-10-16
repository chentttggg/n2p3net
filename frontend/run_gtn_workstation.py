import sys

from gtn_workstation.app import RuntimeDependencyError, main

if __name__ == "__main__":
    try:
        exit_code = main()
    except RuntimeDependencyError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(exit_code)
        