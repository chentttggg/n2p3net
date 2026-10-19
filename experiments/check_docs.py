"""Check repository-owned links from the active guidance and evidence entry points."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DOCUMENTS = (
    "README.md", "AGENTS.md", "doc/README.md", "doc/research_program.zh.md",
    "doc/technical_structure.zh.md", "doc/input_contract_math.zh.md", "doc/roadmap.zh.md",
    "doc/engineering.zh.md", "doc/workstation.zh.md", "doc/evidence/README.md",
    "doc/evidence/calibration_and_selection.zh.md", "doc/evidence/response_and_noise.zh.md",
    "doc/evidence/public_evaluation.zh.md", "doc/evidence/execution_and_integration.zh.md",
    "frozen/README.md",
)
LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+[^)]*)?\)")


def check_links(root: Path) -> dict:
    """Validate local navigation without requiring private runtime payloads on CI.

    Absolute workstation paths are local retrieval pointers. Historical payload
    provenance and archived-member identities have their own checked locators;
    this command checks the active repository navigation rather than decoding them.
    """
    root = root.resolve()
    missing, private_pointers = [], []
    checked = 0
    for name in ACTIVE_DOCUMENTS:
        source = root / name
        if not source.is_file():
            missing.append({"source": name, "target": name, "reason": "missing_entry_point"})
            continue
        text = re.sub(r"```[^\n]*\n[\s\S]*?```", "", source.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match[1].strip("<>")
            if PureWindowsPath(target).is_absolute() or target.startswith("/"):
                private_pointers.append({"source": name, "target": target})
                continue
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = (source.parent / unquote(parsed.path)).resolve()
            if not path.is_relative_to(root):
                private_pointers.append({"source": name, "target": target})
                continue
            checked += 1
            if not path.exists():
                missing.append({"source": name, "target": target, "reason": "missing_local_target"})
    return {"schema": "n2p3_active_document_links/1", "documents": len(ACTIVE_DOCUMENTS),
            "repository_links_checked": checked, "missing": missing,
            "private_retrieval_pointers": private_pointers, "passed": not missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = check_links(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
