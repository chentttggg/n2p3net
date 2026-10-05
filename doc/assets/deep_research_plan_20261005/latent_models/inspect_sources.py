"""Extract acquired primary text and bind read-only local source provenance."""

from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        value = data.strip()
        if value:
            self.parts.append(value)


def main():
    records = []
    for name in ("kalman_gp_author", "student_t_gp_full"):
        path = ROOT / "sources" / f"{name}.pdf"
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        (ROOT / "sources" / f"{name}.txt").write_text("\n\n".join(f"PAGE {i+1}\n{text}" for i, text in enumerate(pages)), encoding="utf-8", newline="\n")
        records.append({"id": name, "level": "primary_fulltext_extracted", "pages": len(pages), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for name in ("autoregression_official", "student_t_gp", "unfold_official"):
        parser = TextParser()
        parser.feed((ROOT / "sources" / f"{name}.html").read_text(encoding="utf-8"))
        (ROOT / "sources" / f"{name}.txt").write_text("\n".join(parser.parts)+"\n", encoding="utf-8", newline="\n")
    local_root = Path(r"D:\BCI\n2p3-net\doc\assets\system_research_20261004\literature")
    for name in ("unfold_full.txt", "bsm_local_full.txt", "toeplitz_code_full.md"):
        path = local_root / name
        records.append({"id": "local_"+name, "level": "existing_primary_text_readonly", "path": str(path),
                        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    source_root = Path(r"D:\BCI\n2p3-net\doc\assets\operator_research_20261004")
    for name in ("continuous_model.py", "selection_erp_model.py", "phase4_report.json"):
        path = source_root / name
        records.append({"id": name, "level": "existing_research_helper_or_development_evidence_readonly", "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / "read_source_manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")
    for record in records:
        print(record["id"], record["level"], record.get("pages", ""))


if __name__ == "__main__":
    main()
