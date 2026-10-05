"""Verify cached primary bytes and exact points after bounded network failures."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader
from retrieve_sources import fetch

OUT = Path(__file__).resolve().parent
BASE = Path("D:/BCI/n2p3-net/doc/assets/operator_research_20261004/literature")


def main():
    entries = []
    accesses = []
    for path in [BASE / "access_manifest.json", BASE / "access_bnci_conditional1.json"]:
        accesses.extend(json.loads(path.read_text(encoding="utf-8")))
    for name, pages in [("sarkka_book.pdf", [229, 230]),
                        ("bnci_official_description.pdf", [1, 2])]:
        source = BASE / name
        body = source.read_bytes()
        reader = PdfReader(io.BytesIO(body))
        entry = next(v for v in accesses if v["file"] == name)
        sha = hashlib.sha256(body).hexdigest()
        assert sha == entry["sha256"], name
        entries.append({"path": str(source), "sha256": sha,
                        "access_level": "cached_primary_full_pdf_text_reinspected",
                        "original_access": entry, "pages_total": len(reader.pages),
                        "reinspected_pages": [{"pdf_page": p,
                                               "page_text_sha256": hashlib.sha256(
                                                   reader.pages[p - 1].extract_text().encode("utf-8")).hexdigest()}
                                              for p in pages]})
    source = Path("D:/BCI/n2p3-net/doc/assets/system_research_20261004/literature/unfold_full.xml")
    body = source.read_bytes()
    xml = ET.fromstring(body)
    paragraphs = [" ".join(" ".join(node.itertext()).split()) for node in xml.iter("p")]
    chosen = [{"paragraph_index": i + 1, "paragraph_sha256": hashlib.sha256(v.encode("utf-8")).hexdigest(),
               "matched_topics": sorted(set(x.lower() for x in re.findall(
                   r"overlap|filter|epoch size|time.expansion", v, re.I)))}
              for i, v in enumerate(paragraphs) if re.search(r"overlap|filter|epoch size|time.expansion", v, re.I)]
    entries.append({"path": str(source), "sha256": hashlib.sha256(body).hexdigest(),
                    "access_level": "cached_primary_full_xml_reinspected",
                    "public_url": "https://peerj.com/articles/7838/",
                    "doi_from_article": ["".join(node.itertext()) for node in xml.iter("article-id")
                                         if node.attrib.get("pub-id-type") == "doi"],
                    "relevant_paragraph_locators": chosen})
    source = BASE / "unfold_filter_simulation.m"
    text = source.read_text(encoding="utf-8")
    entries.append({"path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "access_level": "cached_author_code_reinspected",
                    "public_url": "https://github.com/unfoldtoolbox/unfold",
                    "selected_lines": [{"line": i + 1, "text": v}
                                       for i, v in enumerate(text.splitlines())
                                       if re.search(r"filter|filtnew|timelimits|uf_timeexpandDesignmat|0\.8|1\]", v)]})
    requests = [
        ("bnci_wayback_discovery", "https://archive.org/wayback/available?" + urllib.parse.urlencode({
            "url": "https://bnci-horizon-2020.eu/database/data-sets/008-2014/description.pdf"})),
        ("bnci_official_redirect_target", "https://lampx.tugraz.at/~bci/database/008-2014/description.pdf"),
        ("sarkka_author_book_live", "https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf"),
        ("unfold_primary_publisher", "https://peerj.com/articles/7838.xml"),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        supplemental = list(executor.map(fetch, requests))
    for row in supplemental:
        if row["name"].endswith("discovery"):
            row["access_role"] = "snapshot_discovery_only_not_archived_page_content"
    output = {"created_utc": datetime.now(UTC).isoformat(),
              "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cached_primary_sources": entries, "bounded_supplemental_requests": supplemental}
    (OUT / "cached_primary_sources.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"cached_points": len(entries),
                      "supplemental": [{"name": x["name"], "access_level": x["access_level"]}
                                       for x in supplemental]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
