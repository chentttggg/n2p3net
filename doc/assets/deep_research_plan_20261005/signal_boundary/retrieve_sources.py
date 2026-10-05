"""Bounded primary-source and local-paper inspection for this research spec."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
QUERIES = (
    "Practical considerations and pitfalls in preprocessing denoising and artifact rejection",
    "Digital filter design for electrophysiological data a practical approach",
    "How inappropriate high-pass filters can produce artifactual effects and incorrect conclusions in ERP studies",
    "Determining the initial states in forward backward filtering",
    "A New Approach to Linear Filtering and Prediction Problems",
    "Unfold an integrated toolbox for overlap correction nonlinear modeling and regression based EEG analysis",
)
URLS = (
    ("scipy_sosfiltfilt", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html"),
    ("scipy_filtfilt", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html"),
    ("bnci_description", "https://bnci-horizon-2020.eu/database/data-sets/008-2014/description.pdf"),
    ("kalman_original", "https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf"),
    ("unfold_author_docs", "https://www.unfoldtoolbox.org/"),
    ("widmann_exact_metadata", "https://api.crossref.org/works/10.1016%2Fj.jneumeth.2014.08.002"),
)
LOCAL = (
    "Bayesian_Signal_Matching_ERP_Transfer_arXiv_2401.07111.pdf",
    "Block_Toeplitz_LDA_ERP_JNE_2022.pdf",
    "xDAWN_Riemannian_Transfer_P300_2020.pdf",
)


def fetch(item):
    name, url = item
    entry = {"name": name, "requested_url": url,
             "accessed_utc": datetime.now(UTC).isoformat()}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "n2p3-net research evidence audit"})
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read(5_000_001)
            entry.update(status=response.status, final_url=response.url,
                         content_type=response.headers.get("Content-Type", ""),
                         bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        if len(body) > 5_000_000:
            entry["access_level"] = "bounded_response_truncated"
        elif body.startswith(b"%PDF"):
            reader = PdfReader(io.BytesIO(body))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            entry.update(access_level="full_pdf_text", pages=len(reader.pages),
                         text_excerpt=text[:12000])
        elif "api.crossref.org" in url:
            message = json.loads(body)["message"]
            items = message.get("items", [message])
            entry.update(access_level="metadata_only", items=[{
                "title": work.get("title"), "DOI": work.get("DOI"),
                "published": work.get("published"), "author": work.get("author"),
                "link": work.get("link"), "URL": work.get("URL"),
                "abstract": work.get("abstract")}
                for work in items])
        else:
            html = body.decode("utf-8", errors="replace")
            text = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", "", html)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            level = "primary_xml_text" if "xml" in entry["content_type"].lower() else "primary_html"
            entry.update(access_level=level, text_excerpt=" ".join(text.split()[:65]))
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        entry.update(access_level="failed", error=str(exc))
        if isinstance(exc, urllib.error.HTTPError):
            entry["status"] = exc.code
    return entry


def main():
    tasks = list(URLS)
    for index, query in enumerate(QUERIES):
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
            "query.title": query, "rows": 2})
        tasks.append((f"crossref_{index}", url))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        fetched = list(executor.map(fetch, tasks))
    for item, query in zip(fetched[len(URLS):], QUERIES, strict=True):
        item["executed_query"] = query
    local = []
    for name in LOCAL:
        path = Path("D:/BCI/n2p3-net/Paper") / name
        body = path.read_bytes()
        reader = PdfReader(io.BytesIO(body))
        pages = [page.extract_text() or "" for page in reader.pages]
        key = re.compile(r"filter|reference|overlap|convolution|boundary|covariance|latency", re.I)
        matches = [{"page": i + 1,
                    "page_text_sha256": hashlib.sha256(pages[i].encode("utf-8")).hexdigest(),
                    "matched_topics": sorted(set(v.lower() for v in key.findall(pages[i])))}
                   for i in range(len(pages)) if key.search(pages[i])]
        public_urls = {
            LOCAL[0]: "https://arxiv.org/abs/2401.07111",
            LOCAL[1]: "https://arxiv.org/abs/2202.02001",
            LOCAL[2]: "https://doi.org/10.3390/app10051804",
        }
        local.append({"path": str(path), "sha256": hashlib.sha256(body).hexdigest(),
                      "access_level": "local_full_pdf_text", "pages": len(pages),
                      "public_url": public_urls[name],
                      "first_page_excerpt": " ".join(pages[0].split()[:65]),
                      "relevant_page_locators": matches[:5]})
    result = {"purpose": "bounded_methodology_lookup_not_systematic_review",
              "executed_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "retrieval": fetched, "local_papers": local}
    (OUT / "sources.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"retrieval": [{"name": x["name"], "access_level": x["access_level"]}
                                    for x in fetched], "local_pdfs": len(local)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
