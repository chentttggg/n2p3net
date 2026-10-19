"""Fetch bounded public primary sources; preserve response bytes and access status."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "sarkka_book.pdf": "https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf",
    "gpml_ch2.pdf": "https://gaussianprocess.org/gpml/chapters/RW2.pdf",
    "minka_linear.pdf": "https://tminka.github.io/papers/minka-linear.pdf",
    "murphy_gauss.pdf": "https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf",
    "shah_student.pdf": "https://proceedings.mlr.press/v33/shah14.pdf",
    "guo_calibration.pdf": "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf",
    "scipy_filter_code.py": "https://raw.githubusercontent.com/scipy/scipy/v1.16.2/scipy/signal/_signaltools.py",
    "unfold_repo.json": "https://api.github.com/repos/unfoldtoolbox/unfold/git/trees/main?recursive=1",
    "mtrf_repo.json": "https://api.github.com/repos/mickcrosse/mTRF-Toolbox/git/trees/master?recursive=1",
    "epmc_rerp.json": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1111/psyp.12320&format=json&resultType=core",
    "epmc_mtrf.json": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22multivariate%20temporal%20response%20function%22&format=json&resultType=core&pageSize=5",
    "epmc_regression_filter.json": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(EEG%20AND%20deconvolution%20AND%20filtering)&format=json&resultType=core&pageSize=10",
}


def fetch(item):
    name, url = item
    receipt = {"file": name, "requested_url": url,
               "fetched_utc": datetime.now(UTC).isoformat()}
    try:
        response = requests.get(url, timeout=45, headers={"User-Agent": "n2p3-net-methodology-research/1.0"})
        content = response.content
        (ROOT / name).write_bytes(content)
        receipt.update(status=response.status_code, final_url=response.url,
                       content_type=response.headers.get("content-type"), bytes=len(content),
                       sha256=hashlib.sha256(content).hexdigest())
    except requests.RequestException as exc:
        receipt.update(error=str(exc), status=None)
    return receipt


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(fetch, SOURCES.items()))
    (ROOT / "access_manifest.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    for receipt in receipts:
        print(receipt["file"], receipt.get("status"), receipt.get("bytes"), receipt.get("error", ""))
