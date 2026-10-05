"""Read-only primary-source retrieval for the latent-model research specification."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("g_priors", "crossref_metadata", "https://api.crossref.org/works/10.1198/016214507000001337"),
    ("unfold", "crossref_metadata", "https://api.crossref.org/works/10.7717/peerj.7838"),
    ("xdawn", "crossref_metadata", "https://api.crossref.org/works/10.1109/TBME.2009.2012869"),
    ("toeplitz_lda", "crossref_metadata", "https://api.crossref.org/works/10.1088/1741-2552/ac9c98"),
    ("gmrf_spde", "crossref_metadata", "https://api.crossref.org/works/10.1111/j.1467-9868.2011.00777.x"),
    ("kalman_gp", "crossref_metadata", "https://api.crossref.org/works/10.1109/MLSP.2010.5589113"),
    ("matrix_variate", "crossref_metadata", "https://api.crossref.org/works/10.1093/biomet/68.1.265"),
    ("state_space", "crossref_metadata", "https://api.crossref.org/works/10.1093/acprof:oso/9780199641178.001.0001"),
    ("bsm", "crossref_metadata", "https://api.crossref.org/works/10.1080/01621459.2025.2563189"),
    ("ride_query", "search_results", "https://api.crossref.org/works?query.title=" + quote("Residue iteration decomposition RIDE latency variability Ouyang") + "&rows=3"),
    ("woody_query", "search_results", "https://api.crossref.org/works?query.title=" + quote("Characterization adaptive filter analysis variable latency neuroelectric signals Woody") + "&rows=3"),
    ("factor_query", "search_results", "https://api.crossref.org/works?query.title=" + quote("State Space approach dynamic factor analysis Watson Engle 1983") + "&rows=3"),
    ("student_t_gp", "primary_abstract_html", "https://proceedings.mlr.press/v33/shah14.html"),
    ("burg_official", "official_implementation", "https://raw.githubusercontent.com/cokelaer/spectrum/master/src/spectrum/burg.py"),
    ("autoregression_official", "official_documentation", "https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html"),
    ("toeplitz_official", "official_implementation", "https://raw.githubusercontent.com/jsosulski/toeplitzlda/master/toeplitzlda/classification/lda.py"),
    ("unfold_official", "official_documentation", "https://www.unfoldtoolbox.org/"),
]
EXTRA_SOURCES = [
    ("g_priors_epmc", "metadata_or_abstract", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="+quote('DOI:10.1198/016214507000001337')+"&format=json&resultType=core"),
    ("g_priors_author", "primary_fulltext_pdf", "https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/GpriorsLiangJASAFinal.pdf"),
    ("ride_epmc", "metadata_or_abstract", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="+quote('DOI:10.1111/j.1469-8986.2011.01269.x')+"&format=json&resultType=core"),
    ("xdawn_epmc", "metadata_or_abstract", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="+quote('DOI:10.1109/TBME.2009.2012869')+"&format=json&resultType=core"),
    ("kalman_gp_author", "primary_fulltext_pdf", "https://users.aalto.fi/~ssarkka/pub/gp-ts-kfrts.pdf"),
    ("student_t_gp_full", "primary_fulltext_pdf", "https://proceedings.mlr.press/v33/shah14.pdf"),
    ("toeplitz_github_tree", "official_repository_metadata", "https://api.github.com/repos/jsosulski/toeplitzlda/git/trees/main?recursive=1"),
    ("g_priors_query", "search_results", "https://api.crossref.org/works?query.bibliographic="+quote('Mixtures of g Priors for Bayesian Variable Selection Liang Paulo Molina Clyde Berger 2008')+"&rows=3"),
    ("toeplitz_official_resolved", "official_implementation", "https://raw.githubusercontent.com/jsosulski/toeplitzlda/main/toeplitzlda/classification/toeplitzlda.py"),
    ("xdawn_official_mne", "official_implementation", "https://raw.githubusercontent.com/mne-tools/mne-python/main/mne/preprocessing/xdawn.py"),
]


def fetch(source):
    key, level, url = source
    record = {"id": key, "requested_level": level, "url": url,
              "retrieved_utc": datetime.now(UTC).isoformat()}
    try:
        response = requests.get(url, timeout=(10, 30), headers={"User-Agent": "n2p3-research-method-review/1.0"})
        record.update(http_status=response.status_code, resolved_url=response.url,
                      content_type=response.headers.get("content-type"), bytes=len(response.content),
                      response_sha256=hashlib.sha256(response.content).hexdigest())
        suffix = ".json" if "json" in response.headers.get("content-type", "") else ".txt" if level == "official_implementation" else ".pdf" if "pdf" in response.headers.get("content-type", "") else ".html"
        if response.ok:
            (ROOT / "sources" / (key + suffix)).write_bytes(response.content)
            record["saved_file"] = "sources/" + key + suffix
            record["retrieved_level"] = level
            if level == "crossref_metadata":
                message = response.json()["message"]
                record.update(title=message.get("title"), doi=message.get("DOI"),
                              published=message.get("published"), links=message.get("link"))
            elif level == "search_results":
                record["top_results"] = [{"doi": item.get("DOI"), "title": item.get("title"),
                                           "author": item.get("author", [])[:2]} for item in response.json()["message"]["items"]]
            elif level == "metadata_or_abstract":
                results = response.json()["resultList"]["result"]
                record["result_count"] = len(results)
                record["retrieved_level"] = ("primary_abstract" if any(item.get("abstractText") for item in results)
                                             else "metadata_only" if results else "empty_search_result")
        else:
            record["retrieved_level"] = "request_failed"
    except requests.RequestException as exc:
        record.update(retrieved_level="request_failed", error=str(exc))
    return record


def main():
    (ROOT / "sources").mkdir(parents=True, exist_ok=True)
    if "--extra" in __import__("sys").argv:
        records = json.loads((ROOT / "source_ledger.json").read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=4) as executor:
            extra = list(executor.map(fetch, EXTRA_SOURCES))
        for record in extra:
            previous = next((item for item in records if item["id"] == record["id"]), None)
            records = [item for item in records if item["id"] != record["id"]]
            if previous is not None:
                attempts = previous.get("attempt_history", [])
                attempts.append({key: value for key, value in previous.items() if key != "attempt_history"})
                if record["retrieved_level"] == "request_failed" and previous["retrieved_level"] != "request_failed":
                    attempts.append(record)
                    record = {key: value for key, value in previous.items() if key != "attempt_history"}
                record["attempt_history"] = attempts
            records.append(record)
    else:
        with ThreadPoolExecutor(max_workers=6) as executor:
            records = list(executor.map(fetch, SOURCES))
    (ROOT / "source_ledger.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "query_log.json").write_text(json.dumps({"scope": "targeted primary-source method review, not a systematic review",
                                                   "date_user_timezone": "2026-10-05 Asia/Shanghai",
                                                   "queries": [record for record in records if record["requested_level"] == "search_results"],
                                                   "retrieval_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for record in records:
        print(record["id"], record.get("http_status"), record["retrieved_level"], record.get("title", record.get("top_results", "")))


if __name__ == "__main__":
    main()
