# Receptive-field literature retrieval ledger

Access date: 2026-08-28 (Asia/Shanghai)

## Retrieval contract

- Scope: convolutional total/theoretical receptive-field arithmetic, effective receptive field (ERF), temporal CNN precedent, and the interpretation boundary for EEGNet/MS-EEGNet/P300.
- Retrieval type: bounded, targeted evidence search; not a systematic review.
- Acceptance rule: an exact title plus DOI/arXiv/OpenAlex identity was required. Search rank or HTTP 200 alone was not accepted.
- Evidence rule: computer-vision/sequence papers are method precedent only. Only EEG/P300 papers can supply direct task evidence, and even those do not establish a GTN-optimal receptive field without a matched experiment.

## Reproducible API ledger

| Surface | Endpoint and parameters | Returned / boundary | Validation |
|---|---|---:|---|
| OpenAlex RF arithmetic | `GET https://api.openalex.org/works?search=\"receptive field\" arithmetic convolutional neural networks&per_page=20&sort=cited_by_count:desc` | 20 of 3,718 | Intentionally bounded; broad result set, not exhaustive. |
| OpenAlex ERF | `GET https://api.openalex.org/works?search=\"effective receptive field\" deep convolutional neural networks&per_page=20&sort=cited_by_count:desc` | 20 of 3,768 | Luo exact title/ID retained; unrelated high-citation results rejected. |
| OpenAlex temporal CNN | `GET https://api.openalex.org/works?search=\"temporal convolutional network\" receptive field sequence modeling&per_page=20&sort=cited_by_count:desc` | 20 of 4,134 | Intentionally bounded; used only to locate known method papers. |
| OpenAlex exact entities | `GET /works/doi:10.48550/arxiv.1603.07285`, `/works/doi:10.48550/arxiv.1701.04128`, `/works/doi:10.48550/arxiv.1803.01271` | 3/3 | Dumoulin `W2304648132`, Luo `W2556967412`, Bai `W2792764867`; all non-retracted and OA in OpenAlex. |
| Crossref exact DOI | `GET https://api.crossref.org/works/{doi}` for `10.23915/distill.00021`, `10.1109/CVPR.2017.113`, `10.1016/j.clinph.2007.04.019`, `10.1088/1741-2560/11/3/035008` | 4/4 | `status=ok`; titles and publication years matched. |
| PubMed P300 latency | `GET /entrez/eutils/esearch.fcgi?db=pubmed&term=((P300[Title/Abstract] OR P3b[Title/Abstract]) AND (latency[Title/Abstract] OR morphology[Title/Abstract] OR duration[Title/Abstract] OR \"time window\"[Title/Abstract]))&retmax=20&sort=relevance&retmode=json` | 20 of 3,218 | One bounded page; biomedical `p300` false positives were screened by title/DOI. |
| PubMed single-trial P300 | `GET /entrez/eutils/esearch.fcgi?db=pubmed&term=(P300[Title/Abstract] AND (\"single-trial\"[Title/Abstract] OR \"single trial\"[Title/Abstract]) AND (latency[Title/Abstract] OR jitter[Title/Abstract] OR alignment[Title/Abstract] OR morphology[Title/Abstract]))&retmax=20&sort=relevance&retmode=json` | 20 of 60 | Exact records were fetched in one PubMed XML batch. |
| Europe PMC exact P300 | `GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=(DOI:\"10.1016/j.clinph.2007.04.019\" OR DOI:\"10.1088/1741-2560/11/3/035008\")&pageSize=10&resultType=core&format=json` | 2/2 | Echoed query matched. P300-jitter article reported `OA=N`, `inEPMC=N`, `hasPDF=N`. |
| Unpaywall exact DOI | `GET https://api.unpaywall.org/v2/{doi}?email=<redacted>` for Araujo, Polich, and Arico | 3/3 | Araujo: gold OA HTML, no PDF; Polich: green landing page, no direct PDF; Arico: closed. |
| arXiv API | `GET https://export.arxiv.org/api/query?id_list=1603.07285,1701.04128,1803.01271&max_results=3` | 0 accepted | Repeated Schannel TLS handshake failure. No arXiv API response was treated as evidence; identifiers were independently resolved in OpenAlex and official repositories. |

OpenAlex `search.exact` still returned very broad result sets in this session. It was therefore not treated as exact retrieval; records were accepted only after identifier/title reconciliation.

## Retained primary sources

### 1. Araujo, Norris, and Sim (2019)

- Title: *Computing Receptive Fields of Convolutional Neural Networks*
- DOI: `10.23915/distill.00021`
- Full text: official Distill HTML, saved as `tmp/receptive_field_research_20260828/api/Computing_Receptive_Fields_Araujo_Distill_2019.html` (106,347 bytes).
- Evidence class: mathematical/implementation method precedent; not EEG or P300 performance evidence.
- Core recurrence, written from output layer `L` back toward input:

  `r_(l-1) = s_l r_l + (k_l - s_l)`

  and the closed form:

  `r_0 = 1 + sum_(l=1)^L [(k_l - 1) product_(i=1)^(l-1) s_i]`.

- For dilation `d_l`, replace `k_l` by the effective span `1 + d_l (k_l - 1)`.
- Padding does not change nominal receptive-field size, but it changes receptive-field coordinates and boundary padding contamination. Branches in a multi-path graph can also be center-misaligned; a union width alone is insufficient.

### 2. Luo et al. (NeurIPS 2016; arXiv version 2017)

- Title: *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*
- arXiv: `1701.04128`; OpenAlex: `W2556967412`; DataCite-style arXiv DOI: `10.48550/arXiv.1701.04128`.
- OA PDF: `Paper/Effective_Receptive_Field_Luo_NeurIPS_2016.pdf` (9 pages; 615,339 bytes; `%PDF-` verified).
- Evidence class: ERF theory plus non-EEG image experiments; not P300 performance evidence.
- Core result: influence inside the theoretical receptive field is non-uniform and approximately Gaussian-like around the center, so the ERF occupies only a fraction of the nominal theoretical field. Nonlinearities, subsampling, dilation, skip connections, initialization, and training can change it.
- Boundary: a trunk whose theoretical field covers the P300 window may still assign negligible effective influence to parts of that window. Conversely, theoretical coverage does not identify what the trained model actually uses.

### 3. Dumoulin and Visin (2016)

- Title: *A Guide to Convolution Arithmetic for Deep Learning*
- arXiv: `1603.07285`; OpenAlex: `W2304648132`; DataCite-style arXiv DOI: `10.48550/arXiv.1603.07285`.
- Verified source: official `vdumoulin/conv_arithmetic` GitHub repository; TeX and README saved under `tmp/receptive_field_research_20260828/api/`.
- Evidence class: convolution output-shape/kernel-stride-padding-dilation arithmetic; not ERF theory and not task-performance evidence.
- Retrieval boundary: arXiv API and direct arXiv PDF failed at TLS transport. No corrupt/HTML-disguised PDF was retained or reported as downloaded.

### 4. Lea et al. (CVPR 2017)

- Title: *Temporal Convolutional Networks for Action Segmentation and Detection*
- DOI: `10.1109/CVPR.2017.113`; arXiv: `1611.05267`.
- OA PDF: `Paper/Temporal_Convolutional_Networks_Lea_CVPR_2017.pdf` (733,157 bytes; `%PDF-` verified).
- Evidence class: temporal-CNN architecture precedent on action segmentation; not EEG/P300 performance evidence.
- Boundary: it supports reasoning about temporal hierarchy and long context, but it cannot justify a particular P300 receptive field or GTN model choice.

### 5. Existing direct EEG/P300 sources (not redownloaded)

- Lawhern et al. (2018), EEGNet, DOI `10.1088/1741-2552/aace8c`; local `Paper/EEGNet_Compact_CNN_JNE_2018.pdf` (30 pages).
- Borra et al. (2021), MS-EEGNet, DOI `10.3389/fnhum.2021.655840`; local `Paper/MS_EEGNet_P300_Frontiers_2021.pdf` (22 pages).
- Evidence boundary: EEGNet supplies a compact EEG baseline and MS-EEGNet supplies direct multi-scale P300 evidence, but neither isolates total input-domain receptive field as the causal mechanism. MS-EEGNet's advantage is dataset-dependent, so parallel kernel widths are a hypothesis, not a universal improvement.
- Existing direct latency source: Quattrociocchi et al. (2026), DOI `10.1088/1741-2552/ae7766`; local `Paper/Improving_P300_Morphology_Single_Trial_Latency_Realignment_JNE_2026.pdf`. It supports explicit latency-variability analysis, not an optimal CNN receptive-field width.

## Main scientific conclusions

1. **Use total input-domain receptive field, not a layer's kernel width.** For a forward stack, track jump `j_l = j_(l-1) s_l` and nominal field `r_l = r_(l-1) + d_l (k_l - 1) j_(l-1)` from `j_0=r_0=1`.
2. **Report geometry, not one scalar.** Each branch needs receptive-field span in samples/ms, center offset, output jump, valid-input interval, and padding fraction. Parallel branches also need an alignment check before fusion.
3. **Separate theoretical RF from trained ERF.** Measure held-out target-logit input gradients and perturbation/occlusion sensitivity. Gradient ERF alone is vulnerable to saturation, so it is not causal proof.
4. **Do not infer P300 scale from computer-vision evidence.** Araujo/Luo/Dumoulin/Lea justify calculations and diagnostics only. EEGNet/MS-EEGNet and direct P300 latency work define task-specific hypotheses, not a GTN verdict.
5. **Do not infer performance from one dataset.** The next claim-eligible comparison must be parameter/readout matched, use the same preprocessing and outer subjects, and reserve GTN as confirmation rather than tuning data.

## Required counterexamples

- Same branch kernel, different upstream pooling/stride: different total input-domain RF.
- Same theoretical RF, different trained weights/nonlinearities: different ERF.
- Same nominal RF size, different padding/center offset: different valid evidence near epoch boundaries.
- Same union width, misaligned parallel branches: fusion combines different physical times.
- Same P300 morphology shifted in latency: a center-concentrated ERF can change the logit even though nominal coverage is unchanged.
- Large nominal RF spanning baseline and post-stimulus regions: a timestamped feature is not necessarily local evidence for either region.

## Download and exclusion record

- Added exactly two verified OA PDFs: Luo ERF and Lea TCN. No existing EEGNet/MS-EEGNet PDF was duplicated.
- Retained Distill as official OA HTML and Dumoulin as official TeX source because no verified publisher PDF was available through the working transport.
- Failed arXiv payloads and the PMC proof-of-work HTML page were checked by magic bytes and deleted individually.
- Bai et al. `arXiv:1803.01271` remained metadata-only in this pass because the only resolved PDF location was arXiv and the TLS failure persisted.
- Arico et al. (2014), DOI `10.1088/1741-2560/11/3/035008`, is directly relevant to P300 latency jitter but was closed in Europe PMC and Unpaywall; no numerical claim was extracted from unavailable full text.

## Search limits

- No Scopus, Web of Science, IEEE Xplore full-text crawl, citation-chaining census, or formal risk-of-bias assessment.
- Broad searches were intentionally truncated at 20 results; absence outside the retained exact sources is not evidence of non-existence.
- Citation counts and OA locations can drift after the access date.
- No result here establishes an optimal receptive field for GTN. The ledger establishes the calculation, diagnostic, and falsification framework only.
