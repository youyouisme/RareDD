# Multi-source, knowledge-guided rare disease diagnosis

## Workflow

1. **Candidate discovery:** five HPO tools return Top-5 each; the LLM generates Top-10 from patient phenotypes.
2. **Candidate integration:** match disease IDs and exact names/synonyms. Resolve remaining LLM names using TF-IDF Top-10 concepts and LLM identity matching.
3. **Source-rank fusion:** combine original source ranks with RRF-60, counting each source once per disease.
4. **Dual-view screening:** rank candidates without (D) and with (R) disease descriptions. Combine both Top-5 rankings with RRF-60 to select the shortlist.
5. **Differential reranking:** rerank the shortlist using disease information, clinical descriptions, HPO profiles and BM25 passages.

Input: observed and optional excluded HPO IDs and names. Gold labels are used only for evaluation.

## Setup

Requires Python 3.10+.

```bash
pip install -e .
python -m examples.synthetic
```

The example uses synthetic data. Supply dataset records and resource implementations through `ours/contracts.py`.

## Files

| File | Purpose |
| --- | --- |
| `ours/pipeline.py` | Diagnostic workflow |
| `ours/core.py` | Disease matching and rank fusion |
| `ours/evidence.py` | Evidence retrieval and PMID filtering |
| `ours/prompts.py` | Inference and evaluation prompts |
| `ours/evaluation.py` | Prediction evaluation and metrics |
| `ours/contracts.py` | Data structures and resource interfaces |
| `ours/tools.py` | Tool interfaces |
| `ours/transport.py` | Chat-completions client |

## Tools

Use the official tools in the order below. Connect local tools through `ours/tools.py`, or load saved rankings with `FrozenTool`.

| Source | Upstream / installation |
| --- | --- |
| LIRICAL | [Repository](https://github.com/TheJacksonLaboratory/LIRICAL), [releases](https://github.com/TheJacksonLaboratory/LIRICAL/releases) |
| PhenoApt | [Official client](https://github.com/phenoapt/phenoapt); `pip install phenoapt` |
| Phrank | [Source](https://bitbucket.org/bejerano/phrank/src/master/), [package](https://pypi.org/project/phrank/); `pip install phrank` |
| PhenoDP-Ranker | [Repository and resource instructions](https://github.com/TianLab-Bioinfo/PhenoDP) |
| VC-Ranker | [Repository and resource instructions](https://github.com/cloudna-AI4LS/VC-RDAgent) |

## LLM configuration

Configure the diagnostic backbone and evaluation judge separately. `ChatLLM` accepts an endpoint, model, token counter and provider options; the API key defaults to `LLM_API_KEY`.
Input limit: **46,000 tokens** per call. Default output limit: **8,192 tokens**. If the final reranking input exceeds the limit, the lowest-ranked retrieved passages are dropped until it fits.

## Evaluation

- **Tool predictions:** check native disease IDs and verified equivalents first.
- **LLM predictions:** evaluate returned disease names against Gold names and verified identities.
- **Unresolved predictions:** use the adapted RareArena 0/1/2 judge with reasoning effort set to low. Only score 2 counts as correct; unparsed judgments remain pending.
- **Caching:** separate identity judgments by backbone; key evaluation judgments by prediction, Gold reference, judge settings and resource version.

Report Hit@1/5 for RareBench, MyGene2 and PhenoDP-Lit, and their equally weighted macro-average. For repeated runs, report the mean and sample standard deviation.
