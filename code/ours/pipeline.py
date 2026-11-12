"""Multi-source discovery -> dual-view screening -> differential reranking."""
from .contracts import Prediction
from .core import integrate, source_rrf, rrf, parse_ranking
from .evidence import matched_first, retrieve
from .prompts import ZERO_SHOT, SCREEN, EVIDENCE_RERANK

SELECTED_TOOLS = ("LIRICAL", "PhenoApt", "Phrank", "PhenoDP-Ranker", "VC-Ranker")
INPUT_LIMIT = 46000


def fit_passages(backbone, payload, passages, limit=INPUT_LIMIT):
    """Drop the lowest-ranked passages until the final input fits the token limit."""
    kept = len(passages)
    while kept and backbone.input_tokens(
            EVIDENCE_RERANK, dict(payload, literature_passages=passages[:kept])) > limit:
        kept -= 1
    return passages[:kept]


def diagnose(patient, tools, registry, knowledge, retriever, backbone, *,
             identity_cache=None, source_pmids=()):
    """Run diagnosis with a 46k input-token limit per call. If the final reranking
    input exceeds the limit, the lowest-ranked retrieved passages are dropped.

    source_pmids is used only for filtering and is excluded from LLM inputs."""
    if tuple(t.name for t in tools) != SELECTED_TOOLS:
        raise ValueError("Supply the five selected tools in their fixed source order")
    sources = {t.name: t.rank(patient, 5)[:5] for t in tools}
    zero = backbone.json(ZERO_SHOT, patient.payload())
    names = zero.get("diagnoses")
    if not isinstance(names, list) or not names or len(names) > 10:
        raise ValueError("Expected one to ten Zero-shot disease names")
    if any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Invalid Zero-shot disease name")
    sources["LLM"] = [Prediction(name=n) for n in names]
    cache = {} if identity_cache is None else identity_cache
    pool = integrate(sources, registry, backbone, cache)
    by_id = {c.candidate_id: c for c in pool}
    prior = source_rrf(pool)
    ordered = [by_id[cid] for cid in prior]
    common = dict(patient=patient.payload(), candidates=[c.payload() for c in ordered],
                  source_rrf_order=prior)
    n = min(5, len(pool))
    d = parse_ranking(backbone.json(SCREEN, common), prior, n)
    descriptions = {c.candidate_id: knowledge.descriptions(c.entity_key) for c in ordered}
    r = parse_ranking(backbone.json(SCREEN, dict(common, disease_descriptions=descriptions)), prior, n)
    fused = rrf([{row["candidate_id"]: i for i, row in enumerate(view, 1)} for view in (d, r)])[:5]
    shortlist = [by_id[cid] for cid in fused]
    evidence = {
        c.candidate_id: dict(
            disease_information=registry.concept(c.entity_key),
            clinical_descriptions=descriptions[c.candidate_id],
            phenotype_profile=matched_first(knowledge.phenotype_profile(c.entity_key), patient),
        ) for c in shortlist
    }
    passages, audit = retrieve(shortlist, patient, retriever, source_pmids)
    payload = dict(patient=patient.payload(), candidates=[c.payload() for c in shortlist],
                   disease_knowledge=evidence, literature_passages=passages,
                   shortlist_order=fused)
    kept = fit_passages(backbone, payload, passages)
    payload["literature_passages"] = kept
    final = parse_ranking(backbone.json(EVIDENCE_RERANK, payload, input_limit=INPUT_LIMIT),
                          fused, len(fused))
    return dict(zero_shot=names, initial_candidates=[c.payload() for c in pool],
                source_rrf=prior, D=d, R=r, fused_shortlist=fused,
                final_ranking=final, retrieval_audit=audit,
                truncated_passages=len(passages) - len(kept),
                resource_versions=dict(identity=registry.version, knowledge=knowledge.version,
                                       retrieval=retriever.version, backbone=backbone.fingerprint))
