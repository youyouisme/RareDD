"""Disease profiles and literature evidence."""
import re


def matched_first(profile, patient):
    """Order exact patient HPO matches first, preserving frequencies and NOT qualifiers."""
    ids = {p.id for p in (*patient.observed, *patient.excluded)}
    return sorted(profile, key=lambda row: row["hpo_id"] not in ids)


def pmid(value):
    if value is None:
        return None
    match = re.fullmatch(r"(?:PMID:\s*)?(\d+)", str(value).strip(), re.I)
    return str(int(match[1])) if match else None


def retrieve(candidates, patient, retriever, source_pmids=(), depth=5):
    """Query disease and observed HPO names; exclude source PMIDs after Top-k retrieval.
    Retain missing PMIDs as unverifiable; do not refill filtered results."""
    blocked = {pmid(x) for x in source_pmids if pmid(x)}
    searches, audit = [], []
    for candidate in candidates:
        query = " ".join([candidate.name, *(p.name for p in patient.observed)])
        retained = []
        for hit in retriever.search(query, depth)[:depth]:
            hit_pmid = pmid(hit.get("pmid"))
            status = ("excluded_same_source" if hit_pmid in blocked else
                      "unverifiable" if not hit_pmid or not blocked else "retained")
            audit.append(dict(candidate_id=candidate.candidate_id, pmid=hit_pmid, status=status))
            if status != "excluded_same_source":
                retained.append(hit)
        searches.append((candidate.candidate_id, retained))
    # Rank-major interleaving; deduplicate identical text only, not same PMID.
    packed = {}
    for rank in range(depth):
        for cid, hits in searches:
            if rank >= len(hits):
                continue
            hit = hits[rank]
            text = hit["text"].strip()
            if not text:
                continue
            key = " ".join(text.split())
            if key not in packed:
                packed[key] = dict(title=hit.get("title", ""), text=text, retrieved_for=[])
            if cid not in packed[key]["retrieved_for"]:
                packed[key]["retrieved_for"].append(cid)
    return list(packed.values()), audit
