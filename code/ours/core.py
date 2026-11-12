"""Disease identity resolution and rank fusion."""
import hashlib
import json
import unicodedata

from .contracts import Candidate
from .prompts import IDENTITY, RELATIONS


def normalize(name):
    # Preserve subtype numbers, hyphens, negation and genes.
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def integrate(source_lists, registry, backbone, cache):
    """Retain tool identities and source ranks; resolve LLM names with the backbone.

    Cache keys include model settings, prompt and registry version."""
    grouped = {}
    for source, predictions in source_lists.items():
        for rank, prediction in enumerate(predictions, 1):
            name = prediction.name.strip()
            if not name:
                raise ValueError("Empty disease name: repair metadata before inference")
            if prediction.native_id:
                key = registry.resolve_id(prediction.native_id) or prediction.native_id
            else:
                key = registry.exact_name(name)
                if key is None:
                    options = registry.tfidf(name, 10)
                    ck = digest([backbone.fingerprint, registry.version, IDENTITY, name, options])
                    if ck not in cache:
                        match = (backbone.json(IDENTITY, dict(name=name, concepts=options))
                                 if options else dict(entity_key=None, relation="no_match"))
                        if not isinstance(match, dict) or match.get("relation") not in RELATIONS:
                            raise ValueError("Invalid identity judgment; not cached")
                        if match.get("entity_key") is not None and match["entity_key"] not in {
                                c["entity_key"] for c in options}:
                            raise ValueError("Identity matcher selected an unsupplied concept")
                        if match["relation"] == "same_entity" and match.get("entity_key") is None:
                            raise ValueError("A same-entity judgment must select a concept")
                        cache[ck] = match
                    match = cache[ck]
                    valid = {c["entity_key"] for c in options}
                    if match.get("relation") == "same_entity":
                        if match.get("entity_key") not in valid:
                            raise ValueError("Identity matcher selected an unsupplied concept")
                        key = match["entity_key"]
                key = key or "NAME:" + normalize(name)
            if key not in grouped:
                concept = registry.concept(key)
                grouped[key] = Candidate(f"C{len(grouped)+1:03d}", key, concept.get("name") or name)
            grouped[key].sources.append(dict(source=source, rank=rank, original_name=name,
                                            native_id=prediction.native_id))
    return list(grouped.values())


def rrf(rankings, constant=60):
    """Fuse {candidate_id: original_1_based_rank} mappings; ties follow first appearance."""
    scores = {}
    for ranking in rankings:
        for cid, rank in ranking.items():
            if rank < 1:
                raise ValueError("Ranks must be one-based")
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=lambda cid: -scores[cid])


def source_rrf(candidates):
    sources = {}
    for candidate in candidates:
        for entry in candidate.sources:
            ranks = sources.setdefault(entry["source"], {})
            cid = candidate.candidate_id
            ranks[cid] = min(ranks.get(cid, entry["rank"]), entry["rank"])
    return rrf(sources.values())


def parse_ranking(response, allowed, expected):
    """Validate ranking length, candidate IDs and returned names without filling gaps."""
    rows = response.get("ranking")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError("Incomplete ranking; save raw response for explicit recovery")
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(allowed):
        raise ValueError("Duplicate or out-of-pool candidate ID")
    if any(not isinstance(r.get("disease_name"), str) or not r["disease_name"].strip() for r in rows):
        raise ValueError("No reliable returned disease name")
    # Preserve returned names for evaluation.
    return rows
