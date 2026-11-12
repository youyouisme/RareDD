"""Offline synthetic demo: python -m examples.synthetic."""
import json

from ours import diagnose
from ours.contracts import Patient, Phenotype, Prediction
from ours.core import normalize
from ours.pipeline import SELECTED_TOOLS
from ours.prompts import ZERO_SHOT, IDENTITY, EVIDENCE_RERANK
from ours.tools import FrozenTool


class DemoRegistry:
    version = "synthetic-registry-v1"

    def resolve_id(self, value):
        return value if value in {f"DEMO:{i}" for i in range(1, 11)} else None

    def exact_name(self, name):
        return next((f"DEMO:{i}" for i in range(1, 11)
                     if normalize(name) == normalize(f"Fictional disease {i}")), None)

    def concept(self, key):
        return (dict(entity_key=key, name=f"Fictional disease {key.split(':')[1]}",
                     synonyms=[], definition="Synthetic placeholder, not medical knowledge.", parents=[])
                if self.resolve_id(key) else {})

    def tfidf(self, name, k):
        # Synthetic retrieval fixture.
        return [self.concept(f"DEMO:{i}") for i in range(1, min(10, k) + 1)]

    def compare_ids(self, predicted, gold_ids):
        if predicted in gold_ids:
            return True
        if self.resolve_id(predicted) and all(self.resolve_id(g) for g in gold_ids):
            return False
        return None


class DemoKnowledge:
    version = "synthetic-knowledge-v1"

    def descriptions(self, key):
        return dict(medgen="Fictional description only.", orphanet="")

    def phenotype_profile(self, key):
        return [dict(hpo_id="DEMO-HP:2", name="Fictional feature B", frequency=None, qualifier=""),
                dict(hpo_id="DEMO-HP:1", name="Fictional feature A", frequency=None, qualifier="")]


class DemoRetriever:
    version = "synthetic-retriever-v1"

    def search(self, query, k):
        return [dict(title="Synthetic passage", text="Fictional evidence: " + query)]


class DemoLLM:
    fingerprint = "synthetic-backbone-v1"

    def __init__(self):
        self.calls = []

    def input_tokens(self, prompt, payload):
        return len(json.dumps(payload, ensure_ascii=False)) // 4

    def json(self, prompt, payload, input_limit=46000):
        self.calls.append((prompt, payload, input_limit))
        if prompt == ZERO_SHOT:
            return dict(diagnoses=[f"Fictional disease {i}" for i in range(1, 11)])
        if prompt == IDENTITY:
            return dict(entity_key=None, relation="no_match")
        candidates = payload["candidates"][:5]
        if prompt == EVIDENCE_RERANK:
            candidates = list(reversed(candidates))
        return dict(ranking=[dict(candidate_id=c["candidate_id"], disease_name=c["disease_name"])
                             for c in candidates])


def run():
    patient = Patient((Phenotype("DEMO-HP:1", "Fictional feature A"),))
    tools = [FrozenTool(name, [Prediction(f"Fictional disease {i}", f"DEMO:{i}")
                              for i in range(1, 6)]) for name in SELECTED_TOOLS]
    llm = DemoLLM()
    result = diagnose(patient, tools, DemoRegistry(), DemoKnowledge(), DemoRetriever(), llm)
    return result, llm


if __name__ == "__main__":
    result, llm = run()
    print(json.dumps(dict(synthetic=True, calls=len(llm.calls), source_rrf=result["source_rrf"],
                         fused_shortlist=result["fused_shortlist"],
                         final_ranking=result["final_ranking"]), indent=2))
