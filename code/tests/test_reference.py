import unittest
from unittest.mock import patch
from io import BytesIO
import json

from examples.synthetic import DemoRegistry, DemoKnowledge, DemoRetriever, DemoLLM, run
from ours import diagnose
from ours.contracts import Patient, Phenotype, Prediction, Candidate
from ours.core import integrate, source_rrf, rrf, parse_ranking
from ours.evaluation import Evaluator, Gold, hit_at, Decision, summarize, across_runs
from ours.evidence import matched_first, retrieve
from ours.prompts import SCREEN
from ours.pipeline import SELECTED_TOOLS
from ours.tools import FrozenTool, PhenoAptAPI


class Judge:
    fingerprint = "synthetic-judge-low"

    def __init__(self, answer="Disease name: score 2;"):
        self.answer, self.calls = answer, []

    def text(self, prompt):
        self.calls.append(prompt)
        return self.answer


class ReferenceTests(unittest.TestCase):
    def test_complete_pipeline_is_offline_and_frozen(self):
        result, llm = run()
        self.assertEqual(len(llm.calls), 4)  # No unresolved names in this fixture.
        self.assertEqual(len(result["zero_shot"]), 10)
        self.assertEqual({r["candidate_id"] for r in result["final_ranking"]},
                         set(result["fused_shortlist"]))
        d, r = [p for prompt, p, _ in llm.calls if prompt == SCREEN]
        self.assertNotIn("disease_descriptions", d)
        self.assertIn("disease_descriptions", r)
        self.assertEqual(d["candidates"], r["candidates"])
        self.assertNotIn("gold", json.dumps(llm.calls).lower())
        self.assertEqual(llm.calls[-1][2], 46000)

    def test_final_context_drops_lowest_ranked_passages(self):
        class Tight(DemoLLM):
            def input_tokens(self, prompt, payload):
                return 45998 + len(payload.get("literature_passages", ()))
        llm = Tight()
        patient = Patient((Phenotype("DEMO-HP:1", "Fictional feature A"),))
        tools = [FrozenTool(name, [Prediction(f"Fictional disease {i}", f"DEMO:{i}")
                                   for i in range(1, 6)]) for name in SELECTED_TOOLS]
        result = diagnose(patient, tools, DemoRegistry(), DemoKnowledge(), DemoRetriever(), llm)
        self.assertEqual(len(llm.calls[-1][1]["literature_passages"]), 2)
        self.assertEqual(result["truncated_passages"], 3)

    def test_rrf_uses_original_ranks(self):
        self.assertEqual(rrf([{"A": 5, "B": 1}, {"A": 1}]), ["A", "B"])
        a = Candidate("A", "DEMO:1", "A", [dict(source="S", rank=1), dict(source="S", rank=2)])
        b = Candidate("B", "DEMO:2", "B", [dict(source="S", rank=2), dict(source="T", rank=3)])
        self.assertEqual(source_rrf([a, b]), ["B", "A"])

    def test_id_conflict_does_not_merge_same_text(self):
        llm = DemoLLM()
        pool = integrate({"T": [Prediction("same text", "DEMO:1"), Prediction("same text", "DEMO:2")]},
                         DemoRegistry(), llm, {})
        self.assertEqual(len(pool), 2)
        self.assertFalse(llm.calls)

    def test_only_same_entity_merges(self):
        class Matcher(DemoLLM):
            relation = "query_broader"
            def json(self, *args, **kwargs):
                return dict(entity_key="DEMO:1", relation=self.relation)
        llm = Matcher()
        lists = {"T": [Prediction("Fictional disease 1", "DEMO:1")],
                 "LLM": [Prediction("unresolved alias")]}
        self.assertEqual(len(integrate(lists, DemoRegistry(), llm, {})), 2)
        llm.relation = "same_entity"
        self.assertEqual(len(integrate(lists, DemoRegistry(), llm, {})), 1)

    def test_identity_cache_separates_backbones(self):
        cache, registry = {}, DemoRegistry()
        one, two = DemoLLM(), DemoLLM()
        two.fingerprint = "other-backbone"
        lists = {"LLM": [Prediction("unresolved alias")]}
        integrate(lists, registry, one, cache)
        integrate(lists, registry, one, cache)
        integrate(lists, registry, two, cache)
        self.assertEqual((len(one.calls), len(two.calls)), (1, 1))

    def test_invalid_identity_response_not_cached(self):
        class Broken(DemoLLM):
            def json(self, *args, **kwargs):
                return dict(entity_key="invented", relation="same_entity")
        cache = {}
        with self.assertRaises(ValueError):
            integrate({"LLM": [Prediction("unknown")]}, DemoRegistry(), Broken(), cache)
        self.assertEqual(cache, {})

    def test_matched_first_preserves_qualifier_and_order(self):
        patient = Patient((Phenotype("HP:X", "X"),))
        profile = [dict(hpo_id="HP:Y"), dict(hpo_id="HP:X", qualifier="NOT", frequency="rare")]
        ordered = matched_first(profile, patient)
        self.assertEqual(ordered, [profile[1], profile[0]])
        self.assertEqual(profile[0]["hpo_id"], "HP:Y")

    def test_pmid_filter_retains_unknown_and_distinct_same_pmid(self):
        class Retrieval:
            def search(self, query, k):
                return [dict(text="same-source", pmid="PMID:12"),
                        dict(text="unknown"), dict(text="part a", pmid="13"),
                        dict(text="part b", pmid="13"), dict(text="unknown")]
        docs, audit = retrieve([Candidate("A", "DEMO:1", "Disease")], Patient(()), Retrieval(), ["12"])
        self.assertEqual([d["text"] for d in docs], ["unknown", "part a", "part b"])
        self.assertEqual(audit[0]["status"], "excluded_same_source")
        self.assertEqual(audit[1]["status"], "unverifiable")

    def test_evaluator_locks_native_id_verdict(self):
        judge = Judge()
        evaluation = Evaluator(DemoRegistry(), judge)
        result = evaluation.evaluate(Prediction("Fictional disease 1", "DEMO:2"),
                                     Gold(("Fictional disease 1",), ("DEMO:1",)), native_tool_output=True)
        self.assertFalse(result.correct)
        self.assertFalse(judge.calls)

    def test_llm_evaluation_ignores_inherited_id(self):
        judge = Judge()
        result = Evaluator(DemoRegistry(), judge).evaluate(
            Prediction("Fictional disease 2", "DEMO:1"), Gold(("Fictional disease 1",), ("DEMO:1",)))
        self.assertFalse(result.correct)
        self.assertFalse(judge.calls)

    def test_unknown_crossmap_is_not_false_by_default(self):
        judge = Judge()
        result = Evaluator(DemoRegistry(), judge).evaluate(
            Prediction("unknown disease", "UNMAPPED:1"), Gold(("reference",), ("DEMO:1",)), native_tool_output=True)
        self.assertTrue(result.correct)
        self.assertEqual(len(judge.calls), 1)

    def test_judge_cache_includes_gold(self):
        judge, pred = Judge(), Prediction("unknown disease")
        evaluation = Evaluator(DemoRegistry(), judge)
        evaluation.evaluate(pred, Gold(("Reference A",)))
        evaluation.evaluate(pred, Gold(("Reference A",)))
        evaluation.evaluate(pred, Gold(("Reference B",)))
        self.assertEqual(len(judge.calls), 2)

    def test_only_score_two_is_correct(self):
        evaluation = Evaluator(DemoRegistry(), Judge("Disease name: score 1;"))
        self.assertFalse(evaluation.evaluate(Prediction("unknown"), Gold(("reference",))).correct)

    def test_unparseable_judgment_stays_pending(self):
        result = Evaluator(DemoRegistry(), Judge("no answer")).evaluate(
            Prediction("unknown"), Gold(("reference",)))
        self.assertIsNone(result.correct)
        self.assertIsNone(hit_at([result], 5))
        self.assertTrue(hit_at([result, Decision(True, "known")], 5))
        self.assertFalse(hit_at([], 5))

    def test_ranking_does_not_fill_partial_answer(self):
        row = dict(candidate_id="A", disease_name="Actual returned name")
        with self.assertRaises(ValueError):
            parse_ranking(dict(ranking=[row]), ["A", "B"], 2)
        self.assertEqual(parse_ranking(dict(ranking=[row]), ["A"], 1)[0]["disease_name"],
                         "Actual returned name")

    def test_metric_denominators_and_sample_sd(self):
        stats = summarize({"RareBench": [True, False, None], "MyGene2": [True], "PhenoDP-Lit": [False]})
        self.assertEqual(set(stats), {"RareBench", "MyGene2", "PhenoDP-Lit", "Macro Avg."})
        self.assertEqual(stats["RareBench"]["n"], 3)
        self.assertAlmostEqual(stats["RareBench"]["lower"], 100 / 3)
        self.assertAlmostEqual(stats["RareBench"]["upper"], 200 / 3)
        self.assertAlmostEqual(stats["Macro Avg."]["lower"], 400 / 9)
        self.assertEqual(across_runs([1, 2, 3]), dict(mean=2, sample_sd=1))

    def test_phenoapt_parser_without_network(self):
        body = json.dumps(dict(data=dict(rank=[dict(rank=1, disease_id="OMD:123456", disease_name="Synthetic")]))).encode()
        with patch("ours.tools.urlopen", return_value=BytesIO(body)) as mocked:
            result = PhenoAptAPI().rank(Patient((Phenotype("HP:0000001", "synthetic"),)), 5)
        self.assertEqual(result[0].native_id, "OMIM:123456")
        self.assertIn("n=5", mocked.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
