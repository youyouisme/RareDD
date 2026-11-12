"""Prediction evaluation and aggregate metrics."""
import re
from dataclasses import dataclass
from statistics import mean, stdev

from .core import normalize, digest
from .prompts import SINGLE_NAME_JUDGE


@dataclass(frozen=True)
class Gold:
    names: tuple[str, ...]  # Original dataset names.
    ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    correct: bool | None  # None means pending.
    basis: str


class Evaluator:
    """Independent judge with text(prompt) and low reasoning effort.

    Its fingerprint includes model, revision and reasoning settings.
    Persistent caches must implement MutableMapping."""
    def __init__(self, registry, judge, cache=None):
        self.registry, self.judge = registry, judge
        self.cache = {} if cache is None else cache

    def evaluate(self, prediction, gold, *, native_tool_output=False):
        if not gold.names and not gold.ids:
            raise ValueError("Missing Gold reference")
        if prediction is None or not prediction.name.strip():
            return Decision(False, "no_returned_answer")
        # Verified native IDs take precedence.
        if native_tool_output and prediction.native_id and gold.ids:
            match = self.registry.compare_ids(prediction.native_id, gold.ids)
            if match is not None:
                return Decision(match, "native_id")
        if normalize(prediction.name) in {normalize(n) for n in gold.names}:
            return Decision(True, "exact_gold_name")
        # Evaluate LLM outputs by returned name, not inherited IDs.
        key = self.registry.exact_name(prediction.name)
        if key is not None and gold.ids:
            match = self.registry.compare_ids(key, gold.ids)
            if match is not None:
                return Decision(match, "unique_authoritative_name")
        reference = "; ".join(gold.names) if gold.names else "; ".join(gold.ids)
        prompt = SINGLE_NAME_JUDGE.format(student_answer=f"1. {prediction.name}",
                                          reference_diagnosis=reference)
        cache_key = digest([self.registry.version, self.judge.fingerprint, prompt,
                            gold.names, gold.ids])
        if cache_key not in self.cache:
            # Propagate transport failures for caller retry.
            answer = self.judge.text(prompt)
            scores = re.findall(r"\bscore\s*[:=]?\s*([012])\b", answer, re.I)
            if not scores:
                bare = re.fullmatch(r"\s*Disease name\s*:\s*([012])\s*;?\s*", answer, re.I)
                scores = [bare[1]] if bare else []
            if len(scores) != 1:
                return Decision(None, "unparsed_judge_response")
            self.cache[cache_key] = int(scores[0])
        return Decision(self.cache[cache_key] == 2, "single_name_judge")


def hit_at(decisions, k):
    """Missing predictions count as misses; unresolved judgments remain pending."""
    if k < 1:
        raise ValueError("k must be positive")
    values = [d.correct for d in decisions[:k]]
    return True if True in values else None if None in values else False


def summarize(groups):
    """Summarize {dataset: [per-case bool or None]} with an equal-weight macro-average.
    Pending cases remain in the denominator and determine the bounds."""
    if not groups or any(not values for values in groups.values()):
        raise ValueError("Nonempty evaluation groups required")
    def bounds(values):
        return dict(n=len(values), pending=values.count(None),
                    lower=100 * values.count(True) / len(values),
                    upper=100 * (values.count(True) + values.count(None)) / len(values))
    result = {name: bounds(values) for name, values in groups.items()}
    result["Macro Avg."] = {key: mean(x[key] for x in result.values()) for key in ("lower", "upper")}
    return result


def across_runs(values):
    """Return the mean and sample standard deviation of per-run metrics."""
    return dict(mean=mean(values), sample_sd=stdev(values) if len(values) > 1 else None)
