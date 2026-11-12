"""Patient data and external resource interfaces. Inference excludes Gold labels."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Phenotype:
    id: str
    name: str


@dataclass(frozen=True)
class Patient:
    observed: tuple[Phenotype, ...]
    excluded: tuple[Phenotype, ...] = ()

    def payload(self):
        return {
            "observed_phenotypes": [vars(p) for p in self.observed],
            "excluded_phenotypes": [vars(p) for p in self.excluded],
        }


@dataclass(frozen=True)
class Prediction:
    name: str
    native_id: str | None = None


@dataclass
class Candidate:
    candidate_id: str
    entity_key: str  # Lookup key; not an evaluation identity.
    name: str
    sources: list[dict] = field(default_factory=list)

    def payload(self):
        return dict(candidate_id=self.candidate_id, entity_key=self.entity_key,
                    disease_name=self.name, sources=self.sources)


class IdentityStore(Protocol):
    version: str

    def resolve_id(self, native_id: str) -> str | None:
        """Resolve strict equivalents, preferring MONDO; otherwise retain native identity.

        Related, parent/child and shared-gene links are not equivalence.
        Missing mappings remain unresolved."""
        ...

    def exact_name(self, name: str) -> str | None:
        """Unique canonical-name / reliable exact-synonym match, else None."""
        ...

    def concept(self, key: str) -> dict:
        """Return ID, name, synonyms, definition and parents; unknown keys return {}."""
        ...

    def tfidf(self, name: str, k: int) -> list[dict]:
        """Return Top-k registry concepts with entity_key and concept fields; exclude Gold."""
        ...

    def compare_ids(self, predicted: str, gold_ids: tuple[str, ...]) -> bool | None:
        """True: equivalent to a Gold ID. False: verified different from all Gold IDs.
        None: unresolved mapping; namespace string inequality alone is insufficient."""
        ...


class Knowledge(Protocol):
    version: str

    def descriptions(self, key: str) -> dict:
        """Available MedGen / Orphanet descriptions; {} if unavailable."""
        ...

    def phenotype_profile(self, key: str) -> list[dict]:
        """Return hpo_id, name, frequency and qualifier (e.g. NOT); keep provenance separate."""
        ...


class Retriever(Protocol):
    version: str

    def search(self, query: str, k: int) -> list[dict]:
        """Return ranked {title, text, pmid?} passages from a frozen BM25 index.
        PMIDs must be explicit or authoritatively mapped from PMCID."""
        ...


class LLM(Protocol):
    fingerprint: str  # Model + revision + sampling/reasoning settings.

    def input_tokens(self, prompt: str, payload: dict) -> int:
        """Count input tokens, including chat-template overhead."""
        ...

    def json(self, prompt: str, payload: dict, input_limit: int = 46000) -> dict:
        """Return parsed JSON; raise on empty, malformed, or over-budget input."""
        ...


class Tool(Protocol):
    name: str

    def rank(self, patient: Patient, k: int) -> list[Prediction]: ...
