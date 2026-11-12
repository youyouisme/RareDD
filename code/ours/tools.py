"""Official tool interfaces. Implement local rank() adapters or use FrozenTool."""
import json
from urllib.parse import urlencode
from urllib.request import urlopen

from .contracts import Prediction


class FrozenTool:
    """Return saved native tool rankings."""
    def __init__(self, name, predictions):
        self.name, self.predictions = name, predictions

    def rank(self, patient, k):
        return list(self.predictions[:k])


class PhenoAptAPI:
    """PhenoApt v3 disease ranking with observed HPO IDs.
    Freeze responses for repeated experiments.
    Official client: https://github.com/phenoapt/phenoapt"""
    name = "PhenoApt"

    def __init__(self, endpoint="https://www.phenoapt.org/phenoapt/api/v3/rank-disease", timeout=60):
        self.endpoint, self.timeout = endpoint, timeout

    def rank(self, patient, k):
        ids = list(dict.fromkeys(p.id for p in patient.observed))
        if not ids:
            raise ValueError("PhenoApt requires observed HPO IDs")
        params = urlencode(dict(p=",".join(ids), weights=",".join("1" for _ in ids), n=k))
        with urlopen(self.endpoint + "?" + params, timeout=self.timeout) as response:
            rows = json.load(response)["data"]["rank"]
        ranks = [int(row["rank"]) for row in rows]
        if len(ranks) != len(set(ranks)) or any(r < 1 for r in ranks):
            raise ValueError("Invalid PhenoApt ranks")
        result = []
        for row in sorted(rows, key=lambda r: int(r["rank"]))[:k]:
            native_id = row["disease_id"]
            # Official API's OMD prefix denotes an OMIM disease record.
            native_id = "OMIM:" + native_id[4:] if native_id.startswith("OMD:") else native_id
            result.append(Prediction(row["disease_name"], native_id))
        if not result:
            raise ValueError("Empty PhenoApt ranking")
        return result


class LocalTool:
    def rank(self, patient, k):
        raise NotImplementedError(f"Implement {self.name} using the upstream pseudocode below")


class LIRICAL(LocalTool):
    name = "LIRICAL"
    # Source: https://github.com/TheJacksonLaboratory/LIRICAL
    # Download: https://github.com/TheJacksonLaboratory/LIRICAL/releases
    # PSEUDOCODE: load upstream ontology/annotations; run phenotype-only ranking
    # with observed and excluded HPO; return native Top-k disease IDs and names.


class Phrank(LocalTool):
    name = "Phrank"
    # Source: https://bitbucket.org/bejerano/phrank/src/master/
    # Package / download: https://pypi.org/project/phrank/ ; pip install phrank
    # PSEUDOCODE: initialize with HPO DAG, disease-HPO and gene-disease data;
    # rank diseases with observed HPO and return native Top-k IDs and names.
    # If genes are required, use the full KB gene set for every patient.
    # Exclude NOT annotations from positive features.


class PhenoDPRanker(LocalTool):
    name = "PhenoDP-Ranker"
    # Source / installation: https://github.com/TianLab-Bioinfo/PhenoDP
    # PSEUDOCODE: load official model/KB resources; use the disease ranker
    # with observed HPO; return native Top-k disease IDs and names.


class VCRanker(LocalTool):
    name = "VC-Ranker"
    # Source / data instructions: https://github.com/cloudna-AI4LS/VC-RDAgent
    # PSEUDOCODE: load official resources; run VC-Ranker on observed HPO;
    # return native Top-k OMIM/ORPHA IDs and names. Use only the ranker component.
