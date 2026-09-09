"""
Tests for compound ID mapping and KEGG pathway enrichment.

Everything here runs offline. The two services involved are free, unauthenticated
and maintained by other people, so a test suite that calls them on every run
would be both flaky and rude. Responses are stubbed with payloads captured
verbatim from the live APIs, which also means these tests keep working when
MetaboAnalyst or KEGG are down.

The live services are exercised by ``TestLiveServices``, which is skipped unless
``MORTIS_TEST_NETWORK=1``. Run it after changing anything about request shape:

    MORTIS_TEST_NETWORK=1 pytest tests/test_pathway.py -k Live
"""

from __future__ import annotations

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mortis as mt
from mortis import pathway as pathway_module
from mortis.exceptions import InvalidParameterError, MortisError

# --- payloads captured from the live APIs -----------------------------------

MAPCOMPOUNDS_RESPONSE = json.dumps({
    "Query": ["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"],
    "Match": ["Spermidine", "Lysine", "Citric acid", "NA"],
    "HMDB": ["HMDB0001257", "HMDB0000182", "HMDB0000094", "NA"],
    "PubChem": ["1102", "5962", "311", "NA"],
    "ChEBI": ["16610", "18019", "30769", "NA"],
    "KEGG": ["C00315", "C00047", "C00158", "NA"],
    "METLIN": ["254", "5200", "16", "NA"],
    "SMILES": ["NCCCCNCCCN", "NCCCC[C@H](N)C(O)=O", "OC(=O)CC(O)(CC(O)=O)C(O)=O", "NA"],
    "Comment": ["1", "1", "1", "0"],
})

KEGG_LINKS = "\n".join([
    "cpd:C00315\tpath:map00330",   # Arginine and proline metabolism (metabolic)
    "cpd:C00315\tpath:map00480",   # Glutathione metabolism (metabolic)
    "cpd:C00315\tpath:map01100",   # Metabolic pathways (global - must be dropped)
    "cpd:C00315\tpath:map02010",   # ABC transporters (transport - must be dropped)
    "cpd:C00047\tpath:map00330",
    "cpd:C00047\tpath:map00480",
    "cpd:C00047\tpath:map00970",   # Aminoacyl-tRNA (translation - must be dropped)
    "cpd:C00047\tpath:map02010",
    "cpd:C00158\tpath:map00330",
    "cpd:C00158\tpath:map00480",
])

KEGG_NAMES = "\n".join([
    "map00330\tArginine and proline metabolism",
    "map00480\tGlutathione metabolism",
    "map01100\tMetabolic pathways",
    "map02010\tABC transporters",
    "map00970\tAminoacyl-tRNA biosynthesis",
])

KEGG_BRITE = "\n".join([
    "+C\tMap number",
    "!",
    "AMetabolism",
    "B  Global and overview maps",
    "C    01100  Metabolic pathways",
    "B  Amino acid metabolism",
    "C    00330  Arginine and proline metabolism",
    "B  Metabolism of other amino acids",
    "C    00480  Glutathione metabolism",
    "AGenetic Information Processing",
    "B  Translation",
    "C    00970  Aminoacyl-tRNA biosynthesis",
    "AEnvironmental Information Processing",
    "B  Membrane transport",
    "C    02010  ABC transporters",
])


@pytest.fixture
def offline(monkeypatch):
    """Serve every request from the captured payloads, and count the calls."""
    calls = []

    def fake_request(url, body, timeout, retries):
        calls.append(url)
        if "mapcompounds" in url:
            return MAPCOMPOUNDS_RESPONSE
        if "br:br08901" in url:
            return KEGG_BRITE
        if "link/pathway/compound" in url:
            return KEGG_LINKS
        if "list/pathway/map" in url:
            return KEGG_NAMES
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(pathway_module, "_request", fake_request)
    return calls


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "cache")


def _adata(names):
    adata = ad.AnnData(X=np.ones((5, len(names)), dtype=np.float32))
    adata.var_names = names
    return adata


def _result(names, deltas):
    return pd.DataFrame({
        "metabolite": names,
        "delta": np.asarray(deltas, dtype=float),
        "pval_adj": np.where(np.abs(deltas) > 0.5, 0.001, 0.9),
    })


# ---------------------------------------------------------------------------
# Compound mapping
# ---------------------------------------------------------------------------

class TestMapCompoundIds:

    def test_parses_all_identifier_columns(self, offline, cache):
        table = mt.map_compound_ids(
            ["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"], cache=cache
        )
        assert list(table["kegg"]) == ["C00315", "C00047", "C00158", "NA"]
        assert list(table["hmdb"])[:1] == ["HMDB0001257"]
        assert {"query", "match", "hmdb", "kegg", "pubchem", "chebi", "metlin", "smiles"} <= set(table.columns)

    def test_unmatched_rows_are_kept_and_flagged(self, offline, cache):
        """
        Dropping them would hide the match rate, which is the first thing that
        needs checking before any pathway result is believed.
        """
        table = mt.map_compound_ids(
            ["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"], cache=cache
        )
        assert len(table) == 4
        assert list(table["matched"]) == [True, True, True, False]

    def test_cache_prevents_a_second_call(self, offline, cache):
        names = ["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"]
        mt.map_compound_ids(names, cache=cache)
        assert len(offline) == 1
        mt.map_compound_ids(names, cache=cache)
        assert len(offline) == 1, "second call should have been served from cache"

    def test_cache_can_be_disabled(self, offline):
        names = ["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"]
        mt.map_compound_ids(names, cache=False)
        mt.map_compound_ids(names, cache=False)
        assert len(offline) == 2

    def test_rejects_empty_input(self, offline, cache):
        with pytest.raises(InvalidParameterError, match="names is empty"):
            mt.map_compound_ids([], cache=cache)

    def test_bad_json_raises_network_error(self, monkeypatch, cache):
        monkeypatch.setattr(
            pathway_module, "_request", lambda *a, **k: "<html>503 Service Unavailable</html>"
        )
        with pytest.raises(pathway_module.NetworkError, match="not JSON"):
            mt.map_compound_ids(["Spermidine"], cache=cache)

    def test_unreachable_service_raises_rather_than_returning_partial(self, monkeypatch, cache):
        def always_fail(url, body, timeout, retries):
            raise pathway_module.NetworkError("simulated outage")

        monkeypatch.setattr(pathway_module, "_request", always_fail)
        with pytest.raises(pathway_module.NetworkError):
            mt.map_compound_ids(["Spermidine"], cache=cache)


# ---------------------------------------------------------------------------
# KEGG pathway sets
# ---------------------------------------------------------------------------

class TestKeggPathwaySets:

    def test_keeps_only_metabolic_pathways(self, offline, cache):
        """
        The reason metabolic_only exists. Transport and translation maps carry
        plenty of metabolites without being metabolic pathways, and on a real
        21-compound panel all three of ABC transporters, Aminoacyl-tRNA
        biosynthesis and Protein digestion came out significant.
        """
        sets = mt.fetch_kegg_pathway_sets(cache=cache)
        assert set(sets) == {"Arginine and proline metabolism", "Glutathione metabolism"}
        assert "ABC transporters" not in sets
        assert "Aminoacyl-tRNA biosynthesis" not in sets

    def test_global_maps_are_dropped(self, offline, cache):
        """'Metabolic pathways' spans thousands of compounds and is always 'enriched'."""
        assert "Metabolic pathways" not in mt.fetch_kegg_pathway_sets(cache=cache)

    def test_metabolic_only_false_keeps_everything(self, offline, cache):
        sets = mt.fetch_kegg_pathway_sets(cache=cache, metabolic_only=False)
        assert "ABC transporters" in sets
        assert "Metabolic pathways" in sets

    def test_restricting_to_given_compounds(self, offline, cache):
        sets = mt.fetch_kegg_pathway_sets(["C00315"], cache=cache)
        assert all(members == ["C00315"] for members in sets.values())

    def test_cached_across_calls(self, offline, cache):
        mt.fetch_kegg_pathway_sets(cache=cache)
        first = len(offline)
        mt.fetch_kegg_pathway_sets(cache=cache)
        assert len(offline) == first


class TestBriteParsing:

    def test_classifies_from_the_hierarchy(self):
        keep = pathway_module._metabolic_pathway_ids(KEGG_BRITE)
        assert keep == {"map00330", "map00480"}

    def test_empty_hierarchy_is_survivable(self):
        assert pathway_module._metabolic_pathway_ids("") == set()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestAnnotatePathways:

    NAMES = ["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"]

    def test_writes_identifiers_back_to_var(self, offline, cache):
        adata = _adata(self.NAMES)
        mt.annotate_pathways(
            adata, _result(self.NAMES, [0.9, 0.9, 0.0, 0.0]), cache=cache, min_size=2
        )
        assert list(adata.var["compound_ids_kegg"]) == ["C00315", "C00047", "C00158", "NA"]
        assert adata.var["compound_ids_hmdb"].iloc[0] == "HMDB0001257"

    def test_returns_mapping_alongside_enrichment(self, offline, cache):
        adata = _adata(self.NAMES)
        identifiers, enrichment = mt.annotate_pathways(
            adata, _result(self.NAMES, [0.9, 0.9, 0.0, 0.0]), cache=cache, min_size=2
        )
        assert len(identifiers) == 4
        assert "pathway" in enrichment.columns

    def test_enrichment_uses_compound_names_not_kegg_ids(self, offline, cache):
        adata = _adata(self.NAMES)
        _, enrichment = mt.annotate_pathways(
            adata, _result(self.NAMES, [0.9, 0.9, 0.0, 0.0]), cache=cache, min_size=2
        )
        assert not enrichment.empty
        assert "Spermidine" in " ".join(enrichment["hits"].astype(str))

    def test_raises_when_nothing_maps(self, monkeypatch, cache):
        empty = json.dumps({
            "Query": ["zzz"], "Match": ["NA"], "HMDB": ["NA"], "KEGG": ["NA"],
            "PubChem": ["NA"], "ChEBI": ["NA"], "METLIN": ["NA"], "SMILES": ["NA"],
        })
        monkeypatch.setattr(pathway_module, "_request", lambda *a, **k: empty)
        with pytest.raises(MortisError, match="KEGG identifier"):
            mt.annotate_pathways(_adata(["zzz"]), _result(["zzz"], [0.9]), cache=cache)

    def test_warns_when_no_pathway_is_large_enough(self, offline, cache):
        adata = _adata(self.NAMES)
        with pytest.warns(UserWarning, match="min_size"):
            mt.annotate_pathways(
                adata, _result(self.NAMES, [0.9, 0.9, 0.0, 0.0]), cache=cache, min_size=50
            )

    def test_rejects_result_without_metabolite_column(self, offline, cache):
        with pytest.raises(InvalidParameterError, match="metabolite"):
            mt.annotate_pathways(_adata(self.NAMES), pd.DataFrame({"delta": [1.0]}), cache=cache)


class TestClearCache:

    def test_removes_cached_files(self, offline, cache):
        mt.map_compound_ids(["Spermidine", "L-Lysine", "Citric acid", "NotARealCompoundXYZ"],
                            cache=cache)
        assert mt.clear_cache(cache) >= 1
        assert mt.clear_cache(cache) == 0


# ---------------------------------------------------------------------------
# Live services (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("MORTIS_TEST_NETWORK") != "1",
    reason="set MORTIS_TEST_NETWORK=1 to exercise the live APIs",
)
class TestLiveServices:
    """Guards the request shape against silent API changes."""

    def test_metaboanalyst_contract(self, cache):
        table = mt.map_compound_ids(["Spermidine", "Glucose"], cache=cache)
        assert table.loc[table["query"] == "Spermidine", "kegg"].iloc[0] == "C00315"

    def test_kegg_contract(self, cache):
        sets = mt.fetch_kegg_pathway_sets(cache=cache)
        assert len(sets) > 50
        assert "ABC transporters" not in sets
        assert any("Arginine" in name for name in sets)
