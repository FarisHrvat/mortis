"""
Tests for chemical-class assignment and enrichment.

The compound names in ``REAL_NAMES`` are taken verbatim from a facility MSI
export, because that is the naming the classifier actually has to survive —
a curated list of textbook metabolite names would not exercise the cases that
break rule-based classifiers.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mortis as mt
from mortis.exceptions import InvalidParameterError

#: Verbatim from a facility export, plus lipid shorthand from a lipidomics panel.
REAL_NAMES = [
    "Heptanenitrile", "1-Ethylpiperidine", "1-Amino-4-methylpiperazine",
    "Morpholine-4-carboxamide", "1-Methylspermidine", "N1-Acetylspermidine",
    "N-(1-Adamantyl)urea", "Canavalmine", "Lysylalanine", "Homocarnosine",
    "Lysylproline", "N1-Acetylspermine", "Arginylalanine", "Lysyl-Lysine",
    "L-Lysine", "L-Glutamic acid", "L-Glutamine", "O-Phosphoethanolamine",
    "Spermidine", "4-Hydroxyquinoline", "Cinnamamide",
    "5alpha-Cholesta-7,24-dien-3beta-ol", "PC(34:1)", "SM(d18:1/16:0)",
    "TG(16:0_18:1_18:2)", "Cer(d18:1/24:0)", "LPC(18:0)", "Citric acid",
    "Glucose", "ATP", "Cholesterol", "Glutathione", "Carnitine", "Ascorbic acid",
]


def _adata(names=None):
    names = names or REAL_NAMES
    adata = ad.AnnData(X=np.ones((6, len(names)), dtype=np.float32))
    adata.var_names = names
    return adata


def _result(names, deltas, padj=None):
    return pd.DataFrame({
        "metabolite": names,
        "delta": np.asarray(deltas, dtype=float),
        "pval_adj": np.asarray(padj if padj is not None else np.full(len(names), 0.01)),
    })


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassifyCompounds:

    @pytest.fixture
    def classified(self):
        return mt.classify_compounds(_adata())

    @pytest.mark.parametrize("name,expected", [
        ("PC(34:1)", "Glycerophospholipid"),
        ("LPC(18:0)", "Glycerophospholipid"),
        ("SM(d18:1/16:0)", "Sphingolipid"),
        ("Cer(d18:1/24:0)", "Sphingolipid"),
        ("TG(16:0_18:1_18:2)", "Glycerolipid"),
        ("Cholesterol", "Sterol"),
        ("5alpha-Cholesta-7,24-dien-3beta-ol", "Sterol"),
        ("L-Lysine", "Amino acid"),
        ("L-Glutamic acid", "Amino acid"),
        ("Spermidine", "Polyamine"),
        ("N1-Acetylspermidine", "Polyamine"),
        ("Lysylalanine", "Peptide"),
        ("Glutathione", "Peptide"),
        ("ATP", "Nucleotide"),
        ("Glucose", "Carbohydrate"),
        ("Citric acid", "Organic acid"),
        ("Ascorbic acid", "Vitamin/cofactor"),
    ])
    def test_known_compounds(self, name, expected):
        adata = mt.classify_compounds(_adata([name]))
        assert adata.var["chemical_class"].astype(str).iloc[0] == expected

    def test_most_real_names_are_classified(self, classified):
        """
        The failure mode this guards against is a classifier that quietly dumps
        a third of the panel into a catch-all and takes every class-level
        result down with it.
        """
        classes = classified.var["chemical_class"].astype(str)
        unclassified = (classes == "Unclassified").mean()
        assert unclassified < 0.25, f"{unclassified:.0%} unclassified on real names"

    def test_acyl_and_free_carnitine_split(self):
        """Free carnitine is a quaternary amine; acylcarnitines are fatty acyls."""
        adata = mt.classify_compounds(_adata(["Carnitine", "Palmitoylcarnitine"]))
        classes = adata.var["chemical_class"].astype(str).tolist()
        assert classes == ["Amine", "Fatty acyl"]

    def test_prefix_stripping(self):
        """L-/D-/N-acetyl prefixes must not defeat the exact-name layer."""
        adata = mt.classify_compounds(_adata(["Lysine", "L-Lysine", "D-Lysine"]))
        assert set(adata.var["chemical_class"].astype(str)) == {"Amino acid"}

    def test_overrides_win(self):
        adata = mt.classify_compounds(
            _adata(["PC(34:1)", "Mystery-Compound-7"]),
            overrides={"PC(34:1)": "Xenobiotic", "mystery-compound-7": "Sterol"},
        )
        assert adata.var["chemical_class"].astype(str).tolist() == ["Xenobiotic", "Sterol"]

    def test_unknown_override_class_warns(self):
        with pytest.warns(UserWarning, match="outside CHEMICAL_CLASSES"):
            mt.classify_compounds(_adata(["PC(34:1)"]), overrides={"PC(34:1)": "Quark"})

    def test_warns_when_mostly_unclassified(self):
        with pytest.warns(UserWarning, match="Unclassified"):
            mt.classify_compounds(_adata(["zzz1", "zzz2", "zzz3", "zzz4"]))

    def test_copy_leaves_input_untouched(self):
        adata = _adata()
        mt.classify_compounds(adata, copy=True)
        assert "chemical_class" not in adata.var.columns

    def test_custom_key(self):
        adata = mt.classify_compounds(_adata(), key_added="lipid_group")
        assert "lipid_group" in adata.var.columns


class TestClassificationReport:

    def test_counts_sum_to_panel(self):
        adata = mt.classify_compounds(_adata())
        report = mt.classification_report(adata)
        assert report["n_compounds"].sum() == adata.n_vars
        assert report["fraction"].sum() == pytest.approx(1.0)

    def test_requires_classification_first(self):
        with pytest.raises(InvalidParameterError, match="classify_compounds"):
            mt.classification_report(_adata())


# ---------------------------------------------------------------------------
# Class enrichment
# ---------------------------------------------------------------------------

class TestClassEnrichment:

    def test_detects_a_shifted_class(self):
        """Every polyamine up, everything else flat."""
        names = ["Spermidine", "Spermine", "Putrescine", "Cadaverine", "Agmatine",
                 "L-Lysine", "L-Glutamine", "L-Alanine", "L-Serine", "L-Proline",
                 "Glucose", "Fructose", "Galactose", "Mannose", "Sucrose"]
        deltas = [0.95, 0.9, 0.88, 0.92, 0.86] + [0.02, -0.05, 0.01, 0.03, -0.02] * 2
        adata = mt.classify_compounds(_adata(names))
        report = mt.class_enrichment(_result(names, deltas), adata).set_index("chemical_class")

        assert report.loc["Polyamine", "median_delta"] > 0.8
        assert report.loc["Polyamine", "pval_adj"] < 0.05
        assert report.loc["Amino acid", "pval_adj"] > 0.05

    def test_skips_small_and_unclassified_classes(self):
        names = ["Spermidine", "Spermine", "Putrescine", "zzz1", "zzz2", "L-Lysine"]
        report = mt.class_enrichment(
            _result(names, [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]),
            mt.classify_compounds(_adata(names)),
            min_size=3,
        )
        assert "Unclassified" not in report["chemical_class"].tolist()

    def test_requires_classification(self):
        with pytest.raises(InvalidParameterError, match="classify_compounds"):
            mt.class_enrichment(_result(REAL_NAMES, np.zeros(len(REAL_NAMES))), _adata())

    def test_rejects_disjoint_result(self):
        adata = mt.classify_compounds(_adata())
        with pytest.raises(InvalidParameterError, match="matched"):
            mt.class_enrichment(_result(["nope_a", "nope_b"], [0.5, 0.5]), adata)


# ---------------------------------------------------------------------------
# Pathway ORA
# ---------------------------------------------------------------------------

class TestPathwayORA:

    @pytest.fixture
    def setup(self):
        names = [f"m{i:03d}" for i in range(60)]
        deltas = np.concatenate([np.full(10, 0.9), np.full(10, -0.9), np.zeros(40)])
        sets = {
            "up_pathway": names[:10],
            "down_pathway": names[10:20],
            "flat_pathway": names[20:30],
            "mixed_pathway": names[:5] + names[10:15],
        }
        return _result(names, deltas), sets

    def test_finds_the_enriched_pathway(self, setup):
        result, sets = setup
        report = mt.pathway_ora(result, sets)
        up = report[(report["pathway"] == "up_pathway") & (report["direction"] == "up")]
        assert len(up) == 1
        assert up["pval_adj"].iloc[0] < 0.05
        assert up["n_hits"].iloc[0] == 10

    def test_direction_separates_up_from_down(self, setup):
        result, sets = setup
        report = mt.pathway_ora(result, sets).set_index(["pathway", "direction"])
        assert ("up_pathway", "up") in report.index
        assert ("up_pathway", "down") not in report.index
        assert ("down_pathway", "down") in report.index

    def test_mixed_pathway_is_not_directionally_enriched(self, setup):
        """Half up and half down is not coherent dysregulation."""
        result, sets = setup
        report = mt.pathway_ora(result, sets).set_index(["pathway", "direction"])
        mixed_up = report.loc[("mixed_pathway", "up"), "pval_adj"]
        clean_up = report.loc[("up_pathway", "up"), "pval_adj"]
        assert clean_up < mixed_up

    def test_flat_pathway_absent(self, setup):
        result, sets = setup
        report = mt.pathway_ora(result, sets)
        assert "flat_pathway" not in report["pathway"].tolist()

    def test_background_is_the_measured_panel(self, setup):
        """A pathway member never measured must not count in the background."""
        result, sets = setup
        sets = dict(sets, up_pathway=sets["up_pathway"] + ["never_measured_1", "never_measured_2"])
        report = mt.pathway_ora(result, sets)
        up = report[(report["pathway"] == "up_pathway") & (report["direction"] == "up")]
        assert up["n_in_set"].iloc[0] == 10, "unmeasured members leaked into the set size"
        assert up["n_background"].iloc[0] == 60

    def test_case_insensitive_matching(self, setup):
        result, sets = setup
        upper = {k: [m.upper() for m in v] for k, v in sets.items()}
        assert len(mt.pathway_ora(result, upper)) == len(mt.pathway_ora(result, sets))

    def test_direction_false_gives_only_any(self, setup):
        result, sets = setup
        report = mt.pathway_ora(result, sets, direction=False)
        assert set(report["direction"]) == {"any"}

    def test_fdr_filter_requires_column(self, setup):
        result, sets = setup
        with pytest.raises(InvalidParameterError, match="pval_adj"):
            mt.pathway_ora(result.drop(columns="pval_adj"), sets, fdr_threshold=0.05)

    def test_rejects_empty_sets(self, setup):
        result, _ = setup
        with pytest.raises(InvalidParameterError, match="non-empty"):
            mt.pathway_ora(result, {})

    def test_empty_report_has_full_schema(self, setup):
        """Downstream code should not need to special-case 'nothing found'."""
        result, sets = setup
        report = mt.pathway_ora(result, sets, delta_threshold=0.999)
        assert list(report.columns) == [
            "pathway", "direction", "n_in_set", "n_hits", "n_shifted",
            "n_background", "odds_ratio", "pval", "pval_adj", "hits",
        ]


class TestReferenceTable:
    """
    The database escape hatch. Name rules have a hard ceiling on untargeted
    panels, so a supplied mapping must take precedence over them.
    """

    def test_reference_dict_wins_over_rules(self):
        adata = mt.classify_compounds(
            _adata(["PC(34:1)", "Erysotrine"]),
            reference={"PC(34:1)": "Xenobiotic", "erysotrine": "Alkaloid"},
        )
        assert adata.var["chemical_class"].astype(str).tolist() == ["Xenobiotic", "Alkaloid"]

    def test_reference_dataframe(self):
        reference = pd.DataFrame({
            "compound": ["Slaframine", "Mahanimbine"],
            "class": ["Alkaloid", "Alkaloid"],
        })
        adata = mt.classify_compounds(_adata(["Slaframine", "Mahanimbine"]), reference=reference)
        assert set(adata.var["chemical_class"].astype(str)) == {"Alkaloid"}

    def test_rules_still_apply_to_uncovered_compounds(self):
        adata = mt.classify_compounds(
            _adata(["PC(34:1)", "L-Lysine"]), reference={"PC(34:1)": "Xenobiotic"}
        )
        assert adata.var["chemical_class"].astype(str).tolist() == ["Xenobiotic", "Amino acid"]

    def test_overrides_beat_reference(self):
        adata = mt.classify_compounds(
            _adata(["PC(34:1)"]),
            reference={"PC(34:1)": "Xenobiotic"},
            overrides={"PC(34:1)": "Sterol"},
        )
        assert adata.var["chemical_class"].astype(str).iloc[0] == "Sterol"

    def test_rejects_bad_reference(self):
        with pytest.raises(InvalidParameterError, match="two columns"):
            mt.classify_compounds(_adata(["PC(34:1)"]), reference=pd.DataFrame({"a": ["x"]}))
        with pytest.raises(InvalidParameterError, match="DataFrame or dict"):
            mt.classify_compounds(_adata(["PC(34:1)"]), reference=["PC(34:1)", "Sterol"])

    def test_stereodescriptors_are_stripped(self):
        """(+)-, (2R,3S)- and friends must not block a match."""
        adata = mt.classify_compounds(_adata(["(+)-Cholesterol", "(2R,3S)-Glucose"]))
        assert adata.var["chemical_class"].astype(str).tolist() == ["Sterol", "Carbohydrate"]
