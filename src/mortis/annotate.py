"""
MORTIS Annotation & Enrichment Module
=====================================
Chemical-class assignment from compound names, and enrichment testing at the
class and pathway level.

Why classification is done in layers
------------------------------------
Class-level results are often the most readable thing in a spatial metabolomics
paper, "phospholipids collapse in the inflamed region" says more than a list
of forty m/z values. But they are only as good as the classifier, and the usual
failure is quiet: a rule set built from a handful of lipid prefixes drops a
third of the compounds into "Other", and every class-level conclusion is then
computed on whatever happened to match.

The package already had a version of this problem. ``lipid_class_summary()``
classifies by lipid prefix alone, so on a typical polar-metabolite panel almost
everything lands in "Other". That function is kept for backward compatibility;
:func:`classify_compounds` is what to use instead.

Three layers, in order of precedence:

1. **Exact-name lookup** for compounds where a pattern would guess wrong.
   "Glutathione" contains no amino-acid marker; "Carnitine" looks like an amino
   acid but is not.
2. **Pattern rules** on shorthand and systematic names, ``PC(34:1)``,
   ``SM(d18:1/16:0)``, ``N1-Acetylspermidine``.
3. **Suffix and substructure heuristics** as a last resort, so a compound with
   a recognisable chemical ending is placed rather than discarded.

Anything still unmatched is labelled ``"Unclassified"`` rather than "Other",
the point being that it is a gap in the classifier, not a chemical category.
:func:`classification_report` tells you how large that gap is, and you should
look at it before quoting any class-level result.

Where name rules stop working
-----------------------------
On targeted or curated panels these rules place almost everything. On an
*untargeted* annotation list they do not, and it is worth knowing the size of
the effect before relying on them: measured against a real 2,231-compound
METASPACE-style panel, the built-in rules leave **about 55% unclassified**.

That is not a tuning problem. The unplaced remainder is dominated by plant
alkaloids and natural products whose names carry no usable stem,
"(+)-Erysotrine", "(-)-Slaframine", "(+)-Mahanimbine", together with fully
systematic IUPAC names. Nothing short of a database resolves those, which is
what the ``reference=`` argument of :func:`classify_compounds` is for: pass an
HMDB, LIPID MAPS or ClassyFire export and it is consulted before the rules.
Supplying a table covering half the unplaced compounds took the same panel from
55% to 28% unclassified.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .exceptions import InvalidParameterError, listing

__all__ = [
    "classify_compounds",
    "classification_report",
    "class_enrichment",
    "pathway_ora",
    "CHEMICAL_CLASSES",
]

#: The classes assigned by :func:`classify_compounds`, in reporting order.
CHEMICAL_CLASSES = (
    "Glycerophospholipid",
    "Sphingolipid",
    "Glycerolipid",
    "Sterol",
    "Fatty acyl",
    "Amino acid",
    "Peptide",
    "Nucleotide",
    "Polyamine",
    "Carbohydrate",
    "Organic acid",
    "Vitamin/cofactor",
    "Amine",
    "Alkaloid",
    "Xenobiotic",
    "Unclassified",
)

# --- Layer 1: exact names that patterns would misclassify -------------------
_EXACT: Dict[str, str] = {}


def _register(names: Sequence[str], klass: str) -> None:
    for name in names:
        _EXACT[name.lower()] = klass


_register([
    "glycine", "alanine", "serine", "threonine", "cysteine", "valine", "leucine",
    "isoleucine", "methionine", "proline", "phenylalanine", "tyrosine", "tryptophan",
    "aspartic acid", "glutamic acid", "asparagine", "glutamine", "lysine", "arginine",
    "histidine", "ornithine", "citrulline", "taurine", "sarcosine", "betaine",
    "hydroxyproline", "homocysteine", "homoserine", "cystathionine", "theanine",
], "Amino acid")

_register([
    "glutathione", "oxidized glutathione", "glutathione disulfide", "carnosine",
    "homocarnosine", "anserine", "ophthalmic acid",
], "Peptide")

_register([
    "putrescine", "cadaverine", "spermidine", "spermine", "agmatine",
    "n1-acetylspermidine", "n1-acetylspermine", "n8-acetylspermidine",
    "1-methylspermidine", "canavalmine", "thermospermine",
], "Polyamine")

_register([
    "carnitine", "acetylcarnitine", "choline", "phosphocholine", "glycerophosphocholine",
    "ethanolamine", "o-phosphoethanolamine", "phosphoethanolamine", "sphingosine",
    "sphinganine", "phytosphingosine",
], "Amine")

_register([
    "cholesterol", "cholesteryl ester", "cholic acid", "deoxycholic acid",
    "chenodeoxycholic acid", "taurocholic acid", "glycocholic acid", "lithocholic acid",
    "cortisol", "cortisone", "testosterone", "estradiol", "progesterone", "pregnenolone",
    "dehydroepiandrosterone", "lanosterol", "desmosterol", "7-dehydrocholesterol",
], "Sterol")

_register([
    "citric acid", "citrate", "lactic acid", "lactate", "pyruvic acid", "pyruvate",
    "succinic acid", "succinate", "fumaric acid", "fumarate", "malic acid", "malate",
    "oxaloacetic acid", "alpha-ketoglutaric acid", "2-oxoglutaric acid", "isocitric acid",
    "aconitic acid", "acetoacetic acid", "beta-hydroxybutyric acid", "urocanic acid",
], "Organic acid")

_register([
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "udp", "ump", "ctp", "cdp", "cmp",
    "nad", "nadh", "nadp", "nadph", "fad", "fadh2", "adenine", "adenosine", "guanine",
    "guanosine", "cytosine", "cytidine", "thymine", "thymidine", "uracil", "uridine",
    "inosine", "hypoxanthine", "xanthine", "coenzyme a", "acetyl-coa",
], "Nucleotide")

_register([
    "glucose", "fructose", "galactose", "mannose", "sucrose", "lactose", "maltose",
    "ribose", "glucose-6-phosphate", "fructose-6-phosphate", "glucosamine",
    "n-acetylglucosamine", "n-acetylneuraminic acid", "sorbitol", "mannitol", "inositol",
    "myo-inositol", "glycogen", "trehalose",
], "Carbohydrate")

_register([
    "ascorbic acid", "retinol", "retinoic acid", "thiamine", "riboflavin", "niacin",
    "nicotinamide", "nicotinic acid", "pantothenic acid", "pyridoxine", "pyridoxal",
    "biotin", "folic acid", "cobalamin", "tocopherol", "alpha-tocopherol",
    "phylloquinone", "menaquinone", "heme", "biliverdin", "bilirubin",
], "Vitamin/cofactor")

# --- Layer 2: pattern rules -------------------------------------------------
# Lipid shorthand is checked first: PC(34:1), SM(d18:1/16:0), TG(16:0_18:1_18:2).
_LIPID_SHORTHAND = re.compile(
    r"^(?P<head>[A-Za-z][A-Za-z0-9\-]{0,9})\s*[\(\[]\s*[dtoOP]?\d+:\d+", re.IGNORECASE
)

_LIPID_HEADS = {
    "pc": "Glycerophospholipid", "pe": "Glycerophospholipid", "ps": "Glycerophospholipid",
    "pi": "Glycerophospholipid", "pg": "Glycerophospholipid", "pa": "Glycerophospholipid",
    "lpc": "Glycerophospholipid", "lpe": "Glycerophospholipid", "lps": "Glycerophospholipid",
    "lpi": "Glycerophospholipid", "lpg": "Glycerophospholipid", "lpa": "Glycerophospholipid",
    "plasmenyl-pc": "Glycerophospholipid", "plasmenyl-pe": "Glycerophospholipid",
    "cl": "Glycerophospholipid", "bmp": "Glycerophospholipid",
    "sm": "Sphingolipid", "cer": "Sphingolipid", "hexcer": "Sphingolipid",
    "glccer": "Sphingolipid", "galcer": "Sphingolipid", "laccer": "Sphingolipid",
    "s1p": "Sphingolipid", "so": "Sphingolipid", "sph": "Sphingolipid",
    "gb3": "Sphingolipid", "gm1": "Sphingolipid", "gm3": "Sphingolipid",
    "tg": "Glycerolipid", "dg": "Glycerolipid", "mg": "Glycerolipid",
    "tag": "Glycerolipid", "dag": "Glycerolipid", "mag": "Glycerolipid",
    "ce": "Sterol", "st": "Sterol",
    "fa": "Fatty acyl", "car": "Fatty acyl", "acylcarnitine": "Fatty acyl",
    "coa": "Fatty acyl", "wax": "Fatty acyl",
}

_PATTERNS = (
    (re.compile(r"phosphatidylcholine|phosphatidylethanolamine|phosphatidylserine|"
                r"phosphatidylinositol|phosphatidylglycerol|phosphatidic acid|"
                r"lysophosphatid|cardiolipin|plasmalogen", re.I), "Glycerophospholipid"),
    (re.compile(r"sphingomyelin|ceramide|sphingosine|sphinganine|sphingoid|"
                r"cerebroside|ganglioside|sulfatide", re.I), "Sphingolipid"),
    (re.compile(r"triacylglycerol|diacylglycerol|monoacylglycerol|"
                r"triglyceride|diglyceride|monoglyceride", re.I), "Glycerolipid"),
    (re.compile(r"cholest|sterol|steroid|bile acid|cholan|androst|estr[ao]|pregn|lanost|"
                r"cucurbitacin|withanolid|ecdyson", re.I), "Sterol"),
    (re.compile(r"carnitine|acyl-coa|fatty acid|\benoic acid\b|anoic acid|"
                r"prostaglandin|leukotriene|thromboxane|eicosanoid", re.I), "Fatty acyl"),
    (re.compile(r"spermidine|spermine|putrescine|cadaverine|agmatine|polyamine", re.I), "Polyamine"),
    (re.compile(r"adenosine|guanosine|cytidine|thymidine|uridine|inosine|"
                r"nucleotide|nucleoside|purine|pyrimidine|\bdeoxy.*(sine|dine)\b", re.I), "Nucleotide"),
    (re.compile(r"glucos|fructos|galactos|mannos|sucros|lactos|maltos|ribos|"
                r"saccharide|hexose|pentose|glycan|inositol|sorbitol|mannitol", re.I), "Carbohydrate"),
    (re.compile(r"vitamin|tocopherol|retino|thiamin|riboflavin|niacin|folate|folic|"
                r"cobalamin|biotin|pantothen|pyridox|quinone|porphyrin|heme", re.I), "Vitamin/cofactor"),
    # Dipeptides are written as two concatenated residues: "Lysylalanine".
    (re.compile(r"(glycyl|alanyl|seryl|threonyl|cysteinyl|valyl|leucyl|isoleucyl|"
                r"methionyl|prolyl|phenylalanyl|tyrosyl|tryptophyl|aspartyl|glutamyl|"
                r"asparaginyl|glutaminyl|lysyl|arginyl|histidyl)", re.I), "Peptide"),
    (re.compile(r"^(l|d|dl)-\w+|amino acid|amino.*butyric acid", re.I), "Amino acid"),
    # Ring stems without the trailing "e": systematic names write
#: "piperidin-4-yl", not "piperidine".
    (re.compile(r"\bamine\b|piperidin|piperazin|pyrrolidin|morpholin|imidazol|"
                r"hydrazin|anilin|amino", re.I), "Amine"),
    (re.compile(r"alkaloid|quinolin|isoquinolin|indol|carbolin|tropan|"
                r"berberin|morphin|codein|nicotin|caffein|xanthin|pyrrol|piperin|"
                r"vinca|strychn|atropin|quinin|ergot|harman", re.I), "Alkaloid"),
    (re.compile(r"acid$|oate$|\bcarboxyl", re.I), "Organic acid"),
)

#: Stereodescriptors and locant prefixes, stripped before matching:
#: "(+)-Erysotrine", "(2R,3S)-...", "(1E)-1-Phenyltriaz-1-ene".
_LEADING_DESCRIPTOR = re.compile(
    r"^\s*(\((?:[+\-±]|[0-9]*[a-z]?[RSEZ](?:,\s*[0-9]*[a-z]?[RSEZ])*|"
    r"[0-9]+[a-z]*(?:alpha|beta)?(?:,\s*[0-9]+[a-z]*(?:alpha|beta)?)*)\)|"
    r"[+\-±])[-\s]*",
    re.IGNORECASE,
)

# --- Layer 3: suffix heuristics ---------------------------------------------
_SUFFIX = (
    ("carnitine", "Fatty acyl"),
    ("choline", "Amine"),
    ("ethanolamine", "Amine"),
    ("phosphate", "Organic acid"),
    ("sulfate", "Organic acid"),
    ("nitrile", "Xenobiotic"),
    ("urea", "Xenobiotic"),
    ("carbamate", "Xenobiotic"),
    ("ol", "Amine"),
)


def _classify_one(name: str) -> str:
    # peel off leading stereodescriptors; they anchor at the start and
    # block every match
    cleaned = name.strip()
    for _ in range(4):
        stripped_once = _LEADING_DESCRIPTOR.sub("", cleaned, count=1)
        if stripped_once == cleaned:
            break
        cleaned = stripped_once
    lowered = cleaned.lower()

    if lowered in _EXACT:
        return _EXACT[lowered]

    # Strip a leading L-/D-/DL- and common N-acyl prefixes, then retry exactly.
    stripped = re.sub(r"^(l|d|dl)[-\s]", "", lowered)
    stripped = re.sub(r"^(n[0-9]?|o|s)[-\s]?(acetyl|methyl|formyl)[-\s]?", "", stripped)
    if stripped in _EXACT:
        return _EXACT[stripped]

    match = _LIPID_SHORTHAND.match(cleaned)
    if match:
        head = match.group("head").lower()
        if head in _LIPID_HEADS:
            return _LIPID_HEADS[head]

    for pattern, klass in _PATTERNS:
        if pattern.search(lowered):
            return klass

    for suffix, klass in _SUFFIX:
        if lowered.endswith(suffix):
            return klass

    return "Unclassified"


def classify_compounds(
    adata: ad.AnnData,
    key_added: str = "chemical_class",
    reference: Optional[Union[pd.DataFrame, Dict[str, str]]] = None,
    overrides: Optional[Dict[str, str]] = None,
    copy: bool = False,
) -> ad.AnnData:
    """
    Assign a chemical class to every compound from its name.

    Parameters
    ----------
    adata : anndata.AnnData
        Data whose ``var_names`` are compound names. Purely m/z ``var_names``
        cannot be classified, annotate them first.
    key_added : str
        Column created in ``adata.var``. Default ``'chemical_class'``.
    reference : pandas.DataFrame or dict, optional
        A name-to-class mapping consulted **before** the built-in rules, the
        way to bring a real database in. Either a dict, or a DataFrame with a
        compound-name column and a class column (the first two columns are
        used). Export from HMDB, LIPID MAPS or ClassyFire and pass it here.

        This matters more than it looks. Name rules do well on targeted and
        curated panels but hit a hard ceiling on untargeted annotation lists:
        on a real 2,231-compound METASPACE-style panel the built-in rules leave
        about 55% unclassified, and the remainder are largely plant alkaloids
        and natural products whose names carry no usable stem
        ("(+)-Erysotrine", "(-)-Slaframine"). No amount of pattern work fixes
        that. Those compounds need a database.
    overrides : dict, optional
        ``{compound_name: class}`` applied last and unconditionally. Use this
        for facility-specific naming the built-in rules do not know, and keep
        it in your analysis script so the choice is visible in review.
    copy : bool
        Return a copy instead of annotating in place.

    Returns
    -------
    anndata.AnnData
        With ``adata.var[key_added]`` set. Check the unclassified fraction with
        :func:`classification_report` before quoting class-level results.
    """
    if copy:
        adata = adata.copy()

    reference_map: Dict[str, str] = {}
    if reference is not None:
        if isinstance(reference, pd.DataFrame):
            if reference.shape[1] < 2:
                raise InvalidParameterError(
                    "reference DataFrame needs at least two columns "
                    "(compound name, chemical class)."
                )
            name_col, class_col = reference.columns[:2]
            reference_map = {
                str(k).strip().lower(): str(v)
                for k, v in zip(reference[name_col], reference[class_col])
                if pd.notna(v)
            }
        elif isinstance(reference, dict):
            reference_map = {str(k).strip().lower(): str(v) for k, v in reference.items()}
        else:
            raise InvalidParameterError(
                f"reference must be a DataFrame or dict, got {type(reference).__name__}."
            )

    classes = [
        reference_map.get(str(name).strip().lower()) or _classify_one(str(name))
        for name in adata.var_names
    ]

    if overrides:
        unknown = sorted(set(overrides.values()) - set(CHEMICAL_CLASSES))
        if unknown:
            warnings.warn(
                f"[MORTIS] overrides use class name(s) outside CHEMICAL_CLASSES: {unknown}. "
                "They will be kept, but class-ordered figures may not include them.",
                stacklevel=2,
            )
        lookup = {k.strip().lower(): v for k, v in overrides.items()}
        classes = [
            lookup.get(str(name).strip().lower(), klass)
            for name, klass in zip(adata.var_names, classes)
        ]

    adata.var[key_added] = pd.Categorical(
        classes,
        categories=[c for c in CHEMICAL_CLASSES if c in set(classes)]
        + sorted(set(classes) - set(CHEMICAL_CLASSES)),
    )

    n_unclassified = classes.count("Unclassified")
    fraction = n_unclassified / max(len(classes), 1)
    print(
        f"[MORTIS] Classified {len(classes) - n_unclassified}/{len(classes)} compounds "
        f"into {len(set(classes) - {'Unclassified'})} classes "
        f"-> adata.var['{key_added}']"
    )
    if fraction > 0.25:
        warnings.warn(
            f"[MORTIS] {fraction:.0%} of compounds are Unclassified. Class-level results "
            "computed on the remainder may not represent the panel, inspect "
            "mortis.classification_report(adata) and consider passing overrides=.",
            stacklevel=2,
        )
    return adata


def classification_report(
    adata: ad.AnnData, key: str = "chemical_class", show_unclassified: int = 20
) -> pd.DataFrame:
    """
    Per-class counts, with the unclassified compounds listed by name.

    The listing is the useful half: it turns "23% unclassified" into a concrete
    set of names you can either add to ``overrides`` or decide to ignore.
    """
    if key not in adata.var.columns:
        raise InvalidParameterError(
            f"There is no {key!r} column in adata.var to group compounds by. "
            f"Run mortis.classify_compounds(adata) to add it. Columns present: "
            f"{listing(adata.var.columns)}."
        )
    counts = adata.var[key].value_counts()
    report = (
        counts.rename_axis("chemical_class").reset_index(name="n_compounds")
        .assign(fraction=lambda d: d["n_compounds"] / adata.n_vars)
    )

    unclassified = adata.var_names[adata.var[key].astype(str) == "Unclassified"].tolist()
    if unclassified:
        shown = unclassified[:show_unclassified]
        print(
            f"[MORTIS] {len(unclassified)} unclassified compound(s); first {len(shown)}:"
        )
        for name in shown:
            print(f"[MORTIS]     {name}")
    return report


def class_enrichment(
    result: pd.DataFrame,
    adata: ad.AnnData,
    key: str = "chemical_class",
    effect_col: str = "delta",
    min_size: int = 3,
) -> pd.DataFrame:
    """
    Which chemical classes shift, rather than which individual compounds.

    Compares each class's effect sizes against all other compounds with a
    Mann-Whitney test, so the question is "do this class's members shift more
    than the rest of the panel?". Classes are the level at which small cohorts
    have something to say, a class of 30 phospholipids moving together is
    evidence that no individual compound at n = 6 could carry.

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`mortis.differential_abundance` or
        :func:`mortis.differential_spatial_organization`.
    adata : anndata.AnnData
        Object carrying ``adata.var[key]`` from :func:`classify_compounds`.
    key : str
        Class column in ``adata.var``.
    effect_col : str
        Effect-size column in ``result``.
    min_size : int
        Classes with fewer members are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per class: ``n_compounds``, ``median_delta``,
        ``mean_abs_delta``, ``n_up``, ``n_down``, ``pval``, ``pval_adj``,
        sorted by ``|median_delta|``.
    """
    if key not in adata.var.columns:
        raise InvalidParameterError(
            f"There is no {key!r} column in adata.var to group compounds by. "
            f"Run mortis.classify_compounds(adata) to add it. Columns present: "
            f"{listing(adata.var.columns)}."
        )
    for column in ("metabolite", effect_col):
        if column not in result.columns:
            raise InvalidParameterError(
                f"The result table has no {column!r} column. This function expects "
                f"the output of mortis.differential_abundance() or "
                f"mortis.compare_groups(); pass effect_col= if your effect size "
                f"sits under a different name. Columns present: "
                f"{listing(result.columns)}."
            )

    classes = adata.var[key].astype(str)
    lookup = dict(zip(adata.var_names.astype(str), classes))
    merged = result.assign(
        **{key: result["metabolite"].astype(str).map(lookup)}
    ).dropna(subset=[key])

    if merged.empty:
        raise InvalidParameterError(
            "No metabolite in 'result' matched adata.var_names. Were they computed "
            "from the same object?"
        )

    rows = []
    for klass, group in merged.groupby(key, observed=True):
        if len(group) < min_size or klass == "Unclassified":
            continue
        inside = group[effect_col].to_numpy(dtype=float)
        outside = merged.loc[merged[key] != klass, effect_col].to_numpy(dtype=float)
        if len(outside) < min_size:
            continue
        try:
            _, pval = stats.mannwhitneyu(inside, outside, alternative="two-sided")
        except ValueError:
            pval = 1.0
        rows.append({
            "chemical_class": klass,
            "n_compounds": len(group),
            "median_delta": float(np.median(inside)),
            "mean_abs_delta": float(np.mean(np.abs(inside))),
            "n_up": int((inside > 0).sum()),
            "n_down": int((inside < 0).sum()),
            "pval": float(pval),
        })

    if not rows:
        return pd.DataFrame(
            columns=["chemical_class", "n_compounds", "median_delta", "mean_abs_delta",
                     "n_up", "n_down", "pval", "pval_adj"]
        )

    report = pd.DataFrame(rows)
    report["pval_adj"] = multipletests(report["pval"].to_numpy(), method="fdr_bh")[1]
    report = report.reindex(
        report["median_delta"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    print(
        f"[MORTIS] Class enrichment: {len(report)} class(es) tested, "
        f"{int((report['pval_adj'] < 0.05).sum())} at FDR < 0.05."
    )
    return report


def pathway_ora(
    result: pd.DataFrame,
    metabolite_sets: Dict[str, List[str]],
    effect_col: str = "delta",
    delta_threshold: float = 0.474,
    fdr_threshold: Optional[float] = None,
    direction: bool = True,
    min_size: int = 3,
) -> pd.DataFrame:
    """
    Over-representation of metabolite sets among the shifted compounds.

    Complements :func:`mortis.metabolite_set_enrichment`, which is rank-based
    over the whole list. This one asks a simpler question, of the compounds
    that passed a threshold, is any pathway over-represented, using Fisher's
    exact test against the background of every compound measured.

    The background matters and is easy to get wrong: it is the compounds in
    ``result``, not everything in the pathway database. A panel that only
    measures polar metabolites will look "enriched" for polar pathways against
    a whole-metabolome background, purely because of what the instrument saw.

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`mortis.differential_abundance`.
    metabolite_sets : dict
        ``{pathway_name: [compound...]}``. Matching is case-insensitive.
    effect_col : str
        Effect-size column. Default ``"delta"``.
    delta_threshold : float
        ``|effect|`` at or above which a compound counts as shifted.
    fdr_threshold : float, optional
        Also require this adjusted p-value. Default ``None`` (effect size
        only), because with small cohorts requiring both often leaves nothing
        to test.
    direction : bool
        Test up-shifted and down-shifted compounds separately as well as
        together. A pathway with half its members up and half down is not
        coherently dysregulated, and a combined test would hide that.
    min_size : int
        Pathways with fewer measured members are skipped.

    Returns
    -------
    pandas.DataFrame
        ``pathway``, ``direction`` (``"any"``/``"up"``/``"down"``),
        ``n_in_set``, ``n_hits``, ``n_shifted``, ``n_background``,
        ``odds_ratio``, ``pval``, ``pval_adj``, ``hits``.
    """
    for column in ("metabolite", effect_col):
        if column not in result.columns:
            raise InvalidParameterError(
                f"The result table has no {column!r} column. This function expects "
                f"the output of mortis.differential_abundance() or "
                f"mortis.compare_groups(); pass effect_col= if your effect size "
                f"sits under a different name. Columns present: "
                f"{listing(result.columns)}."
            )
    if not metabolite_sets:
        raise InvalidParameterError(
            "metabolite_sets is empty, so every test would have zero members. "
            "Build it from mortis.annotate_pathways(), or pass your own "
            "{set_name: [compound...]} mapping."
        )
    if fdr_threshold is not None and "pval_adj" not in result.columns:
        raise InvalidParameterError(
            "fdr_threshold was given but 'result' has no 'pval_adj' column."
        )

    names = result["metabolite"].astype(str)
    lowered = names.str.strip().str.lower()
    effects = result[effect_col].to_numpy(dtype=float)

    shifted = np.abs(effects) >= delta_threshold
    if fdr_threshold is not None:
        shifted &= result["pval_adj"].to_numpy(dtype=float) < fdr_threshold

    directions = {"any": shifted}
    if direction:
        directions["up"] = shifted & (effects > 0)
        directions["down"] = shifted & (effects < 0)

    background = set(lowered)
    rows = []
    for pathway, members in metabolite_sets.items():
        member_set = {str(m).strip().lower() for m in members}
        measured = member_set & background
        if len(measured) < min_size:
            continue
        in_set = lowered.isin(measured).to_numpy()

        for label, mask in directions.items():
            n_hits = int((in_set & mask).sum())
            if n_hits == 0:
                continue
            table = [
                [n_hits, int((in_set & ~mask).sum())],
                [int((~in_set & mask).sum()), int((~in_set & ~mask).sum())],
            ]
            odds, pval = stats.fisher_exact(table, alternative="greater")
            rows.append({
                "pathway": pathway,
                "direction": label,
                "n_in_set": len(measured),
                "n_hits": n_hits,
                "n_shifted": int(mask.sum()),
                "n_background": len(background),
                "odds_ratio": float(odds),
                "pval": float(pval),
                "hits": ", ".join(names[in_set & mask].tolist()[:10]),
            })

    if not rows:
        return pd.DataFrame(
            columns=["pathway", "direction", "n_in_set", "n_hits", "n_shifted",
                     "n_background", "odds_ratio", "pval", "pval_adj", "hits"]
        )

    report = pd.DataFrame(rows)
    # correct within direction; pooling would penalise every pathway
    report["pval_adj"] = np.nan
    for label in report["direction"].unique():
        mask = report["direction"] == label
        report.loc[mask, "pval_adj"] = multipletests(
            report.loc[mask, "pval"].to_numpy(), method="fdr_bh"
        )[1]

    report = report.sort_values(
        ["direction", "pval", "pathway"], kind="stable"
    ).reset_index(drop=True)
    print(
        f"[MORTIS] Pathway ORA: {report['pathway'].nunique()} pathway(s) tested against "
        f"{len(background)} measured compounds, "
        f"{int((report['pval_adj'] < 0.05).sum())} enriched at FDR < 0.05."
    )
    return report

