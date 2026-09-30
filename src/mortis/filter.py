"""
MORTIS Filter Module
====================
Metabolite filtering on annotation quality and on whether a compound is a
known drug.

Two independent strategies:

1. Annotation score filtering. Keep only metabolites whose identification
   confidence score, from the instrument's feature table, meets a minimum.
   Scores live in ``adata.var['score']`` and run from 0 (no confidence) to
   2 (library match).

2. Drug filtering. Remove or flag metabolites that are known drugs, using the
   vocabulary bundled with the package.

Public API
----------
filter_by_score(adata, min_score, score_col, copy)
    Keep metabolites whose annotation score meets a threshold.

filter_drugs(adata, db_path, remove_all, drug_names, copy)
    Remove metabolites that match the drug vocabulary.

list_drug_matches(adata, db_path)
    Report which metabolites match, without removing anything.
"""

from __future__ import annotations

import sqlite3
import warnings
from importlib.resources import as_file, files
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .exceptions import FileFormatError, InvalidParameterError

_VOCABULARY_FILE = "drug_names.db"

# Sold as drugs but made by any tissue. filter_drugs() names them first.
_ALSO_ENDOGENOUS = frozenset({
    "alanine", "arginine", "asparagine", "aspartate", "betaine", "biotin",
    "carnitine", "choline", "cholesterol", "citrulline", "creatine", "cysteine",
    "folate", "fructose", "galactose", "glucose", "glutamate", "glutamine",
    "glycine", "histidine", "inositol", "isoleucine", "lactate", "leucine",
    "lysine", "malate", "mannitol", "methionine", "niacin", "ornithine",
    "phenylalanine", "proline", "pyruvate", "riboflavin", "serine", "sorbitol",
    "spermidine", "spermine", "taurine", "threonine", "thiamine", "tryptophan",
    "tyrosine", "urea", "valine",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_vocabulary_path() -> str:
    """
    Find the drug vocabulary shipped inside the installed package.

    Falls back to a bare filename resolved against the working directory, for
    editable installs where package data can sit somewhere else, or if the
    bundled copy is missing.
    """
    try:
        with as_file(files("mortis").joinpath("data", _VOCABULARY_FILE)) as bundled:
            if bundled.is_file():
                return str(bundled)
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return _VOCABULARY_FILE


def _load_drug_names(db_path: str) -> set:
    """Every drug name and synonym in the vocabulary, lower-cased."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Drug vocabulary not found: '{db_path}'. The database ships with "
            "the package, so if you did not pass db_path yourself this install "
            "is missing its data files. Reinstall with 'pip install "
            "--force-reinstall mortis-spatial', or point db_path at your own "
            "SQLite file with a drug_compounds(name) table."
        )
    try:
        conn = sqlite3.connect(str(path))
        names = {row[0].lower().strip() for row in conn.execute("SELECT name FROM drug_compounds")}
        names |= {row[0].lower().strip() for row in conn.execute("SELECT synonym FROM drug_synonyms")}
        conn.close()
    except sqlite3.Error as exc:
        raise FileFormatError(
            f"Could not read the drug vocabulary at '{db_path}'. It should be a "
            "SQLite file with a drug_compounds(name) and a drug_synonyms(synonym) "
            f"table.\nOriginal error: {exc}"
        ) from exc
    return names


def _match_metabolites_to_drugs(var_names: pd.Index, drug_names: set) -> np.ndarray:
    """Boolean mask, True where a metabolite name matches the vocabulary."""
    return np.array([name.lower().strip() in drug_names for name in var_names], dtype=bool)


def _warn_about_endogenous(removed: List[str]) -> None:
    """Name any removed compound the body makes for itself."""
    overlap = sorted(n for n in removed if n.lower().strip() in _ALSO_ENDOGENOUS)
    if overlap:
        shown = ", ".join(overlap[:8]) + ("..." if len(overlap) > 8 else "")
        warnings.warn(
            f"{len(overlap)} of the removed compounds are made by the body as "
            f"well as sold as drugs: {shown}. They are in the vocabulary because "
            "they have drug identifiers, not because they are xenobiotic. If you "
            "are studying tissue metabolism, keep them with remove_all=False.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_by_score(
    adata: ad.AnnData,
    min_score: float = 0.3,
    score_col: str = "score",
    copy: bool = False,
) -> ad.AnnData:
    """
    Keep only metabolites whose annotation confidence meets a threshold.

    Annotation scores come from the MSI software (SCiLS, METASPACE and
    similar) and live in ``adata.var[score_col]``, running from 0 (no match)
    to 2 (library match).

    Parameters
    ----------
    adata : anndata.AnnData
        Input data. Must have ``score_col`` in ``adata.var``.
    min_score : float, optional
        Lowest score to keep. Default 0.3. Useful values are 0.0 to keep
        everything, 0.3 for moderate confidence, 0.5 for high, and 0.8 for
        library matches only.
    score_col : str, optional
        Column in ``adata.var`` holding the scores. Default ``'score'``.
    copy : bool, optional
        Return a copy instead of filtering in place. Default False.

    Returns
    -------
    anndata.AnnData
        Data with low-confidence metabolites dropped.
        ``adata.uns['filter_score']`` records what was used.

    Raises
    ------
    InvalidParameterError
        If ``score_col`` is missing, or ``min_score`` is outside [0, 2].

    Examples
    --------
    >>> adata = mt.filter_by_score(adata, min_score=0.5)
    >>> adata = mt.filter_by_score(adata, min_score=0.3, score_col='fdr_score')
    """
    if score_col not in adata.var.columns:
        raise InvalidParameterError(
            f"Score column '{score_col}' not found in adata.var. "
            f"Available columns: {adata.var.columns.tolist()}. "
            "Either your file was loaded without annotation scores, or the "
            "column goes by another name: pass it as score_col."
        )
    if not (0.0 <= min_score <= 2.0):
        raise InvalidParameterError(
            f"min_score must be between 0.0 and 2.0, got {min_score}. "
            "Typical values: 0.3 (moderate), 0.5 (high), 0.8 (library match)."
        )

    scores = adata.var[score_col].to_numpy(dtype=float)
    keep = scores >= min_score
    n_before = adata.n_vars
    n_kept = int(keep.sum())

    if copy:
        adata = adata.copy()

    adata = adata[:, keep].copy()
    adata.uns["filter_score"] = {
        "min_score": min_score,
        "score_col": score_col,
        "n_before": n_before,
        "n_kept": n_kept,
        "n_removed": n_before - n_kept,
    }
    print(f"[MORTIS] Score filter (>={min_score}): kept {n_kept} / {n_before} metabolites.")
    return adata


def filter_drugs(
    adata: ad.AnnData,
    db_path: Optional[str] = None,
    remove_all: bool = True,
    drug_names: Optional[List[str]] = None,
    copy: bool = False,
) -> ad.AnnData:
    """
    Remove metabolites that are known drugs.

    The bundled vocabulary holds about 20,000 drug names and 44,000 synonyms,
    built from Wikidata: an item counts as a drug when Wikidata gives it a
    DrugBank or ATC identifier, or files it under medication or pharmaceutical
    product. Wikidata is CC0, so the database ships inside the wheel. Rebuild
    it with ``tools/build_drug_vocabulary.py``.

    Matching is on the compound name, case-insensitively, against names and
    synonyms alike.

    A warning worth reading: plenty of ordinary metabolites are also sold as
    drugs, so taurine, glycine, carnitine and cholesterol are all in the
    vocabulary. With ``remove_all=True`` they go too. Call
    :func:`list_drug_matches` first and look at what you are about to lose.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data.
    db_path : str or None, optional
        A SQLite file with ``drug_compounds(name)`` and
        ``drug_synonyms(synonym)`` tables. Default None, which uses the
        bundled copy. Pass a path to use your own, for instance a DrugBank
        export if you hold a licence for one.
    remove_all : bool, optional
        True removes every match. False removes only what ``drug_names``
        lists. Default True.
    drug_names : list of str or None, optional
        Names to remove when ``remove_all=False``. Matched case-insensitively.
    copy : bool, optional
        Return a copy instead of filtering in place. Default False.

    Returns
    -------
    anndata.AnnData
        Data with drug metabolites dropped. ``adata.uns['filter_drugs']``
        records which ones went.

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist.
    FileFormatError
        If the vocabulary cannot be read.
    InvalidParameterError
        If ``remove_all=False`` and ``drug_names`` is empty.

    Examples
    --------
    >>> adata = mt.filter_drugs(adata)
    >>> adata = mt.filter_drugs(
    ...     adata, remove_all=False, drug_names=['Aspirin', 'Metformin']
    ... )
    """
    if not remove_all and not drug_names:
        raise InvalidParameterError(
            "remove_all=False means you choose what to drop, but drug_names was "
            "empty. Pass the names, for example "
            "drug_names=['Aspirin', 'Metformin'], or set remove_all=True to "
            "drop every match in the vocabulary."
        )
    if db_path is None:
        db_path = _default_vocabulary_path()

    if copy:
        adata = adata.copy()

    if remove_all:
        targets = _load_drug_names(db_path)
    else:
        targets = {n.lower().strip() for n in drug_names}
        unknown = targets - _load_drug_names(db_path)
        if unknown:
            shown = ", ".join(sorted(unknown)[:5]) + ("..." if len(unknown) > 5 else "")
            warnings.warn(
                f"{len(unknown)} of the names you passed are not in the drug "
                f"vocabulary: {shown}. They are still removed if they appear in "
                "the data, so check the spelling if nothing goes.",
                UserWarning,
                stacklevel=2,
            )

    drug_mask = _match_metabolites_to_drugs(adata.var_names, targets)
    removed_names = adata.var_names[drug_mask].tolist()
    n_removed = int(drug_mask.sum())
    n_before = adata.n_vars

    if remove_all:
        _warn_about_endogenous(removed_names)

    adata = adata[:, ~drug_mask].copy()
    adata.uns["filter_drugs"] = {
        "db_path": db_path,
        "remove_all": remove_all,
        "n_before": n_before,
        "n_removed": n_removed,
        "n_kept": n_before - n_removed,
        "removed_metabolites": removed_names,
    }
    print(f"[MORTIS] Drug filter: removed {n_removed} / {n_before} metabolites.")
    return adata


def list_drug_matches(
    adata: ad.AnnData,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Report which metabolites match the drug vocabulary, removing nothing.

    Run this before :func:`filter_drugs` to see what it would take. The
    ``endogenous`` column flags compounds the body makes anyway, which are
    usually the ones you want to keep.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data.
    db_path : str or None, optional
        Path to a vocabulary file. Default None, which uses the bundled copy.

    Returns
    -------
    pandas.DataFrame
        One row per match, with columns ``metabolite``, ``is_drug`` and
        ``endogenous``.

    Examples
    --------
    >>> matches = mt.list_drug_matches(adata)
    >>> matches[~matches.endogenous]
    """
    if db_path is None:
        db_path = _default_vocabulary_path()
    drug_names = _load_drug_names(db_path)
    mask = _match_metabolites_to_drugs(adata.var_names, drug_names)
    matched = adata.var_names[mask].tolist()
    result = pd.DataFrame({
        "metabolite": matched,
        "is_drug": True,
        "endogenous": [n.lower().strip() in _ALSO_ENDOGENOUS for n in matched],
    })
    print(f"[MORTIS] {len(matched)} / {adata.n_vars} metabolites match the drug vocabulary.")
    return result
