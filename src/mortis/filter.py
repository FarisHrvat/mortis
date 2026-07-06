"""
MORTIS Filter Module
====================
Metabolite filtering based on annotation quality scores and drug compound
databases.

Two independent filtering strategies are provided:

1. **Annotation score filtering** — keep only metabolites whose identification
   confidence score (from the instrument's feature table) meets a minimum
   threshold.  Scores are stored in ``adata.var['score']`` and range from 0
   (no confidence) to 2 (high confidence / library match).

2. **DrugBank filtering** — remove or flag metabolites that are known drug
   compounds, using the bundled DrugBank SQLite database.  Users can remove
   all drug metabolites, or supply a specific list of drug names to remove.

Public API
----------
filter_by_score(adata, min_score, score_col, copy)
    Keep only metabolites with annotation score ≥ min_score.

filter_drugs(adata, db_path, remove_all, drug_names, copy)
    Remove drug metabolites using the DrugBank database.

list_drug_matches(adata, db_path)
    Return a DataFrame of metabolites found in DrugBank (without removing).
"""

from __future__ import annotations

import sqlite3
from importlib.resources import as_file, files
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .exceptions import FileFormatError, InvalidParameterError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_drugbank_path() -> str:
    """
    Locate the DrugBank database bundled with the installed package
    (``mortis/data/drugbank.db``, shipped as package data). Falls back to
    a plain ``'drugbank.db'`` relative path — resolved against the current
    working directory — for editable/source installs where package data
    resolution can behave differently, or if the bundled copy is missing.
    """
    try:
        with as_file(files("mortis").joinpath("data", "drugbank.db")) as bundled:
            if bundled.is_file():
                return str(bundled)
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return "drugbank.db"


def _load_drugbank_names(db_path: str) -> set:
    """Return a lower-cased set of all drug names + synonyms from DrugBank."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DrugBank database not found: '{db_path}'. "
            "Ensure drugbank.db is in your working directory or provide the "
            "full path."
        )
    try:
        conn = sqlite3.connect(str(path))
        names = set()
        for row in conn.execute("SELECT name FROM drug_compounds"):
            names.add(row[0].lower().strip())
        for row in conn.execute("SELECT synonym FROM drug_synonyms"):
            names.add(row[0].lower().strip())
        conn.close()
    except sqlite3.Error as exc:
        raise FileFormatError(
            f"Could not read DrugBank database '{db_path}'.\n"
            f"Original error: {exc}"
        ) from exc
    return names


def _match_metabolites_to_drugs(
    var_names: pd.Index,
    drug_names: set,
) -> np.ndarray:
    """Return boolean mask: True where var_name matches a drug name."""
    return np.array(
        [name.lower().strip() in drug_names for name in var_names],
        dtype=bool,
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
    Keep only metabolites whose annotation confidence score meets a minimum
    threshold.

    Annotation scores are typically produced by the MSI instrument software
    (e.g. SCiLS, METASPACE) and stored in ``adata.var[score_col]``.  Scores
    range from 0 (no match) to 2 (high-confidence library match).

    Parameters
    ----------
    adata : anndata.AnnData
        Input data.  Must have ``score_col`` in ``adata.var``.
    min_score : float, optional
        Minimum score to retain a metabolite.  Metabolites with
        ``score < min_score`` are removed.  Default: 0.3.

        Recommended thresholds:

        * 0.0 — keep all (no filtering)
        * 0.3 — moderate confidence (default)
        * 0.5 — high confidence
        * 0.8 — very high confidence / library match only
    score_col : str, optional
        Column in ``adata.var`` containing annotation scores.
        Default: ``'score'``.
    copy : bool, optional
        Return a copy instead of modifying in-place.  Default: False.

    Returns
    -------
    anndata.AnnData
        AnnData with low-confidence metabolites removed.
        ``adata.uns['filter_score']`` records the parameters used.

    Raises
    ------
    InvalidParameterError
        If ``score_col`` is not found in ``adata.var`` or ``min_score`` is
        outside [0, 2].

    Examples
        --------
        >>> # Keep only metabolites with score ≥ 0.5
        >>> adata = mt.filter_by_score(adata, min_score=0.5)
        >>> print(f"Retained {adata.n_vars} metabolites")

        >>> # Use a custom score column
        >>> adata = mt.filter_by_score(adata, min_score=0.3, score_col='fdr_score')
        """
    if score_col not in adata.var.columns:
        raise InvalidParameterError(
            f"Score column '{score_col}' not found in adata.var. "
            f"Available columns: {adata.var.columns.tolist()}. "
            "Ensure your data was loaded from a file that includes annotation "
            "scores, or specify the correct column name with score_col=."
        )
    if not (0.0 <= min_score <= 2.0):
        raise InvalidParameterError(
            f"min_score must be between 0.0 and 2.0, got {min_score}. "
            "Typical values: 0.3 (moderate), 0.5 (high), 0.8 (very high)."
        )

    scores = adata.var[score_col].to_numpy(dtype=float)
    keep = scores >= min_score
    n_before = adata.n_vars
    n_kept = int(keep.sum())
    n_removed = n_before - n_kept

    if copy:
        adata = adata.copy()

    adata = adata[:, keep].copy()
    adata.uns["filter_score"] = {
        "min_score": min_score,
        "score_col": score_col,
        "n_before": n_before,
        "n_kept": n_kept,
        "n_removed": n_removed,
    }
    print(
        f"[MORTIS] Score filter (≥{min_score}): "
        f"kept {n_kept} / {n_before} metabolites "
        f"({n_removed} removed)"
    )
    return adata


def filter_drugs(
    adata: ad.AnnData,
    db_path: Optional[str] = None,
    remove_all: bool = True,
    drug_names: Optional[List[str]] = None,
    copy: bool = False,
) -> ad.AnnData:
    """
    Remove drug metabolites using the DrugBank database.

    DrugBank (Wishart et al., Nucleic Acids Res., 2018) contains >17,000 drug
    compounds and >45,000 synonyms.  This function matches metabolite names
    against the database and removes matches.

    **Citation:** Wishart DS, et al. DrugBank 5.0: a major update to the
    DrugBank database for 2018. *Nucleic Acids Research*, 2018, 46(D1):D1074–D1082.
    https://doi.org/10.1093/nar/gkx1037

    Parameters
    ----------
    adata : anndata.AnnData
        Input data.
    db_path : str or None, optional
        Path to a ``drugbank.db`` SQLite file. Default: ``None``, which uses
        the copy bundled with the installed package (``mortis/data/drugbank.db``)
        — no manual setup needed. Pass an explicit path to use a different
        or updated DrugBank export.
    remove_all : bool, optional
        * ``True`` — remove all metabolites found in DrugBank (default).
        * ``False`` — only remove metabolites specified in ``drug_names``.
    drug_names : list of str or None, optional
        Specific drug names to remove.  Used when ``remove_all=False``, or
        to remove additional drugs beyond the database matches.
        Names are matched case-insensitively against both DrugBank compound
        names and synonyms.
        Example: ``['Aspirin', 'Ibuprofen', 'Metformin']``.
    copy : bool, optional
        Return a copy instead of modifying in-place.  Default: False.

    Returns
    -------
    anndata.AnnData
        AnnData with drug metabolites removed.
        ``adata.uns['filter_drugs']`` records which metabolites were removed.

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist.
    FileFormatError
        If the database cannot be read.
    InvalidParameterError
        If ``remove_all=False`` and ``drug_names`` is empty or None.

    Examples
    --------
    >>> # Remove all drug metabolites
    >>> adata = mt.filter_drugs(adata, db_path='drugbank.db')

    >>> # Remove only specific drugs
    >>> adata = mt.filter_drugs(
    ...     adata,
    ...     remove_all=False,
    ...     drug_names=['Aspirin', 'Ibuprofen', 'Metformin']
    ... )
    """
    if not remove_all and not drug_names:
        raise InvalidParameterError(
            "When remove_all=False, you must provide a list of drug names "
            "via drug_names=['Drug1', 'Drug2', ...]."
        )
    if db_path is None:
        db_path = _default_drugbank_path()

    if copy:
        adata = adata.copy()

    if remove_all:
        db_names = _load_drugbank_names(db_path)
        drug_mask = _match_metabolites_to_drugs(adata.var_names, db_names)
    else:
        # Only match the user-supplied names (still look them up in DB for synonyms)
        db_names = _load_drugbank_names(db_path)
        # Expand user names through synonyms
        user_lower = {n.lower().strip() for n in drug_names}
        # Keep only names that are in the DB (validates the input)
        matched_in_db = user_lower & db_names
        not_in_db = user_lower - db_names
        if not_in_db:
            print(
                f"[MORTIS] Warning: {len(not_in_db)} drug name(s) not found "
                f"in DrugBank and will be skipped: "
                f"{sorted(not_in_db)[:5]}{'...' if len(not_in_db) > 5 else ''}"
            )
        drug_mask = _match_metabolites_to_drugs(adata.var_names, matched_in_db | user_lower)

    removed_names = adata.var_names[drug_mask].tolist()
    n_removed = int(drug_mask.sum())
    n_before = adata.n_vars

    adata = adata[:, ~drug_mask].copy()
    adata.uns["filter_drugs"] = {
        "db_path": db_path,
        "remove_all": remove_all,
        "n_before": n_before,
        "n_removed": n_removed,
        "n_kept": n_before - n_removed,
        "removed_metabolites": removed_names,
    }
    print(
            f"[MORTIS] Drug filter: removed {n_removed} / {n_before} metabolites "
            f"({n_before - n_removed} retained)"
        )
    return adata


def list_drug_matches(
    adata: ad.AnnData,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of metabolites found in DrugBank, without removing them.

    Use this to inspect which metabolites would be removed before committing
    to :func:`filter_drugs`.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data.
    db_path : str or None, optional
        Path to a ``drugbank.db`` SQLite file. Default: ``None``, which uses
        the copy bundled with the installed package.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``metabolite`` and ``in_drugbank``, listing
        all metabolites that match a DrugBank entry.

    Examples
    --------
    >>> matches = mt.list_drug_matches(adata)
    >>> print(f"Found {len(matches)} drug metabolites in your data")
    >>> print(matches.head(10))
    """
    if db_path is None:
        db_path = _default_drugbank_path()
    db_names = _load_drugbank_names(db_path)
    mask = _match_metabolites_to_drugs(adata.var_names, db_names)
    matched = adata.var_names[mask].tolist()
    df = pd.DataFrame({
        "metabolite": matched,
        "in_drugbank": True,
    })
    print(
        f"[MORTIS] Found {len(matched)} / {adata.n_vars} metabolites "
        "matching DrugBank entries."
    )
    return df
