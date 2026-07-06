"""
Tests for mortis.filter — annotation score filtering and DrugBank filtering.
"""

import sqlite3

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.exceptions import InvalidParameterError
from mortis.filter import filter_by_score, filter_drugs, list_drug_matches

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scored_adata():
    """AnnData with 5 metabolites and annotation scores."""
    X = np.random.rand(10, 5).astype(np.float32)
    obs = pd.DataFrame({"x": range(10), "y": [0] * 10})
    obs.index = [f"{i}_0" for i in range(10)]
    var = pd.DataFrame({
        "score": [0.1, 0.4, 0.6, 0.8, 0.05],
    }, index=["met_low1", "met_mid", "met_high1", "met_high2", "met_low2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


@pytest.fixture
def drug_adata():
    """AnnData with metabolites including known drug names."""
    X = np.random.rand(10, 4).astype(np.float32)
    obs = pd.DataFrame({"x": range(10), "y": [0] * 10})
    obs.index = [f"{i}_0" for i in range(10)]
    # 'aspirin' and 'ibuprofen' are common drug synonyms in DrugBank
    var = pd.DataFrame(index=["Palmitic acid", "aspirin", "Oleic acid", "ibuprofen"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


@pytest.fixture
def mock_db(tmp_path):
    """Create a minimal mock DrugBank SQLite database."""
    db_path = tmp_path / "test_drugbank.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE drug_compounds (name TEXT)")
    conn.execute("CREATE TABLE drug_synonyms (synonym TEXT)")
    conn.executemany("INSERT INTO drug_compounds VALUES (?)",
                     [("Aspirin",), ("Ibuprofen",), ("Metformin",)])
    conn.executemany("INSERT INTO drug_synonyms VALUES (?)",
                     [("aspirin",), ("ibuprofen",), ("acetylsalicylic acid",)])
    conn.commit()
    conn.close()
    return str(db_path)


# ---------------------------------------------------------------------------
# filter_by_score
# ---------------------------------------------------------------------------

class TestFilterByScore:
    def test_keeps_correct_metabolites(self, scored_adata):
        result = filter_by_score(scored_adata, min_score=0.5)
        assert result.n_vars == 2  # met_high1 (0.6) and met_high2 (0.8)
        assert "met_high1" in result.var_names
        assert "met_high2" in result.var_names

    def test_zero_score_keeps_all(self, scored_adata):
        result = filter_by_score(scored_adata, min_score=0.0)
        assert result.n_vars == 5

    def test_high_score_removes_all(self, scored_adata):
        result = filter_by_score(scored_adata, min_score=1.5)
        assert result.n_vars == 0

    def test_missing_score_col_raises(self, scored_adata):
        with pytest.raises(InvalidParameterError, match="not found in adata.var"):
            filter_by_score(scored_adata, score_col="nonexistent")

    def test_invalid_min_score_raises(self, scored_adata):
        with pytest.raises(InvalidParameterError, match="between 0.0 and 2.0"):
            filter_by_score(scored_adata, min_score=3.0)

    def test_uns_records_params(self, scored_adata):
        result = filter_by_score(scored_adata, min_score=0.5)
        assert "filter_score" in result.uns
        assert result.uns["filter_score"]["min_score"] == 0.5
        assert result.uns["filter_score"]["n_kept"] == 2

    def test_copy_flag(self, scored_adata):
        original_n = scored_adata.n_vars
        filter_by_score(scored_adata, min_score=0.5, copy=True)
        assert scored_adata.n_vars == original_n

    def test_custom_score_col(self, scored_adata):
        scored_adata.var["custom_score"] = [0.9, 0.1, 0.9, 0.1, 0.9]
        result = filter_by_score(scored_adata, min_score=0.5, score_col="custom_score")
        assert result.n_vars == 3


# ---------------------------------------------------------------------------
# filter_drugs
# ---------------------------------------------------------------------------

class TestFilterDrugs:
    def test_removes_drug_metabolites(self, drug_adata, mock_db):
        result = filter_drugs(drug_adata, db_path=mock_db, remove_all=True)
        assert "aspirin" not in result.var_names
        assert "ibuprofen" not in result.var_names
        assert "Palmitic acid" in result.var_names
        assert "Oleic acid" in result.var_names

    def test_remove_specific_drugs(self, drug_adata, mock_db):
        result = filter_drugs(
            drug_adata, db_path=mock_db,
            remove_all=False, drug_names=["aspirin"]
        )
        assert "aspirin" not in result.var_names
        assert "ibuprofen" in result.var_names  # not in the list

    def test_uns_records_removed(self, drug_adata, mock_db):
        result = filter_drugs(drug_adata, db_path=mock_db)
        assert "filter_drugs" in result.uns
        assert result.uns["filter_drugs"]["n_removed"] >= 1

    def test_missing_db_raises(self, drug_adata):
        with pytest.raises(FileNotFoundError, match="not found"):
            filter_drugs(drug_adata, db_path="/nonexistent/drugbank.db")

    def test_remove_all_false_no_names_raises(self, drug_adata, mock_db):
        with pytest.raises(InvalidParameterError, match="drug_names"):
            filter_drugs(drug_adata, db_path=mock_db, remove_all=False)

    def test_copy_flag(self, drug_adata, mock_db):
        original_n = drug_adata.n_vars
        filter_drugs(drug_adata, db_path=mock_db, copy=True)
        assert drug_adata.n_vars == original_n

    def test_unknown_drug_name_warns(self, drug_adata, mock_db, capsys):
        filter_drugs(
            drug_adata, db_path=mock_db,
            remove_all=False, drug_names=["totally_fake_drug_xyz"]
        )
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ---------------------------------------------------------------------------
# list_drug_matches
# ---------------------------------------------------------------------------

class TestListDrugMatches:
    def test_returns_dataframe(self, drug_adata, mock_db):
        df = list_drug_matches(drug_adata, db_path=mock_db)
        assert isinstance(df, pd.DataFrame)
        assert "metabolite" in df.columns
        assert "in_drugbank" in df.columns

    def test_finds_correct_matches(self, drug_adata, mock_db):
        df = list_drug_matches(drug_adata, db_path=mock_db)
        assert "aspirin" in df["metabolite"].values
        assert "ibuprofen" in df["metabolite"].values
        assert "Palmitic acid" not in df["metabolite"].values

    def test_no_matches_returns_empty(self, mock_db):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        obs.index = ["0_0"]
        var = pd.DataFrame(index=["Palmitic acid", "Oleic acid"])
        adata = ad.AnnData(X=np.ones((1, 2), dtype=np.float32), obs=obs, var=var)
        df = list_drug_matches(adata, db_path=mock_db)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Bundled DrugBank database (package data, no db_path required)
# ---------------------------------------------------------------------------

class TestBundledDrugbank:
    def test_default_path_resolves_to_bundled_file(self):
        from pathlib import Path

        from mortis.filter import _default_drugbank_path
        resolved = _default_drugbank_path()
        assert Path(resolved).is_file()
        assert Path(resolved).name == "drugbank.db"

    def test_list_drug_matches_without_db_path(self):
        # Regression test: filter_drugs/list_drug_matches must work without
        # requiring the caller to have a drugbank.db in their cwd — it's
        # bundled as package data (mortis/data/drugbank.db) and resolved by
        # default via importlib.resources.
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        var = pd.DataFrame(index=["Aspirin", "Palmitic acid"])
        adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32), obs=obs, var=var)
        df = list_drug_matches(adata)
        assert "Aspirin" in df["metabolite"].values

    def test_filter_drugs_without_db_path(self):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        var = pd.DataFrame(index=["Aspirin", "Nonexistent Compound Xyz123"])
        adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32), obs=obs, var=var)
        result = filter_drugs(adata)
        assert "Aspirin" not in result.var_names
        assert "Nonexistent Compound Xyz123" in result.var_names
