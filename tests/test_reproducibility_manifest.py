"""
Tests for verification manifests.

The claim is narrow and testable: a manifest lets someone check that a re-run
produced the same numbers, without giving them the data. So the tests are
(a) an honest re-run passes, (b) every kind of divergence is caught and named,
and (c) nothing recoverable leaks into the file.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mortis as mt
from mortis.exceptions import InvalidParameterError, MortisError


def _adata(seed=0, n_obs=40, n_vars=12):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=(rng.random((n_obs, n_vars)) * 100).astype(np.float32))
    adata.obs["patient"] = [f"P{i % 8}" for i in range(n_obs)]
    adata.obs["response"] = ["R" if i % 8 < 4 else "NR" for i in range(n_obs)]
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]
    return adata


def _result(seed=0, n=12):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "metabolite": [f"m{j:03d}" for j in range(n)],
        "delta": rng.uniform(-1, 1, n),
        "pval_adj": rng.uniform(0, 1, n),
    })


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------

class TestVerification:

    def test_honest_rerun_passes_everything(self, tmp_path):
        adata, result = _adata(), _result()
        path = mt.export_manifest(tmp_path / "m", adata=adata, results={"da": result})
        report = mt.verify_manifest(path, adata=_adata(), results={"da": _result()})
        assert not (report["status"] == "fail").any()

    def test_changed_input_is_caught(self, tmp_path):
        adata = _adata()
        path = mt.export_manifest(tmp_path / "m", adata=adata)

        altered = _adata()
        X = np.asarray(altered.X).copy()
        X[0, 0] += 50.0
        altered.X = X

        report = mt.verify_manifest(path, adata=altered).set_index("check")
        assert report.loc["input matrix contents", "status"] == "fail"
        assert report.loc["input row count", "status"] == "pass"

    def test_changed_result_is_caught_and_described(self, tmp_path):
        path = mt.export_manifest(tmp_path / "m", results={"da": _result(0)})
        report = mt.verify_manifest(path, results={"da": _result(1)}).set_index("check")
        assert report.loc["result 'da'", "status"] == "fail"
        assert "different values" in report.loc["result 'da'", "detail"]

    def test_row_count_change_is_named_specifically(self, tmp_path):
        path = mt.export_manifest(tmp_path / "m", results={"da": _result(0, n=12)})
        report = mt.verify_manifest(path, results={"da": _result(0, n=8)}).set_index("check")
        assert "row count" in report.loc["result 'da'", "detail"]

    def test_renamed_features_are_caught(self, tmp_path):
        adata = _adata()
        path = mt.export_manifest(tmp_path / "m", adata=adata)
        renamed = _adata()
        renamed.var_names = [f"x{j:03d}" for j in range(renamed.n_vars)]
        report = mt.verify_manifest(path, adata=renamed).set_index("check")
        assert report.loc["input feature names", "status"] == "fail"
        assert report.loc["input matrix contents", "status"] == "pass"

    def test_tampering_is_refused(self, tmp_path):
        path = mt.export_manifest(tmp_path / "m", adata=_adata())
        body = json.loads(path.read_text())
        body["analysis"] = "a different analysis"
        path.write_text(json.dumps(body))
        with pytest.raises(MortisError, match="seal"):
            mt.verify_manifest(path, adata=_adata())

    def test_missing_pieces_are_skipped_not_failed(self, tmp_path):
        path = mt.export_manifest(tmp_path / "m", adata=_adata(), results={"da": _result()})
        report = mt.verify_manifest(path).set_index("check")
        assert report.loc["input data", "status"] == "skipped"
        assert report.loc["result 'da'", "status"] == "skipped"
        assert not (report["status"] == "fail").any()

    def test_tolerates_floating_point_noise(self, tmp_path):
        """
        BLAS scheduling can move the last bit or two between runs. A checksum
        that trips on that would fail every honest verification.
        """
        result = _result()
        path = mt.export_manifest(tmp_path / "m", results={"da": result})
        jittered = result.copy()
        jittered["delta"] = jittered["delta"] + 1e-12
        report = mt.verify_manifest(path, results={"da": jittered}).set_index("check")
        assert report.loc["result 'da'", "status"] == "pass"

    def test_real_difference_still_fails(self, tmp_path):
        """The tolerance must not be so loose it hides a genuine change."""
        result = _result()
        path = mt.export_manifest(tmp_path / "m", results={"da": result})
        moved = result.copy()
        moved.loc[0, "delta"] += 0.001
        report = mt.verify_manifest(path, results={"da": moved}).set_index("check")
        assert report.loc["result 'da'", "status"] == "fail"


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

class TestNoDataLeaks:

    def test_no_intensity_appears_in_the_manifest(self, tmp_path):
        """
        The whole point is that this file is safe to publish for a cohort that
        is not. If any raw value survived into it, that would be false.
        """
        rng = np.random.default_rng(3)
        adata = _adata()
        marker = 123456.789
        X = np.asarray(adata.X).copy().astype(np.float64)
        X[0, 0] = marker
        adata.X = X

        path = mt.export_manifest(tmp_path / "m", adata=adata)
        text = path.read_text()
        assert "123456" not in text
        for value in rng.choice(X.ravel(), 25):
            assert f"{value:.4f}" not in text

    def test_fingerprint_is_not_invertible_but_is_sensitive(self):
        a, b = _adata(0), _adata(0)
        assert mt.data_fingerprint(a)["matrix_sha256"] == mt.data_fingerprint(b)["matrix_sha256"]
        assert mt.data_fingerprint(_adata(1))["matrix_sha256"] != mt.data_fingerprint(a)["matrix_sha256"]

    def test_obs_values_are_not_recorded_only_column_names(self, tmp_path):
        adata = _adata()
        adata.obs["patient_id_from_hospital"] = [f"NHS-{i:06d}" for i in range(adata.n_obs)]
        path = mt.export_manifest(tmp_path / "m", adata=adata)
        text = path.read_text()
        assert "patient_id_from_hospital" in text, "column names are metadata and are recorded"
        assert "NHS-000000" not in text, "column *values* must never be recorded"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:

    def test_pseudobulk_records_itself(self):
        pb = mt.pseudobulk(_adata(n_obs=80), sample_key="patient")
        chain = mt.provenance(pb)
        assert chain[-1]["step"] == "pseudobulk"
        assert chain[-1]["params"]["sample_key"] == "patient"

    def test_a_recorded_object_can_still_be_saved(self, tmp_path):
        """
        Regression: provenance used to be stored as a list of dicts, which HDF5
        cannot represent — so recording a step quietly broke write_h5ad() on the
        very object it was recorded on. Found by the CLI, which saves its
        pseudobulk output.
        """
        import anndata as ad

        pb = mt.pseudobulk(_adata(n_obs=80), sample_key="patient")
        mt.record_step(pb, "a manual step", {"why": "because"})
        path = tmp_path / "pb.h5ad"
        pb.write_h5ad(path)

        reloaded = ad.read_h5ad(path)
        assert [s["step"] for s in mt.provenance(reloaded)] == ["pseudobulk", "a manual step"]
        assert mt.provenance(reloaded)[-1]["params"]["why"] == "because"

    def test_reads_the_old_dict_format(self):
        """Objects written by an earlier version stay readable."""
        adata = _adata()
        adata.uns["mortis_provenance"] = [{"step": "legacy", "params": {}, "at": ""}]
        assert mt.provenance(adata)[0]["step"] == "legacy"

    def test_manual_steps_can_be_recorded(self):
        adata = _adata()
        mt.record_step(adata, "dropped section S07", {"reason": "fold artefact"})
        assert mt.provenance(adata)[-1]["step"] == "dropped section S07"

    def test_chain_appends_in_order(self):
        adata = _adata()
        mt.record_step(adata, "first")
        mt.record_step(adata, "second")
        assert [s["step"] for s in mt.provenance(adata)] == ["first", "second"]

    def test_provenance_reaches_the_manifest(self, tmp_path):
        pb = mt.pseudobulk(_adata(n_obs=80), sample_key="patient")
        mt.record_step(pb, "excluded an outlier", {"sample": "P3"})
        path = mt.export_manifest(tmp_path / "m", adata=pb)
        steps = [s["step"] for s in json.loads(path.read_text())["provenance"]]
        assert steps == ["pseudobulk", "excluded an outlier"]


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

class TestManifestFile:

    def test_environment_is_recorded(self, tmp_path):
        path = mt.export_manifest(tmp_path / "m", adata=_adata())
        env = json.loads(path.read_text())["environment"]
        assert "numpy" in env["packages"]
        assert env["python"]

    def test_json_extension_is_added(self, tmp_path):
        assert mt.export_manifest(tmp_path / "m", adata=_adata()).name == "m.json"

    def test_needs_something_to_record(self, tmp_path):
        with pytest.raises(InvalidParameterError, match="Nothing to record"):
            mt.export_manifest(tmp_path / "m")

    def test_rejects_a_non_manifest(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"hello": "world"}))
        with pytest.raises(MortisError, match="not a MORTIS manifest"):
            mt.verify_manifest(path)

    def test_result_fingerprint_rejects_non_dataframe(self):
        with pytest.raises(InvalidParameterError, match="DataFrame"):
            mt.result_fingerprint({"delta": [1, 2, 3]})

    def test_strict_environment_flags_version_drift(self, tmp_path, monkeypatch):
        path = mt.export_manifest(tmp_path / "m", adata=_adata())
        from mortis import reproducibility

        monkeypatch.setattr(reproducibility, "_versions", lambda: {"numpy": "0.0.0-fake"})
        lenient = mt.verify_manifest(path, adata=_adata()).set_index("check")
        strict = mt.verify_manifest(path, adata=_adata(), strict_environment=True).set_index("check")
        assert lenient.loc["package versions", "status"] == "info"
        assert strict.loc["package versions", "status"] == "fail"
