"""
Tests for the command line.

The CLI is what runs on a cluster node and inside a container, where nobody is
watching and a stack trace is all you get. So these lean on the two things that
matter there: a bad config fails with a sentence you can act on, and a good one
produces the same numbers the Python API would.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest

from mortis import cli

yaml = pytest.importorskip("yaml")


@pytest.fixture
def cohort(tmp_path):
    """Twelve sections, two arms, on a small grid, a miniature of the real thing."""
    rng = np.random.default_rng(0)
    side, n_vars, n_sections = 12, 15, 12
    n = side * side
    coords = np.array([[i % side, i // side] for i in range(n)], dtype=np.float64)

    blocks, sections, groups, patients = [], [], [], []
    for i in range(n_sections):
        group = "R" if i < n_sections // 2 else "NR"
        offset = rng.normal(0, 1.0, n_vars)
        if group == "R":
            offset[:3] += 6.0
        blocks.append(rng.normal(offset, 1.0, (n, n_vars)))
        sections += [f"S{i:02d}"] * n
        groups += [group] * n
        patients += [f"P{i:02d}"] * n

    X = np.vstack(blocks).astype(np.float32)
    X -= X.min()
    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = np.tile(coords, (n_sections, 1))
    adata.obs["section"], adata.obs["response"] = sections, groups
    adata.obs["patient"] = patients
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]

    path = tmp_path / "cohort.h5ad"
    adata.write_h5ad(path)
    return path


def _config(tmp_path, cohort, **overrides):
    config = {
        "input": {
            "path": str(cohort), "sample_key": "patient", "section_key": "section",
            "group_key": "response", "groups": ["R", "NR"],
        },
        "preprocess": {"normalize": "tic", "log1p": True},
        "analysis": {
            "abundance": True, "organization": True,
            "organization_metrics": ["morans_i"], "classify_compounds": False,
            "pathways": False,
        },
        "output": {"dir": str(tmp_path / "out"), "figures": False, "manifest": True},
        "runtime": {"seed": 0},
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


# ---------------------------------------------------------------------------
# It runs
# ---------------------------------------------------------------------------

class TestRun:

    def test_produces_the_expected_files(self, tmp_path, cohort):
        assert cli.main(["run", str(_config(tmp_path, cohort))]) == 0
        out = tmp_path / "out"
        for name in ("differential_abundance.csv", "differential_organization.csv",
                     "two_axis.csv", "pseudobulk.h5ad", "manifest.json", "config.yaml"):
            assert (out / name).exists(), f"{name} was not written"

    def test_matches_the_python_api(self, tmp_path, cohort):
        """The CLI is a front end, not a second implementation."""
        import matplotlib
        matplotlib.use("Agg")
        import pandas as pd

        import mortis as mt

        cli.main(["run", str(_config(tmp_path, cohort))])
        from_cli = pd.read_csv(tmp_path / "out" / "differential_abundance.csv")

        adata = mt.log1p_transform(mt.tic_normalize(mt.read_metabolomics_data(str(cohort))))
        direct = mt.differential_abundance(
            mt.pseudobulk(adata, sample_key="patient"), "response", "R", "NR"
        )
        np.testing.assert_allclose(
            from_cli.sort_values("metabolite")["delta"].to_numpy(),
            direct.sort_values("metabolite")["delta"].to_numpy(),
            rtol=1e-6,
        )

    def test_config_is_written_back(self, tmp_path, cohort):
        """What ran, not what you meant to run."""
        cli.main(["run", str(_config(tmp_path, cohort))])
        written = yaml.safe_load((tmp_path / "out" / "config.yaml").read_text())
        assert written["input"]["group_key"] == "response"

    def test_manifest_verifies_against_its_own_output(self, tmp_path, cohort):
        cli.main(["run", str(_config(tmp_path, cohort))])
        out = tmp_path / "out"
        assert cli.main([
            "verify", str(out / "manifest.json"), "--data", str(out / "pseudobulk.h5ad"),
            "--result", f"abundance={out / 'differential_abundance.csv'}",
        ]) == 0

    def test_figures_are_written_when_asked(self, tmp_path, cohort):
        config = _config(tmp_path, cohort, output={
            "dir": str(tmp_path / "out"), "figures": True, "formats": ["png"], "manifest": False,
        })
        cli.main(["run", str(config)])
        assert list((tmp_path / "out").glob("*.png"))


# ---------------------------------------------------------------------------
# It fails usefully
# ---------------------------------------------------------------------------

class TestErrors:
    """
    Nobody is watching when this runs on a cluster, so every failure has to say
    what went wrong and what to do about it.
    """

    def test_missing_config_names_the_fix(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["run", str(tmp_path / "nope.yaml")])
        assert "mortis template" in str(excinfo.value)

    def test_unknown_column_lists_the_real_ones(self, tmp_path, cohort):
        config = _config(tmp_path, cohort, input={
            "path": str(cohort), "sample_key": "not_a_column", "section_key": "section",
            "group_key": "response", "groups": ["R", "NR"],
        })
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["run", str(config)])
        message = str(excinfo.value)
        assert "not_a_column" in message and "patient" in message

    def test_wrong_number_of_groups_is_caught(self, tmp_path, cohort):
        config = _config(tmp_path, cohort, input={
            "path": str(cohort), "sample_key": "patient", "section_key": "section",
            "group_key": "response", "groups": ["R"],
        })
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["run", str(config)])
        assert "two group names" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The other subcommands
# ---------------------------------------------------------------------------

class TestOtherCommands:

    def test_template_is_valid_yaml_and_runnable(self, capsys):
        assert cli.main(["template"]) == 0
        config = yaml.safe_load(capsys.readouterr().out)
        # The template is the first thing a new user edits, so it must parse
        # and it must contain the keys `run` actually reads.
        assert config["input"]["sample_key"]
        assert config["output"]["dir"]
        assert "organization" in config["analysis"]

    def test_info_reports_the_environment(self, capsys):
        assert cli.main(["info"]) == 0
        out = capsys.readouterr().out
        assert "MORTIS" in out and "numpy" in out

    def test_verify_detects_a_changed_result(self, tmp_path, cohort):
        import pandas as pd

        cli.main(["run", str(_config(tmp_path, cohort))])
        out = tmp_path / "out"
        table = pd.read_csv(out / "differential_abundance.csv")
        table.loc[0, "delta"] += 0.5
        tampered = tmp_path / "tampered.csv"
        table.to_csv(tampered, index=False)

        assert cli.main([
            "verify", str(out / "manifest.json"), "--result", f"abundance={tampered}",
        ]) == 1

    def test_manifest_carries_the_cli_provenance(self, tmp_path, cohort):
        cli.main(["run", str(_config(tmp_path, cohort))])
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        steps = [entry["step"] for entry in manifest["provenance"]]
        assert "pseudobulk" in steps

