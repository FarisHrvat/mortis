"""
The error messages are part of the interface, so they get tested like the rest
of it.

A user who mistypes a column name should be told that the name is wrong, what
the real names are, and, when the mistake looks like a typo, which one they
probably meant. These tests pin that down for the functions people hit first,
so a later refactor cannot quietly drop back to "'x' not found".
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

import mortis
from mortis.exceptions import (
    InvalidParameterError,
    MissingSpatialError,
    NotPreprocessedError,
    listing,
    suggest,
)


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.random((40, 4), dtype=np.float32))
    a.var_names = ["Taurine", "Creatine", "Glutamate", "Choline"]
    a.obs["response"] = ["Responder"] * 20 + ["Non Responder"] * 20
    a.obs["section"] = ["s1"] * 20 + ["s2"] * 20
    a.obsm["spatial"] = rng.random((40, 2)) * 100
    return a


class TestHelpers:
    def test_listing_truncates_long_panels(self):
        text = listing([f"m{i}" for i in range(50)], limit=5)
        assert text.startswith("'m0', 'm1'")
        assert "50 in total" in text

    def test_listing_handles_nothing(self):
        assert listing([]) == "none"

    def test_suggest_finds_the_typo(self):
        assert "'response'" in suggest("respons", ["response", "section"])

    def test_suggest_stays_quiet_when_nothing_is_close(self):
        assert suggest("zzzzzz", ["response", "section"]) == ""


class TestColumnMistakes:
    def test_group_column_lists_what_exists(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.compare_groups(
                adata, "respons", "Responder", "Non Responder",
                acknowledge_pixel_level=True,
            )
        message = str(err.value)
        assert "'respons'" in message
        assert "'section'" in message          # says what is actually there
        assert "Did you mean 'response'?" in message

    def test_group_value_lists_the_labels(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.compare_groups(
                adata, "response", "Responders", "Non Responder",
                acknowledge_pixel_level=True,
            )
        message = str(err.value)
        assert "'Responder', 'Non Responder'" in message
        assert "Did you mean" in message and "'Responder'" in message

    def test_subset_on_a_value_nobody_has(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.subset_obs(adata, "response", "Remission")
        assert "would be empty" in str(err.value)

    def test_split_names_the_columns(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.split_by_obs(adata, "sample")
        assert "'section'" in str(err.value)


class TestMetaboliteMistakes:
    def test_unknown_metabolite_suggests_the_near_miss(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.getis_ord_gi(adata, "Tauryne")
        assert "Did you mean 'Taurine'?" in str(err.value)

    def test_unknown_metabolite_says_how_many_there_are(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.getis_ord_gi(adata, "Cholesterol")
        assert "4 metabolites" in str(err.value)


class TestMissingPrerequisites:
    def test_missing_coordinates_says_how_to_set_them(self, adata):
        del adata.obsm["spatial"]
        with pytest.raises(MissingSpatialError) as err:
            mortis.spatial_neighbors(adata)
        assert "obsm['spatial']" in str(err.value)
        assert "read_file" in str(err.value)

    def test_unpreprocessed_names_the_function_to_run(self, adata):
        adata.obs["cluster"] = ["0"] * 20 + ["1"] * 20
        with pytest.raises(NotPreprocessedError) as err:
            mortis.find_markers(adata)
        assert "mortis.preprocess(adata)" in str(err.value)

    def test_colocalization_says_which_step_is_missing(self, adata):
        with pytest.raises(InvalidParameterError) as err:
            mortis.metabolite_colocalization(adata)
        assert "spatial_autocorrelation" in str(err.value)


class TestNoStaleCapitalisation:
    """The package is imported as ``mortis``; messages used to say ``MORTIS.``."""

    def test_messages_do_not_tell_users_to_call_MORTIS_dot_something(self):
        import pathlib
        import re

        offenders = []
        for path in (pathlib.Path(mortis.__file__).parent).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"MORTIS\.[a-z_]+\(", text):
                offenders.append(path.name)
        assert not offenders, f"MORTIS.foo() should be mortis.foo(): {offenders}"


class TestSourceStaysPrintable:
    """
    A Windows console on a legacy code page cannot encode characters outside
    cp1252, so an arrow or a >= sign inside a print() crashes the run there
    instead of reporting progress. The package source is kept to ASCII, with
    two deliberate exceptions.
    """

    ALLOWED = {
        "\u00b1",  # matches "(+-)-" stereodescriptors in real compound names
        "\u00b5",  # the micrometre axis label, drawn by matplotlib, never printed
    }

    def _source_files(self):
        import mortis
        root = Path(mortis.__file__).parent
        return sorted(root.glob("*.py"))

    def test_no_unencodable_characters(self):
        offenders = []
        for path in self._source_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for char in line:
                    if ord(char) > 127 and char not in self.ALLOWED:
                        offenders.append(f"{path.name}:{lineno}: {char!r}")
        assert not offenders, "non-ASCII outside the allowed set:\n" + "\n".join(offenders)

    def test_every_message_survives_a_cp1252_console(self):
        for path in self._source_files():
            text = path.read_text(encoding="utf-8")
            for char in self.ALLOWED:
                text = text.replace(char, "")
            text.encode("cp1252")
