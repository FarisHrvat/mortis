"""
MORTIS Exceptions
=================
The error types MORTIS raises, plus two small helpers (:func:`suggest` and
:func:`listing`) used to build the messages.

The rule the messages follow: say what was wrong with the call, say what the
data actually contains, and where a typo is plausible, guess the intended
name. A user who mistypes a column should not have to open a REPL to find out
what the columns are called.
"""

import difflib
from typing import Iterable


def listing(options: Iterable, limit: int = 12) -> str:
    """Render valid names for an error message, truncated so it stays readable.

    A 400-metabolite panel printed in full is not an error message, it is a
    wall, so anything past ``limit`` becomes a count.
    """
    names = [str(o) for o in options]
    if not names:
        return "none"
    shown = ", ".join(repr(n) for n in names[:limit])
    if len(names) > limit:
        shown += f", ... ({len(names)} in total)"
    return shown


def suggest(value, options: Iterable, limit: int = 3) -> str:
    """Guess what the user meant, as a sentence ready to append to a message.

    Returns an empty string when nothing is close, so it can be dropped into
    an f-string unconditionally.
    """
    close = difflib.get_close_matches(
        str(value), [str(o) for o in options], n=limit, cutoff=0.6
    )
    if not close:
        return ""
    if len(close) == 1:
        return f" Did you mean {close[0]!r}?"
    return " Did you mean " + " or ".join(repr(c) for c in close) + "?"


class PseudoreplicationWarning(UserWarning):
    """
    Raised when a test is about to treat pixels as independent replicates.

    Its own category (rather than a plain UserWarning) so that a user who has
    genuinely confirmed a within-sample comparison can filter exactly this and
    nothing else::

        warnings.filterwarnings("ignore", category=mortis.PseudoreplicationWarning)
    """


class MortisError(Exception):
    """Base class for all MORTIS errors."""
    pass


class MissingROIError(MortisError):
    """
    Raised when an AnnData object is missing 'is_tissue' / 'is_background'
    columns in .obs, which are required before background filtering.

    Fix: Run ``mortis.draw_ROIs(adata)`` or load paired tissue/background
    files so ROI labels are assigned automatically.
    """
    pass


class MissingSpatialError(MortisError):
    """
    Raised when an AnnData object has no spatial coordinates.

    Fix: Ensure your file contains 'x' and 'y' columns, or that
    ``adata.obsm['spatial']`` is populated before calling this function.
    """
    pass


class NotPreprocessedError(MortisError):
    """
    Raised when a downstream analysis step is called on data that has not
    been preprocessed (normalized / log-transformed).

    Fix: Run ``mortis.preprocess(adata)`` or call ``tic_normalize`` and
    ``log1p_transform`` before proceeding.
    """
    pass


class NoClustersError(MortisError):
    """
    Raised when cluster labels are required but have not been computed yet.

    Fix: Run ``mortis.cluster(adata)`` before calling this function.
    """
    pass


class NoEmbeddingError(MortisError):
    """
    Raised when a dimensionality-reduction embedding (PCA / UMAP) is required
    but has not been computed yet.

    Fix: Run ``mortis.run_pca(adata)`` or ``mortis.run_umap(adata)`` before calling this function.
    """
    pass


class InsufficientSamplesError(MortisError):
    """
    Raised when a statistical test requires at least two groups but fewer
    were found in the data.

    Fix: Ensure your AnnData contains observations from at least two distinct
    groups (check ``adata.obs['group']``).
    """
    pass


class InvalidParameterError(MortisError):
    """
    Raised when a parameter value is outside the accepted range or is of the
    wrong type.

    The error message will specify which parameter is invalid and what the
    accepted values are.
    """
    pass


class FileFormatError(MortisError):
    """
    Raised when a file cannot be parsed because its format is unrecognised
    or its required columns are missing.

    Fix: Verify that your file contains 'x', 'y', and at least one metabolite
    column, and that the file extension is .csv, .xlsx, or .h5ad.
    """
    pass
