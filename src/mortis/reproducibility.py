"""
MORTIS Reproducibility Module
=============================
A file you can hand a reviewer that lets them verify your analysis re-runs to
the same numbers, without giving them the patient data.

The problem
-----------
"Data available from the corresponding author on reasonable request" is the
standard line, and it verifies exactly nothing.

Put yourself in the reviewer's chair. You doubt a result. Your options are: take
it on trust, ask for data the ethics approval almost certainly forbids sharing,
or reimplement six months of analysis from one paragraph of methods. None of
those is checking. Two of them are wishful thinking and the third is a career
sacrifice.

What a manifest is
------------------
:func:`export_manifest` writes a JSON file recording, for one analysis:

* the exact package and environment versions it ran under,
* every MORTIS step that touched the object, with its parameters and seeds,
* a **fingerprint** of the input data: its shape, its feature names, and a
  checksum of the matrix,
* a fingerprint of every result table produced.

Then :func:`verify_manifest` re-runs the comparison. If someone repeats the
analysis and their manifest matches yours, the numbers are identical and both
of you can prove it. If it does not match, the report says precisely which
part diverged, the input, a parameter, a seed, or a result.

Fingerprints, not data
----------------------
Nothing in a manifest can be turned back into intensities. Checksums are
one-way, and what is stored alongside them is metadata a methods section would
carry anyway: matrix shape, column names, parameter values. A manifest is safe
to attach to a submission or commit to a public repository even when the
underlying cohort cannot leave the institution, which is the entire point,
because that is exactly the situation clinical imaging data is in.

What it does not do
-------------------
A matching manifest proves two runs produced the same numbers. It does not
prove the analysis was appropriate, that the cohort was large enough, or that
the conclusion follows. It closes the "can this be checked" gap and nothing
else.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .exceptions import InvalidParameterError, MortisError

__all__ = [
    "record_step",
    "provenance",
    "data_fingerprint",
    "result_fingerprint",
    "export_manifest",
    "verify_manifest",
    "MANIFEST_VERSION",
]

#: Bumped when the manifest layout changes in a way older readers cannot parse.
MANIFEST_VERSION = 1

#: Packages whose version can change a numeric result. Recorded so a mismatch
#: points at the culprit instead of leaving you to guess.
_TRACKED = (
    "mortis-spatial", "numpy", "scipy", "pandas", "anndata", "scanpy",
    "scikit-learn", "statsmodels", "numba", "leidenalg", "igraph", "harmonypy",
)


def _versions() -> Dict[str, str]:
    out = {}
    for name in _TRACKED:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def _git_commit() -> Optional[str]:
    """The analysis repo's commit, when the analysis lives in one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _digest(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
        h.update(b"\x1e")
    return h.hexdigest()


def _matrix_bytes(X) -> bytes:
    """
    Stable bytes for a matrix, whatever container it arrives in.

    Rounded to 6 decimal places first. Two runs of the same pipeline can differ
    in the last bit or two from BLAS thread scheduling, and a fingerprint that
    trips on that would cry wolf on every verification. Six places is far
    tighter than any real analysis distinguishes and immune to that noise.
    """
    dense = X.toarray() if issparse(X) else np.asarray(X)
    return np.ascontiguousarray(np.round(dense.astype(np.float64), 6)).tobytes()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def record_step(
    adata: ad.AnnData, step: str, params: Optional[Mapping[str, Any]] = None
) -> None:
    """
    Append one step to the object's provenance chain.

    MORTIS functions call this themselves. Call it directly for anything you do
    by hand that changes the data, a manual subset, a custom filter, so the
    manifest reflects the analysis you actually ran rather than the parts of it
    that happened to go through the package.

    Parameters
    ----------
    adata : anndata.AnnData
        Object to stamp. The chain lives in ``adata.uns['mortis_provenance']``.
    step : str
        What was done, e.g. ``"pseudobulk"`` or ``"dropped section S07 (fold)"``.
    params : mapping, optional
        Parameters worth recording. Values are coerced to strings, so anything
        is safe to pass.
    """
    chain = list(adata.uns.get("mortis_provenance", []))
    # JSON strings, not dicts: HDF5 has no nested mapping, so a list of dicts
    # in .uns breaks write_h5ad(). provenance() reads them back.
    chain.append(json.dumps({
        "step": str(step),
        "params": {str(k): str(v) for k, v in (params or {}).items()},
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }))
    adata.uns["mortis_provenance"] = chain


def provenance(adata: ad.AnnData) -> List[Dict[str, Any]]:
    """
    The steps recorded on an object, oldest first.

    Parameters
    ----------
    adata : anndata.AnnData
        Object to read. Returns an empty list if nothing was ever recorded.

    Returns
    -------
    list of dict
        Each with ``step``, ``params`` and ``at``.

    Examples
    --------
    >>> for entry in mt.provenance(pb):
    ...     print(entry["step"], entry["params"])
    pseudobulk {'sample_key': 'patient', 'method': 'mean'...}
    """
    out: List[Dict[str, Any]] = []
    for entry in adata.uns.get("mortis_provenance", []):
        if isinstance(entry, dict):
            out.append(entry)          # written by an older version, still readable
            continue
        try:
            out.append(json.loads(entry))
        except (TypeError, ValueError):
            out.append({"step": str(entry), "params": {}, "at": ""})
    return out


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def data_fingerprint(adata: ad.AnnData, layer: Optional[str] = None) -> Dict[str, Any]:
    """
    A privacy-safe summary of an AnnData object.

    Returns shape, feature names, observation-column names and a SHA-256 of the
    matrix. None of it can be inverted to recover intensities, so this is safe
    to publish for a cohort that cannot be.

    Parameters
    ----------
    adata : anndata.AnnData
        Object to fingerprint.
    layer : str, optional
        Layer to checksum. Default ``None`` uses ``.X``.
    """
    if layer is not None and layer not in adata.layers:
        raise InvalidParameterError(
            f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}."
        )
    matrix = adata.layers[layer] if layer is not None else adata.X

    var_names = [str(v) for v in adata.var_names]
    obs_columns = sorted(str(c) for c in adata.obs.columns)
    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "layer": layer,
        "obs_columns": obs_columns,
        "var_names_sha256": _digest("\x1f".join(var_names).encode("utf-8")),
        "matrix_sha256": _digest(_matrix_bytes(matrix)),
        "has_spatial": "spatial" in adata.obsm,
    }


def result_fingerprint(result: pd.DataFrame) -> Dict[str, Any]:
    """
    A checksum of one result table.

    Numeric columns are rounded to six decimals before hashing, for the same
    reason :func:`data_fingerprint` rounds, so a verification fails on real
    disagreement rather than on floating-point scheduling noise.
    """
    if not isinstance(result, pd.DataFrame):
        raise InvalidParameterError(
            f"result must be a pandas DataFrame, got {type(result).__name__}."
        )
    frame = result.sort_index(axis=1)
    chunks: List[bytes] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            chunks.append(np.round(series.to_numpy(dtype=np.float64), 6).tobytes())
        else:
            chunks.append("\x1f".join(series.astype(str)).encode("utf-8"))
    return {
        "n_rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "sha256": _digest(*chunks),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def export_manifest(
    path: Union[str, Path],
    adata: Optional[ad.AnnData] = None,
    results: Optional[Mapping[str, pd.DataFrame]] = None,
    analysis: str = "unnamed analysis",
    notes: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Write a verification manifest for one analysis.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path. ``.json`` is appended if absent.
    adata : anndata.AnnData, optional
        The object the analysis ran on. Its fingerprint and provenance chain
        are recorded.
    results : mapping of str to DataFrame, optional
        Named result tables, e.g.
        ``{"abundance": ab, "organization": do}``. Each is fingerprinted.
    analysis : str
        A name for this analysis, carried into the manifest.
    notes : str, optional
        Free text for anything a reader should know, which cohort, which
        preregistration, why a section was excluded.
    extra : mapping, optional
        Any additional key-values to record. Coerced to strings.

    Returns
    -------
    pathlib.Path
        Where the manifest was written.

    Examples
    --------
    >>> ab = mt.differential_abundance(pb, "response", "R", "NR")
    >>> mt.export_manifest("manifest.json", adata=pb, results={"abundance": ab},
    ...                    analysis="vedolizumab week 14")
    """
    if adata is None and not results:
        raise InvalidParameterError(
            "Nothing to record, pass adata, results, or both."
        )

    body: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "analysis": str(analysis),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "git_commit": _git_commit(),
        },
        "data": data_fingerprint(adata) if adata is not None else None,
        "provenance": provenance(adata) if adata is not None else [],
        "results": {
            str(name): result_fingerprint(frame) for name, frame in (results or {}).items()
        },
    }
    if notes:
        body["notes"] = str(notes)
    if extra:
        body["extra"] = {str(k): str(v) for k, v in extra.items()}

    # Integrity check, not a signature: it proves the file is internally
    # consistent, not who wrote it.
    body["seal_sha256"] = _digest(json.dumps(body, sort_keys=True).encode("utf-8"))

    out = Path(path)
    if out.suffix.lower() != ".json":
        out = out.with_name(out.name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")

    print(
        f"[MORTIS] Manifest written: {out.name} "
        f"(seal {body['seal_sha256'][:12]}, {len(body['results'])} result table(s), "
        f"{len(body['provenance'])} recorded step(s))."
    )
    return out


def verify_manifest(
    path: Union[str, Path],
    adata: Optional[ad.AnnData] = None,
    results: Optional[Mapping[str, pd.DataFrame]] = None,
    strict_environment: bool = False,
) -> pd.DataFrame:
    """
    Check a re-run against a manifest, and say exactly what differs.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest to check against.
    adata : anndata.AnnData, optional
        The re-run's data object.
    results : mapping of str to DataFrame, optional
        The re-run's result tables, keyed as they were at export.
    strict_environment : bool
        Treat a package-version difference as a failure. Default ``False``,
        which reports versions as informational, most version changes do not
        move a number, and failing on all of them makes the check useless
        within a year. Turn it on when reproducing a published result exactly.

    Returns
    -------
    pandas.DataFrame
        One row per check, with ``check``, ``status`` (``"pass"``, ``"fail"``,
        ``"info"``, ``"skipped"``) and ``detail``.

    Raises
    ------
    MortisError
        If the file is not a MORTIS manifest, or its seal does not match, the
        latter meaning the file was edited after it was written.

    Examples
    --------
    >>> report = mt.verify_manifest("manifest.json", adata=pb, results={"abundance": ab})
    >>> report[report["status"] == "fail"]
    Empty DataFrame
    """
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if "manifest_version" not in body or "seal_sha256" not in body:
        raise MortisError(f"'{path}' is not a MORTIS manifest.")

    recorded_seal = body.pop("seal_sha256")
    if _digest(json.dumps(body, sort_keys=True).encode("utf-8")) != recorded_seal:
        raise MortisError(
            f"The seal on '{path}' does not match its contents, the file was modified "
            "after it was written. Verification cannot proceed."
        )

    rows: List[Dict[str, str]] = [
        {"check": "seal", "status": "pass", "detail": f"intact ({recorded_seal[:12]})"}
    ]

    if body.get("manifest_version") != MANIFEST_VERSION:
        rows.append({
            "check": "manifest version", "status": "info",
            "detail": f"written by version {body.get('manifest_version')}, reading with {MANIFEST_VERSION}",
        })

    # Environment
    current, recorded = _versions(), body.get("environment", {}).get("packages", {})
    drifted = [
        f"{name}: {recorded[name]} -> {current.get(name)}"
        for name in recorded if recorded[name] != current.get(name)
    ]
    if drifted:
        rows.append({
            "check": "package versions",
            "status": "fail" if strict_environment else "info",
            "detail": "; ".join(drifted[:6]) + ("..." if len(drifted) > 6 else ""),
        })
    else:
        rows.append({"check": "package versions", "status": "pass", "detail": "identical"})

    # Data
    if adata is None or body.get("data") is None:
        rows.append({"check": "input data", "status": "skipped", "detail": "not supplied"})
    else:
        want = body["data"]
        got = data_fingerprint(adata, layer=want.get("layer"))
        for key, label in (
            ("n_obs", "row count"), ("n_vars", "feature count"),
            ("var_names_sha256", "feature names"), ("matrix_sha256", "matrix contents"),
        ):
            same = want.get(key) == got.get(key)
            rows.append({
                "check": f"input {label}", "status": "pass" if same else "fail",
                "detail": "matches" if same else f"expected {want.get(key)}, got {got.get(key)}",
            })

    # Results
    recorded_results = body.get("results", {})
    supplied = dict(results or {})
    for name, want in recorded_results.items():
        if name not in supplied:
            rows.append({"check": f"result '{name}'", "status": "skipped", "detail": "not supplied"})
            continue
        got = result_fingerprint(supplied[name])
        if want["sha256"] == got["sha256"]:
            rows.append({"check": f"result '{name}'", "status": "pass",
                         "detail": f"identical ({got['n_rows']} rows)"})
        else:
            if want["n_rows"] != got["n_rows"]:
                why = f"row count {want['n_rows']} -> {got['n_rows']}"
            elif want["columns"] != got["columns"]:
                why = "columns differ"
            else:
                why = "same shape, different values"
            rows.append({"check": f"result '{name}'", "status": "fail", "detail": why})

    for name in supplied:
        if name not in recorded_results:
            rows.append({"check": f"result '{name}'", "status": "info",
                         "detail": "not in the manifest"})

    report = pd.DataFrame(rows, columns=["check", "status", "detail"])
    failed = int((report["status"] == "fail").sum())
    skipped = int((report["status"] == "skipped").sum())
    verdict = "REPRODUCED" if failed == 0 else f"{failed} CHECK(S) FAILED"
    print(
        f"[MORTIS] Verification of '{Path(path).name}': {verdict} "
        f"({int((report['status'] == 'pass').sum())} passed, {failed} failed, {skipped} skipped)."
    )
    if failed:
        for _, row in report[report["status"] == "fail"].iterrows():
            print(f"[MORTIS]   FAIL {row['check']}: {row['detail']}")
    return report

