"""
MORTIS Compound Identifier & Pathway Module
===========================================
Turning a list of compound names into pathway results, in one step instead of
three manual ones.

The manual version of this is: paste names into MetaboAnalyst's name-mapping
tool, download the ID table, work out which pathways those IDs belong to, then
run enrichment somewhere else. Every step is a browser round-trip and none of it
is reproducible six months later. :func:`annotate_pathways` does all of it from
the compound names already in ``adata.var_names``.

Where the pieces come from
--------------------------
**Names to identifiers** — MetaboAnalyst's public REST endpoint
(``rest.xialab.ca/api/mapcompounds``). It returns HMDB, KEGG, PubChem, ChEBI,
METLIN and SMILES for each name, and flags what it could not resolve. This is
the same service the web tool uses.

**Identifiers to pathways** — the KEGG REST API (``rest.kegg.jp``). One call
returns every compound-to-pathway link KEGG holds (about 19,600 of them) and a
second returns the pathway names.

**Enrichment** — done here, by :func:`mortis.pathway_ora`, not remotely. This
was not the original plan. MetaboAnalyst documents exactly one REST endpoint,
the name mapper; there is no public enrichment endpoint (six candidate paths
all return 404). Keeping the statistics local turns out to be the better
outcome anyway, because the background set matters enormously and a web tool
cannot know yours. Enrichment must be tested against **the compounds you
measured**, not against every compound in the database — otherwise a panel that
only detects polar metabolites looks dramatically enriched for polar pathways,
purely because of what the instrument could see. ``pathway_ora`` uses the
measured panel as background.

Caching and courtesy
--------------------
Both services are free and neither asks for a key, which is worth not abusing.
Every response is cached on disk (default ``~/.cache/mortis``), so a re-run
costs nothing and a reviewer re-running your analysis does not hammer someone
else's server. Requests are batched, retried with backoff on transient
failures, and never issued at all when the cache already answers.

KEGG's terms permit academic use; commercial users need a licence from
Pathway Solutions. MetaboAnalyst asks that you cite it, which
:func:`annotate_pathways` prints on first use.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import pandas as pd

from .annotate import pathway_ora
from .exceptions import InvalidParameterError, MortisError

__all__ = [
    "map_compound_ids",
    "fetch_kegg_pathway_sets",
    "annotate_pathways",
    "clear_cache",
]

_METABOANALYST_URL = "https://rest.xialab.ca/api/mapcompounds"
_KEGG_LINK_URL = "https://rest.kegg.jp/link/pathway/compound"
_KEGG_NAMES_URL = "https://rest.kegg.jp/list/pathway/map"
#: KEGG's own classification of every pathway map, one call.
_KEGG_BRITE_URL = "https://rest.kegg.jp/get/br:br08901"

#: Names are sent in batches. Large enough to keep the number of round-trips
#: down, small enough that one unlucky request does not lose much work.
_BATCH_SIZE = 200

_CITATION = (
    "[MORTIS] Compound mapping by MetaboAnalyst (Pang et al., Nucleic Acids Res 2024); "
    "pathway membership from KEGG (Kanehisa & Goto, Nucleic Acids Res 2000). "
    "Please cite both. KEGG is free for academic use; commercial use needs a licence."
)


class NetworkError(MortisError):
    """A remote service could not be reached or returned something unusable."""


def _cache_dir(cache: Optional[str]) -> Optional[Path]:
    if cache is False:  # explicit opt-out
        return None
    path = Path(cache) if cache else Path.home() / ".cache" / "mortis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached_read(directory: Optional[Path], name: str) -> Optional[str]:
    if directory is None:
        return None
    path = directory / name
    return path.read_text() if path.exists() else None


def _cached_write(directory: Optional[Path], name: str, payload: str) -> None:
    if directory is not None:
        (directory / name).write_text(payload)


def _request(url: str, body: Optional[dict], timeout: float, retries: int) -> str:
    """One HTTP call with backoff. Raises NetworkError rather than leaking urllib."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}

    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise NetworkError(
        f"Could not reach {url} after {retries + 1} attempt(s): {last}. "
        "Check the connection, or pass a previously cached result. Nothing was "
        "computed from partial data."
    )


def map_compound_ids(
    names: Sequence[str],
    cache: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
) -> pd.DataFrame:
    """
    Resolve compound names to database identifiers via MetaboAnalyst.

    Parameters
    ----------
    names : sequence of str
        Compound names, typically ``adata.var_names``.
    cache : str or None or False
        Directory for cached responses. ``None`` uses ``~/.cache/mortis``;
        ``False`` disables caching entirely.
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Retries on transient network failure, with backoff.

    Returns
    -------
    pandas.DataFrame
        One row per input name, with columns ``query``, ``match``, ``hmdb``,
        ``kegg``, ``pubchem``, ``chebi``, ``metlin``, ``smiles`` and
        ``matched``. Unresolved names keep their row with ``matched=False``
        and ``NA`` identifiers — dropping them silently would make the
        match rate invisible, and the match rate is the first thing to check.

    Examples
    --------
    >>> ids = mt.map_compound_ids(adata.var_names)
    >>> ids["matched"].mean()
    0.71
    """
    names = [str(n) for n in names]
    if not names:
        raise InvalidParameterError("names must be a non-empty sequence.")

    directory = _cache_dir(cache)
    frames = []
    for start in range(0, len(names), _BATCH_SIZE):
        batch = names[start:start + _BATCH_SIZE]
        key = f"mapcompounds_{abs(hash(tuple(batch)))}.json"

        payload = _cached_read(directory, key)
        if payload is None:
            payload = _request(
                _METABOANALYST_URL,
                {"queryList": ";".join(batch) + ";", "inputType": "name"},
                timeout, retries,
            )
            _cached_write(directory, key, payload)

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise NetworkError(
                f"MetaboAnalyst returned something that is not JSON: {payload[:200]!r}"
            ) from exc

        frames.append(pd.DataFrame({
            "query": parsed.get("Query", batch),
            "match": parsed.get("Match", ["NA"] * len(batch)),
            "hmdb": parsed.get("HMDB", ["NA"] * len(batch)),
            "kegg": parsed.get("KEGG", ["NA"] * len(batch)),
            "pubchem": parsed.get("PubChem", ["NA"] * len(batch)),
            "chebi": parsed.get("ChEBI", ["NA"] * len(batch)),
            "metlin": parsed.get("METLIN", ["NA"] * len(batch)),
            "smiles": parsed.get("SMILES", ["NA"] * len(batch)),
        }))

    table = pd.concat(frames, ignore_index=True)
    table["matched"] = table["hmdb"].astype(str).ne("NA") | table["kegg"].astype(str).ne("NA")
    print(
        f"[MORTIS] Compound mapping: {int(table['matched'].sum())}/{len(table)} names resolved "
        f"({table['kegg'].astype(str).ne('NA').sum()} with a KEGG ID)."
    )
    return table


def fetch_kegg_pathway_sets(
    kegg_ids: Optional[Sequence[str]] = None,
    cache: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
    metabolic_only: bool = True,
) -> Dict[str, List[str]]:
    """
    Build ``{pathway_name: [KEGG compound ids]}`` from the KEGG REST API.

    Parameters
    ----------
    kegg_ids : sequence of str, optional
        Restrict to pathways containing at least one of these compounds.
        Default ``None`` returns every pathway.
    cache : str or None or False
        Cache directory, as in :func:`map_compound_ids`.
    timeout, retries : float, int
        Network behaviour.
    metabolic_only : bool
        Keep only pathways KEGG classifies under **Metabolism**, and drop its
        "Global and overview maps". Default ``True``, and worth leaving on.

        Without it the top hits fill up with maps that contain many metabolites
        without being metabolic pathways — "ABC transporters" (membrane
        transport), "Protein digestion and absorption" (digestive system),
        "Aminoacyl-tRNA biosynthesis" (translation). All three came out
        significant on a 21-compound test panel and none of them means what a
        reader would take it to mean. The global maps are excluded for the
        opposite reason: "Metabolic pathways" spans thousands of compounds, so
        it is enriched for almost any input and says nothing.

        The classification comes from KEGG's own BRITE hierarchy rather than a
        hardcoded list, so it stays correct as KEGG changes.

    Returns
    -------
    dict
        Pathway name to KEGG compound IDs.
    """
    directory = _cache_dir(cache)

    links = _cached_read(directory, "kegg_compound_pathway.tsv")
    if links is None:
        links = _request(_KEGG_LINK_URL, None, timeout, retries)
        _cached_write(directory, "kegg_compound_pathway.tsv", links)

    names = _cached_read(directory, "kegg_pathway_names.tsv")
    if names is None:
        names = _request(_KEGG_NAMES_URL, None, timeout, retries)
        _cached_write(directory, "kegg_pathway_names.tsv", names)

    pathway_names: Dict[str, str] = {}
    for line in names.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            pathway_names[parts[0].strip()] = parts[1].strip()

    keep: Optional[set] = None
    if metabolic_only:
        brite = _cached_read(directory, "kegg_brite_pathway_classes.txt")
        if brite is None:
            brite = _request(_KEGG_BRITE_URL, None, timeout, retries)
            _cached_write(directory, "kegg_brite_pathway_classes.txt", brite)
        keep = _metabolic_pathway_ids(brite)

    wanted = {str(k).upper() for k in kegg_ids} if kegg_ids is not None else None
    sets: Dict[str, List[str]] = {}
    for line in links.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        compound = parts[0].replace("cpd:", "").strip()
        pathway = parts[1].replace("path:", "").strip()
        if not pathway.startswith("map"):
            continue
        if keep is not None and pathway not in keep:
            continue
        if wanted is not None and compound.upper() not in wanted:
            continue
        sets.setdefault(pathway_names.get(pathway, pathway), []).append(compound)

    print(f"[MORTIS] KEGG: {len(sets)} pathway(s) assembled.")
    return sets


def _metabolic_pathway_ids(brite: str) -> set:
    """
    Parse KEGG's BRITE pathway hierarchy into the set of genuinely metabolic
    map IDs.

    The file is a flat indented listing: ``A`` lines are top categories
    ("Metabolism", "Genetic Information Processing", ...), ``B`` lines are
    subcategories ("Carbohydrate metabolism", "Global and overview maps"), and
    ``C`` lines are the maps themselves. Tracking the most recent A and B while
    walking the file is enough to classify every map.
    """
    keep: set = set()
    top = sub = ""
    for line in brite.splitlines():
        if not line:
            continue
        marker, rest = line[0], line[1:].strip()
        if marker == "A":
            top, sub = rest, ""
        elif marker == "B":
            sub = rest
        elif marker == "C":
            parts = rest.split(None, 1)
            if not parts:
                continue
            if top.lower().startswith("metabolism") and "global and overview" not in sub.lower():
                keep.add(f"map{parts[0]}")
    return keep


def annotate_pathways(
    adata: ad.AnnData,
    result: pd.DataFrame,
    cache: Optional[str] = None,
    delta_threshold: float = 0.474,
    fdr_threshold: Optional[float] = None,
    direction: bool = True,
    min_size: int = 3,
    key_added: str = "compound_ids",
    timeout: float = 60.0,
    retries: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Name to identifier to pathway enrichment, in one call.

    Combines the two lookups a user would otherwise do by hand — resolving
    compound names to KEGG IDs, then finding which pathways those belong to —
    and runs :func:`mortis.pathway_ora` on the result with the measured panel
    as background.

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose ``var_names`` are compound names. Identifiers are written
        back to ``adata.var`` under ``{key_added}_hmdb`` and ``{key_added}_kegg``.
    result : pandas.DataFrame
        Output of :func:`mortis.differential_abundance` or
        :func:`mortis.differential_spatial_organization`.
    cache : str or None or False
        Cache directory. ``None`` uses ``~/.cache/mortis``.
    delta_threshold, fdr_threshold, direction, min_size
        Passed through to :func:`mortis.pathway_ora`.
    key_added : str
        Prefix for the identifier columns added to ``adata.var``.
    timeout, retries : float, int
        Network behaviour.

    Returns
    -------
    (identifiers, enrichment)
        The full ID mapping table, and the pathway ORA report. The mapping is
        returned rather than hidden so the match rate stays visible — a
        pathway result computed from 30% of a panel needs reading differently
        from one computed from 90%.

    Raises
    ------
    NetworkError
        If either service is unreachable. Nothing partial is returned.

    Examples
    --------
    >>> res = mt.differential_abundance(pb, "response", "R", "NR")
    >>> ids, paths = mt.annotate_pathways(adata, res)
    >>> mt.plot_pathway_dotplot(paths)
    """
    if "metabolite" not in result.columns:
        raise InvalidParameterError("'result' must have a 'metabolite' column.")

    print(_CITATION)
    identifiers = map_compound_ids(
        adata.var_names, cache=cache, timeout=timeout, retries=retries
    )

    lookup = dict(zip(identifiers["query"].astype(str), identifiers["hmdb"].astype(str)))
    adata.var[f"{key_added}_hmdb"] = [
        lookup.get(str(n), "NA") for n in adata.var_names
    ]
    kegg_lookup = dict(zip(identifiers["query"].astype(str), identifiers["kegg"].astype(str)))
    adata.var[f"{key_added}_kegg"] = [
        kegg_lookup.get(str(n), "NA") for n in adata.var_names
    ]

    resolved = identifiers[identifiers["kegg"].astype(str).ne("NA")]
    if resolved.empty:
        raise MortisError(
            "No compound name resolved to a KEGG identifier, so no pathway analysis is "
            "possible. Check that adata.var_names are compound names rather than m/z "
            "values, and inspect the returned mapping table."
        )

    kegg_sets = fetch_kegg_pathway_sets(
        resolved["kegg"].astype(str).tolist(), cache=cache, timeout=timeout, retries=retries
    )

    # pathway_ora matches on compound name, so translate the KEGG sets back
    # into the names used in `result`.
    name_for_kegg: Dict[str, List[str]] = {}
    for query, kegg in zip(resolved["query"].astype(str), resolved["kegg"].astype(str)):
        name_for_kegg.setdefault(kegg.upper(), []).append(query)

    named_sets = {
        pathway: [n for cid in compounds for n in name_for_kegg.get(cid.upper(), [])]
        for pathway, compounds in kegg_sets.items()
    }
    named_sets = {p: c for p, c in named_sets.items() if len(c) >= min_size}
    if not named_sets:
        warnings.warn(
            f"[MORTIS] No KEGG pathway had at least min_size={min_size} measured compounds. "
            "The panel may be too small or too specialised for pathway-level analysis.",
            stacklevel=2,
        )
        return identifiers, pathway_ora(
            result, {"__none__": []}, min_size=10 ** 9, direction=direction
        )

    enrichment = pathway_ora(
        result, named_sets, delta_threshold=delta_threshold,
        fdr_threshold=fdr_threshold, direction=direction, min_size=min_size,
    )
    return identifiers, enrichment


def clear_cache(cache: Optional[str] = None) -> int:
    """
    Delete cached API responses. Returns how many files were removed.

    Worth doing when KEGG has been updated and you want the current release
    rather than whatever was current when the analysis first ran. For a
    finished analysis, keeping the cache is the reproducible choice.
    """
    directory = _cache_dir(cache)
    if directory is None:
        return 0
    removed = 0
    for path in directory.glob("*"):
        if path.is_file():
            path.unlink()
            removed += 1
    print(f"[MORTIS] Cleared {removed} cached response(s) from {directory}.")
    return removed
