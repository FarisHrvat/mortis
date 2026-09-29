"""
Rebuild the drug-name vocabulary that filter_drugs() matches against.

The names come from Wikidata, which releases its content under CC0, so the
resulting database can be shipped inside the wheel without a data licence
attached. DrugBank's own downloads are CC BY-NC and cannot be redistributed
that way, which is why this is built from Wikidata instead. An item counts as
a drug here if Wikidata gives it a DrugBank or ATC identifier, or classifies
it under medication or pharmaceutical product.

Run it when the vocabulary needs refreshing. It takes a few minutes and writes
src/mortis/data/drug_names.db in place.

    python tools/build_drug_vocabulary.py
"""

import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
AGENT = "mortis-vocabulary-build/1.0 (https://github.com/FarisHrvat/mortis)"
OUT = Path(__file__).resolve().parent.parent / "src" / "mortis" / "data" / "drug_names.db"

# P715 is the DrugBank identifier, P267 the ATC code. Q12140 is medication and
# Q28885102 pharmaceutical product. Using the identifiers as a membership test
# keeps the selection factual; none of DrugBank's own content is copied.
SELECTOR = """
  { ?item wdt:P715 ?drugbank_id }
  UNION { ?item wdt:P267 ?atc_code }
  UNION { ?item wdt:P31/wdt:P279* wd:Q12140 }
  UNION { ?item wdt:P31/wdt:P279* wd:Q28885102 }
"""

PAGE = 4000


def ask(query: str, attempts: int = 4) -> list:
    """One SPARQL call, retried on the timeouts the public endpoint throws."""
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': query})}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": AGENT}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())["results"]["bindings"]
        except Exception as exc:
            if attempt == attempts - 1:
                raise SystemExit(f"Wikidata did not answer: {exc}")
            time.sleep(5 * (attempt + 1))
    return []


def collect(predicate: str, label: str) -> set:
    """Page through every English name reachable by one predicate."""
    found, offset = set(), 0
    while True:
        rows = ask(f"""
            SELECT DISTINCT ?name WHERE {{
              {SELECTOR}
              ?item {predicate} ?name .
              FILTER(LANG(?name) = "en")
            }}
            ORDER BY ?name
            LIMIT {PAGE} OFFSET {offset}
        """)
        if not rows:
            break
        found.update(r["name"]["value"] for r in rows)
        offset += PAGE
        print(f"  {label}: {len(found):,}", file=sys.stderr)
        if len(rows) < PAGE:
            break
    return found


def usable(name: str) -> bool:
    """Drop names too short or too generic to match a compound safely."""
    name = name.strip()
    return 3 < len(name) < 120 and not name.startswith("Q")


def main() -> None:
    names = {n.strip() for n in collect("rdfs:label", "names") if usable(n)}
    synonyms = {n.strip() for n in collect("skos:altLabel", "synonyms") if usable(n)}
    synonyms -= names

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    conn = sqlite3.connect(OUT)
    conn.executescript("""
        CREATE TABLE drug_compounds (name TEXT PRIMARY KEY);
        CREATE TABLE drug_synonyms  (synonym TEXT PRIMARY KEY);
    """)
    # Sorted so a rebuild from unchanged input gives a byte-identical file.
    conn.executemany("INSERT INTO drug_compounds VALUES (?)", [(n,) for n in sorted(names)])
    conn.executemany("INSERT INTO drug_synonyms VALUES (?)", [(s,) for s in sorted(synonyms)])
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    print(f"{OUT}: {len(names):,} names, {len(synonyms):,} synonyms, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
