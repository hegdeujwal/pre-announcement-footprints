"""What followed an alert, in the filer's own words.

When a live alert is graded as followed by an 8-K, the dashboard can say only
"5.02 director or officer change" from the item codes, and news is not
collected for the live period. This module fetches that filing once, from the
SEC, and keeps the opening sentences of each substantive item, quoted word for
word: "On September 29, 2026, Ynon Kreiz resigned as Chief Executive Officer…".

It is DISPLAY ONLY. Nothing here scores, grades or trains anything; the
excerpt is attached to an alert after its outcome is already decided. And it
is extractive on purpose: a quotation cannot misstate the filing, where a
written summary could.

Three choices worth knowing:

  - The item's official SEC title is stripped from the start of its text
    (`items.titles`), so a heading is never mistaken for the announcement.
  - Sentences of legal furnishing language and cross-references
    (`live.excerpts.boilerplate`) are skipped: they say nothing about the
    event. An item left with nothing but boilerplate gets no excerpt.
  - A filing whose document cannot be fetched or parsed is recorded with no
    excerpt and a reason, so it is not fetched again every night, and the
    dashboard falls back to the item labels and the sec.gov link.

The record is committed as live-log/filing_excerpts.csv beside the alert log,
so a fresh clone shows the same quotations.

Usage:
  python -m src.live.excerpts            # fetch what is missing, export
"""

from __future__ import annotations

import argparse
import html
import logging
import re
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.timeutils import utc_now_ts

log = logging.getLogger(__name__)

#: A block-level HTML tag: where a line break belongs in the plain text.
_BLOCK = re.compile(r"(?i)<\s*(br|/p|/div|/tr|/li|/h\d|/table)\b[^>]*>")
#: An item heading at the start of a line: "Item 5.02.", "ITEM 5.02 Departure…"
_HEAD = re.compile(r"(?im)^\s*item\s*(\d{1,2}\.\d{2})\b\.?")
#: Where an item's text ends besides the next item: a "Section 9 — …" heading
#: or the signature block.
_STOP = re.compile(r"(?im)^\s*(section\s+\d+\b|signatures?\s*$)")
#: Abbreviations a sentence does not end on.
_ABBREV = {"inc", "corp", "co", "ltd", "llc", "l.p", "lp", "n.a", "mr", "ms",
           "mrs", "dr", "no", "u.s", "jr", "sr", "st", "plc", "s.a", "n.v",
           "vs", "approx", "e.g", "i.e"}
#: The record's columns, in file order.
COLUMNS = ("accession_no", "cik", "item", "excerpt", "fetched_utc", "note")


def to_text(body: bytes) -> str:
    """A filing's HTML as plain text, one block per line."""
    s = body.decode("utf-8", "ignore")
    s = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", s)
    s = _BLOCK.sub("\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r"\n\s*", "\n", s)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def strip_title(section: str, title: str | None) -> str:
    """Remove the item's official title from the start of its text.

    Matched word by word, tolerant of the small variations filers introduce
    (a dropped "an", a semicolon for a full stop). Only a match covering most
    of the title is removed, so a section that does not open with it is left
    untouched.
    """
    if not title:
        return section
    want = _words(title)
    tokens = list(re.finditer(r"[A-Za-z0-9']+", section))
    i = j = 0
    while i < len(tokens) and j < len(want):
        w = tokens[i].group(0).lower()
        ahead = want[j:j + 3]
        if w not in ahead:
            break
        j += ahead.index(w) + 1
        i += 1
    if i == 0 or j < 0.6 * len(want):
        return section
    cut = tokens[i - 1].end()
    return section[cut:].lstrip(" .:;-–—\n")


def sentences(text: str) -> list[str]:
    """Split prose into sentences, not breaking after "Inc." or "Mr."."""
    text = re.sub(r"\s+", " ", text).strip()
    out, start = [], 0
    for m in re.finditer(r"[.!?][\"”’)]?\s+(?=[A-Z\"“(])", text):
        before = text[start:m.start()].rsplit(" ", 1)[-1].lower().strip("(\"“")
        if before.rstrip(".") in _ABBREV or re.fullmatch(r"[a-z]", before):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def excerpt_items(body: bytes, items: list[str], cfg: dict) -> dict[str, str]:
    """The opening sentences of each substantive item, word for word."""
    ecfg = cfg["live"]["excerpts"]
    skip = {str(c) for c in ecfg["skip_items"]}
    titles = {str(k): v for k, v in (cfg["items"].get("titles") or {}).items()}
    junk = [re.compile(p, re.I) for p in ecfg["boilerplate"]]
    limit = int(ecfg["max_chars"])
    text = to_text(body)
    heads = list(_HEAD.finditer(text))
    out: dict[str, str] = {}
    for k, m in enumerate(heads):
        code = m.group(1)
        if code in skip or code in out or code not in items:
            continue
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        raw = text[m.end():end]
        stop = _STOP.search(raw)
        section = strip_title(raw[:stop.start()] if stop else raw, titles.get(code))
        title_words = set(_words(titles.get(code, "")))
        kept, size = [], 0
        for sent in sentences(section):
            if any(p.search(sent) for p in junk) or len(sent) < 25:
                continue
            # A heading that survived as a "sentence" (filers vary the
            # official wording slightly): mostly the title's own words.
            w = _words(sent)
            if (title_words and len(sent) < 220
                    and sum(x in title_words for x in w) >= 0.7 * len(w)):
                continue
            if kept and size + len(sent) > limit:
                break
            kept.append(sent)
            size += len(sent) + 1
            if size >= limit:
                break
        if not kept:
            continue
        quote = " ".join(kept)
        if len(quote) > limit:
            quote = quote[:limit].rsplit(" ", 1)[0] + "…"
        out[code] = quote
    return out


def _followed(conn, cfg: dict, extra: list[str] | None = None) -> pd.DataFrame:
    """Every filing an alert was graded against, with what is needed to fetch it.

    `extra` adds accessions from elsewhere — the committed outcomes file, when
    this database's own grades lag it.
    """
    df = pd.read_sql(
        "SELECT DISTINCT o.accession_no, f.cik, f.primary_doc, f.items "
        "FROM alert_outcomes o JOIN filings f ON f.accession_no = o.accession_no "
        "WHERE o.filed = 1 AND o.accession_no IS NOT NULL", conn)
    if extra:
        marks = ",".join("?" * len(extra))
        more = pd.read_sql(
            f"SELECT accession_no, cik, primary_doc, items FROM filings "
            f"WHERE accession_no IN ({marks})", conn, params=list(extra))
        df = pd.concat([df, more]).drop_duplicates("accession_no")
    return df


def backfill(cfg: dict, conn, client=None, have: set[str] | None = None,
             limit: int | None = None, extra: list[str] | None = None) -> dict:
    """Fetch and excerpt every followed filing not yet on record.

    `have` is the set of accessions already on record (the committed file and
    the database); those are never fetched again.
    """
    from src.collectors.edgar import EdgarClient

    have = set(have or ()) | {r[0] for r in conn.execute(
        "SELECT DISTINCT accession_no FROM filing_excerpts")}
    todo = _followed(conn, cfg, extra)
    todo = todo[~todo["accession_no"].isin(have)]
    limit = int(limit if limit is not None else cfg["live"]["excerpts"]["batch"])
    todo = todo.head(limit)
    if todo.empty:
        return {"fetched": 0, "quoted": 0, "empty": 0, "failed": 0}
    client = client or EdgarClient(cfg)
    base = cfg["edgar"]["archives_base"]
    now = utc_now_ts()
    fetched = quoted = empty = failed = 0
    for r in todo.itertuples():
        items = [c.strip() for c in str(r.items or "").split(",") if c.strip()]
        rows: list[tuple] = []
        try:
            if not r.primary_doc:
                raise ValueError("no primary document on record")
            url = (f"{base}/data/{int(r.cik)}/{r.accession_no.replace('-', '')}/"
                   f"{r.primary_doc}")
            found = excerpt_items(client.get_bytes(url), items, cfg)
            fetched += 1
            rows = [(r.accession_no, str(r.cik), code, text, now, "")
                    for code, text in found.items()]
            if rows:
                quoted += 1
            else:
                empty += 1
                rows = [(r.accession_no, str(r.cik), "", "", now,
                         "no quotable item text in the filing")]
        except Exception as exc:                 # one bad filing must not stop the rest
            failed += 1
            log.warning("excerpts: %s failed: %s", r.accession_no, exc)
            rows = [(r.accession_no, str(r.cik), "", "", now,
                     f"not fetched: {type(exc).__name__}")]
        conn.executemany(
            "INSERT OR REPLACE INTO filing_excerpts (accession_no, cik, item, "
            "excerpt, fetched_utc, note) VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    if fetched == 0 and failed:
        raise SystemExit(f"excerpts: every one of {failed} filings failed to "
                         f"fetch — EDGAR is not answering.")
    return {"fetched": fetched, "quoted": quoted, "empty": empty, "failed": failed}


def read_csv(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.read_csv(path, dtype={"cik": "string", "item": "string",
                                    "excerpt": "string", "note": "string"},
                       keep_default_na=False)


def import_csv(conn, path) -> int:
    """Load the committed record into the database, so nothing is refetched."""
    df = read_csv(path)
    conn.executemany(
        "INSERT OR IGNORE INTO filing_excerpts (accession_no, cik, item, "
        "excerpt, fetched_utc, note) VALUES (?, ?, ?, ?, ?, ?)",
        [tuple(r) for r in df[list(COLUMNS)].itertuples(index=False)])
    conn.commit()
    return len(df)


def export_csv(conn, path) -> int:
    """Write the record, merged with the file, so it never loses a row."""
    path = Path(path)
    now = pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM filing_excerpts", conn)
    merged = pd.concat([read_csv(path), now], ignore_index=True)
    merged = (merged.drop_duplicates(["accession_no", "item"], keep="last")
                    .sort_values(["accession_no", "item"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    merged[list(COLUMNS)].to_csv(path, index=False)
    return int(len(merged))


def main() -> None:
    from src import db
    from src.live.catchup import DEFAULT_EXCERPTS_CSV

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_EXCERPTS_CSV)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    import_csv(conn, args.csv)
    from src.live.catchup import DEFAULT_OUTCOMES_CSV
    extra = []
    if Path(DEFAULT_OUTCOMES_CSV).exists():
        o = pd.read_csv(DEFAULT_OUTCOMES_CSV, dtype={"accession_no": str})
        extra = sorted(o.loc[o["filed"] == 1, "accession_no"].dropna().unique())
    print(backfill(cfg, conn, limit=args.limit, extra=extra))
    print("exported", export_csv(conn, args.csv), "rows ->", args.csv)


if __name__ == "__main__":
    main()
