"""A plain-language summary of an 8-K, written by an LLM on Groq, on request.

The dashboard's "What followed" panel quotes a filing word for word
(`src/live/excerpts.py`). Quotes are safe but can be hard going: a results
8-K's body often says only "see Exhibit 99.1", and legal prose is long. This
module adds a short summary in plain English, generated only when a user
clicks "Generate summary", and saved so the same filing is never sent twice.

It is DISPLAY ONLY, like the excerpts. The text it reads is the filing itself,
which exists only after t0, so nothing that detects, scores, trains or labels
may read it (`tests/test_live_summaries.py` checks that no such module imports
it). The labels stay the SEC item codes.

Where a summary lives: one JSON file per filing, `live.summaries.dir/
<accession>.json`, written atomically. A file per filing rather than a table
in the main database because the dashboard opens that database read-only,
and a JSON file can be read, checked, or deleted by hand. Deleting it is how
a summary is regenerated.

Usage:
  python -m src.live.summaries ACCESSION_NO      # generate (or show) one
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import requests

from src.live.excerpts import _HEAD, to_text
from src.utils.config import load_config
from src.utils.timeutils import utc_now_ts

SYSTEM_PROMPT = (
    "You summarise SEC Form 8-K filings for a research dashboard. Write plain "
    "English for a reader who is not a finance expert. Use only facts stated "
    "in the filing text you are given; if something is not stated, leave it "
    "out rather than guess. Give no investment advice, no view on where the "
    "stock will go, and no speculation about anyone's motives or trading. "
    "Format: one headline sentence, then 2 to 4 short bullet points with the "
    "key facts (who, what, when, and any amounts or figures the filing "
    "states). Keep it under 150 words."
)


class SummaryError(RuntimeError):
    """A summary that could not be made, with a reason a person can act on."""


def summary_path(cfg: dict, accession_no: str) -> Path:
    return Path(cfg["live"]["summaries"]["dir"]) / f"{accession_no}.json"


def load(cfg: dict, accession_no: str) -> dict | None:
    """The saved summary for one filing, or None if it was never generated."""
    path = summary_path(cfg, accession_no)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save(path: Path, record: dict) -> None:
    """Atomically, so a crash mid-write never leaves half a summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)


def _folder_url(cfg: dict, cik: str, accession_no: str) -> str:
    return (f"{cfg['edgar']['archives_base']}/data/{int(cik)}/"
            f"{accession_no.replace('-', '')}")


def exhibit_names(cfg: dict, client, cik: str, accession_no: str) -> list[str]:
    """Press-release exhibits in the filing's folder, by file name.

    A missing or unreadable folder listing is not an error: the summary is
    then made from the 8-K body alone.
    """
    scfg = cfg["live"]["summaries"]
    pattern = re.compile(scfg["exhibit_pattern"], re.I)
    try:
        listing = client.get_json(f"{_folder_url(cfg, cik, accession_no)}/index.json")
        names = [i["name"] for i in listing["directory"]["item"]]
    except Exception:
        return []
    picked = [n for n in sorted(names)
              if pattern.search(n) and n.lower().endswith((".htm", ".html", ".txt"))]
    return picked[:int(scfg["max_exhibits"])]


def filing_text(cfg: dict, client, cik: str, accession_no: str,
                primary_doc: str) -> tuple[str, list[str]]:
    """The 8-K body, then its press-release exhibits, cut to the input budget.

    Every document comes through the EDGAR client, so it is rate-limited,
    sent with our User-Agent, and cached under data/raw like every other SEC
    response. Returns the text and the URLs it came from.
    """
    if not primary_doc:
        raise SummaryError("no primary document on record for this filing")
    budget = int(cfg["live"]["summaries"]["max_input_chars"])
    folder = _folder_url(cfg, cik, accession_no)
    docs = [primary_doc] + [n for n in exhibit_names(cfg, client, cik, accession_no)
                            if n != primary_doc]
    parts, sources = [], []
    for k, name in enumerate(docs):
        url = f"{folder}/{name}"
        try:
            text = to_text(client.get_bytes(url)).strip()
        except Exception as exc:
            if k == 0:
                raise SummaryError(f"could not fetch the filing: {exc}") from exc
            continue                      # a missing exhibit is not fatal
        if k == 0 and (first := _HEAD.search(text)):
            # Skip the cover page — addresses, file numbers, checkboxes — and
            # spend the budget from the first "Item x.xx" on.
            text = text[first.start():]
        if not text:
            continue
        label = "8-K" if k == 0 else f"Exhibit ({name})"
        parts.append(f"--- {label} ---\n{text}")
        sources.append(url)
        if sum(len(p) for p in parts) >= budget:
            break
    text = "\n\n".join(parts)[:budget]
    # Rule 8: a document that parses to nothing must not become a summary of
    # nothing. An SEC error page is the usual cause.
    if len(text.split()) < 20:
        raise SummaryError("the filing has no readable text to summarise")
    return text, sources


def messages(cfg: dict, text: str, items: list[str], ticker: str | None) -> list[dict]:
    names = {str(k): v for k, v in (cfg["items"].get("names") or {}).items()}
    item_line = ", ".join(f"{c} ({names[c]})" if c in names else c for c in items)
    head = (f"Ticker: {ticker or 'unknown'}\n8-K items: {item_line or 'not listed'}"
            f"\n\nFiling text:\n<<<\n{text}\n>>>")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": head}]


def api_key(cfg: dict) -> str:
    name = cfg["live"]["summaries"]["api_key_env"]
    key = os.getenv(name, "").strip()
    if not key:
        raise SummaryError(f"{name} is not set — add it to .env (a free key "
                           f"from console.groq.com) and restart the dashboard")
    return key


def call_groq(cfg: dict, msgs: list[dict], session=None) -> str:
    """One chat completion from Groq's OpenAI-compatible endpoint."""
    scfg = cfg["live"]["summaries"]
    key = api_key(cfg)
    http = session or requests
    body = {"model": scfg["model"], "messages": msgs,
            "temperature": float(scfg["temperature"]),
            "max_tokens": int(scfg["max_output_tokens"])}
    if scfg.get("reasoning_effort"):
        body["reasoning_effort"] = scfg["reasoning_effort"]
    try:
        resp = http.post(
            scfg["api_url"], timeout=float(scfg["timeout_s"]),
            headers={"Authorization": f"Bearer {key}"},
            json=body)
    except requests.RequestException as exc:
        raise SummaryError(f"could not reach Groq: {exc}") from exc
    if resp.status_code == 401:
        raise SummaryError("Groq rejected the API key (HTTP 401)")
    if resp.status_code == 413:
        raise SummaryError("the filing is too long for Groq's per-minute limit "
                           "— lower live.summaries.max_input_chars in config")
    if resp.status_code == 429:
        wait = str((getattr(resp, "headers", None) or {}).get("retry-after", ""))
        when = f"{wait} seconds" if wait.isdigit() else "a minute"
        raise SummaryError(f"Groq rate limit reached — try again in {when}")
    if resp.status_code != 200:
        raise SummaryError(f"Groq returned HTTP {resp.status_code}: "
                           f"{resp.text[:200]}")
    try:
        out = resp.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise SummaryError("Groq's reply had no summary in it") from exc
    if not out:
        raise SummaryError("Groq returned an empty summary")
    return out


def generate(cfg: dict, accession_no: str, cik: str, primary_doc: str,
             items: list[str] | str | None, ticker: str | None = None,
             client=None, session=None, force: bool = False) -> dict:
    """The summary for one filing: from disk if saved, else made and saved."""
    if not force:
        saved = load(cfg, accession_no)
        if saved is not None:
            return saved
    api_key(cfg)               # before any SEC request: fail fast, fetch nothing
    if isinstance(items, str) or items is None:
        items = [c.strip() for c in str(items or "").split(",") if c.strip()]
    if client is None:
        from src.collectors.edgar import EdgarClient
        client = EdgarClient(cfg)
    text, sources = filing_text(cfg, client, cik, accession_no, primary_doc)
    summary = call_groq(cfg, messages(cfg, text, items, ticker), session=session)
    scfg = cfg["live"]["summaries"]
    record = {"accession_no": accession_no, "ticker": ticker, "items": items,
              "summary": summary, "model": scfg["model"], "provider": "groq",
              "prompt_version": int(scfg["prompt_version"]),
              "sources": sources, "input_chars": len(text),
              "created_utc": utc_now_ts()}
    _save(summary_path(cfg, accession_no), record)
    return record


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accession_no")
    ap.add_argument("--force", action="store_true", help="regenerate even if saved")
    args = ap.parse_args()
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"], readonly=True)
    row = conn.execute("SELECT cik, primary_doc, items, ticker FROM filings "
                       "WHERE accession_no = ?", (args.accession_no,)).fetchone()
    if row is None:
        raise SystemExit(f"{args.accession_no} is not in the filings table")
    rec = generate(cfg, args.accession_no, row[0], row[1], row[2], ticker=row[3],
                   force=args.force)
    print(rec["summary"])
    print(f"\n[{rec['provider']} {rec['model']} · saved {summary_path(cfg, args.accession_no)}]")


if __name__ == "__main__":
    main()
