#!/usr/bin/env python3
"""Copy the final result tables into the committed `results/` folder.

`data/` is never committed (AGENTS.md), which left a fresh clone with an empty
Evaluation screen: the dashboard reads the Phase 10 table and the Phase 8 pair
from `data/processed/`, and nobody else has them. This copies exactly the files
listed under `results_export` in config into `paths.results`, and writes
`MANIFEST.md` beside them with each file's size, SHA-256 and source path, so a
reader can check the committed copy against the original byte for byte.

A listed file that is missing stops the export rather than shipping a partial
set: an Evaluation screen that silently falls back to an older table is worse
than one that says nothing is there.

Usage:
  .venv/bin/python scripts/export_results.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config              # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(cfg: dict) -> list[tuple[str, int, str]]:
    src_root = REPO / cfg["paths"]["processed"]
    dst_root = REPO / cfg["paths"]["results"]
    names = list(cfg["results_export"])
    missing = [n for n in names if not (src_root / n).exists()]
    if missing:
        raise SystemExit(f"not exporting: {len(missing)} listed result file(s) "
                         f"missing under {src_root}: {', '.join(missing)}")
    done = []
    for name in names:
        src, dst = src_root / name, dst_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        done.append((name, src.stat().st_size, sha256(dst)))
    lines = ["# Results — committed copies of the final tables", "",
             "Copied from `" + cfg["paths"]["processed"] + "/` by "
             "`scripts/export_results.py`. The dashboard reads these first. "
             "Regenerate them with the commands in `README.md`, never by "
             "editing a copy here.", "",
             "| file | bytes | sha256 |", "|---|---|---|"]
    lines += [f"| `{n}` | {size:,} | `{digest}` |" for n, size, digest in done]
    (dst_root / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    return done


if __name__ == "__main__":
    for name, size, digest in export(load_config()):
        print(f"{size:>9,}  {digest[:12]}  {name}")
