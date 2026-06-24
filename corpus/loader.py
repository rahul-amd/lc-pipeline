"""Shared FinePDFs parquet loading + quality filtering (Phase 2).

Both Phase-2 stages read the same corpus the same way — detect the text/id
columns, stream row-group batches (a single FinePDFs file's text column is
multi-GB, so we never materialise a whole file), and drop low-value rows using
the metadata FinePDFs already ships:

  * short docs      : token_count < min_tokens
  * near-duplicates : minhash_cluster_size > 1   (when drop_dups)
  * low quality     : mean(fw_edu_scores) < min_edu

`cluster.py` collects the survivors into arrays to embed; `untie/knots.py`
consumes them lazily. Both go through `iter_docs` so the filtering stays in one
place.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq

TEXT_CANDIDATES = ["text", "content", "raw_content", "document", "body"]
ID_CANDIDATES = ["id", "doc_id", "url", "_id"]


def pick(colnames, candidates):
    """First of `candidates` present in `colnames`, else None."""
    for c in candidates:
        if c in colnames:
            return c
    return None


def doc_mean(scores):
    """Mean of a per-page score list; tolerant of scalars / None."""
    if scores is None:
        return 0.0
    if isinstance(scores, (list, tuple)):
        return float(np.mean(scores)) if len(scores) else 0.0
    return float(scores)


def schema(input_dir):
    """(parquet_files, columns, text_col, id_col) for `input_dir`.

    Raises SystemExit if there are no parquet files or no recognisable text
    column. `id_col` may be None.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {input_dir!r}")
    cols = pq.ParquetFile(files[0]).schema_arrow.names
    text_col = pick(cols, TEXT_CANDIDATES)
    id_col = pick(cols, ID_CANDIDATES)
    if text_col is None:
        raise SystemExit(f"no text column found; columns are {cols}")
    return files, cols, text_col, id_col


def iter_docs(input_dir, *, min_tokens=200, min_edu=0.75, drop_dups=True,
              char_cap=None, batch_size=4096, max_docs=None, stats=None):
    """Stream `(doc_id, source_tag, text, n_tokens)` for quality-filtered docs.

    `char_cap` clips each text at load time (lossless for callers that only look
    at a prefix, and essential to keep RAM bounded for 500k+ rows). `stats`, if
    given, is a dict-like counter incremented with seen/empty/short/dup/lowedu/
    kept so callers can report drop reasons. `source_tag` is always
    "<file>:<row>" (stable provenance); `doc_id` falls back to it when the
    corpus has no id column.
    """
    files, cols, text_col, id_col = schema(input_dir)
    has_tok = "token_count" in cols
    has_edu = "fw_edu_scores" in cols
    has_mh = "minhash_cluster_size" in cols

    read_cols = [text_col] + ([id_col] if id_col else [])
    for c in ("token_count", "fw_edu_scores", "minhash_cluster_size"):
        if c in cols:
            read_cols.append(c)

    def bump(key):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    kept = 0
    for f in files:
        stem = os.path.basename(f)
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_size, columns=read_cols):
            d = batch.to_pydict()
            t = d[text_col]
            ids = d[id_col] if id_col else [None] * len(t)
            tc = d.get("token_count", [None] * len(t))
            ed = d.get("fw_edu_scores", [None] * len(t))
            mh = d.get("minhash_cluster_size", [None] * len(t))
            for row in range(len(t)):
                bump("seen")
                txt = t[row]
                if not txt:
                    bump("empty")
                    continue
                ntok = int(tc[row]) if tc[row] is not None else 0
                if has_tok and ntok < min_tokens:
                    bump("short")
                    continue
                if drop_dups and has_mh and mh[row] is not None and int(mh[row]) > 1:
                    bump("dup")
                    continue
                if has_edu and doc_mean(ed[row]) < min_edu:
                    bump("lowedu")
                    continue
                bump("kept")
                src = f"{stem}:{row}"
                did = str(ids[row]) if ids[row] is not None else src
                yield did, src, (txt[:char_cap] if char_cap else txt), ntok
                kept += 1
                if max_docs and kept >= max_docs:
                    return


def fetch_texts_by_id(input_dir, needed):
    """{doc_id -> text} for the ids in `needed` (a set), read in one streaming pass.

    Used to join cluster member texts back from the corpus by id.
    """
    files, _cols, text_col, id_col = schema(input_dir)
    if id_col is None:
        raise SystemExit("this corpus has no id column to join member texts")
    texts = {}
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=[text_col, id_col]):
            d = batch.to_pydict()
            for did, txt in zip(d[id_col], d[text_col]):
                sid = str(did)
                if sid in needed and txt:
                    texts[sid] = txt
            if len(texts) >= len(needed):
                return texts
    return texts
