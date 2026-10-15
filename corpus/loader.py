"""Shared FinePDFs parquet loading + quality filtering.

Both corpus generators read the same corpus the same way — detect the text/id
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
    try:
        values = np.asarray(scores, dtype=float)
        if not values.size or not np.isfinite(values).all():
            return 0.0
        return float(values.mean())
    except (TypeError, ValueError):
        return 0.0


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
    files, _, _, _ = schema(input_dir)
    if batch_size < 1 or (max_docs is not None and max_docs < 1):
        raise ValueError("batch_size and max_docs must be positive")

    def bump(key):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    kept = 0
    for f in files:
        parquet = pq.ParquetFile(f)
        cols = parquet.schema_arrow.names
        text_col, id_col = pick(cols, TEXT_CANDIDATES), pick(cols, ID_CANDIDATES)
        if text_col is None:
            raise ValueError(f"no text column in {f}")
        has_tok, has_edu, has_mh = (c in cols for c in
                                   ("token_count", "fw_edu_scores", "minhash_cluster_size"))
        read_cols = [text_col] + ([id_col] if id_col else [])
        read_cols += [c for c in ("token_count", "fw_edu_scores", "minhash_cluster_size") if c in cols]
        stem = os.path.relpath(f, input_dir)
        offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=read_cols):
            d = batch.to_pydict()
            t = d[text_col]
            ids = d[id_col] if id_col else [None] * len(t)
            tc = d.get("token_count", [None] * len(t))
            ed = d.get("fw_edu_scores", [None] * len(t))
            mh = d.get("minhash_cluster_size", [None] * len(t))
            for row in range(len(t)):
                bump("seen")
                txt = t[row]
                if not isinstance(txt, str) or not txt.strip():
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
                src = f"{stem}:{offset + row}"
                did = str(ids[row]) if ids[row] is not None else src
                yield did, src, (txt[:char_cap] if char_cap else txt), ntok
                kept += 1
                if max_docs and kept >= max_docs:
                    return
            offset += len(t)


def fetch_texts_by_id(input_dir, needed):
    """{doc_id -> text} for the ids in `needed` (a set), read in one streaming pass.

    Used to join cluster member texts back from the corpus by id.
    """
    needed = set(map(str, needed))
    if not needed:
        return {}
    texts = {}
    for did, _src, text, _tokens in iter_docs(
            input_dir, min_tokens=0, min_edu=0, drop_dups=False):
        if did in needed:
            if did in texts:
                raise ValueError(f"duplicate document id in corpus: {did!r}")
            texts[did] = text
    missing = needed - texts.keys()
    if missing:
        raise ValueError(f"{len(missing)} cluster document IDs missing from corpus")
    return texts
