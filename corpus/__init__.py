"""Phase 2 corpus tooling: shared FinePDFs loading + similarity clustering.

`loader` streams/filters the FinePDFs parquet corpus and is reused by both the
clustering stage (`corpus.cluster`) and the unshuffle task (`untie.knots`).
"""

from . import loader

__all__ = ["loader"]
