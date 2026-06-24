"""Untie the Knots (Phase 2b): paragraph-unshuffle long-context datagen.

`knots` builds the samples (a `ruler_pp` CorpusTask); `viz` is a Gradio viewer
for the generated jsonl. Run generation with `python -m untie`.
"""

from .knots import UntieKnots, make_units

__all__ = ["UntieKnots", "make_units"]
