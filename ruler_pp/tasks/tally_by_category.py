"""Histogram: count records per category (a multi-value answer)."""

from __future__ import annotations

from collections import Counter

from ..base import Sample, Task
from .. import wordbank


class TallyByCategory(Task):
    name = "tally_by_category"
    reserve_tokens = 200
    min_size = 10

    def build(self, size, rng):
        recs = wordbank.make_records(size, rng)
        context = wordbank.render_records(recs)

        counts = Counter(r.category for r in recs)
        cats = sorted(counts)
        gold = {c: counts[c] for c in cats}
        answers = [f"{c}: {counts[c]}" for c in cats]

        instruction = "Below is a list of records, each tagged with a category."
        question = (
            "Question: How many records fall into each category? List every "
            "category that appears together with its count."
        )
        return Sample(
            instruction, context, question,
            answer_prefix="",
            answers=answers, gold=gold, answer_type="map",
            meta={"num_categories": len(cats)},
        )
