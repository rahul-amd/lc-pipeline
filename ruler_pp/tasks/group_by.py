"""SQL-style GROUP BY: sum of values per category."""

from __future__ import annotations

from collections import defaultdict

from ..base import Sample, Task
from .. import wordbank


class GroupBy(Task):
    name = "group_by"
    reserve_tokens = 200
    min_size = 10

    def build(self, size, rng):
        recs = wordbank.make_records(size, rng)
        context = wordbank.render_records(recs)

        totals: dict[str, int] = defaultdict(int)
        for r in recs:
            totals[r.category] += r.value
        cats = sorted(totals)
        gold = {c: totals[c] for c in cats}
        answers = [f"{c}: {totals[c]}" for c in cats]

        instruction = (
            "Below is a list of records, each with a category and a numeric value."
        )
        question = (
            "Question: For each category, what is the sum of values across all "
            "records in that category?"
        )
        return Sample(
            instruction, context, question,
            answer_prefix="",
            answers=answers, gold=gold, answer_type="map",
            meta={"num_categories": len(cats)},
        )
