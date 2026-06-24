"""Count records satisfying a predicate (>, <, ==category, range)."""

from __future__ import annotations

from ..base import Sample, Task
from .. import wordbank


class CountPredicate(Task):
    name = "count_predicate"
    reserve_tokens = 32
    min_size = 10

    def build(self, size, rng):
        recs = wordbank.make_records(size, rng)
        context = wordbank.render_records(recs)
        kind = rng.choice(["gt", "lt", "category", "between"])

        if kind == "gt":
            t = rng.randint(200, 800)
            gold = sum(r.value > t for r in recs)
            question = f"Question: How many records have a value greater than {t}?"
            meta = {"predicate": "value >", "threshold": t}
        elif kind == "lt":
            t = rng.randint(200, 800)
            gold = sum(r.value < t for r in recs)
            question = f"Question: How many records have a value less than {t}?"
            meta = {"predicate": "value <", "threshold": t}
        elif kind == "category":
            c = rng.choice(wordbank.CATEGORIES)
            gold = sum(r.category == c for r in recs)
            question = f"Question: How many records have category '{c}'?"
            meta = {"predicate": "category ==", "category": c}
        else:
            a = rng.randint(100, 400)
            b = rng.randint(500, 900)
            gold = sum(a <= r.value <= b for r in recs)
            question = f"Question: How many records have a value between {a} and {b} inclusive?"
            meta = {"predicate": "between", "low": a, "high": b}

        instruction = (
            "Below is a list of records. Each record has a name, a category, and "
            "a numeric value."
        )
        return Sample(
            instruction, context, question,
            answer_prefix="The count is",
            answers=[str(gold)], gold=gold, answer_type="int", meta=meta,
        )
