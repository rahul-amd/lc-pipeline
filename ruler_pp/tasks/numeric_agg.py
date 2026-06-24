"""Aggregate a list of integers (sum / min / max / mean / median)."""

from __future__ import annotations

import statistics

from ..base import Sample, Task


class NumericAgg(Task):
    name = "numeric_agg"
    reserve_tokens = 32
    min_size = 10

    def build(self, size, rng):
        nums = [rng.randint(1, 999) for _ in range(size)]
        context = "\n".join(f"{i + 1}. {v}" for i, v in enumerate(nums))
        agg = rng.choice(["sum", "min", "max", "mean", "median"])

        if agg == "sum":
            gold = sum(nums)
        elif agg == "min":
            gold = min(nums)
        elif agg == "max":
            gold = max(nums)
        elif agg == "mean":
            gold = round(sum(nums) / len(nums), 2)
        else:
            gold = statistics.median(nums)

        instruction = "Below is a numbered list of integers."
        question = f"Question: What is the {agg} of all the integers in the list?"
        return Sample(
            instruction, context, question,
            answer_prefix=f"The {agg} is",
            answers=[str(gold)], gold=gold, answer_type="number",
            meta={"agg": agg, "n": size},
        )
