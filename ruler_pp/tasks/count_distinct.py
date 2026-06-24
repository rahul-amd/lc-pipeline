"""Count the number of distinct words in a list with repeats."""

from __future__ import annotations

from ..base import Sample, Task
from .. import wordbank


class CountDistinct(Task):
    name = "count_distinct"
    reserve_tokens = 32
    min_size = 20

    def build(self, size, rng):
        distinct = max(5, size // 4)
        pool = wordbank.sample_words(distinct, rng)

        stream: list[str] = []
        for w in pool:
            stream.extend([w] * rng.randint(1, 5))
        rng.shuffle(stream)

        gold = len(set(stream))  # empirical, robust to any duplicate sampling
        context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(stream))
        instruction = "Below is a numbered list of words, where some words are repeated."
        question = "Question: How many distinct (unique) words appear in the list?"
        return Sample(
            instruction, context, question,
            answer_prefix="The number of distinct words is",
            answers=[str(gold)], gold=gold, answer_type="int",
            meta={"distinct": gold},
        )
