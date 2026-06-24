"""Set operations over two word lists: intersection / union size / difference."""

from __future__ import annotations

from ..base import Sample, Task
from .. import wordbank


class SetOps(Task):
    name = "set_ops"
    reserve_tokens = 200
    min_size = 12

    def build(self, size, rng):
        n = max(4, size // 2)
        half = n // 2
        pool = wordbank.sample_words(2 * n, rng)  # plenty of distinct words

        common = pool[:half]
        only_a = pool[half:n]
        only_b = pool[n:n + (n - half)]
        a, b = common + only_a, common + only_b
        rng.shuffle(a)
        rng.shuffle(b)

        sa, sb = set(a), set(b)
        op = rng.choice(["intersection", "union_size", "difference"])
        if op == "intersection":
            gold = sorted(sa & sb)
            answers, answer_type = gold, "list"
            question = "Question: Which words appear in BOTH Set A and Set B?"
            answer_prefix = "The words in both sets are:"
        elif op == "union_size":
            gold = len(sa | sb)
            answers, answer_type = [str(gold)], "int"
            question = "Question: How many distinct words appear in Set A or Set B (the size of their union)?"
            answer_prefix = "The size of the union is"
        else:
            gold = sorted(sa - sb)
            answers, answer_type = gold, "list"
            question = "Question: Which words appear in Set A but NOT in Set B?"
            answer_prefix = "The words only in Set A are:"

        context = f"Set A: {', '.join(a)}\n\nSet B: {', '.join(b)}"
        instruction = "Below are two sets of words, Set A and Set B."
        return Sample(
            instruction, context, question, answer_prefix,
            answers=answers, gold=gold, answer_type=answer_type, meta={"op": op},
        )
