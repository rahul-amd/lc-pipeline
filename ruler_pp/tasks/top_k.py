"""Return the k most frequent words, ordered most-to-least frequent.

A few "heavy" words get strictly separated high counts; the rest are light
filler (count 1-2), so the top-k boundary is unambiguous.
"""

from __future__ import annotations

from collections import Counter

from ..base import Sample, Task
from .. import wordbank


class TopK(Task):
    name = "top_k"
    reserve_tokens = 96
    min_size = 40

    def build(self, size, rng):
        k = rng.choice([3, 5])
        n_light = max(5, size // 2)
        pool = wordbank.sample_words(k + n_light, rng)  # all distinct
        heavy, light = pool[:k], pool[k:]

        base = max(size // 3, 6)
        heavy_counts = [base + 3 * (k - i) for i in range(k)]  # strictly descending, all > 2

        stream: list[str] = []
        for w, c in zip(heavy, heavy_counts):
            stream.extend([w] * c)
        for w in light:
            stream.extend([w] * rng.randint(1, 2))
        rng.shuffle(stream)

        gold = [w for w, _ in Counter(stream).most_common(k)]  # empirical order
        context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(stream))
        instruction = "Below is a numbered list of words."
        question = (
            f"Question: What are the {k} most frequently occurring words, ordered "
            "from most to least frequent?"
        )
        return Sample(
            instruction, context, question,
            answer_prefix=f"The {k} most frequent words are:",
            answers=gold, gold=gold, answer_type="list",
            meta={"k": k, "heavy_counts": heavy_counts},
        )
