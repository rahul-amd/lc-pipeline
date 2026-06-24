"""Count how many times one specific word appears in a list."""

from __future__ import annotations

from ..base import Sample, Task
from .. import wordbank


class CountOccurrences(Task):
    name = "count_occurrences"
    reserve_tokens = 32
    min_size = 20

    def build(self, size, rng):
        distinct = max(10, size // 4)
        pool = wordbank.sample_words(distinct, rng)
        target = pool[0]

        stream: list[str] = []
        for w in pool[1:]:
            if w == target:
                continue
            stream.extend([w] * rng.randint(1, 4))
        planted = rng.randint(2, max(3, size // 20))
        stream.extend([target] * planted)
        rng.shuffle(stream)

        count = stream.count(target)  # empirical, always exact
        context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(stream))
        instruction = "Below is a numbered list of words."
        question = f"Question: How many times does the word '{target}' appear in the list?"
        return Sample(
            instruction, context, question,
            answer_prefix=f"The word '{target}' appears",
            answers=[f"{count} times", str(count)], gold=count, answer_type="int",
            meta={"target": target, "count": count},
        )
