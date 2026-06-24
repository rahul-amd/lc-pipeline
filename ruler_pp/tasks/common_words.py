"""RULER-faithful CWE: a numbered word list where some words repeat far more."""

from __future__ import annotations

from ..base import Sample, Task
from .. import wordbank


class CommonWords(Task):
    name = "common_words"
    reserve_tokens = 160
    min_size = 20

    def build(self, size, rng):
        num_cw, common_rep, uncommon_rep = 10, 30, 3
        n_uncommon = max(num_cw, size)  # size scales the uncommon tail
        pool = wordbank.sample_words(num_cw + n_uncommon, rng)
        common, uncommon = pool[:num_cw], pool[num_cw:]

        words = common * common_rep + uncommon * uncommon_rep
        rng.shuffle(words)
        context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(words))

        instruction = (
            "Below is a numbered list of words. Some words appear far more "
            "often than others. Memorize the ones that appear most often."
        )
        question = f"Question: What are the {num_cw} most common words in the list above?"
        return Sample(
            instruction, context, question,
            answer_prefix="The most common words are:",
            answers=common, gold=common, answer_type="list",
            meta={"num_cw": num_cw, "common_rep": common_rep,
                  "uncommon_rep": uncommon_rep, "n_uncommon": n_uncommon},
        )
