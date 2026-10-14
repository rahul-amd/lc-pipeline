"""RULER-faithful FWE: coded words drawn from a Zipf/zeta distribution.

The single most frequent token is rendered as '...' noise; the answer is the
next three most frequent coded words.
"""

from __future__ import annotations

import numpy as np
from scipy.special import zeta

from ..base import Sample, Task
from .. import wordbank


class FreqWords(Task):
    name = "freq_words"
    reserve_tokens = 96
    min_size = 60

    def build(self, size, rng):
        alpha = 1.5
        vocab_size = max(20, size // 20)
        vocab = wordbank.coded_words(vocab_size, rng, length=6)
        vocab[0] = "..."  # rank-0 token becomes pure noise

        ranks = np.arange(1, len(vocab) + 1)
        counts = (size * (ranks ** -alpha) / zeta(alpha)).astype(int)

        # Integer rounding can tie the third real word with the tail at small sizes.
        for rank in (3, 2, 1):
            counts[rank] = max(counts[rank], counts[rank + 1] + 1)

        stream: list[str] = []
        for w, c in zip(vocab, counts):
            stream.extend([w] * int(c))
        rng.shuffle(stream)
        context = " ".join(stream)

        answer = vocab[1:4]  # the three most frequent real words
        instruction = (
            "Read the following stream of coded words and track how often each "
            "one appears. Ignore the dots '...'."
        )
        question = (
            "Question: What are the three most frequently appearing coded words "
            "(excluding '...')?"
        )
        return Sample(
            instruction, context, question,
            answer_prefix="The three most frequent words are:",
            answers=answer, gold=answer, answer_type="list",
            meta={"alpha": alpha, "vocab_size": vocab_size},
        )
