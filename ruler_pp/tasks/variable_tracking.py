"""RULER-faithful VT: a chain of variable copies hidden in noise.

VAR A = 12345 ; VAR B = VAR A ; VAR C = VAR B ; ...  All variables in the chain
share the same value; the answer is every variable name assigned that value.
"""

from __future__ import annotations

import string

from ..base import Sample, Task
from .. import wordbank


def _new_var(rng) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=5))


class VariableTracking(Task):
    name = "variable_tracking"
    reserve_tokens = 96
    min_size = 12

    def build(self, size, rng):
        num_hops = 4
        names: set[str] = set()
        while len(names) < num_hops + 1:
            names.add(_new_var(rng))
        names = list(names)
        rng.shuffle(names)

        value = str(rng.randint(10000, 99999))
        chain = [f"VAR {names[0]} = {value}"]
        for j in range(num_hops):
            chain.append(f"VAR {names[j + 1]} = VAR {names[j]}")

        sentences = [wordbank.NOISE] * size
        for offset, pos in enumerate(sorted(rng.sample(range(len(sentences)), len(chain)))):
            sentences.insert(pos + offset, chain[offset])
        context = "\n".join(sentences)

        answer = names[: num_hops + 1]
        instruction = (
            "Memorize and track the chain of variable assignments hidden in the "
            "following text."
        )
        question = f"Question: Find all variables that are assigned the value {value}."
        return Sample(
            instruction, context, question,
            answer_prefix=f"The variables assigned the value {value} are:",
            answers=answer, gold=answer, answer_type="list",
            meta={"num_chains": 1, "num_hops": num_hops, "value": value},
        )
