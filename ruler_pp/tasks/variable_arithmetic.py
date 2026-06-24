"""Variable chain with arithmetic; compute the final value of the last variable.

VAR A = 7 ; VAR B = VAR A + 3 ; VAR C = VAR B * 2 ; ...  hidden in noise.
"""

from __future__ import annotations

import string

from ..base import Sample, Task
from .. import wordbank


def _new_var(rng) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=5))


class VariableArithmetic(Task):
    name = "variable_arithmetic"
    reserve_tokens = 48
    min_size = 12

    def build(self, size, rng):
        num_hops = 4
        names: set[str] = set()
        while len(names) < num_hops + 1:
            names.add(_new_var(rng))
        names = list(names)

        cur = rng.randint(1, 20)
        chain = [f"VAR {names[0]} = {cur}"]
        for j in range(num_hops):
            op = rng.choice(["+", "-", "*"])
            c = rng.randint(1, 9)
            cur = cur + c if op == "+" else cur - c if op == "-" else cur * c
            chain.append(f"VAR {names[j + 1]} = VAR {names[j]} {op} {c}")

        sentences = [wordbank.NOISE] * size
        for offset, pos in enumerate(sorted(rng.sample(range(len(sentences)), len(chain)))):
            sentences.insert(pos + offset, chain[offset])
        context = "\n".join(sentences)

        target = names[-1]
        instruction = (
            "Track the chain of arithmetic variable assignments hidden in the "
            "text below."
        )
        question = f"Question: What is the final integer value of VAR {target}?"
        return Sample(
            instruction, context, question,
            answer_prefix=f"The value of VAR {target} is",
            answers=[str(cur)], gold=cur, answer_type="int",
            meta={"num_hops": num_hops, "target": target},
        )
