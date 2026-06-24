"""Core abstractions shared by every task.

A `Sample` is one generated example — the common currency across both phases.

Two task families produce `Sample`s:
  * `Task` (synthetic): builds a sample at a given `size`, an opaque monotonic
    knob that scales token count. The driver in `cli.py` searches `size` to fill
    a token budget, so synthetic tasks never deal with tokenization themselves.
  * `CorpusTask` (real text): derives a sample from source text drawn from a
    corpus. There is no `size` knob — length is set by how much source text is
    packed under a token budget by the task's own driver.

Both emit the same `Sample`, so the output shaping (`pt`/`sft`) and JSON schema
in `cli.py` are shared verbatim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    instruction: str          # top-of-prompt task description
    context: str              # the variable-length body (scales with `size`)
    question: str             # the concrete query
    answer_prefix: str        # cue the model continues from (held out of `input`)
    answers: list[str]        # acceptable surface forms of the gold answer
    gold: Any                 # structured gold (int / list / dict / ...)
    answer_type: str          # "int" | "number" | "list" | "map" | "string"
    meta: dict = field(default_factory=dict)

    @property
    def input(self) -> str:
        return f"{self.instruction}\n\n{self.context}\n\n{self.question}"

    def answer_text(self) -> str:
        """Canonical surface form of the gold answer."""
        if self.answer_type in ("list", "map"):
            return ", ".join(self.answers)
        return self.answers[0]

    def fewshot_text(self) -> str:
        """Rendered as an in-context example: prompt + answer_prefix + answer."""
        answer = f"{self.answer_prefix} {self.answer_text()}".strip()
        return f"{self.input}\n{answer}"


class Task(ABC):
    name: str = "base"
    # Tokens reserved for the model's answer (kept free under the budget).
    reserve_tokens: int = 96
    # Smallest size that still yields a valid sample / answer.
    min_size: int = 4

    @abstractmethod
    def build(self, size: int, rng) -> Sample:
        """Return one sample whose context scales with `size`."""

    def fewshot(self, rng, k: int) -> str:
        """Generic in-context examples: build `k` tiny samples at min_size."""
        if k <= 0:
            return ""
        blocks = [self.build(self.min_size, rng).fewshot_text() for _ in range(k)]
        return "\n\n".join(blocks) + "\n\n"


class CorpusTask(ABC):
    """Base for tasks whose samples come from a real text corpus.

    Unlike `Task`, there is no synthetic `size` knob: a sample's length is set by
    how much source text the task packs under a token budget. Subclasses turn one
    piece of prepared source text into a `Sample` (or `None` if it is too small),
    reusing the same `Sample` contract and output shaping as the synthetic tasks.
    """

    name: str = "corpus"

    @abstractmethod
    def build(self, source: Any, rng) -> "Sample | None":
        """Return one sample from `source`, or None if the source is unusable."""
