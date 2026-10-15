"""Pluggable tokenizers and the binary-search length fitter (the engine)."""

from __future__ import annotations

import re
from typing import Callable


class _Whitespace:
    """Dependency-free fallback: count word/punct chunks."""

    _re = re.compile(r"\w+|[^\w\s]")

    def count(self, text: str) -> int:
        return len(self._re.findall(text))


class _Tiktoken:
    def __init__(self, encoding: str = "cl100k_base"):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text, disallowed_special=()))


class _HF:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


def get_tokenizer(spec: str):
    """Resolve the requested tokenizer; never silently change the length unit."""
    if spec.startswith("hf:") and spec[3:]:
        return _HF(spec[3:])
    if spec == "tiktoken" or spec.startswith("tiktoken:"):
        return _Tiktoken(spec.split(":", 1)[1] if ":" in spec else "cl100k_base")
    if spec == "whitespace":
        return _Whitespace()
    raise ValueError(f"unknown tokenizer spec: {spec!r}")


def fit_size(measure: Callable[[int], int], budget: int, min_size: int) -> int:
    """Largest `size` with ``measure(size) <= budget``.

    `measure` builds a sample at `size` and returns its token length. We probe
    once to estimate tokens-per-size, set a generous upper bound, then binary
    search. Tasks need only be roughly monotonic in `size`; the per-sample loop
    in the driver handles any residual overflow.
    """
    if min_size < 1 or budget < 1 or measure(min_size) > budget:
        raise ValueError(f"minimum task size {min_size} does not fit token budget {budget}")
    probe = max(min_size, 256)
    tokens_per_size = max(measure(probe) / probe, 1e-6)
    upper = max(int(budget / tokens_per_size * 2), min_size * 2)

    best = min_size
    lo, hi = min_size, upper
    while lo <= hi:
        mid = (lo + hi) // 2
        if measure(mid) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
