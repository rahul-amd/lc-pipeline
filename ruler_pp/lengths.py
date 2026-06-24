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
        return len(self.enc.encode(text))


class _HF:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


def get_tokenizer(spec: str):
    """Resolve a tokenizer spec, falling back gracefully if it can't load.

    Specs: ``hf:<name_or_path>`` | ``tiktoken[:<encoding>]`` | ``whitespace``.
    """
    try:
        if spec.startswith("hf:"):
            return _HF(spec[3:])
        if spec.startswith("tiktoken"):
            enc = spec.split(":", 1)[1] if ":" in spec else "cl100k_base"
            return _Tiktoken(enc)
        if spec == "whitespace":
            return _Whitespace()
        raise ValueError(f"unknown tokenizer spec: {spec!r}")
    except Exception as e:  # noqa: BLE001 - fall back rather than crash datagen
        print(f"[ruler++] tokenizer {spec!r} unavailable ({e}); falling back.", flush=True)
    try:
        return _Tiktoken()
    except Exception:  # noqa: BLE001
        return _Whitespace()


def fit_size(measure: Callable[[int], int], budget: int, min_size: int) -> int:
    """Largest `size` with ``measure(size) <= budget``.

    `measure` builds a sample at `size` and returns its token length. We probe
    once to estimate tokens-per-size, set a generous upper bound, then binary
    search. Tasks need only be roughly monotonic in `size`; the per-sample loop
    in the driver handles any residual overflow.
    """
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
