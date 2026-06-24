"""Word sources and the shared "record table" substrate.

All randomness flows through a caller-supplied `random.Random` so generation is
fully reproducible per sample.
"""

from __future__ import annotations

import json
import os
import string
from dataclasses import dataclass

# A repeated noise sentence used by the variable-tracking tasks as filler.
NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

# Arbitrary category labels. Membership is assigned at random — these tasks are
# about counting/aggregation over long context, not real-world semantics.
CATEGORIES = [
    "fruit", "metal", "animal", "color", "country", "planet",
    "tool", "gem", "flower", "fish", "bird", "vehicle",
]

_WORDS: list[str] | None = None


def _default_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "res", "english_words.json")


def load_words(path: str | None = None) -> list[str]:
    """Load and cache a clean subset of english words (lowercase a-z, len 3-12)."""
    global _WORDS
    if _WORDS is None:
        with open(path or _default_path(), encoding="utf-8") as f:
            raw = json.load(f).values()
        seen: set[str] = set()
        out: list[str] = []
        for w in raw:
            if w.isascii() and w.isalpha() and 3 <= len(w) <= 12:
                wl = w.lower()
                if wl not in seen:
                    seen.add(wl)
                    out.append(wl)
        _WORDS = out
    return _WORDS


def sample_words(n: int, rng) -> list[str]:
    """`n` distinct words when possible; falls back to sampling with replacement."""
    pool = load_words()
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def coded_words(n: int, rng, length: int = 6) -> list[str]:
    """`n` distinct synthetic words of random lowercase letters."""
    s: set[str] = set()
    while len(s) < n:
        s.add("".join(rng.choices(string.ascii_lowercase, k=length)))
    out = list(s)
    rng.shuffle(out)
    return out


@dataclass
class Record:
    name: str
    category: str
    value: int


def make_records(n: int, rng, categories=None, val_range=(1, 999)) -> list[Record]:
    cats = categories or CATEGORIES
    names = sample_words(n, rng)
    return [Record(nm, rng.choice(cats), rng.randint(*val_range)) for nm in names]


def render_records(records: list[Record], numbered: bool = True) -> str:
    lines = []
    for i, r in enumerate(records):
        prefix = f"{i + 1}. " if numbered else "- "
        lines.append(f"{prefix}{r.name} | category: {r.category} | value: {r.value}")
    return "\n".join(lines)
