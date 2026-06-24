"""ruler++ : synthetic long-context datagen (Phase 1, no LLM inference).

A small, reusable engine that fills a token budget (RULER-style) with
synthetic tasks whose gold answers are computed deterministically.
"""

from .base import CorpusTask, Sample, Task

__all__ = ["Sample", "Task", "CorpusTask"]
