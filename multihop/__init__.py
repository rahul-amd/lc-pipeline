"""Multi-hop QA datagen over FinePDFs documents (LLM-in-the-loop, two phases)."""

from .multihop import (
    assemble, build_messages, chunk_doc, extract_text, main, parse_pair,
    prepare, strip_thinking,
)

__all__ = [
    "assemble", "build_messages", "chunk_doc", "extract_text", "main",
    "parse_pair", "prepare", "strip_thinking",
]
