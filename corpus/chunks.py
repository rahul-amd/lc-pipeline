"""Bounded text chunks without flattening source line breaks."""
import re


def split_text(text, count, max_tokens):
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    while text:
        if count(text) <= max_tokens:
            yield text
            return
        lo, hi, best = 1, len(text), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if count(text[:mid]) <= max_tokens:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if not best:
            raise ValueError("one character exceeds chunk token budget")
        # Prefer a word boundary without increasing the counted prefix.
        boundaries = list(re.finditer(r"\s+", text[:best]))
        cut = boundaries[-1].end() if boundaries else best
        # BPE token counts need not be monotonic when shortening a prefix.
        if cut != best and count(text[:cut]) > max_tokens:
            cut = best
        chunk, text = text[:cut], text[cut:]
        if chunk.strip():
            yield chunk
