"""Shared training-length, identity and atomic JSONL contracts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile


def training_length(row, count):
    """Count PT text or the HF training chat template, including the answer.

    Without a chat template, count newline-joined message content. This is a
    content-only budget, not a guarantee for a downstream trainer's template.
    """
    if row["format"] == "pt":
        return count(row["text"]), "text"
    tokenizer = getattr(getattr(count, "__self__", None), "tok", None)
    if tokenizer is not None and tokenizer.chat_template:
        return len(tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False,
            return_dict=False)), "chat_template"
    return count("\n".join(m["content"] for m in row["messages"])), "message_content"


def finalize_row(row, count, limit):
    length, mode = training_length(row, count)
    if limit is not None and length > limit:
        raise ValueError(f"{row.get('task')}: training length {length} exceeds {limit}")
    row["training_length"] = length
    row["length_mode"] = mode
    identity = [row.get("id"), row.get("seed"), limit, row["format"],
                row.get("text", row.get("messages"))]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                     sort_keys=True).encode()).hexdigest()[:20]
    row["id"] = f"{row['task']}-{digest}"
    return row


def write_jsonl(path, rows):
    """Publish only after every row has been generated and serialized."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".jsonl-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
