"""Run prepared requests through an OpenAI-compatible chat-completions API.

Offline dispatcher outputs remain supported by the assemble commands. This
small runner provides a bounded, unattended alternative for smoke/small runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from corpus.llm import load_jsonl, validate_response
from ruler_pp.output import write_jsonl


def run_request(row, *, base_url, model, api_key, max_tokens, timeout,
                retries, chat_template_kwargs=None):
    payload = {"model": model, "messages": row["messages"], "temperature": 0,
               "max_tokens": max_tokens}
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for attempt in range(retries + 1):
        try:
            request = Request(base_url.rstrip("/") + "/chat/completions",
                              json.dumps(payload).encode(), headers)
            with urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            result = {"custom_id": row["id"], "response": {"body": body}}
            validate_response(result)
            return result
        except (HTTPError, URLError, TimeoutError) as exc:
            transient = not isinstance(exc, HTTPError) or exc.code in (408, 429, 500, 502, 503, 504)
            if not transient or attempt == retries:
                raise RuntimeError(f"inference failed for request {row['id']}") from exc
            time.sleep(min(2 ** attempt, 8))


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_url", required=True, help="API root including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--chat_template_kwargs", type=json.loads, default=None)
    args = parser.parse_args(argv)
    if min(args.concurrency, args.max_tokens, args.timeout) <= 0 or args.retries < 0:
        parser.error("concurrency, max_tokens and timeout must be positive; retries nonnegative")
    rows = list(load_jsonl(args.requests))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("requests must be nonempty with unique IDs")
    options = dict(base_url=args.base_url, model=args.model,
                   api_key=os.environ.get(args.api_key_env), max_tokens=args.max_tokens,
                   timeout=args.timeout, retries=args.retries,
                   chat_template_kwargs=args.chat_template_kwargs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Ordered iteration and atomic publication make failures visible. This
        # runner intentionally does not implement large-scale resume/dispatch.
        write_jsonl(args.output, pool.map(lambda row: run_request(row, **options), rows))
    print(json.dumps({"responses": len(rows), "output": args.output, "model": args.model}))


if __name__ == "__main__":
    main()
