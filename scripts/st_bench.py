"""Benchmark SentenceTransformers embedding throughput on the cu128 toolchain.

Run in WSL with the cu128 venv:
    ~/emb-cu128/bin/python -u /mnt/c/Users/rahul/code/lc-pipeline/scripts/st_bench.py
"""
import time, glob, numpy as np
import pyarrow.parquet as pq

INPUT = "/mnt/c/Users/rahul/code/lc-pipeline/finpdf_sample"
N = 2000
EMBED_TOKENS = 1024
CHAR_CAP = EMBED_TOKENS * 12


def load(n):
    files = sorted(glob.glob(INPUT + "/**/*.parquet", recursive=True))
    docs = []
    for f in files:
        tb = pq.read_table(f, columns=["text", "token_count", "minhash_cluster_size", "fw_edu_scores"]).to_pydict()
        for txt, tc, mh, ed in zip(tb["text"], tb["token_count"], tb["minhash_cluster_size"], tb["fw_edu_scores"]):
            if not txt or not tc or tc < 200:
                continue
            if mh and mh > 1:
                continue
            if ed and float(np.mean(ed)) < 0.75:
                continue
            docs.append(txt[:CHAR_CAP])
            if len(docs) >= n:
                return docs
    return docs


def main():
    import torch
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | dev {torch.cuda.get_device_name(0)}", flush=True)

    docs = load(N)
    print(f"loaded {len(docs)} docs", flush=True)

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda",
                                model_kwargs={"torch_dtype": torch.float16,
                                              "attn_implementation": "sdpa"})
    model.max_seq_length = EMBED_TOKENS
    print(f"model load: {time.time()-t0:.1f}s | max_seq_len {model.max_seq_length}", flush=True)

    # warmup
    model.encode(docs[:64], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    print("warmup done", flush=True)

    sub = docs[:512]
    for sl in [256, 512, 768, 1024]:
        model.max_seq_length = sl
        model.encode(sub[:64], batch_size=32, normalize_embeddings=True, show_progress_bar=False)  # warm shape
        t = time.time()
        model.encode(sub, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        dt = time.time() - t
        dps = len(sub) / dt
        print(f"seq={sl:4d}: {len(sub)} docs in {dt:5.1f}s -> {dps:6.1f} docs/s | 500k in {500000/dps/3600:5.2f}h", flush=True)


if __name__ == "__main__":
    main()
