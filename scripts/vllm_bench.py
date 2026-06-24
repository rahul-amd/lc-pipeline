import os
_CU = os.path.expanduser("~/vllm-env/lib/python3.12/site-packages/nvidia/cu13")
os.environ["CUDA_HOME"] = _CU
os.environ["PATH"] = _CU + "/bin:" + os.environ.get("PATH", "")
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time, glob, numpy as np
import pyarrow.parquet as pq

INPUT = "/mnt/c/Users/rahul/code/lc-pipeline/finpdf_sample"
N = 2000
CHAR_CAP = 6144


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
    docs = load(N)
    print(f"loaded {len(docs)} docs", flush=True)

    from vllm import LLM
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen3-Embedding-0.6B", runner="pooling",
              max_model_len=512, gpu_memory_utilization=0.6,
              dtype="float16",
              compilation_config={"mode": 3, "cudagraph_mode": "NONE"})
    print(f"model load: {time.time()-t0:.1f}s", flush=True)

    tk = {"truncation": True, "max_length": 512}
    llm.embed(docs[:64], tokenization_kwargs=tk)  # warmup
    t = time.time()
    out = llm.embed(docs, tokenization_kwargs=tk)
    dt = time.time() - t
    emb = np.array([o.outputs.embedding for o in out], dtype=np.float32)
    print(f"embed {len(docs)} docs in {dt:.1f}s -> {len(docs)/dt:.1f} docs/s", flush=True)
    print(f"-> 500k in {500000/(len(docs)/dt)/3600:.2f}h | emb shape {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
