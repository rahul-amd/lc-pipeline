# Validation results — 2026-10-14

The pipeline runs end to end with real GPU embeddings and API inference. This
establishes software/data integrity for the tested configuration; it does **not**
establish that the model-generated answers are consistently high quality.

| Check | Result | Evidence |
|---|---|---|
| Regression suite | 108 passed | `runs/regression.xml`, `runs/regression.log` |
| Full integration matrix | 278 rows, 123 final files, 41 commands; passed in 379.3 seconds | `runs/e2e-02/RESULT.json` |
| Main shell orchestrator on final QA assembly code | 36 rows, 14 API responses; all stages completed | `runs/orchestrator-03/RESULT.json` |
| Resume with completed stages | All 34 JSONL files unchanged | `runs/resume-result.json`, `runs/resume.log` |
| Duplicate-question fix against real responses | Both formats: 2 documents, 6 QA pairs, 2 duplicates removed | `runs/question-dedup/RESULT.json` |
| Syntax / whitespace | Python AST parse, shell syntax checks and `git diff --check` passed | final repository checks |

The full matrix covers all 22 synthetic tasks at 8K and 32K in PT and SFT,
including few-shot examples; both Untie flows from clustered and single sources;
MRCR and all three IDK cases; and prepare → real API → assembly for multi-hop QA
and summarization. Three synthetic tasks also generated 128K-budget outputs.
The maximum measured training length was **131,054 tokens**. All final IDs were
unique and all samples fit their requested limits. Deterministic golds were
recomputed from the rendered context; Untie reconstruction was round-tripped.

There were 136 PT and 142 SFT rows in the matrix. IDK included 16 absent,
4 answerable and 4 subtle cases; Untie included 16 rows per flow. The 20 real LLM
requests in this matrix produced 16 final document rows across PT/SFT. Model
inference used whole documents of roughly 3K–14K tokens for the sampled requests;
128K validation tested generation/serialization, **not 128K model inference**.

The normalized-question deduplication fix was added after the matrix's 8K QA
assembly. It was checked against those same saved real responses in both formats,
covered by regression, and included in the fresh unattended shell orchestration
run. The original matrix artifacts are retained unchanged for comparison.

## Environment

Slurm job `55239`, partition `amd-tw-verification`, node `tus1-p15-g55`; two GPUs,
16 CPUs and 128 GiB host RAM allocated. No `hissu` allocation was used. GPU work
ran in the existing ROCm Apptainer image, on gfx942 hardware with approximately
256 GiB visible GPU memory. The allocation and API server were released after
validation.

- Authoring model/tokenizer: `Qwen/Qwen3-8B`; thinking disabled for inference.
- Embedding model: `Qwen/Qwen3-Embedding-0.6B`, revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, tested on GPU and with cache reuse.
- PyTorch `2.10.0+git8514f05`, HIP `7.2.53211`, Transformers `5.5.0`,
  SentenceTransformers `5.5.1`, vLLM `0.18.2rc1.dev71+ge92668e83.rocm721`.
- NumPy `2.1.3`, SciPy `1.17.1`, PyArrow `23.0.1`, pytest `9.0.2`.
- Full versions: `runs/environment.json`. Stage commands/logs and source IDs/hashes
  are in `runs/e2e-02/commands.json` and `provenance.json`.

This reused the existing ROCm environment. A clean installation was not tested.

## Failures discovered during validation

The initial regression found tied word frequencies with an invalid assumed gold;
the generator now separates the top-three counts. The first integration attempt
also exposed a Transformers return-type incompatibility in the new chat-length
code: the default result was a dictionary, so counting its keys was wrong.
Affected test steps were cancelled and the fresh matrix ran after the explicit
`return_dict=False` correction. The cancelled run is retained in `runs/e2e-01/`;
it is not a passing result.

Spot review found a factual error in generated QA and weak multi-hop structure.
See [QUALITY_REVIEW.md](QUALITY_REVIEW.md) for evidence and proposed experiments,
and [AUDIT.md](AUDIT.md) for the full list of fixes and remaining release work.
Large-scale generation should wait for that quality calibration.
