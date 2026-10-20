# Long-context pipeline audit

Scope: core generators, shared corpus/length/serialization code, response import,
gold verifiers and shell orchestration. The Gradio viewers, download utilities and
standalone performance benchmarks are not covered by the integration matrix.
Existing user outputs were not modified. The existing `WORKERS=16` edit in
`scripts/datagen.sh` is preserved.

## Corrected issues

| Severity | Finding | Correction |
|---|---|---|
| High | All ordinary non-thinking API answers were dropped because they lacked `</think>`. | Accept plain content; remove complete reasoning blocks; reject visible unterminated blocks. |
| High | Fixed answer reserves did not bound complete training sequences; minimum-size overflow was silently accepted. | Fit and validate actual prompt + answer, including HF chat templates; reject impossible budgets. |
| High | Untie reconstruction could exceed its window and missed quotas exited successfully. | Refit complete examples; fail on unmet targets; publish flow files after success. |
| High | Fallback corpus IDs restarted at every batch and collided between same-named shards. | Stable relative shard paths + absolute row offsets; consistent cluster joins without explicit source IDs. |
| High | LLM imports silently accepted missing/duplicate/unknown IDs, failed/truncated responses and empty datasets. | Strict bundle joins, explicit partial-import option, completion/status checks, manifest hashes and nonempty publication. |
| High | Requested model tokenizer errors silently changed the tokenizer. | Fail on unavailable/unknown specs; keep whitespace counting explicitly selectable. |
| Medium | Variable names/coded words depended on hash seed despite seeded RNGs. | Sort string sets before random selection/shuffling. |
| Medium | Small frequency tasks had ambiguous top-three golds after integer rounding. | Ensure strict counts at the top-three boundary; independently verify minimum-size golds. |
| Medium | Word-pool exhaustion violated distinctness assumptions in common/top-k tasks. | Extend with distinct generated words instead of sampling with replacement. |
| Medium | IDs repeated across buckets/seeds and unrelated source runs. | Content/settings-based sample IDs and prepared-request IDs. |
| Medium | Large CLI runs materialized all long examples before writing. | Stream serial and pooled generation into atomic output files. |
| Medium | Chunking flattened newlines and word windows could exceed token caps. | Shared measured text splitting that preserves internal formatting. |
| Medium | NaN education scores passed filters; later Parquet shards inherited the first shard's schema. | Treat invalid scores as zero; resolve each shard's schema independently. |
| Medium | Cluster joins silently omitted missing documents; zero-token/empty/oversized inputs had invalid behavior. | Validate IDs/metadata, fail incomplete joins, exclude oversized docs with counts; handle singleton neighbors and mask self with negative infinity. |
| Medium | MRCR `--k` values other than two crashed; invalid fractions/IDK choices failed obscurely. | Explicit supported-configuration checks; exclude IDK passages containing conflicting binding syntax. |
| Medium | Nonempty partial output directories were treated as completed shell stages. | Completion markers only after success; configurable cluster location and context buckets. |
| Medium | Shell orchestration stopped at LLM preparation; length buckets had reserve-sized gaps. | Optional API inference + assembly; contiguous document-length thresholds. |
| Low | Empty gold-verifier inputs passed as 0/0; summary cleanup stripped legitimate “Summary statistics…” text. | Nonempty verification and punctuation-delimited preamble cleanup. |
| Medium | Repeated QA questions with differently worded answers survived exact-pair deduplication. | Keep the first normalized question per source document and count removed duplicates. |
| Low | Mean rounding and cyclic graph self-membership were underspecified. | State two-decimal rounding and exclude the queried node explicitly. |

During validation, a new HF length-counting implementation initially counted
Transformers 5's default `BatchEncoding` keys. The first integration run exposed
the error; affected steps were cancelled. Passing `return_dict=False` explicitly
fixes the contract, and a real-model-tokenizer regression guards it.

## Remaining quality and release work

1. **Multi-hop is a prompt requirement, not a guarantee.** Random chunks often
   lack a useful connection. Accepted format does not prove groundedness or that
   every chunk is necessary. Evaluate a stratified sample with evidence spans,
   answer support checks and leave-one-chunk-out tests before scaling. Compare
   random chunk selection with entity-linked sampling. Keep abstentions visible.
2. **Summary faithfulness and full-document coverage remain unscored.** Review
   factual claims and coverage across beginning/middle/end. Calibrate any judge
   against a human-labeled sample before making it a rejection gate.
3. **Untie labels can be semantically ambiguous.** Source order is deterministic,
   but shuffled independent paragraphs may admit many valid orders; concatenated
   document rank is especially arbitrary. Measure order recoverability and
   compare single-document and clustered training before claiming gains.
4. **Split leakage and final dataset deduplication need a dataset-level policy.**
   Passage reuse, repeated source spans and multiple seeds can generate related
   examples. Split by source/cluster before generating train/eval sets; audit
   exact and near-duplicate final examples. Content-derived IDs are not semantic
   deduplication. Synthetic tasks should be kept separate from benchmark inputs.
5. **Source metadata is only a filter.** Missing education fields are not scored;
   near-duplicate filtering drops all members marked with cluster size > 1.
   Corpus token counts may come from a different tokenizer, and a shard named
   `eng_Latn` may contain multilingual text. Select and validate the intended
   language/domain mix explicitly. Cluster budgets use source token metadata;
   the final generators enforce their chosen tokenizer's limit.
6. **Resume and scale remain bounded.** Completion markers are not configuration
   cache keys; use a fresh output root after changing settings. Exact neighbor
   search is quadratic. Cluster text lookup and Untie's reuse pool can consume
   substantial RAM; the API runner materializes the request list and is intended
   for small runs. Large inference should use the external dispatcher.
7. **Release packaging/legal review is still needed.** Establish the code/data
   licenses, exclude local corpus/outputs, and test installation in a clean CPU
   environment plus a documented GPU stack. Existing hardware-specific install
   comments are not a portable environment specification. No package publication,
   code push, model training or downstream benchmark evaluation was performed.

Software smoke tests establish runnable paths and data integrity. They cannot
establish that the resulting mix improves long-context model behavior.
