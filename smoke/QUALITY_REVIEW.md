# Spot review of model-generated data

This is a small diagnostic review, not a calibrated quality estimate. Evidence
comes from the real Qwen3-8B API responses in `runs/e2e-02/8192-llm/` with thinking
disabled. The source snapshots are verbatim FinePDFs text; IDs/hashes are recorded
in the run's `provenance.json`. No downstream training was performed.

## Confirmed factual error

Source ID: `<urn:uuid:0991ad98-2509-490f-b4d7-5fbef82f9a10>`.
The source's English curriculum timeline says **September 2022**, and its Welsh
version says **Medi 2022**. The generated answer about the Wales curriculum
timeline instead says **June 2022**. The summary of the same document correctly
says September. This is direct evidence that format validation and a successful
inference call do not establish answer correctness.

## Weak multi-hop structure and repeated questions

Source ID: `<urn:uuid:69acf9dd-5622-4b44-b3fe-397f734b6a92>`.
Several generated questions combine independent newsletter facts into a multipart
question: event dates, home-blessing arrangements, and membership dues. That can
exercise multi-span retrieval, but does not demonstrate a chain of dependent
reasoning. One answer also omits the requested dates while listing other details.

The initial matrix produced the same question twice with differently worded
answers. Assembly now deduplicates normalized questions per document, keeping the
first request's answer and counting discarded duplicates. That prevents repeated
questions but is not an answer-quality selection rule. Random chunk sampling can
still repeat a chunk combination and waste inference; request-level uniqueness
is a separate optimization.

## Suggested next experiment

1. Sample 100 QA pairs and 50 summaries across source domains and 8K/32K/128K
   document bands. Have human reviewers label factual support, missing requested
   information, contradiction and question type (single-span, multi-span, true
   dependent multi-hop). Include multilingual/mixed-language documents.
2. Ask the authoring model for exact source evidence per answer claim and per
   chunk. Check quoted spans programmatically. Evaluate claim support with a
   separate judge calibrated against those human labels; span existence alone
   does not establish entailment.
3. Test each QA with one supporting chunk removed, checking whether the remaining
   chunks still answer it. Compare random chunks with related chunks chosen by
   shared entities. Report retained yield and error rate together.
4. For summaries, measure support and coverage separately across the beginning,
   middle and end of documents. The current broad “faithful summary” prompt should
   not be treated as a quality gate.

Do not automatically turn an uncalibrated judge into a release gate. Preserve
source evidence and rejection reasons so threshold choices can be reviewed.
