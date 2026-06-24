# ruler++ — Phase 1 (synthetic, no LLM inference)

RULER-style long-context datagen, extended with more diverse algorithmic tasks.
Every sample is generated procedurally and its gold answer is computed
deterministically, so **no LLM is needed to label data**.

## Design in one paragraph

Each task exposes a single monotonic **size** knob that scales how much text it
produces. A shared driver tokenizes a probe, binary-searches `size` to **fill a
token budget** (`--max_seq_length`), then emits N samples at that size (shrinking
on the rare overflow). Tasks never touch tokenizers or length logic — they just
implement `build(size, rng) -> Sample`. This is the same engine across all 22
tasks, which is what keeps it small and reusable.

```
ruler_pp/
  base.py        Sample dataclass + Task ABC (the only interface tasks implement)
  wordbank.py    word sources (english_words.json, coded words) + record table substrate
  lengths.py     pluggable tokenizers (hf / tiktoken / whitespace) + binary-search fitter
  registry.py    name -> Task   (add a task here)
  cli.py         the driver: fit budget, generate, write jsonl
  tasks/         one self-contained file per task
```

## Tasks

RULER-faithful: `common_words` (CWE), `freq_words` (FWE, Zipf/zeta),
`variable_tracking` (VT).

Extras: `count_occurrences`, `count_predicate`, `count_distinct`,
`tally_by_category`, `numeric_agg`, `group_by`, `top_k`, `set_ops`,
`variable_arithmetic`.

Most extras share one substrate — a **record table** (`name | category: c | value: v`)
— so counting/aggregation tasks reuse the same generator. Each task also samples
among several variants (predicate kind, aggregation, set op, k, …) for diversity;
the choice is recorded in `meta`.

Graph reasoning (GraphWalks-style, all in `tasks/graph.py`): `graph_bfs`,
`graph_parents`, `graph_children`, `graph_descendants`, `graph_ancestors`,
`graph_shortest_path`, `graph_reachable`, `graph_cycle`, `graph_component_size`,
`graph_max_degree`. One random directed graph (`size` scales node/edge count) is
rendered as a shuffled `A -> B` edge list; each op recomputes its gold from the
graph. Answers follow GraphWalks' `Final Answer:` cue — a JSON node list, an
integer, or Yes/No (`answer_type` `list`/`int`/`string`); op params live in `meta`.

## Usage

```bash
python -m ruler_pp --list                       # list task names
python -m ruler_pp --task all --max_seq_length 4096 --num_samples 20
python -m ruler_pp --task count_predicate --max_seq_length 16384 \
    --tokenizer hf:Qwen/Qwen2.5-0.5B --output_dir data
```

Flags: `--task` (name or `all`), `--max_seq_length` (token budget incl. answer),
`--num_samples`, `--seed`, `--num_fewshot` (0 = zero-shot), `--tokenizer`
(`hf:<name>` | `tiktoken[:enc]` | `whitespace`; unavailable specs fall back to
tiktoken then whitespace), `--output_dir` (defaults to a `YYYY-MM-DD_HH-MM-SS`
timestamp dir; one `<task>.jsonl` file is written per task), `--format`
(`pt` | `sft`).

## Output format (`--format`)

Both formats carry the same metadata; they differ only in the trainable field:

- **`pt`** (default, regular/pretraining data): a single flat `"text"` field =
  `input + "\n" + answer_prefix + " " + answer` (one continuous string).
- **`sft`**: an OpenAI-style `"messages"` array so chat templates apply cleanly:
  `[{"role":"user","content": <input>}, {"role":"assistant","content": <answer_prefix + answer>}]`.
  (Few-shot examples, if any, are prepended into the user turn.)

## Output schema (one JSON object per line)

```jsonc
{
  "id": "count_predicate-000003",
  "task": "count_predicate",
  "input": "...",            // full prompt the model sees, EXCLUDING answer_prefix
  "instruction": "...",      // top line
  "context": "...",          // variable-length body (used by the verifier)
  "question": "...",
  "answer_prefix": "The count is",  // optional answer lead-in (may be empty)
  "answers": ["85"],         // acceptable surface forms
  "gold": 85,                // structured gold (int / list / map / number)
  "answer_type": "int",      // int | number | list | map | string
  "length": 4043,            // token length of input + answer_prefix
  "max_seq_length": 4096,
  "num_fewshot": 0,
  "seed": 42,
  "meta": { "predicate": "between", "low": 259, "high": 610 }
}
```

## Viewing samples (Gradio)

Eyeball generated data one example at a time — a dropdown per task, a slider over
N random samples, SFT rendered as a user/assistant chat and PT as flat text.

```bash
python -m ruler_pp.viz --data_dir data            # local
python -m ruler_pp.viz --data_dir data --share    # public link others can open
```

For `graph_*` tasks the viewer also **draws the graph** (the operation's focus
node(s) in red, the gold-answer nodes in green) when `networkx` + `matplotlib`
are installed; without them it falls back to the text view. Flags: `--data_dir`,
`--num` (samples per task, default 10), `--seed`, `--share`, `--port`. To extend
the rendering, edit `_render()` in `ruler_pp/viz.py` (one function maps a row ->
panel updates).

## Verifying golds

`scripts/verify_golds.py` independently re-parses each rendered `context` and
recomputes the answer (it does not trust the generator's bookkeeping); it covers
every synthetic task, including the graph family:

```bash
python scripts/verify_golds.py data    # -> TOTAL: N/N samples verified
```

## Adding a task

1. Add `ruler_pp/tasks/my_task.py` with a `Task` subclass implementing
   `build(size, rng) -> Sample` and a `name`. Compute the gold from the data you
   actually rendered (robust to sampling quirks).
2. Register it in `ruler_pp/registry.py`.
3. Optionally add a branch to `scripts/verify_golds.py`.
