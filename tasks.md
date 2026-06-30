# Tasks — what each one is, and the long-context skill it trains

This pipeline produces long-context training data from several generators. They
fall into two camps:

* **`ruler_pp` synthetic tasks** — algorithmic, no LLM. A shared driver binary-searches
  an opaque `size` knob to fill a token budget, and every gold is computed
  deterministically from the generated context (an independent re-checker,
  `scripts/verify_golds.py`, re-parses the context to confirm). 22 tasks in three
  families.
* **Corpus generators** — run over a FinePDFs sample through one shared loader
  (`corpus/loader.py`: parquet streaming + quality filters). Some are deterministic
  (`untie`, `mrcr`, `idk`); two are LLM-in-the-loop (`multihop`, `summarize`) and run
  in two offline phases around an inference call.

The unifying idea: each task isolates a *specific* long-context competency and forces
the model to use the **whole** window — answers are constructed so that no local
shortcut (recency, a single span, topical gist) suffices.

---

## 1. `ruler_pp` synthetic family

All synthetic tasks share the `Sample` contract (`instruction` + `context` +
`question`, with a structured `gold`). `size` scales the context length; the answer
is small and exact. Because the content is random tokens / coded words / numbers,
there is **no pretraining leakage** — the model genuinely has to read this context.

### 1a. Retrieval & tracking (RULER-faithful)

**`common_words` (CWE).** The context is a shuffled, numbered list of words. Ten
"common" words are each repeated 30 times; a tail of `n_uncommon` words (the tail is
what `size` scales) are each repeated 3 times. The question asks for the 10 most
common words. *Long-context skill:* frequency aggregation over the entire window. The
signal (30×) is buried among a growing crowd of 3× distractors and **shuffled**, so
the model cannot rely on position or a local window — it must maintain counts across
the full context and separate signal from a heavy distractor floor.

**`freq_words` (FWE).** A stream of 6-character coded words sampled from a Zipf/zeta
distribution (α=1.5); the single most frequent token is rendered as `...` noise. The
answer is the next three most frequent coded words. *Long-context skill:* counting
under a heavy-tailed distribution while **ignoring a designated noise token**, and
discriminating between ranks whose frequencies are close. Coded words can't be guessed
— only counted.

**`variable_tracking` (VT).** A chain of assignments — `VAR A = 12345`, `VAR B = VAR A`,
`VAR C = VAR B`, … (4 hops) — is scattered line-by-line through `size` lines of noise.
All variables in the chain resolve to the same value; the answer is every variable name
holding it. *Long-context skill:* **multi-hop coreference / dereferencing** across long
distances. The links sit far apart, so the model must follow a pointer chain through
noise and resolve transitive identity — the canonical "track the entity through the
document" probe.

### 1b. Counting, filtering & aggregation

These build context either as numbered word/number lists or as **records** (each a
`name`, a `category`, and a numeric `value`, via `wordbank.make_records`). The skill
gradient runs from exact counting → predicate filtering → grouped aggregation.

**`count_occurrences`.** Numbered word list; count how many times one target word
appears (planted 2–`size/20` times among other words repeated 1–4×). Gold is the
*empirical* count. *Skill:* an exhaustive, exact scan — the model must count every hit,
not estimate, and not stop early.

**`count_distinct`.** Word list with repeats; count the unique words. *Skill:* maintain
a **set** over the whole stream and deduplicate across the entire span — a different
memory structure (membership) than running tallies.

**`count_predicate`.** A record list; count records matching one of four predicates:
`value > t`, `value < t`, `category == c`, or `a <= value <= b`. *Skill:* uniform
**filtering over structured rows** spread across the full context, applying a numeric or
categorical test consistently to every record.

**`numeric_agg`.** A numbered list of integers; compute `sum` / `min` / `max` / `mean` /
`median`. *Skill:* numeric aggregation over *all* operands — carry running state (sum,
extrema) or hold the full multiset (median) across the window.

**`tally_by_category`.** A record list; return the full histogram — every category with
its count (a `map` answer). *Skill:* grouped counting **plus complete enumeration**: the
model is penalised for missing any category, so it must cover the whole context, not
sample it.

**`group_by`.** SQL-style `GROUP BY`: for each category, sum the values of its records.
*Skill:* grouped aggregation that combines filtering (which group) + arithmetic (sum
within group) + complete enumeration — strictly harder than `tally_by_category`.

**`top_k`.** Word list with a few "heavy" words at strictly separated high counts and
many "light" filler words (count 1–2). Return the `k` (3 or 5) most frequent **in
descending order**. *Skill:* frequency ranking *and* ordering, with an unambiguous top-k
boundary — the model must both count and sort.

**`set_ops`.** Two shuffled word lists, Set A and Set B (with a deliberate overlap).
Compute their intersection, union size, or A∖B. *Skill:* **cross-referencing two large
collections** held in different regions of the context — set reasoning that requires
both lists fully in working memory at once.

**`variable_arithmetic`.** Like VT, but the chain applies arithmetic: `VAR A = 7`,
`VAR B = VAR A + 3`, `VAR C = VAR B * 2`, … (4 hops, `+`/`-`/`*`), hidden in noise. Return
the final integer value of the last variable. *Skill:* multi-hop tracking **fused with
sequential computation** — the model must both retrieve each scattered link and carry an
arithmetic accumulator across hops.

### 1c. Graph-reasoning family (GraphWalks-style)

One random directed graph (node IDs are random 10-hex strings; every node has degree
≥ 1; avg degree ≈ 2.5) is rendered as a **shuffled `A -> B` edge list**, and a family of
ten operations query it. `size` scales the node/edge count. This is the hardest
synthetic family: the graph's structure exists **only** as edges scattered across the
context, so the model must assimilate the *entire* edge list to reconstruct the global
object before it can answer — no single line is sufficient.

* **`graph_children` / `graph_parents`** — direct out-/in-neighbours of a node (1-hop
  local lookup, but the relevant edges are scattered).
* **`graph_bfs`** — every node exactly *d* steps from a start node (bounded-depth
  traversal).
* **`graph_descendants` / `graph_ancestors`** — the full forward / backward transitive
  closure from a node (unbounded traversal).
* **`graph_shortest_path`** — the edge count of the shortest directed path between two
  nodes.
* **`graph_reachable`** — yes/no: is the target reachable from the source? (balanced
  yes/no by construction).
* **`graph_cycle`** — yes/no: does the graph contain a directed cycle? (≈50% of samples
  are DAGs so the answer is balanced; checked by Kahn's topological peel).
* **`graph_component_size`** — size of the weakly-connected component containing a node
  (graph is built sparser, avg degree 1.3, so components actually differ).
* **`graph_max_degree`** — the node(s) with the highest total (in+out) degree.

*Long-context skill (whole family):* **global structure assembly + multi-hop graph
traversal**. Local-attention shortcuts fail completely — reachability, closures, cycles,
and component size each require integrating edges that are deliberately spread through
the window and reasoning over the assembled whole.

---

## 2. Corpus generators (over real FinePDFs text)

### 2a. `corpus.cluster` — long, *coherent* contexts (supporting stage)

Not a training task itself: it embeds docs (Qwen3-Embedding-0.6B), finds exact top-k
cosine neighbours, and greedily chains near-docs into ~128k-token clusters
(`clusters.parquet`). *Why it matters for long context:* it manufactures genuinely long
contexts that are **topically coherent** rather than random concatenation, so downstream
tasks (chiefly `untie`'s cluster source) force the model to integrate related
information spread across many documents — closer to how real long documents read.

### 2b. `untie` — paragraph unshuffle ("Untie the Knots")

Splits a long source (a cluster of related docs, or a single long doc) into paragraph
units, **shuffles** them, and asks the model to recover the original order. Two flows:

* **`reconstruct`**: input = shuffled paragraphs (~n/2 tokens); answer = the full text in
  original order (~n/2 tokens).
* **`permutation`**: input = shuffled paragraphs (~n tokens); answer = the **label order**
  that reconstructs the text (a short list of indices).

*Long-context skill:* **global discourse coherence**. To order paragraphs the model must
attend to *every* unit and reason about narrative/logical flow across the entire window
— a holistic ordering task, not a local one. `reconstruct` additionally trains long
*ordered generation*; `permutation` isolates the same global reasoning behind a compact
index answer (cheap to grade, hard to fake).

### 2c. `mrcr` — marker retrieval among confusable distractors

A long numbered list of FinePDFs passages; each passage's text is fenced `<<<…>>>`,
followed by plain-English sentences saying what it is *marked as* (concrete nouns from
`res/markers.txt`, e.g. "This paragraph is a lemon. It is also a car."), and sometimes
what it is *not* ("but remember that it is not a dune."). **Exactly one** passage (the
target) is positively marked with *both* query markers — guaranteeing a unique gold;
confounders share one marker (some are negation near-misses, "a pepper … not a dune"),
the rest are noise. The query asks for "the paragraph that is both a {a} and a {b}", and
the answer reproduces that passage **verbatim**, prefixed with a per-sample random string
(a guard against degenerate output). Metric = `difflib.SequenceMatcher` ratio ∈ [0,1].
`size` (number of paragraphs) is binary-searched to fill the budget, so the same task
scales across 32k/128k/1M.

*Long-context skill:* **precise retrieval + disambiguation + verbatim reproduction**.
Unlike a single-needle search, the target must satisfy a **conjunction** of two markers,
which defeats single-keyword matching; the negation near-misses punish sloppy reading;
and reproducing the passage verbatim tests exact copying from deep in the context. It's
needle-in-a-haystack with many *confusable* needles.

### 2d. `idk` — calibrated answering & abstention

A long numbered list of FinePDFs passages; a subset get an appended **binding sentence**
stating a random code for a named item ("The access code for the lemon is 7F3A-21." —
markers from `res/markers.txt`, codes random). The question is a 4-way MCQ — *"What is the
{attribute} code for the {marker}?"* — with one option always **"I don't know"**
(shuffled to any letter). Three kinds:

* **answerable (~30%)**: the (attribute, marker) pair is bound → gold = its code;
* **unanswerable / absent**: the marker is in no binding → gold = "I don't know";
* **unanswerable / subtle (~15%)**: the marker *is* bound, but under a *different*
  attribute than asked → gold = "I don't know", and its real code is planted among the
  distractors as **bait**, so ignoring the attribute is punished.

Codes are random (leakage-proof); metric = accuracy on the chosen letter.

*Long-context skill:* **retrieval with calibrated abstention**. The model must locate a
specific binding among many near-identical ones *and* recognise when the asked fact is
genuinely absent — i.e. resist hallucinating an answer. The subtle case forbids the
"topic is present, so guess its code" shortcut and forces exact key (attribute+marker)
matching. Knowing *when not to answer* is a core long-context reliability property.

### 2e. `multihop` — multi-hop QA (LLM-in-the-loop)

The only generators that need an LLM to *write* the data run in two offline phases; the
repo never calls the model. **`prepare`** chunks each doc into ~512-token chunks and, per
doc, emits several requests (default 13, over-generated against an 8–10 pair target),
each sampling 3–4 **random** chunks. The system prompt asks for **one** QA pair whose
answer needs a fact from *every* chunk shown (true multi-hop, not answerable from any
single chunk), with the exact fallback phrase `cannot generate` when the chunks share no
usable common ground. **`assemble`** parses `Question:/Answer:` from the engine's
responses (dropping `cannot generate`), regroups per doc, and emits the final record
(`pt`: `doc\n\nQuestion:…Answer:…`; `sft`: numbered questions → numbered answers). The
final training context is the **whole document**.

*Long-context skill:* **multi-hop synthesis across dispersed regions**. By construction,
each answer requires combining facts from chunks that sit far apart in the document, so
the model is trained to integrate information across the long context rather than
retrieve a single span. Over-generation + the `cannot generate` escape hatch keep the
kept pairs genuinely multi-hop (low-overlap chunk sets are discarded, not forced).

### 2f. `summarize` — long-context summarization (LLM-in-the-loop)

Same two-phase shape as `multihop`, but simpler: **one request per doc** (summarization
always succeeds — no chunking, no over-generation). `prepare` sends the **full document
followed by the instruction** (instruction *at the end*, after the long context — the
layout `mrcr`/`idk` also use, so the model reads the doc before seeing the task);
`--summary_words` can set a target length. `assemble` cleans each summary (strips a stray
"Here is the summary:" preamble / wrapping quotes) and writes the final record
(`pt`: `doc\n\ninstruction\nsummary`; `sft`: user = `doc\n\ninstruction`, assistant =
`summary`).

*Long-context skill:* **global comprehension and faithful compression**. A good summary
forces the model to read and integrate the *entire* document — not just the head — and
to compress it without adding unsupported facts. The system prompt explicitly pushes
coverage of the whole arc and faithfulness, and the instruction-at-end layout is the
long-context-friendly placement.

---

## Quick-reference table

| Task | Generator type | Context (input) | What the model must do | Answer / metric | Long-context skill |
|---|---|---|---|---|---|
| `common_words` (CWE) | synthetic | shuffled numbered word list (10 words @30×, tail @3×) | find the 10 most frequent | word list / exact | frequency aggregation over the whole window vs. distractors |
| `freq_words` (FWE) | synthetic | Zipf-distributed coded-word stream, `...` = noise | top-3 coded words by frequency | word list / exact | counting under heavy tail; ignore designated noise |
| `variable_tracking` (VT) | synthetic | 4-hop `VAR x = VAR y` chain in noise | all vars sharing the value | name list / exact | multi-hop coreference / pointer chasing |
| `count_occurrences` | synthetic | numbered word list | count one target word | int / exact | exhaustive exact scan |
| `count_distinct` | synthetic | word list with repeats | count unique words | int / exact | set membership over full stream |
| `count_predicate` | synthetic | record list (name/category/value) | count rows matching `>`,`<`,`==`,`between` | int / exact | uniform filtering over structured rows |
| `numeric_agg` | synthetic | numbered integer list | sum/min/max/mean/median | number / exact | numeric aggregation over all items |
| `tally_by_category` | synthetic | record list | histogram: count per category | map / exact | grouped counting + complete enumeration |
| `group_by` | synthetic | record list | sum of values per category | map / exact | grouped aggregation (filter + sum + enumerate) |
| `top_k` | synthetic | word list (heavy + light) | top-k words, most→least frequent | ordered list / exact | frequency ranking + ordering |
| `set_ops` | synthetic | two word sets A, B | intersection / union size / A∖B | list or int / exact | cross-referencing two large lists |
| `variable_arithmetic` | synthetic | 4-hop arithmetic var chain in noise | final value of last var | int / exact | multi-hop tracking + sequential arithmetic |
| `graph_children` / `graph_parents` | synthetic | shuffled `A -> B` edge list | direct out-/in-neighbours | node list / exact | local lookup over scattered edges |
| `graph_bfs` | synthetic | edge list | nodes exactly *d* steps from start | node list / exact | bounded-depth traversal |
| `graph_descendants` / `graph_ancestors` | synthetic | edge list | forward / backward transitive closure | node list / exact | unbounded traversal over assembled graph |
| `graph_shortest_path` | synthetic | edge list | shortest-path length between two nodes | int / exact | global path reasoning |
| `graph_reachable` | synthetic | edge list | is target reachable from source? | Yes/No / exact | reachability over assembled graph |
| `graph_cycle` | synthetic | edge list (~50% DAGs) | does a directed cycle exist? | Yes/No / exact | global structural property |
| `graph_component_size` | synthetic | sparser edge list | size of node's weak component | int / exact | connected-component assembly |
| `graph_max_degree` | synthetic | edge list | node(s) of highest total degree | node list / exact | global aggregate over all edges |
| `corpus.cluster` | preprocessing | FinePDFs docs | embed + chain into ~128k clusters | clusters.parquet | builds long, *coherent* multi-doc contexts |
| `untie` (reconstruct) | corpus, deterministic | shuffled paragraphs of a long doc/cluster | output text in original order | text / order-correctness | global discourse coherence + long ordered generation |
| `untie` (permutation) | corpus, deterministic | shuffled paragraphs | output the index permutation | index list / exact | global ordering reasoning (compact answer) |
| `mrcr` | corpus, deterministic | numbered fenced passages + marker sentences | reproduce the passage marked as both *a* and *b* | verbatim passage / SequenceMatcher ratio | conjunctive retrieval + disambiguation + verbatim copy |
| `idk` | corpus, deterministic | numbered passages + random code bindings | 4-way MCQ incl. "I don't know" | option letter / accuracy | calibrated retrieval + abstention (no hallucination) |
| `multihop` | corpus, LLM-in-the-loop | the whole document | answer QA needing facts from dispersed chunks | Q/A pairs / (downstream) | multi-hop synthesis across far-apart regions |
| `summarize` | corpus, LLM-in-the-loop | the whole document | write a faithful whole-doc summary | summary / (downstream) | global comprehension + faithful compression |

> Synthetic tasks (`ruler_pp`) carry exact, independently re-checkable golds and scale to
> any token budget via the `size` search. `mrcr`/`idk` are deterministic corpus tasks with
> re-checkers (`scripts/verify_mrcr.py`, `scripts/verify_idk.py`). `untie`, `multihop`, and
> `summarize` produce open-ended targets (text / QA / summaries) graded downstream rather
> than by an exact-match checker.
