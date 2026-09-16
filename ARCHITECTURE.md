# ClaimRAG — architecture and design note

## The problem, restated

A claim decision is not a question with an answer in the document. It is a set of independent
conditions — is the treatment in scope, has the waiting period run, does an exclusion bite,
does a sub-limit apply, is the paperwork sufficient — each of which lives in a different part
of a long policy wording, and **any one of which can flip the outcome**. A single retrieval
query cannot serve that, and a single LLM call cannot be held to account for it.

So the system is built around three commitments:

1. **Decompose before retrieving.** The case is turned into decision dimensions first; each
   dimension gets its own queries. One claim produces 8–15 retrievals, not one.
2. **Nothing is true because the model said it.** Every material statement is checked against
   retrieved policy text by a component that did not produce it.
3. **Abstention is a first-class output**, not an error path. `NEEDS_REVIEW` is reached by
   design in three distinct ways (missing case facts, unresolved dimensions, failed validation).

## Agent boundaries

The boundaries are drawn where **the failure modes differ**, which is the only reason to split
an agent at all.

- **Case Analysis** fails by mis-reading the case. It sees the claim and no policy text, so it
  structurally cannot assert a policy rule. Its output is a plan, not a conclusion.
- **Policy Evidence** fails by not finding the clause. It contains no LLM at all — it is
  deterministic retrieval code, which makes retrieval quality separately measurable and
  separately fixable.
- **Coverage & Exclusion** fails by mis-reading the clause. It sees case facts and evidence but
  is forbidden from choosing a status, so a coverage error surfaces as a wrong *finding* with a
  citation attached to it, which is inspectable.
- **Decision** fails by weighing findings badly. It sees the specialist's structured output and
  picks exactly one status. It is the only component that may write `decision`.
- **Validation** fails by being lenient. It re-retrieves independently, so it cannot inherit the
  decision agent's mistake, and it is the only component that may demand a revision.

The test is whether the responsibilities are separable, and they are: the Policy Evidence agent
can be swapped for a different retriever without touching a prompt, and the Validation agent can
be run against any decision, including one produced by a human.

## State flow

One typed object, `CaseState` (`app/models.py`), is the entire inter-agent protocol. No agent
passes prose to another. The fields each agent may write are disjoint:

```
CaseState
├── case            ← input
├── analysis        ← Agent 1   (facts, dimensions, missing fields, irrelevant attributes)
├── evidence[]      ← Agent 2   (chunk + dense rank + sparse rank + fusion + rerank score)
├── coverage        ← Agent 3   (findings, limits, exclusions, unresolved dimensions)
├── decision        ← Agent 4   (+ confidence, citations, applicable_limits, next_action)
├── validation      ← Agent 5   (PASS/FAIL + unsupported claims)
└── trace[]         ← every agent, via a context manager that records timings
```

Control flow is an explicit state machine with one conditional edge (`app/graph.py`):
`analyze → retrieve → assess → decide → validate`, with `validate:FAIL → decide` at most
`MAX_VALIDATION_RETRIES` times, then forced abstention. A framework was not used because the
graph is five nodes and the interesting logic is the edge condition, which deserves to be read
directly and unit-tested directly.

## Retrieval design

**Chunking.** Heading-driven. Per-span font metadata gives a body font size; headings are
detected by relative size, boldness, numbering pattern and length; a heading stack yields a
section path; chunks never cross a heading boundary; over-long clauses split at sentence
boundaries with a small overlap; the section path is prepended to the chunk text so it is
matchable and citable. Metadata carried: `chunk_id`, `page`, `page_end`, `section`,
`section_path`, `source`, `kind` (prose/definition), `char_len`.

**Hybrid.** Dense (`bge-small-en-v1.5`, cosine, FAISS with a numpy fallback) and sparse
(BM25 with a tokenizer that preserves `pre-existing`, `sub-limit`, `48`) run in parallel and
are fused with Reciprocal Rank Fusion. RRF rather than score normalisation, because BM25 and
cosine scores are not on a comparable scale and BM25's range moves per query; RRF needs only
the rankings.

**Reranking.** A cross-encoder (`bge-reranker-base`) scores ~20 fused candidates and applies a
score floor, so a query with no good clause returns little rather than returning the
least-bad clause confidently. Degradation is explicit: if the reranker cannot load, fusion
order is kept and `reranker_available: false` appears in `/health` and in every response.

**Observability.** Every `EvidenceItem` carries dense rank, sparse rank, fusion score, rerank
score, final rank and the query that found it — which is what makes `evidence_recall_at_k` and
citation-quality metrics computable offline instead of guessed.

## Important trade-offs

| Choice | Bought | Paid |
|---|---|---|
| Per-dimension retrieval | buried clauses are reachable; multi-section decisions possible | 8–15 retrievals per case; higher latency |
| Cross-encoder rerank | precision where it matters, before the LLM sees anything | ~0.3–1 s per query on CPU |
| Validation re-retrieves | validator cannot inherit the decider's error | ~2× retrieval, +1 LLM call per case |
| Deterministic invariants in code | the abstention guarantees are testable, not promised | some decisions the model got right get downgraded |
| Structured JSON everywhere + repair turn | machine-readable, validated output | occasional extra LLM call on schema failure |
| Hand-written state machine | readable, testable control flow; no framework failure modes | must be rewritten if the graph gains real branching |
| Expected decisions as *sets* | honest about genuinely defensible alternatives | a coarser accuracy number |

## What I would do next, in order

1. **Label the 12 public cases properly** — expected status plus the clause that decides it.
   Decision accuracy means little until the labels are as defensible as the system.
2. **A clause-type classifier at index time** (definition / waiting period / exclusion / limit /
   procedure) to let each dimension retrieve within its own clause family. This is the single
   change most likely to move `evidence_recall_at_k`.
3. **Amount computation** with the arithmetic done in Python from clause-extracted parameters,
   not by the model — the model should extract "1% of Sum Insured per day", not multiply.
4. **Self-consistency on borderline cases only** (n=3 at the Decision node when confidence lands
   in 0.4–0.7), which buys stability where variance actually hurts without tripling cost.
