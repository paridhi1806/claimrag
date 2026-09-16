# ClaimRAG — evidence-grounded health-insurance claim analysis

A small production-style system that analyses health-insurance claim cases against a single
authoritative policy wording (CSC – Individual Health Insurance, Universal Sompo,
UIN `UNIHLIP18004V011718`) using hybrid RAG and a five-agent workflow.

The design goal is not fluency. It is that **every material statement in a decision can be
traced to a clause in the supplied policy, and that the system abstains when it cannot.**

---

## 1. Architecture

```
                 ┌──────────────────── one-time ─────────────────────┐
  policy.pdf ──► │ layout-aware chunker → chunks.jsonl               │
                 │   ├─ BM25 index      (rank_bm25)                  │
                 │   └─ dense index     (bge-small + FAISS)          │
                 └───────────────────────────────────────────────────┘

  claim case JSON
        │
        ▼
  ┌─────────────────────┐  CaseAnalysis         facts, decision dimensions,
  │ 1 CaseAnalysisAgent │ ────────────────────► retrieval plan, missing fields
  └─────────────────────┘
        │  CaseState
        ▼
  ┌─────────────────────┐  per dimension:  dense ─┐
  │ 2 PolicyEvidenceAgent│                        ├─ RRF ─► cross-encoder rerank ─► EvidenceItem[]
  └─────────────────────┘                  BM25 ──┘          (dense/sparse rank, scores kept)
        │  CaseState
        ▼
  ┌─────────────────────┐  CoverageAssessment   findings, waiting-period effects,
  │ 3 CoverageAgent      │ ───────────────────► exclusions, limits, missing evidence
  └─────────────────────┘
        │  CaseState
        ▼
  ┌─────────────────────┐  DecisionDraft  ──► deterministic invariants ──► Decision + citations
  │ 4 DecisionAgent      │◄──────────────────────────────┐
  └─────────────────────┘        feedback: unsupported   │
        │  CaseState                                     │
        ▼                                                │
  ┌─────────────────────┐  independent re-retrieval per claim + entailment test
  │ 5 ValidationAgent    │ ── FAIL ───────────────────────┘  (≤ MAX_VALIDATION_RETRIES)
  └─────────────────────┘ ── FAIL again ──► forced NEEDS_REVIEW
        │
        ▼
   AnalyzeResponse  (decision, confidence, findings, limits, missing evidence,
                     citations, validation, retrieval metadata, trace)
```

Agents communicate through one typed object, `CaseState` (`app/models.py`), never through
free-form text. Each agent reads specific fields and writes specific fields.

| # | Agent | Reads | Writes | Owns |
|---|-------|-------|--------|------|
| 1 | `CaseAnalysisAgent` | raw case | `analysis` | fact extraction, decision dimensions, missing fields, irrelevant attributes |
| 2 | `PolicyEvidenceAgent` | `analysis.dimensions` | `evidence` | hybrid retrieval, fusion, reranking, retrieval metadata (no LLM) |
| 3 | `CoverageAgent` | `analysis`, `evidence` | `coverage` | scope, waiting periods, exclusions, limits, unresolved dimensions |
| 4 | `DecisionAgent` | `analysis`, `coverage`, `evidence` | `decision`, `citations`, `limits` | the single final status + calibrated confidence |
| 5 | `ValidationAgent` | `key_findings`, `citations`, retriever | `validation` | independent evidence check, revision/abstain signal |

---

## 2. Quick start

```bash
git clone <this-repo> && cd claimrag
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
cp .env.example .env                                   # then set LLM_API_KEY

# 1. put the supplied materials in place
cp /path/to/policy.pdf        data/policy/policy.pdf
cp /path/to/PUB-*.json        data/cases/public/

# 2. build the index (downloads ~200 MB of models on first run)
make index                     # python -m app.ingest.build_index --pdf data/policy/policy.pdf

# 3. run
make api                       # http://localhost:8000/docs
make ui                        # http://localhost:8501   (set API_URL if not local)

# 4. verify
make test
make eval
```

No key handy? `LLM_STUB=1` runs the whole pipeline offline with a deterministic stub. It
always abstains, so stub output is never mistaken for a real decision.

---

## 3. API

### `POST /analyze`

Accepts `{"case": {...}}` or a bare case object. Only `case_id` is required; all other
fields are preserved as supplied (public cases carry heterogeneous shapes and none are dropped).

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' -d '{
  "case": {
    "case_id": "CAND-003",
    "policy": {"inception_date": "2019-08-20", "sum_insured": 700000,
               "continuous_months_covered": 61},
    "treatment": {"diagnosis": "Dorsal hump, nasal contour asymmetry",
                  "procedure": "Elective rhinoplasty (aesthetic reshaping)",
                  "admission_date": "2024-11-14"},
    "expenses": {"total_claimed": 212000},
    "documents_available": ["discharge summary", "final bill"],
    "investigation_task": "Determine whether the procedure falls within the exclusions."
  }
}' | jq
```

```jsonc
{
  "case_id": "CAND-003",
  "decision": "NOT_ADMISSIBLE",
  "confidence": 0.86,
  "abstained": false,
  "key_findings": [
    {"dimension": "exclusions",
     "statement": "Cosmetic/plastic surgery is excluded unless necessitated by accident or burns; the file documents no functional impairment.",
     "verdict": "EXCLUDES_COVERAGE",
     "chunk_ids": ["p019-permanent-exclusions-03-9f1c2a"],
     "material": true}
  ],
  "applicable_limits": [],
  "missing_evidence": [],
  "next_action": "Reject citing the cosmetic-surgery exclusion; inform the claimant of the appeal route.",
  "citations": [
    {"claim": "Cosmetic or plastic surgery is excluded unless necessitated by accident or burns.",
     "source": "policy.pdf", "page": 19, "section": "Permanent Exclusions",
     "chunk_id": "p019-permanent-exclusions-03-9f1c2a", "support_score": 4.12}
  ],
  "validation": {"status": "PASS", "unsupported_claims": [], "checked_claims": 3,
                 "notes": "3/3 material claims evidence-backed."},
  "retrieval": {"evidence_count": 18, "pages_cited": [19], "top_rerank_score": 4.12,
                "reranker_available": true, "dimensions": [/* per-dimension queries + status */],
                "llm_calls": 4},
  "trace": [
    {"agent": "CaseAnalysisAgent", "action": "analyze_case", "detail": "dimensions: exclusions, scope_of_cover, …", "results": 5, "elapsed_ms": 2140},
    {"agent": "PolicyEvidenceAgent", "action": "retrieve", "detail": "dimension=exclusions", "results": 6, "elapsed_ms": 310}
  ],
  "elapsed_ms": 11840
}
```

Errors: `422` malformed body or missing `case_id`; `503` index not built or LLM unconfigured;
`500` returns `{"error", "detail"}` and never a decision.

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | readiness: index loaded, chunk/page counts, reranker status, LLM configured |
| `GET` | `/cases` | bundled public + custom case ids (used by the UI) |
| `GET` | `/cases/{id}` | one bundled case |
| `GET` | `/index/stats` | chunk count, page count, section count, models in use |
| `GET` | `/docs` | OpenAPI |

---

## 4. Design decisions and trade-offs

**Chunking is heading-driven, not size-driven.** A policy wording is a clause hierarchy; a
waiting period split from its duration is worse than useless, and a chunk without its heading
cannot be cited. `app/ingest/chunker.py` reads per-span font metadata, infers the body font
size, classifies headings by relative size / boldness / numbering, maintains a heading stack,
and never packs across a heading boundary. The heading path is prepended to the chunk text so
both the embedder and BM25 see it. *Trade-off:* this depends on the PDF having a usable text
layer and consistent typography; a scanned policy would need OCR first, and the builder fails
loudly rather than silently emitting garbage.

**Hybrid retrieval with RRF, then a cross-encoder.** Claim decisions hinge on exact tokens
("48 months", "sub-limit", "co-payment") that dense vectors blur, and on paraphrases ("knee
replacement" vs "joint replacement surgery") that BM25 misses. Reciprocal Rank Fusion is used
instead of score normalisation because BM25 scores and cosine similarities are not comparable
and BM25's range shifts per query. The cross-encoder then does the precision work on ~20
candidates. *Trade-off:* the reranker adds ~0.3–1 s per query on CPU; it degrades explicitly
(fusion order, `reranker_available: false` in `/health` and in every response) rather than
silently.

**One retrieval per decision dimension.** The Case Analysis agent emits dimensions
(`waiting_period_pre_existing`, `room_rent_limit`, …) with their own queries, and evidence is
merged across them. This is what lets one decision combine clauses from several parts of a
long document, and it is why a buried clause is reachable at all. A baseline dimension set is
always appended, so scope/waiting periods/exclusions/limits are investigated even if the model
forgets them.

**Validation re-retrieves instead of re-reading.** The Validation agent runs a *fresh*
retrieval for each material statement, so it does not inherit the decision agent's evidence
pool, and asks a narrow entailment question per claim. One FAIL triggers exactly one revision
with the unsupported claims attached; a second FAIL forces `NEEDS_REVIEW`. *Trade-off:* this
roughly doubles retrieval cost and adds one LLM call per case, which is the price of not
shipping unsupported conclusions.

**Model-proposed, code-enforced.** Statuses are decided by an LLM but the invariants are not:
hallucinated `chunk_id`s are stripped against the retrieved evidence, an uncited decision is
forced to `NEEDS_REVIEW`, `ADMISSIBLE` + any limit becomes `ADMISSIBLE_WITH_LIMITS`,
unresolved dimensions cap confidence at 0.6, abstention caps it at 0.4, and any exception
anywhere degrades to abstention rather than a guess. These are the behaviours covered by
`tests/test_pipeline.py`.

**A hand-written state machine, not an agent framework.** The control flow is five nodes and
one conditional edge. Writing it directly (`app/graph.py`, ~120 lines) keeps the transition
rules — the part that decides when to abstain — readable and unit-testable, and avoids a
dependency whose failure modes would be harder to explain than the code it replaces. LangGraph
would be a drop-in if the graph grew branches.

**No chain-of-thought is exposed.** Traces carry agent name, action, short detail, result
counts and timings. Prompts explicitly instruct the models to output conclusions, not
reasoning steps, and `tests/test_api.py` asserts the trace stays clean.

---

## 5. Evaluation

```bash
make eval                                  # all bundled cases, in-process
python -m evaluation.run_eval --set custom # candidate cases only
python -m evaluation.run_eval --api https://<deployed-api>   # score the deployment
```

Writes `evaluation/results/REPORT.md`, `results.json` and `scores.csv`.

**Metrics** (`evaluation/metrics.py`)

| Metric | Definition |
|---|---|
| `decision_accuracy` | share of labelled cases whose status is in the expected set |
| `evidence_recall_at_k` | share of the clause anchors a human reviewer needs that appear in the evidence actually passed to the agents (measures chunking + fusion + rerank end-to-end, not embeddings alone) |
| `citation_lexical_overlap` | share of citations whose cited clause lexically covers ≥30% of the claim's content words — an LLM-free proxy for citation correctness |
| `citation_support_rate` | share of material claims the independent entailment check confirmed |
| `findings_citation_coverage` | share of material findings carrying ≥1 `chunk_id` |
| `abstention.recall` / `.precision` | did the system abstain when it should, and only then |
| `attribute_leakage_cases` | irrelevant case attributes that leaked into decision text |

**How expected outcomes are established.** Each candidate case was written backwards from one
clause family in the policy (waiting periods, permanent exclusions, sub-limits, definition of
Hospital, documentation requirements); the expected status is whatever that clause dictates,
recorded with its rationale in `evaluation/expected/expectations.json`. Expected decisions are
recorded as a *set* where two statuses are defensible before the sub-limit is read
(e.g. `ADMISSIBLE` vs `ADMISSIBLE_WITH_LIMITS`). For the 12 supplied public cases, add a block
per case to the same file; any case without a label is reported as **UNLABELLED** rather than
silently scored.

**Candidate-created cases** (`evaluation/cases/`, 7 cases, covering every required reliability
scenario):

| Case | Scenario exercised | Expected |
|---|---|---|
| CAND-001 | day-care cataract, category sub-limit changes the payable amount | `ADMISSIBLE_WITH_LIMITS` |
| CAND-002 | undeclared pre-existing disease inside the waiting period | `NOT_ADMISSIBLE` |
| CAND-003 | cosmetic-surgery exclusion; clause buried deep in the document | `NOT_ADMISSIBLE` |
| CAND-004 | **abstention** — maternity option not stated in the file | `NEEDS_REVIEW` |
| CAND-005 | **abstention** — no discharge summary, no diagnosis, hospital definition untestable | `NEEDS_REVIEW` |
| CAND-006 | covered surgery + room-rent proportionate deduction; contains two irrelevant attributes (hobby, employer) that must not appear in findings | `ADMISSIBLE_WITH_LIMITS` |
| CAND-007 | **abstention** — a required condition (date of first diagnosis) cannot be established; file self-contradicts | `NEEDS_REVIEW` |

Three abstention cases are included where two are required, because over-confidence is the
failure mode this system is built to avoid.

> **Results.** `evaluation/results/REPORT.md` is generated by the command above and is the
> single source of reported numbers; it is regenerated per model and per index build, so no
> metric is hard-coded into this README.

### Reliability scenarios → where they are handled

| Scenario | Mechanism |
|---|---|
| Clause buried in a long document | heading-aware chunking + BM25 half of the hybrid + per-dimension queries |
| Decision needs several sections | one retrieval per dimension, merged evidence pool |
| Waiting period flips the outcome | dedicated dimension + prompt requires the arithmetic (months of cover vs clause) in the statement |
| Category sub-limit changes the amount | `applicable_limits[]` with its own citations; `ADMISSIBLE` + limits auto-becomes `ADMISSIBLE_WITH_LIMITS` |
| Insufficient evidence | `missing_fields` → `missing_evidence` → `NEEDS_REVIEW`; confidence capped at 0.4 |
| Irrelevant attribute present | `irrelevant_attributes` in the analysis; leakage measured in the eval |
| Required condition unestablishable | `unresolved_dimensions` caps confidence and pushes toward abstention |
| Model claims what the policy does not say | chunk_id stripping + Validation agent + one revision + forced abstention |

---

## 6. Failure analysis

Three failures observed while building, their root cause, and the change made. Each has a
regression test.

**F1 — Confident decisions citing clauses that did not mention the point.**
Early runs produced statements like "dental treatment is excluded for 24 months" citing the
general exclusions chunk, which says nothing about dental. *Root cause:* the decision model was
free to pick any `chunk_id` from a large evidence pool, and nothing checked the pairing.
*Fix:* (a) `chunk_id`s not present in the retrieved pool are stripped and the finding is
demoted to `INCONCLUSIVE`; (b) the Validation agent re-retrieves per claim and runs a narrow
entailment test; (c) a decision with zero surviving citations is forced to `NEEDS_REVIEW`.
Covered by `test_hallucinated_chunk_ids_are_stripped`, `test_uncited_decision_is_forced_to_abstain`,
`test_validation_failure_forces_needs_review`.

**F2 — Waiting-period clauses missed by dense-only retrieval.**
Queries phrased in claimant vocabulary ("knee replacement after 11 months") retrieved
scope-of-cover prose and missed the specific-illness waiting table, because the decisive tokens
are numerals and the table's language is generic. *Root cause:* single embedding query, no
lexical channel, and chunks that had been split away from their headings. *Fix:* BM25 added and
fused via RRF; the heading path is prepended to every chunk so "Waiting Periods > Specific
Diseases" is itself matchable; per-dimension queries are written in policy vocabulary rather
than the claimant's. `evidence_recall_at_k` is the metric that tracks this.

**F3 — Abstention cases answered confidently.**
CAND-005 (no discharge summary, no diagnosis) initially returned `ADMISSIBLE` at 0.8: the model
retrieved a plausible scope-of-cover clause and filled the gaps from general insurance
knowledge. *Root cause:* nothing in the pipeline distinguished "the policy says X" from "the
policy does not tell us". *Fix:* the Case Analysis agent emits `missing_fields` before any
retrieval happens; the Coverage agent must declare `unresolved_dimensions` when clauses are
absent or ambiguous; both feed `missing_evidence`; unresolved dimensions cap confidence and
abstention caps it at 0.4; and the prompts state explicitly that abstention is a correct
answer. Tracked by `abstention.recall` / `.precision` in the eval.

**F4 — Section hierarchy collapsing on some documents (caught by `tests/test_chunker.py`).**
Every chunk came back with a single-element `section_path`, so citations lost the parent
section ("Pre-existing Diseases" with no "Waiting Periods" above it) and heading-based
retrieval got weaker. *Root cause:* body font size was inferred as the statistical mode over
*lines*. In a page dense with short headings and sparse body text, a heading size wins the
mode, the "is this bigger than body text?" test then matches nothing, and every heading is
treated as top-level. *Fix:* body size is now the size carrying the most **characters**
(`_body_size`), which is robust to heading-heavy pages. `test_sections_and_metadata` asserts
the full parent path.

### Known limitations

- Single-document knowledge base. Endorsements, the customer information sheet and IRDAI
  circulars are not indexed; anything they would change is out of scope by construction.
- Monetary computation is described, not calculated. The system names the applicable limit and
  its basis; it does not produce a settled payable figure.
- Expected outcomes are the author's reading of the policy, not an adjudicator's. The eval
  therefore measures consistency with a documented reading, and each label ships with its
  rationale so a reviewer can disagree with the label rather than with a number.
- The reranker and embedder run on CPU; a cold container pays ~60–90 s of model download and
  ~10–20 s of load. Free-tier hosts that sleep will show this on the first request.
- The LLM is non-deterministic even at `temperature=0`; run-to-run variance on borderline cases
  is real and is why expected decisions are sets, not single values.
- Scanned or image-only policy PDFs are not supported (no OCR); the index builder fails loudly.

---

## 7. Deployment

**API (Render, free tier)** — `render.yaml` is included; it builds the Dockerfile, sets
`healthCheckPath: /health`, and expects `LLM_API_KEY` as a synced-off secret. If the policy PDF
is committed, the index is built at image build time so the first request is fast.

**UI (Streamlit Community Cloud)** — point it at `ui/streamlit_app.py` and set `API_URL` in the
app's secrets to the deployed API. The UI runs standalone and holds no key.

**Hugging Face Spaces** is an alternative for both in one container.

Secrets live only in environment variables; `.env` is git-ignored and `.env.example` documents
every variable.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | any OpenAI-compatible provider |
| `LLM_STUB` | `1` runs the pipeline offline with a deterministic abstaining stub |
| `POLICY_PDF`, `INDEX_DIR`, `PUBLIC_CASES_DIR` | data locations |
| `EMBEDDING_MODEL`, `RERANKER_MODEL`, `DENSE_TOP_K`, `SPARSE_TOP_K`, `RRF_K`, `RERANK_TOP_N` | retrieval tuning |
| `CHUNK_TARGET_CHARS`, `CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS` | chunking |
| `MAX_VALIDATION_RETRIES` | revision budget before forced abstention |

---

## 8. Repository layout

```
app/
  config.py            env-driven settings
  models.py            CaseState + every agent contract (the typed message bus)
  graph.py             the five-node state machine and the abstention rules
  api.py               FastAPI: /analyze, /health, /cases, /index/stats
  ingest/chunker.py    layout-aware, heading-preserving PDF chunker
  ingest/build_index.py  CLI: PDF -> chunks + BM25 + dense index
  retrieval/           dense.py, sparse.py, fusion.py (RRF), rerank.py, hybrid.py
  llm/client.py        OpenAI-compatible JSON client, repair turn, offline stub
  agents/              case_analysis, policy_evidence, coverage, decision, validation
ui/streamlit_app.py    reviewer UI: decision, evidence, trace, abstention banner
evaluation/            run_eval.py, metrics.py, cases/ (7 candidate cases), expected/
tests/                 chunker, retrieval, pipeline invariants, API contract
ARCHITECTURE.md        1–2 page design note
```
