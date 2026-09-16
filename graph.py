"""Orchestrator — an explicit state machine over the five agents.

    analyze -> retrieve -> assess -> decide -> validate
                                       ^          |
                                       |  FAIL    |
                                       +----------+   (at most MAX_VALIDATION_RETRIES)

Kept as a small typed state machine rather than a framework: the control flow is
five nodes and one conditional edge, and the transition rules (when to abstain)
are the part we most need to read, test and explain.
"""
from __future__ import annotations

import time

from app.agents.case_analysis import CaseAnalysisAgent
from app.agents.coverage import CoverageAgent
from app.agents.decision import DecisionAgent
from app.agents.policy_evidence import PolicyEvidenceAgent
from app.agents.validation import ValidationAgent
from app.config import Settings, get_settings
from app.llm.client import LLMClient
from app.models import (
    AnalyzeResponse,
    CaseState,
    ClaimCase,
    Decision,
    TraceStep,
    UnsupportedClaim,
)
from app.retrieval.hybrid import HybridRetriever

ABSTAIN = {Decision.NEEDS_REVIEW}


class ClaimPipeline:
    def __init__(self, retriever: HybridRetriever | None = None,
                 llm: LLMClient | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.retriever = retriever or HybridRetriever.instance(self.s)
        self.llm = llm or LLMClient(self.s)
        self.analysis = CaseAnalysisAgent(self.llm)
        self.evidence = PolicyEvidenceAgent(self.retriever)
        self.coverage = CoverageAgent(self.llm)
        self.decision = DecisionAgent(self.llm)
        self.validator = ValidationAgent(self.llm, self.retriever)

    # ------------------------------------------------------------------ #
    def run(self, case: ClaimCase) -> AnalyzeResponse:
        t0 = time.perf_counter()
        state = CaseState(case=case)
        try:
            state = self.analysis.run(state)
            state = self.evidence.run(state)
            if not state.evidence:
                return self._abstain(
                    state, t0, "Hybrid retrieval returned no policy clauses for this case.",
                    "Check the policy index and the case description."
                )
            state = self.coverage.run(state)
            state = self.decision.run(state)
            state = self.validator.run(state)

            while (
                state.validation.status == "FAIL"
                and state.revision_count < self.s.max_validation_retries
            ):
                state.revision_count += 1
                feedback = "\n".join(
                    f"- UNSUPPORTED: {u.claim}  (reason: {u.reason})"
                    for u in state.validation.unsupported_claims
                )
                state.trace.append(
                    TraceStep(agent="Orchestrator", action="revision_triggered",
                              detail=f"attempt {state.revision_count}: "
                                     f"{len(state.validation.unsupported_claims)} unsupported claims",
                              results=len(state.validation.unsupported_claims))
                )
                state = self.decision.run(state, feedback=feedback)
                state = self.validator.run(state)

            if state.validation.status == "FAIL":
                state.trace.append(
                    TraceStep(agent="Orchestrator", action="force_abstention",
                              detail="validation still failing after revision")
                )
                state.decision = Decision.NEEDS_REVIEW
                state.confidence = min(state.confidence, 0.35)
                for u in state.validation.unsupported_claims:
                    msg = f"Unsupported by policy evidence: {u.claim}"
                    if msg not in state.missing_evidence:
                        state.missing_evidence.append(msg)
                state.next_action = (
                    "Human review required: material statements were not supported by "
                    "the policy wording."
                )
        except Exception as exc:  # never return a confident answer after a failure
            state.errors.append(f"{type(exc).__name__}: {exc}")
            return self._abstain(
                state, t0, f"Pipeline error: {type(exc).__name__}: {exc}",
                "Retry, or inspect server logs; no policy-grounded decision was produced."
            )

        return self._respond(state, t0)

    # ------------------------------------------------------------------ #
    def _abstain(self, state: CaseState, t0: float, reason: str, action: str) -> AnalyzeResponse:
        state.decision = Decision.NEEDS_REVIEW
        state.confidence = 0.0
        if reason not in state.missing_evidence:
            state.missing_evidence.append(reason)
        state.next_action = action
        state.validation.status = "FAIL"
        state.validation.unsupported_claims = state.validation.unsupported_claims or [
            UnsupportedClaim(claim="(no decision produced)", reason=reason)
        ]
        state.trace.append(TraceStep(agent="Orchestrator", action="abstain", detail=reason[:200]))
        return self._respond(state, t0)

    def _respond(self, state: CaseState, t0: float) -> AnalyzeResponse:
        ranks = [e.rerank_score for e in state.evidence if e.rerank_score is not None]
        retrieval = {
            "evidence_count": len(state.evidence),
            "pages_cited": sorted({c.page for c in state.citations}),
            "sections_cited": sorted({c.section for c in state.citations}),
            "pages_retrieved": sorted({e.page for e in state.evidence}),
            "top_rerank_score": max(ranks) if ranks else None,
            "reranker_available": self.retriever.reranker.available,
            "dimensions": [
                {"dimension": d.dimension, "status": d.status, "queries": d.queries}
                for d in (state.analysis.dimensions if state.analysis else [])
            ],
            "llm_calls": self.llm.calls,
        }
        return AnalyzeResponse(
            case_id=state.case.case_id,
            decision=state.decision or Decision.NEEDS_REVIEW,
            confidence=state.confidence,
            abstained=(state.decision or Decision.NEEDS_REVIEW) in ABSTAIN,
            key_findings=state.key_findings,
            applicable_limits=state.applicable_limits,
            missing_evidence=list(dict.fromkeys(state.missing_evidence)),
            next_action=state.next_action,
            citations=state.citations,
            validation=state.validation,
            retrieval=retrieval,
            trace=state.trace,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
