"""End-to-end evaluation harness.

    python -m evaluation.run_eval                 # all bundled cases, in-process
    python -m evaluation.run_eval --set custom    # only candidate-created cases
    python -m evaluation.run_eval --api https://...   # score a deployed backend

Writes evaluation/results/{results.json, scores.csv, REPORT.md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from evaluation.metrics import CaseScore, aggregate, score_case

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"


def load_cases(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            print(f"  ! skipping {p.name}: {exc}")
    return out


def run_local(cases: list[dict]) -> list[dict]:
    from app.graph import ClaimPipeline
    from app.models import ClaimCase

    pipe = ClaimPipeline()
    results = []
    for case in cases:
        cid = case.get("case_id", "?")
        print(f"  -> {cid}", flush=True)
        try:
            resp = pipe.run(ClaimCase(**case)).model_dump(mode="json")
        except Exception as exc:
            resp = {"case_id": cid, "decision": "ERROR", "confidence": 0.0,
                    "_error": f"{type(exc).__name__}: {exc}", "citations": [],
                    "validation": {}, "key_findings": [], "elapsed_ms": 0}
        resp["_evidence"] = _evidence_for(pipe, resp)
        results.append(resp)
    return results


def _evidence_for(pipe, resp: dict) -> list[dict]:
    """Re-expose the evidence the run used, for retrieval metrics. The API
    response deliberately ships citations rather than the whole evidence pool,
    so the harness re-runs the dimension queries against the same retriever."""
    queries = [q for d in resp.get("retrieval", {}).get("dimensions", []) for q in d.get("queries", [])]
    if not queries:
        return []
    items = pipe.retriever.retrieve_multi(queries, cap=24)
    return [{"chunk_id": i.chunk_id, "text": i.text, "page": i.page, "section": i.section} for i in items]


def run_api(cases: list[dict], api: str) -> list[dict]:
    import requests

    results = []
    for case in cases:
        cid = case.get("case_id", "?")
        print(f"  -> {cid} (remote)", flush=True)
        try:
            r = requests.post(f"{api.rstrip('/')}/analyze", json={"case": case}, timeout=300)
            resp = r.json() if r.status_code < 400 else {
                "case_id": cid, "decision": "ERROR", "_error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as exc:
            resp = {"case_id": cid, "decision": "ERROR", "_error": str(exc)}
        resp.setdefault("citations", [])
        resp.setdefault("validation", {})
        resp.setdefault("key_findings", [])
        resp.setdefault("elapsed_ms", 0)
        resp["_evidence"] = []  # retrieval metrics need local index access
        results.append(resp)
    return results


def write_report(scores: list[CaseScore], summary: dict, expectations: dict, path: Path) -> None:
    from tabulate import tabulate

    rows = [s.as_row() for s in scores]
    unlabelled = [s.case_id for s in scores if not s.labelled]
    lines = [
        "# ClaimRAG — evaluation report",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Per-case results",
        "",
        tabulate(rows, headers="keys", tablefmt="github"),
        "",
        "### Metric definitions",
        "",
        "- **decision_accuracy** — share of labelled cases whose status is in the expected set.",
        "- **evidence_recall_at_k** — share of the clause anchors a human reviewer would need,",
        "  found in the evidence actually passed to the agents.",
        "- **citation_lexical_overlap** — share of citations whose cited clause lexically covers",
        "  the claim (LLM-free proxy for citation correctness).",
        "- **citation_support_rate** — share of material claims confirmed by the independent",
        "  entailment check in the Validation agent.",
        "- **abstention.recall / .precision** — did we abstain when we should, and only then.",
        "",
    ]
    if unlabelled:
        lines += [
            "## Unlabelled cases",
            "",
            "These ran but were not scored for decision accuracy — add them to "
            "`evaluation/expected/expectations.json`:",
            "",
            *[f"- `{c}`" for c in unlabelled],
            "",
        ]
    lines += ["## Expected-outcome rationale", ""]
    for cid, exp in expectations.items():
        lines.append(f"- **{cid}** → {'/'.join(exp.get('expected_decision', []))}: {exp.get('rationale','')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Run the ClaimRAG evaluation.")
    ap.add_argument("--set", choices=["all", "public", "custom"], default="all")
    ap.add_argument("--api", default=None, help="score a deployed backend instead of running in-process")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    cases: list[dict] = []
    if args.set in ("all", "public"):
        pub = load_cases(Path(s.public_cases_dir))
        print(f"public cases: {len(pub)}")
        if not pub:
            print(f"  (none found in {s.public_cases_dir} — drop the 12 supplied cases there)")
        cases += pub
    if args.set in ("all", "custom"):
        cust = load_cases(Path(s.custom_cases_dir))
        print(f"custom cases: {len(cust)}")
        cases += cust
    if not cases:
        raise SystemExit("No cases to evaluate.")

    expectations: dict[str, Any] = json.loads(
        (ROOT / "evaluation" / "expected" / "expectations.json").read_text(encoding="utf-8")
    )["cases"]

    t0 = time.perf_counter()
    results = run_api(cases, args.api) if args.api else run_local(cases)
    scores = [score_case(r, expectations.get(r.get("case_id", ""))) for r in results]
    summary = aggregate(scores)
    summary["wall_clock_s"] = round(time.perf_counter() - t0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, default=str), encoding="utf-8"
    )
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scores[0].as_row().keys()))
        w.writeheader()
        for sc in scores:
            w.writerow(sc.as_row())
    write_report(scores, summary, expectations, out / "REPORT.md")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'REPORT.md'}, results.json, scores.csv")


if __name__ == "__main__":
    main()
