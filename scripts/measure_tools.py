"""
Measure how big each tool result would be, before choosing formats and caps.

No LLM calls. Needs GRAPH_RAG_DIR in .env, and the local vector index
(data/vector_index) in the Graph RAG repo. Importing vector_rag also needs
GROQ_API_KEY to be set, though nothing here calls Groq.

Run from the repo root:  python scripts/measure_tools.py
"""

import json
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
GRAPH_RAG_DIR = Path(os.environ["GRAPH_RAG_DIR"])
sys.path.append(str(GRAPH_RAG_DIR / "src"))

import retrieve      # noqa: E402  (must come after the sys.path line)
import vector_rag    # noqa: E402

CHARS_PER_TOKEN = 4
HUBS = ["prudential", "bharti-life-insurance", "magma-general-insurance"]


# ---------------------------------------------------------------------------
# Candidate formats for graph_neighbours
# ---------------------------------------------------------------------------

def edge_line(edge, max_sources):
    """One edge as compact text: facts first, then up to max_sources sources."""
    facts = [f"[{edge['id']}] {edge['src']} {edge['relation']} {edge['dst']}",
             edge["status"]]
    if edge.get("stake_pct") is not None:
        facts.append(f"stake {edge['stake_pct']}%")
    if edge.get("start_date") is not None or edge.get("end_date") is not None:
        facts.append(f"dates {edge.get('start_date')} to {edge.get('end_date')}")
    if edge.get("role"):
        facts.append(f"role {edge['role']}")
    line = " | ".join(facts)

    sources = edge["sources"] if max_sources is None else edge["sources"][:max_sources]
    for s in sources:
        line += (f"\n    {retrieve.short_date(s['pub_date'])} "
                 f"\"{s['evidence']}\" {s['url']}")
    return line


def graph_sizes(node_id, hops):
    edges = retrieve.traverse([node_id], hops=hops)
    return {
        "edges": len(edges),
        "raw_json": len(json.dumps(edges)),
        "compact_1src": len("\n".join(edge_line(e, 1) for e in edges)),
        "compact_2src": len("\n".join(edge_line(e, 2) for e in edges)),
        "compact_all": len("\n".join(edge_line(e, None) for e in edges)),
    }


# ---------------------------------------------------------------------------
# Candidate format for vector_search
# ---------------------------------------------------------------------------

def chunk_line(i, chunk):
    return (f"[v{i}] {retrieve.short_date(chunk['pub_date'])} | "
            f"{chunk['text']} | {chunk['url']}")


def vector_sizes(question, k):
    chunks = vector_rag.search(question, k=k)
    text = "\n".join(chunk_line(i, c) for i, c in enumerate(chunks, 1))
    return {"chars": len(text),
            "max_chunk_text": max(len(c["text"]) for c in chunks)}


# ---------------------------------------------------------------------------

def tok(chars):
    return round(chars / CHARS_PER_TOKEN)


def main():
    questions = json.loads((GRAPH_RAG_DIR / "data" / "eval_questions.json")
                           .read_text(encoding="utf-8"))

    # Every node any eval question seeds, plus the known hubs.
    nodes = []
    for q in questions:
        for n in retrieve.find_seeds(q["question"]):
            if n not in nodes:
                nodes.append(n)
    for n in HUBS:
        if n not in nodes:
            nodes.append(n)

    print("=== graph_neighbours: characters per result (tokens ~ chars/4) ===")
    header = f"{'node':32} {'hops':>4} {'edges':>5} {'raw_json':>9} {'1 src':>7} {'2 src':>7} {'all src':>8}"
    print(header)
    rows = []
    for n in nodes:
        for hops in (1, 2):
            s = graph_sizes(n, hops)
            rows.append((n, hops, s))
            print(f"{n:32} {hops:>4} {s['edges']:>5} {s['raw_json']:>9} "
                  f"{s['compact_1src']:>7} {s['compact_2src']:>7} {s['compact_all']:>8}")

    print("\n=== vector_search: characters per result, per eval question ===")
    print(f"{'q':4} {'k=5':>6} {'k=10':>6} {'longest chunk':>14}")
    v5, v10 = [], []
    for q in questions:
        a, b = vector_sizes(q["question"], 5), vector_sizes(q["question"], 10)
        v5.append(a["chars"])
        v10.append(b["chars"])
        print(f"{q['id']:4} {a['chars']:>6} {b['chars']:>6} {b['max_chunk_text']:>14}")

    print("\n=== summary (characters / approx tokens) ===")
    for hops in (1, 2):
        for key in ("raw_json", "compact_1src", "compact_2src"):
            vals = [s[key] for _, h, s in rows if h == hops]
            print(f"graph hops={hops} {key:13} median {statistics.median(vals):>7.0f} / {tok(statistics.median(vals)):>5}"
                  f"   max {max(vals):>6} / {tok(max(vals)):>5}")
    for k, vals in ((5, v5), (10, v10)):
        print(f"vector k={k:<2}                  median {statistics.median(vals):>7.0f} / {tok(statistics.median(vals)):>5}"
              f"   max {max(vals):>6} / {tok(max(vals)):>5}")


if __name__ == "__main__":
    main()