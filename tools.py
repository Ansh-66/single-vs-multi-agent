"""
The three tools every arm can call, their schemas, and one dispatcher.

Each tool returns plain text, never raises: a bad argument comes back as a
line starting with "Error:" so the model can correct itself and the run log
records it as tool misuse. Settings below were chosen from
scripts/measure_tools.py (see docs/tool_design.md) and are identical for all
arms.
"""

import inspect
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
GRAPH_RAG_DIR = Path(os.environ["GRAPH_RAG_DIR"])
sys.path.append(str(GRAPH_RAG_DIR / "src"))   # append: our own modules win name clashes

import retrieve      # noqa: E402  (needs the sys.path line above)
import vector_rag    # noqa: E402

# ---------------------------------------------------------------------------
# Settings (same for every arm)
# ---------------------------------------------------------------------------

GRAPH_CHAR_CAP = 6000      # fits every non-hub result at 1 and 2 hops
NOTE_ROOM = 120            # kept free for the "+N more edges" note, so the cap holds
DEFAULT_HOPS, MAX_HOPS = 1, 2
DEFAULT_K, MAX_K = 5, 10   # k=5 is ~860 tokens, k=10 ~1,700

NODES = retrieve.NODES["nodes"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def error(message):
    return f"Error: {message}"


def as_int(value, name, low, high):
    """Accept 2 or "2"; return (int, None) or (None, error text)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None, error(f"{name} must be a whole number from {low} to {high}, got {value!r}.")
    if not low <= number <= high:
        return None, error(f"{name} must be from {low} to {high}, got {number}.")
    return number, None


def sortable_date(pub_date):
    """'2026-05-17' for sorting; '' for an unparseable date, so it sorts first."""
    day = retrieve.short_date(pub_date)
    return day if day[:1].isdigit() else ""


def edge_line(edge):
    """One edge as one compact entry: its facts, then its most recent source."""
    facts = [f"[{edge['id']}] {edge['src']} {edge['relation']} {edge['dst']}",
             edge["status"]]
    if edge.get("stake_pct") is not None:
        facts.append(f"stake {edge['stake_pct']}%")
    if edge.get("start_date") is not None or edge.get("end_date") is not None:
        facts.append(f"dates {edge.get('start_date') or '?'} to {edge.get('end_date') or '?'}")
    if edge.get("role"):
        facts.append(f"role {edge['role']}")
    if len(edge["sources"]) > 1:
        facts.append(f"{len(edge['sources'])} sources")

    latest = max(edge["sources"], key=lambda s: sortable_date(s["pub_date"]))
    return (" | ".join(facts)
            + f"\n    {retrieve.short_date(latest['pub_date'])} "
              f"\"{latest['evidence']}\" {latest['url']}")


# Vector chunk ids: the chunk's position in the index, so [v123] always means
# the same chunk for as long as the index is not rebuilt.
_chunk_ids = None


def chunk_id(chunk):
    global _chunk_ids
    if _chunk_ids is None:
        _chunk_ids = {(c["article_id"], c["text"]): i
                      for i, c in enumerate(vector_rag.CHUNKS)}
    return f"v{_chunk_ids[(chunk['article_id'], chunk['text'])]}"


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def find_entities(text):
    if not isinstance(text, str) or not text.strip():
        return error("text must be a non-empty string.")
    ids = retrieve.find_seeds(text)
    if not ids:
        return f"No known company or person found in: {text}"
    return "\n".join(f"{node_id}: {NODES[node_id]['canonical']}" for node_id in ids)


def graph_neighbours(node_id, hops=DEFAULT_HOPS, other_id=None):
    if not isinstance(node_id, str):
        return error(f"node_id must be a string id from find_entities, got {node_id!r}.")
    if other_id is not None and not isinstance(other_id, str):
        return error(f"other_id must be a string id from find_entities, got {other_id!r}.")
    if node_id not in NODES:
        return error(f"unknown node id {node_id!r}. Call find_entities to get ids.")
    if other_id is not None and other_id not in NODES:
        return error(f"unknown other_id {other_id!r}. Call find_entities to get ids.")
    hops, problem = as_int(hops, "hops", 1, MAX_HOPS)
    if problem:
        return problem

    edges = retrieve.traverse([node_id], hops=hops)
    if other_id is not None:
        edges = [e for e in edges if other_id in (e["src"], e["dst"])]
    if not edges:
        scope = f" connecting {node_id} and {other_id}" if other_id else f" for {node_id}"
        return f"No edges found{scope} within {hops} hop(s)."

    # Edges that touch the node itself first, so the cap drops far edges first.
    edges.sort(key=lambda e: node_id not in (e["src"], e["dst"]))

    shown, size = [], 0
    for edge in edges:
        line = edge_line(edge)
        if shown and size + len(line) + 1 > GRAPH_CHAR_CAP - NOTE_ROOM:
            break
        shown.append(line)
        size += len(line) + 1
    text = "\n".join(shown)

    if len(shown) < len(edges):
        text += (f"\n(Showing {len(shown)} of {len(edges)} edges. "
                 f"Pass other_id to get only the edges with one company.)")
    return text


def vector_search(query, k=DEFAULT_K):
    if not isinstance(query, str) or not query.strip():
        return error("query must be a non-empty string.")
    k, problem = as_int(k, "k", 1, MAX_K)
    if problem:
        return problem

    chunks = vector_rag.search(query, k=k)
    return "\n".join(
        f"[{chunk_id(c)}] {retrieve.short_date(c['pub_date'])} | {c['text']} | {c['url']}"
        for c in chunks
    )


# ---------------------------------------------------------------------------
# Schemas: what the model sees
# ---------------------------------------------------------------------------

FIND_ENTITIES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_entities",
        "description": "Resolve company or person names or phrases in free text to node ids in the "
                       "knowledge graph. Returns one 'id: name' line per match, or a line saying "
                       "none was found. It returns no deal facts (stakes, dates, relations). Call "
                       "this first: other graph tools take the returned ids, not names.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "A company or person name, or a sentence mentioning companies or "
                                   "people, e.g. 'Prudential plc' or 'What stake has Prudential "
                                   "agreed to acquire in Bharti Life?'",
                },
            },
            "required": ["text"],
        },
    },
}

GRAPH_NEIGHBOURS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "graph_neighbours",
        "description": "Return the deal edges around a node: relation, status (asserted or "
                       "prospective, i.e. not yet completed), stake, dates, and the most recent "
                       "evidence sentence with its source URL. Each edge starts with an id like "
                       "[e12]; cite it. Use other_id to keep only the edges with one other node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string",
                            "description": "A node id from find_entities, e.g. 'prudential'."},
                "hops": {"type": "integer", "minimum": 1, "maximum": MAX_HOPS,
                         "description": "1 = direct edges only (default). 2 = also edges of "
                                        "neighbours, for chains."},
                "other_id": {"type": "string",
                             "description": "Optional node id: return only edges that involve it."},
            },
            "required": ["node_id"],
        },
    },
}

VECTOR_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "vector_search",
        "description": "Search the news articles by meaning. Returns the k most similar passages, "
                       "each starting with an id like [v123], then its date, text and source URL; "
                       "cite the id. Use for facts the graph does not hold.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for, in plain words."},
                "k": {"type": "integer", "minimum": 1, "maximum": MAX_K,
                      "description": f"Number of passages, default {DEFAULT_K}."},
            },
            "required": ["query"],
        },
    },
}

TOOLS = [FIND_ENTITIES_SCHEMA, GRAPH_NEIGHBOURS_SCHEMA, VECTOR_SEARCH_SCHEMA]

TOOL_FUNCTIONS = {
    "find_entities": find_entities,
    "graph_neighbours": graph_neighbours,
    "vector_search": vector_search,
}


# ---------------------------------------------------------------------------
# Dispatcher: the only function the agent loops call
# ---------------------------------------------------------------------------

def run_tool(name, arguments):
    """
    Run one tool call from the model.

    name: the tool name the model asked for.
    arguments: the raw JSON string from the model.
    Returns (result_text, error_text_or_None). Never raises.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        msg = error(f"unknown tool {name!r}. Available: {', '.join(TOOL_FUNCTIONS)}.")
        return msg, msg

    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        msg = error(f"arguments are not valid JSON ({e.msg}).")
        return msg, msg
    if not isinstance(args, dict):
        msg = error("arguments must be a JSON object.")
        return msg, msg

    # Bind first so only argument-name mistakes become "bad arguments"; a TypeError
    # raised inside a tool is a bug and should surface, not be blamed on the model.
    try:
        inspect.signature(func).bind(**args)
    except TypeError as e:            # missing or unexpected argument names
        msg = error(f"bad arguments for {name}: {e}")
        return msg, msg

    text = func(**args)

    return text, (text if text.startswith("Error:") else None)


if __name__ == "__main__":
    # Smoke test, no LLM calls.
    for name, arguments in [
        ("find_entities", '{"text": "Prudential Bharti Life"}'),
        ("find_entities", '{"text": "weather in Mumbai"}'),
        ("graph_neighbours", '{"node_id": "prudential", "other_id": "bharti-life-insurance"}'),
        ("graph_neighbours", '{"node_id": "prudential", "hops": 2}'),
        ("graph_neighbours", '{"node_id": "prudential-plc"}'),
        ("graph_neighbours", '{"node_id": "aviva", "hops": 5}'),
        ("vector_search", '{"query": "Prudential stake in Bharti Life", "k": 3}'),
        ("get_edges", '{"source_id": "prudential"}'),
        ("find_entities", '{"text": "Aviva"'),
    ]:
        text, err = run_tool(name, arguments)
        print(f"=== {name} {arguments}  ({len(text)} chars, error={err is not None})")
        print(text[:700] + ("\n..." if len(text) > 700 else ""))
        print()
