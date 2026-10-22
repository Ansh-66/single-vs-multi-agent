# Tool design: formats, caps and why

Decided on 2026-10-19 from the output of `scripts/measure_tools.py` (no LLM calls). These settings are identical for every arm and every role; changing one after measured runs start means re-running every arm.

## What was measured

Sizes of one tool result, in characters (about 4 characters per token).

| Result | Median | Max |
| --- | --- | --- |
| `graph_neighbours`, 1 hop, raw JSON | 2,313 | 43,510 |
| `graph_neighbours`, 1 hop, compact, 1 source per edge | 1,129 | 9,710 |
| `graph_neighbours`, 2 hops, raw JSON | 12,367 | 69,558 |
| `graph_neighbours`, 2 hops, compact, 1 source per edge | 2,956 | 17,210 |
| `vector_search`, k=5 | 3,432 | 3,764 |
| `vector_search`, k=10 | 6,799 | 7,338 |

Nodes measured: every node seeded by the 11 Graph RAG eval questions, plus the three hubs `prudential` (21 edges), `bharti-life-insurance` (20) and `magma-general-insurance` (16).

Two observations drive the design:

- **Every tool result is resent on every later call of the same question.** One raw `prudential` 2-hop result (about 17,400 tokens) received at call 2 of a 5-call question would cost about 70,000 tokens, a third of a day's free quota.
- **Vector results are predictable (max/min about 1.2x); graph results are not (about 35x).** The cap is a graph problem, and specifically a hub problem.

## Decisions

| Decision | Setting | Reason |
| --- | --- | --- |
| Output format | Plain text, one entry per edge or chunk, each starting with a citable id (`[e12]`, `[v123]`) | Same facts at about half the median size and a quarter of the worst case. The ids let answers cite their evidence and let scoring check it. |
| Sources per edge | 1, the most recent | A second source adds 30 to 55% on hubs. One evidence sentence is enough to ground a claim. Most recent, because some deals changed over time (announced, then completed); an older article would report a stale status. |
| Default `hops` | 1, maximum 2 | Median 1,129 characters at 1 hop against 2,956 at 2 hops. The model asks for 2 when a question needs a chain. |
| `other_id` filter | Optional argument of `graph_neighbours` | Cuts hub results 4 to 7x (`prudential` with `other_id=bharti-life-insurance`: 21 edges to 5, 9,710 to 2,110 characters). Fewer distractors also means fewer merge and invented-fact errors. In the spike the model invented a tool of exactly this shape (`get_edges(source_id, target_id)`). It selects from `traverse` output; no new retrieval logic. |
| Graph character cap | 6,000 (about 1,500 tokens), with 120 characters kept for the truncation note | Every non-hub result fits at 1 and 2 hops (largest: `greenlife-insurance-broking`, 2 hops). Hubs are cut by whole edges, edges touching the queried node first, and end with "Showing X of Y edges. Pass other_id..." so the model knows the result is partial, how much was cut and how to narrow it. |
| Vector `k` | Default 5, maximum 10 | k=5 is about 860 tokens against 1,700 for k=10, and the difference is paid again on every later call. |
| Vector cap | None | A result cannot exceed about 7,400 characters at k=10. |

## Errors

Model mistakes come back as text; code bugs raise.

- **Model mistakes** (a bad tool name, invalid JSON, a missing or unexpected argument, an unknown node id, an out-of-range `hops` or `k`) return a single line starting with `Error:`. The model reads it and can correct itself, and `run_tool` also returns the error text separately so the run log records it as tool misuse.
- **Argument names are checked before the tool runs**, with `inspect.signature(func).bind(**args)`. Only a failure there is reported as "bad arguments".
- **A bug inside a tool is not caught.** It raises, so it is seen and fixed during development instead of being counted against the model. Catching every `TypeError` would have reported a tool bug (for example `enumerate(None)` on an unloaded index) as model misuse and inflated the misuse count.

Empty results are sentences, not empty lists: "No known company or person found in: ...", "No edges found for ... within 1 hop(s)."

## Known limits

- `[v123]` is the chunk's position in the vector index. It is stable only while the index is not rebuilt.
- Some deals appear as two edges with different statuses (for example `e109` asserted and `e110` prospective for Prudential's Bharti Life stake), because different articles were extracted at different times. Both are shown; the answer rules make the model report the status the evidence supports.