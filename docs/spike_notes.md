# Spike notes: one tool round trip on Groq

Run on 2026-10-18 with `spike.py`, before any tools, cache or arms existed. Purpose: confirm that tool calling works with the project's model, and measure what one call costs.

## Setup

| Setting | Value |
| --- | --- |
| Model | `openai/gpt-oss-120b` on Groq (OpenAI-compatible endpoint) |
| Temperature | 0 |
| Tools | `find_entities(text)` only, wrapping `retrieve.find_seeds` |
| Question | "What stake has Prudential agreed to acquire in Bharti Life?" (a control question) |
| `reasoning_effort` | Not set (Groq default) |

`find_entities` returns node ids and display names only, with no deal facts. A model that follows the answer rules therefore cannot answer the question, and should say "The sources do not say."

## Results

Two system prompts were tried. The first was the draft arm A prompt, which tells the model to "use the other tools" after `find_entities`. The second removed that step and the citation rule, because no other tools or citable ids existed yet.

| Prompt | Call | Prompt tokens | Completion tokens | Of which reasoning | Wall time | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Arm A draft | 1 | 415 | 196 | 164 | 1.10 s | `find_entities({"text": "Prudential Bharti Life"})` |
| Arm A draft | 2 | - | - | - | - | HTTP 400 `tool_use_failed`: model called `get_edges` |
| Spike prompt | 1 | 374 | 317 | 285 | 1.05 s | `find_entities({"text": "Prudential Bharti Life"})` |
| Spike prompt | 2 | 440 | 139 | 124 | 0.68 s | `finish_reason: stop`, "The sources do not say." |

Spike prompt, whole question: 1,270 tokens over 2 LLM calls and 1 tool call.

## Findings

1. **Tool calling works.** The model requested the tool with valid JSON arguments, and after receiving the result it stopped with a final answer.

2. **The evidence rule held on this question.** With no stake in the tool results, the model gave the exact refusal sentence instead of stating a stake from outside knowledge.

3. **Prompt tokens can be predicted from character counts.** Call 2's prompt was 66 tokens larger than call 1's. The estimate was about 65: the replayed assistant message (about 30 tokens), the tool result (110 characters, about 28 tokens) and role markers. Hidden formatting overhead per call is about 40 tokens.

4. **Fixed cost per call is about 375 to 415 prompt tokens** with one tool schema and this system prompt, before any conversation exists. It will rise with three tool schemas and must be measured again then.

5. **Hidden reasoning is a large share of cost, and it is unstable.**
   - 409 of 456 completion tokens (90%) were reasoning. That is 32% of all tokens for the question.
   - Removing two lines from the system prompt raised call 1's reasoning from 164 to 285 tokens (+74%), although the prompt got shorter.
   - Groq counts reasoning tokens against the daily quota.
   - Consequence: budget from measured totals, not from prompt sizes, and freeze the system prompt before any measured run.

6. **The model passed names, not the question.** It sent "Prudential Bharti Life" although the schema's example was a full question. Parameter descriptions are guidance, not instructions. Here it makes no difference, because `find_seeds` handles both.

7. **Repeat runs were identical.** Running the spike prompt twice gave the same arguments, the same answer and the same token counts, reasoning included. This was 2 runs of 1 question; it supports caching, but it does not show that Groq is deterministic in general. Only the tool call ids differed, which confirms that ids are opaque and must be copied, never parsed.

8. **Wall time includes Groq queue time** of 0.21 to 0.35 s per call, which depends on Groq's load rather than on the design. Reported wall times should say so.

## The `get_edges` failure

With the arm A draft prompt, the model received the ids from `find_entities`, then tried to call a tool that does not exist:

```
400 tool_use_failed: attempted to call tool 'get_edges' which was not in request.tools
failed_generation: {"name": "get_edges", "arguments": {"source_id": "prudential", "target_id": "bharti-life-insurance"}}
```

The cause was the prompt: it told the model to "use the other tools", and none were provided, so the model invented a plausible one.

What this means for the arms:

- A system prompt must describe only the tools in that request's `tools` list. In arm B, each role's prompt must match that role's tool set.
- This failure arrives as an exception from `create()`, not as a response, so a loop that only checks `tool_calls` never sees it.
- No `tool` message can be sent back, because the rejected response has no `tool_call_id`.
- `failed_generation` records what the model attempted, and should be logged.

## Decisions opened by the spike

| Decision | Options | Constraint |
| --- | --- | --- |
| Handling `tool_use_failed` | Record as tool misuse and end the question, or retry once with a corrective user message | Same policy in all arms |
| `reasoning_effort` | Groq default, or `low` / `medium` / `high` set explicitly | Same value for all arms and all roles |

## Not settled by the spike

- **Caps on LLM calls and tokens per question.** These need arm A running on real questions with all three tools.
- **The tool-result character cap.** This needs measured output sizes from `graph_neighbours` and `vector_search`, the first step of the tools phase.
- **What the model cites.** This depends on the `graph_neighbours` result format.
