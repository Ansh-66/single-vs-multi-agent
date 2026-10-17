# single-vs-multi-agent: scope

A measured comparison of single-agent and multi-agent designs, built on the [Insurance Graph RAG](https://github.com/Ansh-66/Insurance-Graph-RAG) knowledge base.

## The question

**When does a team of agents beat one agent with the same tools, and what does it cost?**

The project exists to answer this with a measurement. If the answer is "one agent was good enough", that is a result, not a failure.

## What gets built

Three arms. Same model, same temperature (0), same tools, same answer rules, same question set.

| Arm | Design |
|---|---|
| A. Single agent | One tool-calling loop: the model calls tools, reads results, and answers when it has enough. |
| B. Orchestrator + workers | A planner splits the question into sub-questions; a worker agent researches each one with the tools; the planner merges the findings into one answer. |
| C. Orchestrator + workers + verifier | B, plus a verifier that checks every claim in the draft answer against its cited passage. Unsupported claims are sent back once for revision, then dropped. |

Answer rules, shared by every arm (the same as Graph RAG): use only tool results, cite every claim, say when a deal is pending, and say "The sources do not say" when they don't.

## Tools

Thin wrappers over code that already exists. No new retrieval logic.

| Tool | Wraps | Returns |
|---|---|---|
| `find_entities(text)` | `retrieve.find_seeds` | Matching node ids with display names |
| `graph_neighbours(node_id, hops)` | `retrieve.traverse` | Edges: relation, status, stake, dates, evidence sentence, source URL |
| `vector_search(query, k)` | `vector_rag.search` | Top-k chunks with source URLs |

Every tool result is capped at a fixed length (set after the spike) so one call cannot flood the context. The cap is identical for all arms.

## Question set

12 to 15 questions, written and given gold answers **before** arms B and C exist, so the questions cannot be shaped to suit them.

| Type | Count | Example | Why it is in the set |
|---|---|---|---|
| Control (single hop) | 2 | What stake has Prudential agreed to acquire in Bharti Life? | Multi-agent should not help here. Measures pure overhead. |
| Comparison | 3 | Which foreign insurers moved to full ownership of an Indian JV, and at what price each? | Needs several independent lookups. The case for splitting work. |
| Temporal | 3 | Who owned Bharti Life before the Prudential deal, and who will after? | Uses edge dates and pending status. |
| Verification | 3 | Did Allianz sell its entire stake in the Bajaj ventures? | Nuanced answers (23% sold, 3% to follow). The case for a verifier. |
| Aggregation | 2 | List every stake change above 25% since February 2026. | Needs complete coverage, not one good passage. |
| Unanswerable | 2 | (A deal the corpus does not contain) | Measures invented answers. |

Each question records its gold answer, the facts a complete answer must contain, and the edges or articles that hold them. List answers are scored on those required facts, so a partly correct list is visible rather than a plain wrong.

## Metrics

Per question, per arm:

- **Correct**: Y / N, plus required facts found out of required facts for list questions
- **Grounded**: every claim supported by the passage it cites
- **Cost**: input tokens, output tokens, LLM calls, tool calls
- **Time**: wall-clock seconds
- **Failure type**: one of the taxonomy below, for every wrong answer

Failure taxonomy (extended as new ones appear):
- wrong decomposition (the planner split the question badly)
- missed lookup (a needed tool call never happened)
- tool misuse (wrong arguments, wrong tool)
- invented fact (a claim with no supporting tool result)
- lost in merge (a worker found it, the final answer dropped it)
- looping (repeated identical calls until a cap was hit)
- cap hit (stopped by the budget limit)

## Fairness rules

- One model for every role in every arm. A stronger model for the planner would measure the model, not the design.
- The same tool code, result caps and answer rules everywhere.
- Hard caps per question, identical for all arms: a maximum number of LLM calls and of total tokens. Hitting a cap is recorded as a failure, not retried.
- Workers run one after another, not in parallel. Parallel runs would hit the free-tier rate limit, and wall time is reported with this noted.
- Every LLM response is cached by a hash of (model, messages, tools). A re-run of an unchanged arm costs nothing, and the reported numbers come from one recorded run.

## Budget

Groq's free tier caps at about 200k tokens a day. A rough estimate, to be replaced by real numbers from the spike:

| Arm | LLM calls per question | Tokens per question | 12 questions |
|---|---|---|---|
| A | 4 to 6 | 15k to 25k | 180k to 300k |
| B | 8 to 14 | 35k to 60k | 420k to 720k |
| C | 10 to 18 | 45k to 80k | 540k to 960k |

So a full run is roughly 1.2M to 2M tokens: **6 to 10 days of free quota**, before any debugging. Tighter tool-result caps should bring it toward the low end. This is the main constraint on the project, and these rules follow from it:

- The cache above is built first, before any arm.
- Development runs use 3 to 4 questions, not the full set.
- Full runs start as soon as each arm works, in parallel with building the next arm.
- The full eval runs once per arm, resumable across days like `run_eval.py`.
- If the spike shows costs above this estimate, the question count drops, not the arms.

Local inference (Ollama, 7B on a 4GB GPU) is an option for debugging the loop's mechanics, but not for reported results: tool calling at that size is too unreliable to compare designs fairly.

## Who writes what

Revised for speed (Oct 2026). The aim is a resume entry in about 10 to 11 evenings without losing defensibility.

| Part | Who |
|---|---|
| The agent loop (arm A, about 50 lines) | **Ansh.** The piece an interviewer will ask to see written. |
| Tool wrappers, response cache, step logging, run harness | Claude |
| Planner, worker and verifier (arms B and C) | Claude drafts; then a walkthrough of each file |
| Questions, gold answers, predictions, scoring | **Ansh** |
| README findings | **Ansh**, with Claude editing |

**Defense drill.** Before any file Ansh did not write is committed, Claude asks five interview-style questions about it and Ansh makes one small change to it without help. A file is not done until both pass.

## Phases and gates

**Phase 1: the core**
1. Spike: one tool call on Groq with `find_entities`. Measure tokens per call and confirm tool calling is reliable.
2. Tools, cache, logging.
3. Arm A from scratch: loop, caps, stop conditions.
4. Question set and gold answers.
5. Arms B and C.
6. Full run, scoring, results table.
7. README with the finding.

Gate: a results table worth defending in an interview. **The project goes on the resume here**, not before. Phase 2 starts only after it.

**Phase 2: the extensions**
8. LangGraph port of arm B, run on the same questions. Check the results match the from-scratch version, and record what the framework added and what it hid.
9. An MCP server exposing the three tools, shown working from an MCP client.
10. An "Agents" tab in the existing React app showing each arm's steps: the tool called, its arguments, and what came back.

Phase 2 adds no new numbers. It shows framework and tooling skills on top of a proven result, and is added to the same resume entry when done. LangChain enters only here: the LangGraph port uses LangChain's chat model, `@tool` and message types.

## Out of scope

- New data collection. The corpus and graph are fixed at their current state.
- Web search or any tool outside the corpus. Gold answers must be checkable.
- Fine-tuning, or different models per role.
- More than four agent roles.
- Agent frameworks in Phase 1 (LangChain, LangGraph, CrewAI, AutoGen).
- Claims of statistical significance. With 12 to 15 questions the result is reported per question and per type, not as a significance test.

## Decisions to make before coding

1. **Where the code lives.** Decided: this repo, cloned next to Insurance-Graph-RAG, importing its modules in place through `GRAPH_RAG_DIR` in `.env`. Done: Graph RAG's `retrieve`, `vector_rag`, `resolve` and `extract` now resolve data paths and `.env` from their repo root (commit `41125f9`), so they import from any folder. Importing `vector_rag` requires `GROQ_API_KEY`, because it imports `extract` at the top.
2. **Cap values.** The maximum LLM calls and tokens per question, set from the spike's measurements.
3. **Tool result cap.** The maximum characters per tool result, set from the spike.
4. **Partial credit.** Whether list questions count as correct only with every required fact, or are reported as a fraction alongside.

5. **OpenRouter top-up.** A one-time $10 credit raises OpenRouter's free-model limit from 50 to 1,000 requests a day. Faster, but weaker and less reliable models than gpt-oss-120b. Undecided; the plan works without it.

## Predictions

Written before any arm runs, then compared with the results.

- Arm A on control questions: ...
- Arm B against A on comparison questions: ...
- Arm C against B on verification questions: ...
- Cost ratio B : A, and C : A: ...
- The failure type that will be most common in B: ...
