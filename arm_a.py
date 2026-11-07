from cache import cached_completion
from config import FIRST_CALL_TOOL_CHOICE, MODEL, SEED, TEMPERATURE, MAX_LLM_CALLS, MAX_TOKENS, make_client
from prompts import SYSTEM_PROMPT
from tools import TOOLS, run_tool
from logger import RunLog


def run_single_agent(client, log, question_id, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tokens = 0
    calls = 0
    rejections = 0
    tool_ran = False          # becomes True once any tool has run; until then tool_choice is "required"

    for step in range(1, MAX_LLM_CALLS + 1):
        # Checked before each call, so a question can go over MAX_TOKENS by at
        # most one call: the cap means "no new call once the total reaches it".
        if tokens >= MAX_TOKENS:
            break

        result = cached_completion(
            client, model=MODEL, messages=messages, tools=TOOLS,
            temperature=TEMPERATURE, seed=SEED,
            tool_choice=FIRST_CALL_TOOL_CHOICE if not tool_ran else "auto",
        )
        calls += 1
        log.llm_call(question_id, "single", step, result)

        if result.error:
            # Rejections are counted across the whole question: the first one
            # gets a correction and a retry, the second ends the question.
            rejections += 1
            if rejections > 1:
                log.question_end(question_id, None, result.error["code"],
                                 llm_calls=calls, tokens=tokens)
                return None
            names = ", ".join(t["function"]["name"] for t in TOOLS)
            messages.append({
                "role": "user",
                "content": f"Your last tool call was rejected ({result.error['message']}). "
                           f"The only tools you have are: {names}.",
            })
            continue

        usage = result.response.usage
        tokens += usage.prompt_tokens + usage.completion_tokens
        message = result.response.choices[0].message

        if not message.tool_calls:
            log.question_end(question_id, message.content, "answered",
                             llm_calls=calls, tokens=tokens)
            return message.content

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            text, error = run_tool(call.function.name, call.function.arguments)
            tool_ran = True
            log.tool_call(question_id, "single", step, call.function.name,
                          call.function.arguments, text, error)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

    log.question_end(question_id, None, "cap_hit", llm_calls=calls, tokens=tokens)
    return None


if __name__ == "__main__":
    import json
    from tools import GRAPH_RAG_DIR
    from logger import read_log, token_totals

    DEV_IDS = ["q01", "q09", "q10", "q11"]
    path = GRAPH_RAG_DIR / "data" / "eval_questions.json"
    questions = json.loads(path.read_text(encoding="utf-8"))

    client = make_client()
    log = RunLog("dev_a_002", arm="A")

    for q in questions:
        if q["id"].lower() in DEV_IDS:
            answer = run_single_agent(client, log, q["id"], q["question"])
            print(f"{q['id']}: {answer}\n")

    events = read_log(log.path)
    for e in events:
        if e["event"] == "question_end":
            print(e["question"], e["stop_reason"], "calls:", e["llm_calls"], "tokens:", e["tokens"])
    print(token_totals(events))