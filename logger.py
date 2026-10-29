"""
Run log: one JSON object per line in runs/<run_id>.jsonl.

Every number in the results table is computed from these lines. Three events:

  llm_call      after every model call (tokens, cache hit, original time, errors)
  tool_call     after every tool run (arguments, the full result text, errors)
  question_end  when a question finishes (answer, how it stopped, totals)

The file is only ever appended to, so a crash keeps every line written before
it and a resumed run just adds more lines. Tool results are stored in full so
grounding can be checked from the log alone, without the cache.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


class RunLog:
    def __init__(self, run_id, arm):
        self.run_id = run_id
        self.arm = arm
        self.path = RUNS_DIR / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event, **fields):
        line = {"event": event, "run_id": self.run_id, "arm": self.arm,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        # One write call per line, then flush, so each line reaches the disk whole.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()

    def llm_call(self, question, role, step, result):
        """result: the CallResult from cached_completion."""
        response = result.response
        usage = response.usage if response is not None else None
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        choice = response.choices[0] if response is not None else None

        self._write(
            "llm_call",
            question=question, role=role, step=step,
            cache_hit=result.cache_hit, cache_key=result.key,
            finish_reason=choice.finish_reason if choice else None,
            tool_calls=len(choice.message.tool_calls or []) if choice else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            reasoning_tokens=(getattr(details, "reasoning_tokens", None) or 0) if details else 0,
            elapsed=result.elapsed,
            fingerprint=getattr(response, "system_fingerprint", None) if response is not None else None,
            error=result.error["code"] if result.error else None,
            failed_generation=result.error.get("failed_generation") if result.error else None,
        )

    def tool_call(self, question, role, step, name, arguments, result_text, error):
        """arguments: the raw JSON string from the model. error: run_tool's second value."""
        self._write(
            "tool_call",
            question=question, role=role, step=step,
            tool=name, arguments=arguments,
            result_chars=len(result_text), result=result_text,
            error=error,
        )

    def question_end(self, question, answer, stop_reason, **extra):
        """stop_reason: "answered", "cap_hit", "tool_use_failed", ..."""
        self._write("question_end", question=question, answer=answer,
                    stop_reason=stop_reason, **extra)


def read_log(path):
    """
    All events in a log, as a list of dicts.

    A broken LAST line (a crash in the middle of a write) is skipped with a
    warning. A broken line anywhere else means the file was damaged, so that
    raises instead of being skipped.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    events = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"warning: skipped a half-written last line in {path}")
            else:
                raise
    return events


def token_totals(events):
    """What the run costs (all calls) and what it spent now (cache misses only)."""
    calls = [e for e in events if e["event"] == "llm_call"]
    cost = sum(e["prompt_tokens"] + e["completion_tokens"] for e in calls)
    spent = sum(e["prompt_tokens"] + e["completion_tokens"] for e in calls if not e["cache_hit"])
    return {"llm_calls": len(calls), "cache_hits": sum(e["cache_hit"] for e in calls),
            "tokens_cost": cost, "tokens_spent": spent}


if __name__ == "__main__":
    # Self-test: writes runs/selftest.jsonl with fake events, reads it back, deletes it.
    from types import SimpleNamespace as NS

    from cache import CallResult

    def fake_result(hit, prompt, completion, reasoning, tool_calls=0):
        message = NS(tool_calls=[object()] * tool_calls or None)
        choice = NS(finish_reason="tool_calls" if tool_calls else "stop", message=message)
        usage = NS(prompt_tokens=prompt, completion_tokens=completion,
                   completion_tokens_details=NS(reasoning_tokens=reasoning))
        return CallResult(response=NS(choices=[choice], usage=usage), error=None,
                          cache_hit=hit, key="k" * 64, elapsed=0.8)

    log = RunLog("selftest", arm="A")
    try:
        log.llm_call("q01", "single", 1, fake_result(True, 415, 196, 164, tool_calls=1))
        log.tool_call("q01", "single", 1, "find_entities", '{"text": "Prudential"}',
                      "prudential: Prudential plc", None)
        log.llm_call("q01", "single", 2, fake_result(False, 480, 60, 40))
        rejected = CallResult(response=None, cache_hit=False, key="r" * 64, elapsed=0.5,
                              error={"code": "tool_use_failed", "message": "m",
                                     "failed_generation": '{"name": "get_edges"}'})
        log.llm_call("q02", "single", 1, rejected)
        log.question_end("q01", "The sources do not say.", "answered")
        log.question_end("q02", None, "tool_use_failed")

        with open(log.path, "a", encoding="utf-8") as f:   # simulate a crash mid-line
            f.write('{"event": "llm_ca')

        events = read_log(log.path)
        print("events read:", len(events), "(expected 6)")
        print("by type:", {t: sum(e["event"] == t for e in events)
                           for t in ("llm_call", "tool_call", "question_end")})
        print("rejected call logged as:", events[3]["error"], events[3]["failed_generation"])
        print("totals:", token_totals(events), "(expected cost 1151, spent 540)")
    finally:
        log.path.unlink(missing_ok=True)