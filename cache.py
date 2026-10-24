"""
On-disk cache for LLM calls.

Every LLM call in every arm goes through cached_completion(). A request that
was answered before is answered from disk: no API call, no tokens. That makes
re-runs free and makes the reported numbers come from one recorded run.

What is cached:
  - successful responses
  - tool_use_failed rejections (the model's own behaviour for that request;
    not caching them would let re-runs retry failures until they succeed,
    which inflates accuracy)
What is not cached (raised to the caller instead):
  - rate limits, server errors, timeouts: infrastructure, not model behaviour
  - any other 400: a bug in our request

The key is a SHA-256 hash of the whole request (model, messages, tools,
temperature and anything else passed to create). Leaving a field that changes
the answer out of the key would silently replay old answers, so the whole
request is hashed, not a chosen subset.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import BadRequestError
from openai.types.chat import ChatCompletion

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"


@dataclass
class CallResult:
    response: ChatCompletion | None   # None when the model's output was rejected
    error: dict | None                # {"code", "message", "failed_generation"} for tool_use_failed
    cache_hit: bool
    key: str


def cache_key(request):
    """Same request content -> same key, whatever order the dict keys are in."""
    text = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path, data):
    """Write to a temp file, then rename: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)          # atomic on Windows and Linux
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _error_body(exc):
    """The provider's error dict, whether or not it is wrapped in {"error": ...}."""
    body = exc.body if isinstance(exc.body, dict) else {}
    return body.get("error", body) if isinstance(body.get("error"), dict) else body


def _from_record(record, key, hit):
    # construct(), not model_validate(): it rebuilds the nested objects without
    # strict validation. Groq returns values the OpenAI SDK's types do not list
    # (service_tier "on_demand"), and validation would reject a response the
    # SDK itself accepted when the call was made.
    response = (ChatCompletion.construct(**record["response"])
                if record["response"] is not None else None)
    return CallResult(response=response, error=record["error"], cache_hit=hit, key=key)


def cached_completion(client, **request):
    """
    client.chat.completions.create(**request), answered from disk when possible.

    request must be plain JSON data (dicts, lists, strings, numbers). Passing
    SDK objects raises TypeError here, before any call is made.
    """
    key = cache_key(request)
    path = CACHE_DIR / f"{key}.json"

    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return _from_record(record, key, hit=True)

    try:
        response = client.chat.completions.create(**request)
        record = {"response": response.model_dump(), "error": None}
    except BadRequestError as e:
        body = _error_body(e)
        if body.get("code") != "tool_use_failed":
            raise                      # our request is wrong: a bug, not a result
        record = {"response": None,
                  "error": {"code": "tool_use_failed",
                            "message": body.get("message"),
                            "failed_generation": body.get("failed_generation")}}

    record["request"] = request        # for debugging; contains no API key
    record["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_atomic(path, record)
    return _from_record(record, key, hit=False)


if __name__ == "__main__":
    # Self-test with a fake client: no API calls, no tokens.
    import shutil

    import httpx

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **request):
            self.calls += 1
            if request["messages"][-1]["content"] == "fail":
                raise BadRequestError(
                    "tool_use_failed",
                    response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                    body={"error": {"code": "tool_use_failed",
                                    "message": "attempted to call tool 'get_edges'",
                                    "failed_generation": '{"name": "get_edges"}'}})
            return ChatCompletion.construct(**{
                "id": "c1", "object": "chat.completion", "created": 0, "model": request["model"],
                "service_tier": "on_demand",   # what Groq really sends; strict validation rejects it
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {})()
            self.chat.completions = FakeCompletions()

    CACHE_DIR = ROOT / "cache_selftest"
    client = FakeClient()
    ask = {"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hi"}]}
    bad = {"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "fail"}]}

    try:
        a = cached_completion(client, **ask)
        b = cached_completion(client, **ask)
        c = cached_completion(client, **{"temperature": 0, "model": "m",
                                         "messages": [{"role": "user", "content": "hi"}]})
        d = cached_completion(client, **{**ask, "temperature": 1})
        e = cached_completion(client, **bad)
        f = cached_completion(client, **bad)

        print("first call     hit:", a.cache_hit, "| content:", a.response.choices[0].message.content)
        print("same request   hit:", b.cache_hit, "| tokens still reported:", b.response.usage.total_tokens)
        print("keys reordered hit:", c.cache_hit)
        print("temperature=1  hit:", d.cache_hit)
        print("tool_use_failed hit:", e.cache_hit, "->", f.cache_hit, "| error:", f.error["code"])
        print("API calls made:", client.chat.completions.calls, "(expected 3)")
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)