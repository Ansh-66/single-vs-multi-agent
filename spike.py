import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import json
import time
from openai import OpenAI
from cache import cached_completion


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GRAPH_RAG_DIR = os.environ["GRAPH_RAG_DIR"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
sys.path.append(str(Path(GRAPH_RAG_DIR) / "src"))
import retrieve

client =  OpenAI(base_url = "https://api.groq.com/openai/v1", api_key = GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"


def find_entities(text):
    node_ids = retrieve.find_seeds(text)
    return [
        {
        "id": node_id,
        "name": retrieve.NODES["nodes"][node_id]["canonical"]
        }
        for node_id in node_ids
    ]


FIND_ENTITIES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_entities",
        "description": "Resolve company or person names or phrases in free text to node ids "
                       "in the knowledge graph. Returns a list of {id, name} matches, or an "
                       "empty list if no known company or person is mentioned. It returns no "
                       "deal facts (stakes, dates, relations). Call this first: other graph "
                       "tools take the returned ids, not names.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "A company or person name, or a sentence mentioning "
                                   "companies or people, e.g. 'Prudential plc' or "
                                   "'What stake has Prudential "
                                   "agreed to acquire in Bharti Life?'"
                }
            },
            "required": ["text"]
        }
    }
}

TOOLS = [FIND_ENTITIES_SCHEMA]

SYSTEM_PROMPT = """ You answer questions about the Indian Insurance sector using ONLY the results of your tool call.
How to work:
1. First use find_entities to resolve the companies or people in the question to node ids.
2. Answer only once the tool results contain the fact. If they never do, don't guess.

Rules:
1. Use only the facts stated in the tool results. Do not use outside knowledge.
2. If a result describes talks, approvals, or a deal not yet completed (status "prospective"), say so.
3. If the tool results do not contain the answer, say exactly: "The sources do not say."
4. Be brief: two to four sentences.
"""
QUESTION = "What stake has Prudential agreed to acquire in Bharti Life?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": QUESTION},
]

start = time.perf_counter()
result = cached_completion(
    client,
    model=MODEL,
    messages=messages,
    tools=TOOLS,
    temperature=0,
)
response = result.response
print("cache_hit:", result.cache_hit)
elapsed = time.perf_counter() - start
if result.error:
    print("rejected:", result.error)
    raise SystemExit

choice = response.choices[0]
message = choice.message

print("finish_reason:", choice.finish_reason)
print("content:", message.content)
print("tool_calls:")
for call in message.tool_calls or []:  
    print(f"  {call.id}: {call.function.name}({call.function.arguments})")
print("usage:", response.usage)
print(f"elapsed: {elapsed:.2f}s")

messages.append({
    "role": "assistant",
    "content": message.content,
    "tool_calls": [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in message.tool_calls or []
    ],
})

TOOL_FUNCTIONS = {"find_entities": find_entities}

for call in message.tool_calls or []:
    args = json.loads(call.function.arguments)
    func = TOOL_FUNCTIONS[call.function.name]
    result = func(**args)
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result),
    })

start = time.perf_counter()
result2 = cached_completion(
    client,
    model=MODEL,
    messages=messages,
    tools=TOOLS,
    temperature=0,
)
response2 = result2.response
print("cache_hit:", result2.cache_hit)
elapsed2 = time.perf_counter() - start
if result2.error:
    print("rejected:", result2.error)
    raise SystemExit

choice2 = response2.choices[0]
message2 = choice2.message

print("finish_reason:", choice2.finish_reason)
print("content:", message2.content)
print("tool_calls:")
for call in message2.tool_calls or []:
    print(f"  {call.id}: {call.function.name}({call.function.arguments})")
print("usage:", response2.usage)
print(f"elapsed: {elapsed2:.2f}s")

if __name__ == "__main__":
    # print(find_entities("What stake has Prudential agreed to acquire in Bharti Life?"))
    # print(find_entities("Prudential plc"))
    # print(find_entities("What is the weather in Mumbai?"))
    # print(json.dumps(TOOLS, indent=2))
    pass