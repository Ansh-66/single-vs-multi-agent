"""
Settings shared by every arm and every role.

They live in one file so the arms cannot drift apart: the fairness rules say
the same model, temperature, seed and tool settings everywhere. Changing any
value after measured runs start means re-running every arm.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0

# Best effort only. Groq: "Determinism is not guaranteed." The spike showed an
# identical request taking a different path on its third run.
SEED = 42

# The first call of every agent that has tools must call at least one tool,
# so no answer is given without looking anything up. Later calls use "auto".
FIRST_CALL_TOOL_CHOICE = "required"


def make_client():
    return OpenAI(base_url=BASE_URL, api_key=os.environ["GROQ_API_KEY"])