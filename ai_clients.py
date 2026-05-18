import os

from config import GEMINI_MODEL


gemini_client = None
openai_client = None


class GenAIProxy:
    def __init__(self):
        self._module = None

    def _load(self):
        if self._module is None:
            from google import genai as genai_module

            self._module = genai_module
        return self._module

    def Client(self, *args, **kwargs):
        return self._load().Client(*args, **kwargs)


genai = GenAIProxy()


def get_gemini_client():
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        gemini_client = genai.Client(api_key=api_key)
    return gemini_client


def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        from openai import OpenAI

        openai_client = OpenAI(api_key=api_key)
    return openai_client


def gemini_valence_config():
    from google.genai import types

    return types.GenerateContentConfig(
        temperature=0,
        maxOutputTokens=48,
        responseMimeType="application/json",
        thinkingConfig=types.ThinkingConfig(thinkingBudget=0),
    )
