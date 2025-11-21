# gpt_structure.py

from typing import Any
from openai import OpenAI
from omegaconf import DictConfig

_client_cache = None

def get_openai_client(api_key: str) -> OpenAI:
    """
    Return a cached OpenAI client instance.
    """
    global _client_cache
    if _client_cache is None:
        _client_cache = OpenAI(api_key=api_key)
    return _client_cache



def test_chat_completion(cfg: DictConfig, user_text: str) -> str:
    """
    Use cfg.openai.* to call the LLM with a simple test message.
    """
    api_key = cfg.openai.openai_api_key
    client = get_openai_client(api_key)

    # simple test chat
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",   # good, cheap test model
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant. "
                    f"The key owner is {cfg.openai.key_owner}."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        max_tokens=64,
    )

    return resp.choices[0].message.content.strip()




def safe_generate_structured_response(prompt: str, schema: dict):
    """
    Send a prompt to the LLM and force it to return valid JSON.
    If parsing fails, retries by asking the model to correct itself.

    Args:
        prompt: str - user/system prompt
        schema: dict - NOT formally validated, only passed for context if needed

    Returns:
        parsed JSON (Python dict)
    """

    # 1) First attempt
    response = client.chat.completions.create(
        model="gpt-4.1-mini",     
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You MUST return ONLY valid JSON. No explanation."},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content.strip()

    try:
        return json.loads(response)
    except Exception:
        pass  # fall through to retry

    # 2) Retry with correction
    correction_prompt = f"""
The following output was invalid JSON. Fix it and return valid JSON only.

--- INVALID OUTPUT ---
{response}
----------------------

Correct it according to the schema. Return ONLY valid JSON.
"""
    corrected = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Return valid JSON. No explanation."},
            {"role": "user", "content": correction_prompt},
        ],
    ).choices[0].message.content.strip()

    try:
        return json.loads(corrected)
    except Exception:
        # Last resort: give empty valid fallback
        return {"error": "Failed to parse JSON from LLM."}
