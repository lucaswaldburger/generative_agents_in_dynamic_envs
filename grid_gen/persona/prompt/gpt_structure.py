# gpt_structure.py

import json
from openai import OpenAI

client = OpenAI()


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
