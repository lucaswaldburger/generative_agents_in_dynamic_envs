# gpt_structure.py

from typing import Any
from openai import OpenAI
from omegaconf import DictConfig
import json

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

class LLMConversation:
    def __init__(self, cfg: DictConfig, system_prompt: str):
        self.client = get_openai_client(cfg.openai.openai_api_key)
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def ask_llm(self, user_text: str, model: str = "gpt-4.1-mini") -> str:
        self.messages.append({"role": "user", "content": user_text})

        resp = self.client.chat.completions.create(
            model=model,
            messages=self.messages,
            max_tokens=1000,
        )
        answer = resp.choices[0].message.content.strip()

        self.messages.append({"role": "assistant", "content": answer})
        return answer





def llm_decide_intent(conv, agent_cfg, plan_item, perception_desc, external_events, clock_time, valid_locations):


    plan_text = plan_item["activity"] if plan_item else "no scheduled activity"
    plan_loc  = plan_item["location"] if plan_item else None

    prompt = f"""
        Time: {clock_time}

        Agent persona:
        - name: {agent_cfg.name}
        - age: {agent_cfg.age}
        - gender: {agent_cfg.gender}
        - innate: {agent_cfg.innate}
        - dependents: {agent_cfg.dependents}
        - living_area: {agent_cfg.living_area}

        Scheduled plan right now:
        - activity: {plan_text}
        - location: {plan_loc}

        Perception:
        {perception_desc}

        External events:
        {external_events}

        Valid locations in this map (choose exactly one if moving):
        {valid_locations}

        Task:
        Decide the agent's intent and where they will go.
        If staying, target_location should be null.

        Return ONLY JSON:
        {{
        "intent": "<follow_plan|stay|evacuate|check_dependent|help_other|reroute|other>",
        "target_location": "<one string from valid_locations OR null>",
        "command": "<either 'stay' OR 'go to <target_location>'>",
        "reason": "<1-3 sentences>"
        }}
        """
    raw = conv.ask_llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent":"follow_plan", "target_location": plan_loc, "command": f"go to {plan_loc}" if plan_loc else "stay", "reason": raw}



def llm_decide_local_direction(conv, agent_cfg, perception_desc, high_level_goal, valid_dirs):
    prompt = f"""
You are controlling {agent_cfg.name}.
High-level goal: {high_level_goal}.
Perception: {perception_desc}

Valid directions: {valid_dirs}

Choose exactly ONE direction from Valid directions.
Reply JSON:
{{"direction": "<ONE_OF_VALID>", "reason": "..."}}
"""
    return conv.ask_llm(prompt)


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
