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

def _strip_code_fence(text: str):
    return text.replace("```json", "").replace("```", "").strip()



def llm_decide_intent(
    conv,
    agent_cfg,
    plan_item,
    perception_desc,
    external_events,
    clock_time,
    valid_locations,
    current_location: str | None,
):
    plan_text = plan_item["activity"] if plan_item else "no scheduled activity"
    plan_loc  = plan_item["location"] if plan_item else None
    is_at_plan = (plan_loc is not None and current_location is not None and plan_loc == current_location)

    persona = agent_cfg.persona_compact
    valid_locs = ",".join(valid_locations)

    prompt = f"""
    HIGH-LEVEL INTENT PLANNER CALL
    (Triggered because a NEW external event occurred.)

    Time:{clock_time}
    External:{external_events or "none"}

    Persona:{persona}

    Plan:{plan_text}@{plan_loc}
    Current:{current_location},at_plan:{is_at_plan}

    Perception:{perception_desc}
    Valid_locs:{valid_locs}

    Choose intent:
    - ignore
    - evacuate: pack, check_on_dependent, go to <target_location> (choose one from Valid_locs)


    Rules:
    - If intent=ignore: action="stay", next_action = continue current plan, target_location=null
    - If intent=evacuate and pack: action="stay", next_action = pack_belongings, target_location=null
    - If intent=evacuate and check_on_dependent: action="stay", next_action = help dependent, target_location=null
    - If intent=evacuate and go to <target_location>: action="go to <target_location>", next_action = "go to <target_location>", target_location=<target_location>

    Return ONLY JSON:
    {{
    "intent":"...",
    "action":"...",
    "next_action":"...",
    "target_location":"... or null",
    "command":"...",
    "reason":"1-3 sentences"
    }}
    """

    raw = conv.ask_llm(prompt)
    cleaned = _strip_code_fence(raw)
    return json.loads(cleaned)


def llm_decide_local_direction(conv, agent_cfg, perception_desc, high_level_goal, valid_dirs):
    prompt = f"""You are controlling {agent_cfg.name}.
    High-level goal: {high_level_goal}.
    Perception: {perception_desc}
    Valid directions: {valid_dirs}

    Choose exactly ONE direction from Valid directions.
    Reply JSON:
    {{"direction": "<ONE_OF_VALID>", "reason": "..."}}
    """
    return json.loads(conv.ask_llm(prompt))

