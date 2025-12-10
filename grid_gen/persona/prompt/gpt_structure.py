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

def summarize_dependents(agent_cfg):
    """
    Return a natural-language description of dependents and where they are.
    """
    deps = getattr(agent_cfg, "dependents", []) or []
    home = getattr(agent_cfg, "living_area", None)

    if not deps:
        return "This agent has no dependents."

    parts = []
    for d in deps:
        dtype = d.get("type", "dependent")
        name = d.get("name", "unknown")
        # optional extra fields if you have them
        age = d.get("age")
        extra_bits = []
        if age:
            extra_bits.append(f"{age} years old")
        if home:
            extra_bits.append(f"currently at {home}")
        extra = f" ({', '.join(extra_bits)})" if extra_bits else ""
        parts.append(f"a {dtype} named {name}{extra}")

    deps_str = "; ".join(parts)
    home_str = f"Home location: {home}." if home else ""
    return f"This agent has {deps_str}. {home_str}"

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

    home_loc = getattr(agent_cfg, "living_area", None)
    dependents_desc = summarize_dependents(agent_cfg)

    prompt = f"""
    HIGH-LEVEL INTENT PLANNER CALL
    (Triggered because a NEW external event occurred.)

    Time: {clock_time}
    External event: {external_events or "none"}

    Persona (compact): {persona}

    Daily plan: {plan_text} @ {plan_loc}
    Current location: {current_location}
    At planned location: {is_at_plan}

    Home location: {home_loc}
    Dependents: {dependents_desc}

    Perception: {perception_desc}
    Valid locations: {valid_locs}

    You must choose exactly one intent from:
    - "ignore"
    - "evacuate"

    Semantics and constraints:
    - Dependents (children, pets) are physically located at the home location, unless explicitly stated otherwise.
    - The agent can only "pack" or "check_on_dependent" when they are physically at the same location as the dependent (usually Home_*).
    - If the agent is at work and the dependent is at home, then to help the dependent they must first travel from work to home.
    - When intent="ignore", the agent continues their current plan and stays where they are.
    - When intent="evacuate", the agent should choose a concrete evacuation goal in Valid locations
      (for example: Home_A to rescue a pet, or an open safe place like Park).

    Output JSON fields:

    - "intent": must be exactly "ignore" or "evacuate".
    - "target_location":
        - If intent="ignore": null.
        - If intent="evacuate": one of the valid locations (e.g., "Home_A", "Park", "Workplace_A", etc.).
          If the agent has a dependent at home and wants to rescue them, target_location should usually be the home location.
    - "action":
        - If intent="ignore": "stay".
        - If intent="evacuate": MUST be "go to <target_location>".
    - "next_action":
        - Brief natural-language description of what they will do next (e.g., "continue working",
          "go home to rescue my pet", "go to the park to stay safe").
        - It does NOT affect the planner, it is just an explanation.
    - "command":
        - Either "stay" or "go to <target_location>".
        - This will be sent to the motion planner, so keep it simple.

    Return ONLY valid JSON, no extra text. Example of a valid evacuate response from work to rescue a cat at Home_A:

    {{
      "intent": "evacuate",
      "action": "go to Home_A",
      "next_action": "go home to rescue my cat and then follow further instructions",
      "target_location": "Home_A",
      "command": "go to Home_A",
      "reason": "Explain briefly why this decision makes sense given the persona, plan, dependents, and the event."
    }}
    """

    raw = conv.ask_llm(prompt)
    cleaned = _strip_code_fence(raw)
    return json.loads(cleaned)


def llm_decide_local_direction(
    conv,
    agent_cfg,
    perception_desc: str,
    high_level_goal: str,
    valid_dirs: list[str],
    route_priors: dict | None = None,
    ):
    # Build a short natural-language summary of priors (if available)
    priors_text = ""
    if route_priors is not None:
        lines = [f"Route-choice priors for this agent (age group {route_priors['age_bucket']}, {route_priors['gender_key']}):"]

        if route_priors.get("width_pref") is not None:
            lines.append(
                f"- Tends to choose the WIDER corridor with probability about {route_priors['width_pref']:.2f} when widths differ."
            )
        if route_priors.get("transition_pref") is not None:
            lines.append(
                f"- Tends to choose the corridor that contains a TRANSITION CUE (e.g., stairs/entrance) with probability about {route_priors['transition_pref']:.2f}."
            )
        cw = route_priors.get("conflict_follow_width")
        ct = route_priors.get("conflict_follow_transition")
        if cw is not None and ct is not None:
            lines.append(
                f"- When width and transition cues CONFLICT: follows WIDTH about {cw:.2f} vs TRANSITION CUE about {ct:.2f}."
            )

        lines.append(
            "Use these as soft biases when the available directions differ in corridor width or transition cues. "
            "If they do not differ in those ways, ignore these priors."
        )
        priors_text = "\n" + "\n".join(lines) + "\n"

    prompt = f"""You are controlling {agent_cfg.name}.
    High-level goal: {high_level_goal}.
    Perception: {perception_desc}
    Valid directions: {valid_dirs}{priors_text}

    Choose exactly ONE direction from Valid directions.
    Reply JSON:
    {{"direction": "<ONE_OF_VALID>", "reason": "..."}}
    """
    return json.loads(conv.ask_llm(prompt))


import json

def llm_decide_social(
    conv,
    ego_cfg,
    friend_cfg,
    perception_desc: str,
    hazards: list[str],
    current_command: str,
    clock_time: str,
    ):
    """
    SOCIAL-LEVEL decision: should ego agent talk to a nearby friend or keep following their goal?

    Returns a dict:
    {
        "talk": bool,
        "new_command": str | None,
        "reason": str
    }
        """
    hazard_text = "; ".join(hazards) if hazards else "none"

    prompt = f"""
    SOCIAL-LEVEL DECISION

    Time:{clock_time}
    Ego:{ego_cfg.persona_compact}
    Friend:{friend_cfg.name}

    Perception:{perception_desc}
    Hazards:{hazard_text}
    Current_command:{current_command}

    Task:
    Decide if the ego agent should briefly talk to this friend to exchange
    information about the situation, or ignore and continue following the
    current command.

    If the agent SHOULD talk:
    - "talk": true
    - "new_command": you MAY either:
        - leave it as null to keep the current_command, or
        - override with a safer high-level command like "stay" or "go to Home_A".

    If the agent SHOULD NOT talk:
    - "talk": false
    - "new_command": null

    Return ONLY JSON:
    {{
    "talk": true or false,
    "new_command": "<new high-level command string or null>",
    "reason": "<1-2 sentences>"
    }}
    """

    raw = conv.ask_llm(prompt)
    try:
        cleaned = _strip_code_fence(raw)
        return json.loads(cleaned)
    except Exception as e:
        print("[WARN] llm_decide_social parse error:", e, "raw:", raw)
        # Safe fallback: do nothing, keep current behavior
        return {
            "talk": False,
            "new_command": None,
            "reason": "Fallback: keep current command.",
        }
