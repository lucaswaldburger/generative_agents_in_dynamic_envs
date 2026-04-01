# gpt_structure.py

from typing import Any
from openai import OpenAI
from omegaconf import DictConfig
import json
from dataclasses import dataclass, field

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


@dataclass
class LLMCallRecord:
    t: int | None
    agent_name: str | None
    call_type: str  # "intent", "mid_local", "social", etc.
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMConversation:
    def __init__(self, cfg, system_prompt: str, track_tokens: bool = True):
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.track_tokens = track_tokens

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.call_log: list[LLMCallRecord] = []
        self.client = get_openai_client(cfg.openai.openai_api_key)
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def ask_llm(
        self,
        user_text: str,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 150,
        meta: dict | None = None,
    ) -> str:
        """
        meta can contain:
        - "t": simulation step
        - "agent_name"
        - "call_type": "intent" | "mid" | "social" | ...
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        answer = resp.choices[0].message.content.strip()

        usage = getattr(resp, "usage", None)
        if self.track_tokens and usage is not None:
            pt = usage.prompt_tokens
            ct = usage.completion_tokens
            tt = usage.total_tokens
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
            self.total_calls += 1
            m = meta or {}
            self.call_log.append(
                LLMCallRecord(
                    t=m.get("t"),
                    agent_name=m.get("agent_name"),
                    call_type=m.get("call_type", "unknown"),
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                )
            )
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
    t: int = 0,
    urgency_assessment: str | None = None,
):
    plan_text = plan_item["activity"] if plan_item else "no scheduled activity"
    plan_loc  = plan_item["location"] if plan_item else None
    is_at_plan = (plan_loc is not None and current_location is not None and plan_loc == current_location)

    persona = agent_cfg.persona_compact
    valid_locs = ",".join(valid_locations)

    home_loc = getattr(agent_cfg, "living_area", None)
    dependents_desc = summarize_dependents(agent_cfg)

    # Build urgency section separately to avoid f-string backslash issue
    urgency_section = ""
    if urgency_assessment:
        urgency_section = f"\nURGENCY ASSESSMENT:\n{urgency_assessment}\n"

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
    {urgency_section}

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
    - IMPORTANT: Consider the urgency assessment carefully. 
      - If urgency is CRITICAL or HIGH: You MUST choose "evacuate" - staying is not safe.
      - If urgency is MEDIUM: You should strongly consider "evacuate" unless there are compelling reasons to stay (e.g., 
        immediate danger to dependents at current location that requires staying briefly).
      - If urgency is LOW: You may choose "ignore" if the fire is distant and not spreading toward you.
      - Safety assessments indicating "in_danger" or "critical_danger" require evacuation.
      - Always prioritize safety over routine activities when urgency is medium or higher.

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
    - "reason": REQUIRED - A clear explanation of why this decision was made.
        - If intent="ignore" (staying): MUST explain why the agent chooses to stay despite the situation.
          Include: (1) assessment of the urgency/safety level, (2) why staying is appropriate given the urgency,
          (3) how the agent's persona traits influence this decision, (4) what they will do while staying.
          Example: "The urgency is low and fire is distant. Isabella tends to underestimate risks and is skeptical
          of authority warnings, so she will continue working while monitoring the situation."
        - If intent="evacuate": Explain why evacuation is necessary and why the chosen target location was selected.

    Return ONLY valid JSON, no extra text. Example of a valid evacuate response from work to rescue a cat at Home_A:

    # {{
    #   "intent": "evacuate",
    #   "action": "go to Home_A",
    #   "next_action": "go home to rescue my cat and then follow further instructions",
    #   "target_location": "Home_A",
    #   "command": "go to Home_A",
    #   "reason": "Explain briefly why this decision makes sense given the persona, plan, dependents, and the event."
    # }}
    # """

    raw = conv.ask_llm(
        prompt,
        meta={
            "t": t,
            "agent_name": agent_cfg.name,
            "call_type": "intent",
        },
    )
    cleaned = _strip_code_fence(raw)
    return json.loads(cleaned)


def llm_decide_local_direction(
    conv,
    agent_cfg,
    perception_desc: str,
    high_level_goal: str,
    valid_dirs: list[str],
    route_priors: dict | None = None,
    t: int = 0,
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
    
    return json.loads(conv.ask_llm(
        prompt,
        meta={
            "t": t,
            "agent_name": agent_cfg.name,
            "call_type": "mid",
        },
    ))


import json

def llm_decide_social(
    conv,
    ego_cfg,
    friend_cfg,
    perception_desc: str,
    hazards: list[str],
    current_command: str,
    clock_time: str,
    t: int,
):

    if not hazards:
        return {
            "talk": False,
            "new_command": None,
            "reason": "No hazards to discuss.",
        }
 
    hazard_text = "; ".join(hazards)
    prompt = (
        f"SOCIAL DECISION Time:{clock_time}\n"
        f"Ego:{ego_cfg.persona_compact} Friend:{friend_cfg.name}\n"
        f"Perception:{perception_desc}\n"
        f"Hazards:{hazard_text} Cmd:{current_command}\n"
        f"Should ego talk to friend? If yes, optionally override command.\n"
        f'Reply: {{"talk":true/false,"new_command":"<cmd>|null","reason":"..."}}'
    )
 
    raw = conv.ask_llm(
        prompt,
        max_tokens=80,
        meta={"t": t, "agent_name": ego_cfg.name, "call_type": "social"},
    )
    try:
        return json.loads(_strip_code_fence(raw))
    except Exception as e:
        print("[WARN] llm_decide_social parse error:", e, "raw:", raw)
        return {"talk": False, "new_command": None, "reason": "Fallback."}
