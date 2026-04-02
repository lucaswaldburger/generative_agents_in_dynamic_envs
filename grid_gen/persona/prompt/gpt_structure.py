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



SYSTEM_PROMPT = (
    "You are the cognitive model for agents in a fire evacuation sim. "
    "Reply ONLY valid JSON matching the requested schema. No extra text.\n"
    "Persona format: N|A|G|I|R|T|X|L|F|D|H "
    "(Name|Age|Gender|Innate|Risk|Trust|Threat|Learned|Lifestyle|Dependents|Home)"
)
@dataclass
class LLMCallRecord:
    t: int | None
    agent_name: str | None
    call_type: str  # "intent", "mid_local", "social", etc.
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMConversation:
    # now stateless
    def __init__(self, cfg, system_prompt: str = None, track_tokens: bool = True):
            self.cfg = cfg
            self.system_prompt = system_prompt or SYSTEM_PROMPT
            self.track_tokens = track_tokens
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_calls = 0
            self.call_log: list[LLMCallRecord] = []
            self.client = get_openai_client(cfg.openai.openai_api_key)
    

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


def _short_dependents(agent_cfg) -> str:
    """Compact dependents string: 'cat:Whiskers@Home_A' or 'none'."""
    deps = getattr(agent_cfg, "dependents", []) or []
    if not deps:
        return "none"
    home = getattr(agent_cfg, "living_area", "unknown")
    parts = [f"{d.get('type', 'dep')}:{d.get('name', '?')}" for d in deps]
    return f"{','.join(parts)}@{home}"


def summarize_dependents(agent_cfg):
    short = _short_dependents(agent_cfg)
    return "No dependents." if short == "none" else f"Dependents: {short}"

# def summarize_dependents(agent_cfg):
#     """
#     Return a natural-language description of dependents and where they are.
#     """
#     deps = getattr(agent_cfg, "dependents", []) or []
#     home = getattr(agent_cfg, "living_area", None)

#     if not deps:
#         return "This agent has no dependents."

#     parts = []
#     for d in deps:
#         dtype = d.get("type", "dependent")
#         name = d.get("name", "unknown")
#         # optional extra fields if you have them
#         age = d.get("age")
#         extra_bits = []
#         if age:
#             extra_bits.append(f"{age} years old")
#         if home:
#             extra_bits.append(f"currently at {home}")
#         extra = f" ({', '.join(extra_bits)})" if extra_bits else ""
#         parts.append(f"a {dtype} named {name}{extra}")

#     deps_str = "; ".join(parts)
#     home_str = f"Home location: {home}." if home else ""
#     return f"This agent has {deps_str}. {home_str}"

def llm_decide_local_direction(
    conv,
    agent_cfg,
    perception_desc: str,
    high_level_goal: str,
    valid_dirs: list[str],
    route_priors: dict | None = None,
    t: int = 0,
):
    priors_line = ""
    if route_priors:
        parts = []
        wp = route_priors.get("width_pref")
        tp = route_priors.get("transition_pref")
        if wp is not None:
            parts.append(f"wider={wp:.2f}")
        if tp is not None:
            parts.append(f"transition={tp:.2f}")
        if parts:
            priors_line = (
                f"\nPriors({route_priors['age_bucket']},"
                f"{route_priors['gender_key']}): {','.join(parts)}"
            )
 
    prompt = (
        f"MID-LEVEL ROUTE\n"
        f"Agent:{agent_cfg.name} Goal:{high_level_goal}\n"
        f"Perception:{perception_desc}\n"
        f"Valid:{valid_dirs}{priors_line}\n"
        f'Pick ONE direction. Reply: {{"direction":"<DIR>","reason":"..."}}'
    )
 
    raw = conv.ask_llm(
        prompt,
        max_tokens=60,
        meta={"t": t, "agent_name": agent_cfg.name, "call_type": "mid"},
    )
    return json.loads(_strip_code_fence(raw))

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
    plan_text = plan_item["activity"] if plan_item else "none"
    plan_loc = plan_item["location"] if plan_item else None
    persona = agent_cfg.persona_compact
    home_loc = getattr(agent_cfg, "living_area", None)
    deps = _short_dependents(agent_cfg)
    valid_locs = ",".join(valid_locations)
 
    urg = f"\nUrgency:{urgency_assessment}" if urgency_assessment else ""
 
    prompt = (
        f"HIGH-LEVEL INTENT\n"
        f"Time:{clock_time} Event:{external_events or 'none'}\n"
        f"Persona:{persona}\n"
        f"Plan:{plan_text}@{plan_loc} Loc:{current_location} Home:{home_loc}\n"
        f"Deps:{deps}\n"
        f"Perception:{perception_desc}\n"
        f"ValidLocs:{valid_locs}{urg}\n\n"
        f"Rules:\n"
        f"- Urgency HIGH/CRITICAL or safety=in_danger/critical: MUST evacuate.\n"
        f"- Urgency LOW + fire distant: may ignore.\n"
        f"- Dependents are at home; must travel there to help them.\n\n"
        f"If intent=ignore: target_location=null, action='stay', command='stay'.\n"
        f"If intent=evacuate: target_location=one of ValidLocs, action='go to <loc>', command='go to <loc>'.\n"
        f'Reply JSON: {{"intent":"...","target_location":"...","action":"...","next_action":"...","command":"...","reason":"..."}}'
    )
 
    raw = conv.ask_llm(
        prompt,
        max_tokens=120,
        meta={"t": t, "agent_name": agent_cfg.name, "call_type": "intent"},
    )
    result = json.loads(_strip_code_fence(raw))
 
    # Safety check: if command is null/None or contains "null", fix it
    cmd = result.get("command")
    if not cmd or cmd.strip().lower() == "null":
        if result.get("intent") == "evacuate" and result.get("target_location"):
            result["command"] = f"go to {result['target_location']}"
        else:
            result["command"] = "stay"
            result["intent"] = "ignore"
 
    return result




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
