from __future__ import annotations

from agent.llm import SimulationLLM
from agent.schemas import IntentDecision, DirectionDecision, SocialDecision
from agent.prompts import summarize_dependents


def decide_intent(
    sim_llm: SimulationLLM,
    agent_cfg,
    plan_item,
    perception_desc: str,
    external_events,
    clock_time: str,
    valid_locations: list[str],
    current_location: str | None,
    t: int = 0,
    urgency_assessment: str | None = None,
) -> dict:
    """High-level intent decision via LangChain structured output.

    Returns a plain dict (same keys as the original llm_decide_intent).
    """
    plan_text = plan_item["activity"] if plan_item else "no scheduled activity"
    plan_loc = plan_item["location"] if plan_item else None
    is_at_plan = (
        plan_loc is not None
        and current_location is not None
        and plan_loc == current_location
    )

    persona = agent_cfg.persona_compact
    valid_locs = ",".join(valid_locations)
    home_loc = getattr(agent_cfg, "living_area", None)
    dependents_desc = summarize_dependents(agent_cfg)

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

    result = sim_llm.decide(
        prompt,
        IntentDecision,
        meta={"t": t, "agent_name": agent_cfg.name, "call_type": "intent"},
    )
    return result.model_dump()


def decide_local_direction(
    sim_llm: SimulationLLM,
    agent_cfg,
    perception_desc: str,
    high_level_goal: str,
    valid_dirs: list[str],
    route_priors: dict | None = None,
    t: int = 0,
) -> dict:
    """Mid-level direction choice via LangChain structured output.

    Returns a plain dict with keys ``direction`` and ``reason``.
    """
    priors_text = ""
    if route_priors is not None:
        lines = [
            f"Route-choice priors for this agent "
            f"(age group {route_priors['age_bucket']}, {route_priors['gender_key']}):"
        ]
        if route_priors.get("width_pref") is not None:
            lines.append(
                f"- Tends to choose the WIDER corridor with probability "
                f"about {route_priors['width_pref']:.2f} when widths differ."
            )
        if route_priors.get("transition_pref") is not None:
            lines.append(
                f"- Tends to choose the corridor that contains a TRANSITION CUE "
                f"(e.g., stairs/entrance) with probability about "
                f"{route_priors['transition_pref']:.2f}."
            )
        cw = route_priors.get("conflict_follow_width")
        ct = route_priors.get("conflict_follow_transition")
        if cw is not None and ct is not None:
            lines.append(
                f"- When width and transition cues CONFLICT: follows WIDTH "
                f"about {cw:.2f} vs TRANSITION CUE about {ct:.2f}."
            )
        lines.append(
            "Use these as soft biases when the available directions differ "
            "in corridor width or transition cues. "
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

    result = sim_llm.decide(
        prompt,
        DirectionDecision,
        meta={"t": t, "agent_name": agent_cfg.name, "call_type": "mid"},
    )
    return result.model_dump()


def decide_social(
    sim_llm: SimulationLLM,
    ego_cfg,
    friend_cfg,
    perception_desc: str,
    hazards: list[str],
    current_command: str,
    clock_time: str,
    t: int,
) -> dict:
    """Social-level decision via LangChain structured output.

    Returns a plain dict.  Falls back to a safe no-op on parse errors.
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

    try:
        result = sim_llm.decide(
            prompt,
            SocialDecision,
            meta={"t": t, "agent_name": ego_cfg.name, "call_type": "social"},
        )
        return result.model_dump()
    except Exception as e:
        print("[WARN] decide_social parse error:", e)
        return {
            "talk": False,
            "new_command": None,
            "reason": "Fallback: keep current command.",
        }
