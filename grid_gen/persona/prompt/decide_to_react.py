# persona/cognitive/decide_to_react.py

"""
decide_to_react prompt module.

This file is for:
- Call the LLM (through safe_generate_structured_response)
- Decide what the agent should do next in a fire situation
- Decide where the agent should go next (home / work / park)

This file DOES NOT:
- Change the environment
- Save memory by itself

Other scripts that connect to this:
- gpt_structure.py  -> actually calls the LLM and returns JSON
- perceive.py       -> builds obs dict from env
- plan.py           -> A* planning with get_next_plan_and_waypoint(...)
- main.py           -> main loop, where everything is integrated together
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from gpt_structure import safe_generate_structured_response



# Data classes for this module
@dataclass
class PersonaDemographics:
    """
    Simple information about the agent.

    Should be connected to main.py. Example (in main.py):

        from persona.cognitive.decide_to_react import PersonaDemographics

        demo = PersonaDemographics(
            name="Isabella Rodriguez",
            age=34,
            gender="woman",
        )
    """
    name: str      # name from persona_spec or manual
    age: int       # age in years
    gender: str    # "woman", "man", "x", etc.


@dataclass
class Perception:
    """
    Simple perception information to show the LLM.

    CONNECTING TO main.py:

    main.py must:
    1) call get_obs(env, ...)            ← from persona.cognitive.perceive
    2) call build_perception_from_obs()  ← from this file
    3) store the returned Perception object
    4) pass that object into decide_to_react(..)

    Example (in main.py):
        from persona.cognitive.perceive import get_obs
        from persona.cognitive.decide_to_react import build_perception_from_obs

        obs_struct = get_obs(env, include_world_coords=True, include_street=False)

        perception = build_perception_from_obs(
            env,
            obs_struct,
            official_order_level=0,
            location_label=current_location_text,
        )
    """
    location_desc: str
    distance_to_fire: float
    see_fire: bool
    official_order_level: int


@dataclass
class MemorySummary:
    """
    Very small "memory summary".

    The full memory system is not here.
    Should be connected to main.py. Example (in main.py):

        from persona.cognitive.decide_to_react import MemorySummary

        memory = MemorySummary(
            has_seen_fire_before=False,
            last_decision="",
            current_goal_summary="following normal daily routine (going to work)",
        )
    """
    has_seen_fire_before: bool
    last_decision: str        # e.g. "wait", "move_exit"
    current_goal_summary: str # e.g. "try to reach exit A"



# Helper: build Perception from env + obs
def _manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    """Simple Manhattan distance function."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def build_perception_from_obs(
    env,
    obs: Dict[str, Any],
    *,
    official_order_level: int = 0,
    location_label: Optional[str] = None,
) -> Perception:
    """
    Turn MiniGrid obs into a Perception object.

    Typical usage (in main.py):

        from persona.cognitive.perceive import get_obs
        from persona.cognitive.decide_to_react import build_perception_from_obs

        obs_struct = get_obs(env, include_world_coords=True, include_street=False)
        perception = build_perception_from_obs(
            env,
            obs_struct,
            official_order_level=0,
            location_label=current_location_text,
        )
    """

    # obs is expected to have:
    # - "agent_pos": (x, y)
    # - "visible": list of dicts with "sem_name" and "world"/"local" positions
    agent_pos = obs.get("agent_pos", (0, 0))
    visible = obs.get("visible", [])

    # Find nearest fire in visible area
    min_fire_dist = math.inf
    sees_fire = False

    for item in visible:
        sem_name = item.get("sem_name")
        if sem_name != "fire":
            continue

        sees_fire = True

        # prefer world coords if given, else local
        world = item.get("world")
        local = item.get("local", (0, 0))
        ref = world if world is not None else local

        d = _manhattan(agent_pos, ref)
        if d < min_fire_dist:
            min_fire_dist = d

    if min_fire_dist < math.inf:
        distance_to_fire = float(min_fire_dist)
    else:
        # no fire visible
        distance_to_fire = 9999.0

    # Simple location label if none is provided
    if location_label is None:
        ax, ay = agent_pos
        location_label = f"grid_position_{ax}_{ay}"

    return Perception(
        location_desc=location_label,
        distance_to_fire=distance_to_fire,
        see_fire=sees_fire,
        official_order_level=official_order_level,
    )



################################################################ Main LLM policy function
def decide_to_react(
    demo: PersonaDemographics,
    perception: Perception,
    memory: MemorySummary,
    empirical_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the LLM:
    - how to treat the current baseline A* route
      (follow it as is, or reroute away from fire)
    - how to update memory

    It only returns a decision dict.

    main.py should:
    - call this function,
    - read the "action" (route mode),
    - pass this route mode into the A* planner as a cost-bias flag,
    - update MemorySummary with "memory_update".
    """

    # get real-world route choice data analysis as a part of prior memory
    if empirical_hint is None:
        hint_block = "No empirical evacuation hint is available."
    else:
        hint_block = empirical_hint

    ############################################# Build the prompt string for the LLM
    # In this design, the environment does NOT execute low-level movement
    # directly from the LLM. The environment uses A* to plan on the grid.
    # The LLM only decides:
    #   (1) how to treat the baseline A* route (follow / reroute),
    #   (2) optionally how strongly to consider empirical priors
    #       (depending on age, gender, etc.).

    prompt = f"""
You are simulating {demo.name}'s behavior in a small urban world
during a building fire evacuation scenario.

[WORLD LAYOUT]
The world is a small city made of rectangular blocks:
- North blocks: B1–B4
- Central blocks: B5, B7, B8
- South blocks: B9–B12
- There is a Park in the central-south area.
- There are Homes (Home A, Home B) in the south-west area.
- There is a Workplace (Workplace A) in the east-central area.
- The Fire region is located in the north-east area of the city.

[PERSONA]
- Age: {demo.age}
- Gender: {demo.gender}

[CURRENT STATE]
- Current location description: {perception.location_desc}
- Distance to visible fire (grid units): {perception.distance_to_fire:.1f}
- Is fire currently visible in the field of view?: {"yes" if perception.see_fire else "no"}
- Official evacuation order level (0–3): {perception.official_order_level}

[BASELINE ROUTE FROM A*]
The baseline A* planner, ignoring fire risk, would follow this route:
- {memory.current_goal_summary}

You may treat this as the "normal" shortest path in this city
if there were no hazards.

[EMPIRICAL HINT]
This text summarizes empirical route-choice or evacuation behavior.
Use it only as a soft prior, not a hard rule:

{hint_block}

[TASK – VERY IMPORTANT]
You do NOT choose low-level grid moves directly.
The environment will use A* to actually move the agent.

Your job is to decide ONE route decision mode (returned as "action"):

- "follow_baseline"
    → follow the baseline A* route as it is.

- "reroute_away_from_fire"
    → choose another route that increases distance from the fire,
      even if it is longer. You may refer to the empirical hint above,
      considering this person's age ({demo.age}) and gender ({demo.gender}).

Also decide whether this moment should be written into memory as a
salient fire-related event, and how important it is.

Return ONLY valid JSON with the following structure:

{{
  "action": "follow_baseline" | "reroute_away_from_fire",
  "reason": "short explanation referencing the city layout, the current state, and the empirical hint if useful",
  "memory_update": {{
    "store_fire_event": true or false,
    "event_summary": "short description of what was perceived/decided",
    "poignancy": <integer between 1 and 10>
  }}
}}
""".strip()

    schema: Dict[str, Any] = {
        "action": (str, ...),          # "follow_baseline" / "reroute_away_from_fire"
        "reason": (str, ...),
        "memory_update": (
            {
                "store_fire_event": (bool, ...),
                "event_summary": (str, ...),
                "poignancy": (int, ...),
            },
            ...,
        ),
    }

    result = safe_generate_structured_response(prompt, schema)
    return result



#  Very small pseudo-example for main.py usage
#
# from persona.cognitive.perceive import get_obs
# from persona.cognitive.plan import get_next_plan_and_waypoint
# from persona.cognitive.decide_to_react import (
#     PersonaDemographics,
#     MemorySummary,
#     build_perception_from_obs,
#     decide_to_react,
# )
#
# # 1) make persona and memory (once)
# demo = PersonaDemographics(name="Isabella Rodriguez", age=34, gender="woman")
# memory = MemorySummary(
#     has_seen_fire_before=False,
#     last_decision="",
#     current_goal_summary="following normal daily routine (going to work)",
# )
#
# current_location_text = "home"
# next_plan_text = "go to work"
# empirical_hint = None  # or some text from route_choice_priors.json
#
# # 2) inside your simulation loop:
# obs_struct = get_obs(env, include_world_coords=True, include_street=False)
#
# perception = build_perception_from_obs(
#     env,
#     obs_struct,
#     official_order_level=0,
#     location_label=current_location_text,
# )
#
# decision = decide_to_react(
#     demo=demo,
#     perception=perception,
#     memory=memory,
#     empirical_hint=empirical_hint,
# )
#
# action_label = decision["action"]
# next_plan_text = decision["next_plan_text"]
# reason = decision["reason"]
#
# mu = decision.get("memory_update", {}) or {}
# if mu.get("store_fire_event", False):
#     memory.has_seen_fire_before = True
#     memory.last_decision = action_label
#     memory.current_goal_summary = mu.get("event_summary", memory.current_goal_summary)
#
# path = get_next_plan_and_waypoint(current_location_text, next_plan_text, env)
# if isinstance(path, str):
#     print("Planning error:", path)
# else:
#     for action in path:
#         obs, reward, term, trunc, info = env.step(action)
#         # render, step count, etc.
#
#     # update simple location label
#     lg = next_plan_text.lower()
#     if "home" in lg:
#         current_location_text = "home"
#     elif "work" in lg:
#         current_location_text = "work"
#     elif "park" in lg:
#         current_location_text = "park"
#
