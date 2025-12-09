
import re
from typing import Iterable, Tuple, Optional

def seed_static_memory(agent):
    cfg = agent.config
    static_facts = [
        f"My name is {cfg.name}.",
        f"I am {cfg.age} years old and {cfg.gender}.",
        f"My friends are: {', '.join(cfg.friends_with) if cfg.friends_with else 'none listed'}.",
    ]

    for dep in cfg.dependents:
        static_facts.append(f"I have a dependent: {dep}.")

    for fact in static_facts:
        if getattr(agent, "assoc_mem", None) is None:
            continue
        agent.assoc_mem.add_fact(
            text=fact,
            importance=10,   # max/near-max
            tags=["identity", "social", "dependents"],
        )

#---------------------------------------------------------------
# added for social
def add_social_memory(
    agent,
    info: str,
    coords: Optional[Iterable[Tuple[int, int]]] = None,
    ):
    """
    Store a piece of social information in the agent's associative memory.
    Optionally also update known hazard cells if coordinates are provided
    or can be parsed from 'info'.

    Parameters
    ----------
    agent : HumanAgent
    info : str
        Free-text description, e.g. "There is a fire near Home_A at (10, 5)."
    coords : iterable of (x, y), optional
        Explicit hazard coordinates to add to known_hazard_cells.
        If not given, we fall back to parsing a single (x, y) from 'info'.
    """
    # 1) Store in associative memory (if present)
    if getattr(agent, "assoc_mem", None) is not None:
        try:
            tags = ["social", "shared_info"]
            lower = info.lower()
            if "fire" in lower or "smoke" in lower or "traffic" in lower:
                tags.append("hazard")

            agent.assoc_mem.add_fact(
                text=f"Friend said: {info}",
                importance=6,  # moderate importance
                tags=tags,
            )
        except Exception as e:
            print(f"[WARN] add_social_memory: could not add_fact for {getattr(agent, 'config', None)}: {e}")

    # 2) Update known hazard cells
    if not hasattr(agent, "known_hazard_cells"):
        agent.known_hazard_cells = set()

    # If explicit coords are given, prefer those
    if coords is not None:
        for (x, y) in coords:
            agent.known_hazard_cells.add((int(x), int(y)))
        return

    # Otherwise, try to parse a single "(x, y)" from the text
    m = re.search(r"\((\d+),\s*(\d+)\)", info)
    if m:
        x = int(m.group(1))
        y = int(m.group(2))
        agent.known_hazard_cells.add((x, y))


def hazard_to_dialogue(info: str) -> str:
    """
    Turn a structured hazard description into a short line of dialogue.
    Examples of info:
        "Home_A is on fire at (10, 5)."
        "I see smoke near Workplace_A at (3, 4)."
        "There is heavy traffic on the roads."
    """
    lower = info.lower()

    # 1) Specific named-place fire, e.g. "Home_A is on fire"
    m = re.search(r"([A-Za-z0-9_]+)\s+is on fire", info, re.IGNORECASE)
    if m:
        place = m.group(1)
        return f"{place} is on fire!"

    # 2) Specific named-place smoke, e.g. "smoke near Workplace_A"
    m = re.search(r"smoke near\s+([A-Za-z0-9_]+)", info, re.IGNORECASE)
    if m:
        place = m.group(1)
        return f"I see smoke near {place}."

    # 3) Generic fallbacks
    if "fire" in lower:
        return "Help, there's a fire nearby!"
    if "smoke" in lower:
        return "I see smoke nearby."
    if "traffic" in lower:
        return "There is heavy traffic on the roads."

    # 4) Last resort: echo something generic
    return "I heard something important from a neighbor."

#---------------------------------------------------------------

def is_friend(ego_cfg, other_cfg) -> bool:
    """
    Check if other agent is listed as a friend of ego.
    Supports friends_with containing ids (e.g. 'human_2') or names.
    """
    friends = getattr(ego_cfg, "friends_with", []) or []
    return (
        getattr(other_cfg, "id", None) in friends
        or getattr(other_cfg, "name", None) in friends
    )

