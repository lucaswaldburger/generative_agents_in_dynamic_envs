#---------------------------------------------------------------
import re # to extract coordinates (x, y) from shared hazard strings
#---------------------------------------------------------------

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
        agent.assoc_mem.add_fact(
            text=fact,
            importance=10,   # max/near-max
            tags=["identity", "social", "dependents"],
        )

#---------------------------------------------------------------
# added for social
def add_social_memory(agent, info: str):
    if hasattr(agent, "assoc_mem") and agent.assoc_mem is not None:
        try:
            agent.assoc_mem.add_fact(
                text=f"Friend said: {info}",
                importance=6,  # moderate importance
                tags=["social", "shared_info", "hazard"],
            )
        except Exception as e:
            print(f"[WARN] add_social_memory: could not add_fact for {getattr(agent, 'config', None)}: {e}")


    m = re.search(r"\((\d+),\s*(\d+)\)", info)
    if m:
        x = int(m.group(1))
        y = int(m.group(2))

        if not hasattr(agent, "known_hazard_cells"):
            agent.known_hazard_cells = set()

        agent.known_hazard_cells.add((x, y))


def hazard_to_dialogue(info: str) -> str:

    lower = info.lower()

    m = re.search(r"([A-Za-z0-9_]+)\s+is on fire", info)
    m = re.search(r"smoke near\s+([A-Za-z0-9_]+)", info, re.IGNORECASE)
    if m:
        place = m.group(1)
        return f"I see smoke near {place}."

    if "fire" in lower:
        return "Help, there's a fire nearby!"
    if "smoke" in lower:
        return "I see smoke nearby."

    if "traffic" in lower:
        return "There is heavy traffic on the roads."

#---------------------------------------------------------------


