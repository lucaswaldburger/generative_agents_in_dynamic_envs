#---------------------------------------------------------------
#import re # to extract coordinates (x, y) from shared hazard strings
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
#import re  # extract coordinates (x, y) from shared hazard 
#
#def add_social_memory(agent, info: str):
#
#    # store the message 
#    agent.assoc_mem.add_fact(
#        text=f"Friend said: {info}",
#        importance=6,  # moderate importance
#        tags=["social", "shared_info", "hazard"],  
#    )
#
#    # extract coordinates 
#    m = re.search(r"\((\d+),\s*(\d+)\)", info)
#    if m:
#        x = int(m.group(1))
#        y = int(m.group(2))
#
#        if not hasattr(agent, "known_hazard_cells"):
#            agent.known_hazard_cells = set()
#
#        # add  hazard location to the agent’s hazard memory
#        agent.known_hazard_cells.add((x, y))
#---------------------------------------------------------------
