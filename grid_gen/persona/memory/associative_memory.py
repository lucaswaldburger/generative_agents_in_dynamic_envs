

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