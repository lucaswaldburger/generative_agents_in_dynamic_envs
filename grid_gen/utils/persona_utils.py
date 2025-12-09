def encode_persona(agent_cfg):
    """
    Compact persona encoding.
    N|A|G|I|R|T|X|L|F|D|H
    """


    # Summaries
    learned = "owns_WorkplaceA" if "Workplace_A" in str(agent_cfg.learned) else "none"
    lifestyle = "walks_to_work" if "walks" in str(agent_cfg.lifestyle) else "default"

    # Dependents
    if getattr(agent_cfg, "dependents", None):
        d = agent_cfg.dependents[0]
        dependents = f"{d.get('type')}:{d.get('name')}"
    else:
        dependents = "none"

    # Home area
    home = "Home_A" if "Home_A" in str(agent_cfg.living_area) else \
           "Home_B" if "Home_B" in str(agent_cfg.living_area) else "Unknown"

    return (
        f"{agent_cfg.first_name}|{agent_cfg.age}|{agent_cfg.gender[0].upper()}|"
        f"{agent_cfg.innate}|{agent_cfg.risk_perception}|{agent_cfg.authority_trust}|{agent_cfg.threat_response}|"
        f"{learned}|{lifestyle}|{dependents}|{home}"
    )
