from persona.prompt.gpt_structure import llm_decide_intent
from persona.cognitive.plan import normalize_command_for_planner, get_plan_for_time
from persona.cognitive.perceive import describe_perception
from utils.logs import setup_debug_loggers

def react_to_local_fire_smoke(
    *,
    t: int,
    clock_time: str,
    agent_id: int,
    agent,
    fire_smoke_hazards,
    env,
    conv,
    valid_locations,
    intent_logger,
    agent_commands,
):
    """
    Called when THIS agent newly perceives fire/smoke.
    Triggers a fresh high-level intent just for this agent.
    """

    # At t == 0 we ccould get the daily plan.
    # Later on, we rely on current perception + command instead.
    if t == 0:
        plan_item = get_plan_for_time(agent.config, clock_time)
        current_location = plan_item["location"] if plan_item else None
    else:
        plan_item = None
        current_location = None  # LLM infers from perception_desc

    local_external_events = {
        "type": "local_fire_smoke_perception",
        "source_agent": agent.config.name,
        "hazards": fire_smoke_hazards,
        "current_command": agent_commands.get(agent_id, "stay"),
    }

    # Simple description (no valid_dirs); enough for high-level intent.
    intent_desc = describe_perception(env, agent_id, include_decision_info=False)

    decision = llm_decide_intent(
        conv=conv,
        agent_cfg=agent.config,
        plan_item=plan_item,
        perception_desc=intent_desc,
        external_events=local_external_events,   
        clock_time=clock_time,
        valid_locations=valid_locations,
        current_location=current_location,
        t=t,
    )

    print(
        f"[LOCAL INTENT FIRE/SMOKE] t={t} ({clock_time}) "
        f"{agent.config.name} sees fire/smoke, intent={decision.get('intent')}, "
        f"action={decision.get('action')}"
    )

    intent_logger.debug(
        f"t={t} ({clock_time}) [LOCAL_FIRE_SMOKE] agent={agent.config.name} "
        f"intent={decision.get('intent')} "
        f"action={decision.get('action')} "
        f"next_action={decision.get('next_action')} "
        f"target_location={decision.get('target_location')} "
        f"command={decision.get('command')} "
        f"reason={decision.get('reason')} "
        f"plan_item={plan_item} "
        f"perception={intent_desc} "
        f"external_events={local_external_events}"
    )

    cmd = normalize_command_for_planner(decision, agent.config, env)
    if (decision.get("intent") or "").lower() == "stay" or "stay" in cmd.lower():
        agent_commands[agent_id] = "stay"
    else:
        agent_commands[agent_id] = cmd

    agent_commands[agent_id] = cmd
    # Mark that this agent has reacted to fire/smoke so we don’t spam the LLM
    agent.local_fire_smoke_seen = True

    return cmd