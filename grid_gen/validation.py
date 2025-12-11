from __future__ import annotations
import hydra
import sys
import numpy as np
import time
import json
import csv

from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from collections import defaultdict

from pathlib import Path
from typing import Dict, Any
import yaml

from env.load_map import load_map
from persona.cognitive.react_to_hazard import react_to_local_fire_smoke
from persona.cognitive.reflect import assess_urgency, format_urgency_for_llm
from persona.load_personas import load_agent_configs
from env.grid import MultiHumanGridEnv
from env.constants import Action, DIR_TO_VEC 

# this might be temporary map until we move this high-level to the other cognitive models
from persona.cognitive.plan import get_accessible_locations, high_level_planner, astar, get_plan_for_time, normalize_command_for_planner, get_agent_tile
from env.constants import SemanticMap
from persona.cognitive.perceive import describe_perception, is_intersection, valid_move_actions, action_names, get_local_hazards, agents_in_fov, get_local_traffic_cells

from persona.control_state import AgentControlState, PlannerLevel

# load social memory
from persona.memory.associative_memory import add_social_memory, hazard_to_dialogue, is_friend

from persona.prompt.gpt_structure import test_chat_completion, LLMConversation, llm_decide_intent, llm_decide_local_direction, llm_decide_social

# logging imports
from utils.logs import  create_agent_logs, log_agent_step, setup_debug_loggers, setup_sim_output_dir, sim_time_str
from utils.external_events import get_external_events_for_t

from utils.persona_utils import encode_persona
from utils.parse_route_choice_priors import get_agent_route_priors

import os





@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """
    Loads map and runs the MiniGrid environment defined in JSON.
    """
    map_path = to_absolute_path(cfg.map.file)
    personas_path = to_absolute_path(cfg.personas.file)
    priors_path = to_absolute_path(cfg.data.route_choice_priors_file)

    agent_configs = load_agent_configs(personas_path, cfg.personas.personas_in_sim)
    for agent_cfg in agent_configs:
        agent_cfg.persona_compact = encode_persona(agent_cfg)
        # print(f"[Persona] Loaded agent '{agent_cfg.name}' with encoding: {agent_cfg.persona_compact}")
    map_spec = load_map(map_path)

    route_choice_priors = None
    try:
        with open(priors_path, "r") as f:
            priors_json = json.load(f)
            route_choice_priors = priors_json.get("route_choice_priors", {})
    except Exception as e:
        print(f"[WARN] Could not load priors JSON at {priors_path}: {e}")

    env = MultiHumanGridEnv(
        map_spec=map_spec,
        agent_configs=agent_configs,
        max_steps=cfg.sim.steps,
        render_mode=cfg.sim.render_mode,
        traffic_mode=False,
        fire_spread_rate=0.00,
    )

    print("[Env] Resetting...")
    obs, info = env.reset()
    print("Initial state:")
    env.render()


    valid_locations = ['Home_A', 'Home_B', 'fire']


    # testing connecting
    try:
        reply = test_chat_completion(
            cfg,
            "Say: 'OpenAI comms successful.'"
        )
        print("[OpenAI] Response:", reply)
    except Exception as e:
        print("[OpenAI] Error while testing API:", e)

    # Create ONE conversation for this simulation
    # let's move this somewhere else later
    conv = LLMConversation(
        cfg,
        system_prompt=(
        "You are the cognitive model for multiple human agents in a fire evacuation simulation. "
        "You never control the environment directly; instead, you provide decisions for each agent.\n\n"
        "You will be called at multiple decision levels:\n"
        "1) HIGH-LEVEL INTENT: Triggered when a NEW external event occurs (e.g., alerts, visible smoke, alarms). Given an agent's persona, daily plan, current time, perception, and external events, "
        "   decide their high-level intent (e.g., ignore, shelter in place, evacuate) and a high-level goal "
        "   such as 'go to Home_A', 'go to Workplace_A', or 'stay'.\n"
        "2) MID-LEVEL LOCAL ROUTE CHOICE: Triggered at intersections or navigating, given the agent's current high-level goal, local perception, "
        "   and a list of valid directions, choose a single direction from the allowed options (e.g., LEFT, RIGHT, FORWARD, BACK, STAY).\n"
        "3) SOCIAL-LEVEL RESPONSES: Optionally, you may later be asked to generate brief messages the agent might "
        "   say to others about hazards or guidance.\n\n"
        "Your decisions must combine data provided in each prompt, for example:\n"
        "- Empirical route-choice priors (by age, gender, etc.)\n"
        "- Each agent's persona (innate traits, learned traits, lifestyle, dependents)\n"
        "- The current situation (time of day, hazards like smoke/fire, congestion, alerts)\n\n"
        "In emergencies, safety and survival are more important than rouxtine preferences or habits. "
        "When in doubt, favor routes that avoid known hazards and reflect the agent's risk attitude and responsibilities "
        "(e.g., protecting dependents).\n"
        "All outputs must follow the requested JSON schema exactly when prompted (no extra text).\n"
        "PERSONA ENCODING SCHEMA:\n"
        "N|A|G|I|R|T|X|L|F|D|H\n"
        "Where:\n"
        "- N=name\n"
        "- A=age\n"
        "- G=gender(M/F)\n"
        "- I=innate traits\n"
        "- R=risk perception summary\n"
        "- T=authority trust summary\n"
        "- X=threat style summary\n"
        "- L=learned summary\n"
        "- F=lifestyle summary\n"
        "- D=dependents\n"
        "- H=home area\n"
        "\n"
        "Different decision leevel might use some of these encoding schema. All responses must strictly follow the JSON schema when asked."
        ),
    )

    run_dir = setup_sim_output_dir()
    agent_logs = create_agent_logs(run_dir, env)
    intent_logger, local_logger, planner_logger = setup_debug_loggers(run_dir)

    # Initialize list to store urgency data for all agents at each time step
    urgency_data = []

    # initializing variables for simulation loop
    last_external_events = None
    agent_commands = {aid: "stay" for aid in range(env.num_agents)}
    active_goal_cmd = {aid: None for aid in range(env.num_agents)} 
    planned_paths = {aid: [] for aid in range(env.num_agents)}

    control_states = {
        aid: AgentControlState()
        for aid in range(env.num_agents)
    }

    for aid, a in enumerate(env.agents):
        a.known_hazard_cells = set()
        a.traffic_memory = {}
        a.social_ignored_friends = set()
        a.local_fire_smoke_seen = False
        a.last_urgency_level = "low"
        a.current_urgency_assessment = None

        # Optional: attach for debugging
        a.control_state = control_states[aid]

        # Initialize state
        control_states[aid].high_intent = "routine"
        control_states[aid].high_command = "stay"
        control_states[aid].next_level = PlannerLevel.LOW

    for t in range(cfg.sim.steps):
        clock_time = sim_time_str(cfg, t)

        external_events = get_external_events_for_t(t)
        stimulus_triggered = (external_events is not None and external_events != last_external_events)

        # Assess urgency for all agents at every time step
        for agent_id, agent in enumerate(env.agents):
            urgency_assessment = assess_urgency(
                env=env,
                agent_id=agent_id,
                agent=agent,
                t=t,
                fire_start_time=getattr(env, "fire_start_time", None),
            )
            print(f"[URGENCY] t={t} {agent.config.name}: {urgency_assessment.urgency_level.upper()} urgency (score: {urgency_assessment.urgency_score:.2f})")
            
            # Store urgency assessment for later use
            agent.current_urgency_assessment = urgency_assessment
            
            # Store urgency data for output file
            urgency_data.append({
                'time_step': t,
                'clock_time': clock_time,
                'agent_id': agent_id,
                'agent_name': agent.config.name,
                'urgency_level': urgency_assessment.urgency_level,
                'urgency_score': urgency_assessment.urgency_score,
                'safety_assessment': urgency_assessment.safety_assessment,
                'fire_proximity': urgency_assessment.fire_proximity if urgency_assessment.fire_proximity != float('inf') else None,
                'smoke_proximity': urgency_assessment.smoke_proximity if urgency_assessment.smoke_proximity != float('inf') else None,
                'visibility_impact': urgency_assessment.visibility_impact,
                'time_since_awareness': urgency_assessment.time_since_awareness,
                'primary_factors': ', '.join(urgency_assessment.primary_factors) if urgency_assessment.primary_factors else None,
            })

        if stimulus_triggered:
            last_external_events = external_events

            # this will go to plannner eventially
            for agent_id, agent in enumerate(env.agents):
                plan_item = get_plan_for_time(agent.config, clock_time)
                if plan_item:
                    print(f"[PLAN] t={t} ({clock_time}) {agent.config.name} plan is", plan_item["activity"], "in", plan_item["location"])
                desc = describe_perception(env, agent_id)
                
                # Assess urgency and reflect on situation (for LLM decision)
                urgency_assessment = assess_urgency(
                    env=env,
                    agent_id=agent_id,
                    agent=agent,
                    t=t,
                    fire_start_time=getattr(env, "fire_start_time", None),
                )
                urgency_text = format_urgency_for_llm(urgency_assessment)
                
                decision = llm_decide_intent(
                    conv=conv,
                    agent_cfg=agent.config,
                    plan_item=plan_item,
                    perception_desc=desc,
                    external_events=external_events,
                    clock_time=clock_time,
                    valid_locations=valid_locations,
                    current_location=plan_item["location"] if plan_item else None,
                    t=t,
                    urgency_assessment=urgency_text,
                )

                print(f"[LLM HIGH-LEVEL GOAL] t={t} ({clock_time}) {agent.config.name} intent {decision['intent']}, action {decision['action']}")
                
                # Explicitly print reasoning when agent chooses to stay
                if decision.get('intent') == 'ignore' or decision.get('action') == 'stay':
                    reason = decision.get('reason', 'No reason provided')
                    print(f"[STAY DECISION] t={t} ({clock_time}) {agent.config.name} chooses to STAY")
                    print(f"  Reason: {reason}")
                    print(f"  Urgency: {urgency_assessment.urgency_level.upper()} (score: {urgency_assessment.urgency_score:.2f})")
                    print(f"  Safety: {urgency_assessment.safety_assessment}")

                intent_logger.debug(
                    f"t={t} ({clock_time}) agent={agent.config.name} "
                    f"intent={decision.get('intent')} "
                    f"action={decision.get('action')} "
                    f"next_action={decision.get('next_action')} "
                    f"target_location={decision.get('target_location')} "
                    f"command={decision.get('command')} "
                    f"reason={decision.get('reason')} "
                    f"plan_item={plan_item} "
                    f"perception={desc} "
                    f"external_events={external_events}"
                )
                cmd = normalize_command_for_planner(decision, agent.config, env)
                agent_commands[agent_id] = cmd
                active_goal_cmd[agent_id] = None if cmd == "stay" else cmd
                planned_paths[agent_id] = []
                cs = control_states[agent_id]
                cs.set_new_high_command(
                    intent=decision.get("intent", None),
                    cmd=cmd,
                )

                # # If LLM intent is stay or command includes stay → force stay
                # if decision["intent"] == "stay" or "stay" in cmd:
                #     agent_commands[agent_id] = "stay"
                # else:
                #     agent_commands[agent_id] = cmd

        # 2) LOCAL FIRE/SMOKE perception 
        for agent_id, agent in enumerate(env.agents):
            hazards = get_local_hazards(env, agent_id)
            fire_smoke_hazards = [
                h for h in hazards
                if ("fire" in h.lower()) or ("smoke" in h.lower())
            ]
            cs = control_states[agent_id]

            # Get current urgency assessment (already computed above)
            urgency_assessment = getattr(agent, "current_urgency_assessment", None)
            if urgency_assessment is None:
                urgency_assessment = assess_urgency(
                    env=env,
                    agent_id=agent_id,
                    agent=agent,
                    t=t,
                    fire_start_time=getattr(env, "fire_start_time", None),
                )

            # Check if we need to react to fire/smoke
            should_react = False
            react_reason = ""
            
            if fire_smoke_hazards and not getattr(agent, "local_fire_smoke_seen", False):
                # First time seeing fire/smoke - always react
                should_react = True
                react_reason = "first_time_seeing_fire"
            elif fire_smoke_hazards and getattr(agent, "local_fire_smoke_seen", False):
                # Already seen fire/smoke, but check if urgency has increased
                current_cmd = agent_commands.get(agent_id, "stay")
                urgency_level = urgency_assessment.urgency_level.lower()
                
                # Re-evaluate if urgency is medium or high and agent is currently staying
                if urgency_level in ["medium", "high", "critical"]:
                    if current_cmd.strip().lower() == "stay":
                        should_react = True
                        react_reason = f"urgency_{urgency_level}_while_staying"
                    else:
                        # Agent is already moving, but check if urgency increased significantly
                        last_urgency = getattr(agent, "last_urgency_level", "low")
                        if urgency_level in ["high", "critical"] and last_urgency not in ["high", "critical"]:
                            should_react = True
                            react_reason = f"urgency_escalated_to_{urgency_level}"

            # Always update last_urgency_level to track changes
            agent.last_urgency_level = urgency_assessment.urgency_level.lower()
            
            if should_react:
                urgency_text = format_urgency_for_llm(urgency_assessment)
                
                new_cmd = react_to_local_fire_smoke(
                    t=t,
                    clock_time=clock_time,
                    agent_id=agent_id,
                    agent=agent,
                    fire_smoke_hazards=fire_smoke_hazards,
                    env=env,
                    conv=conv,
                    valid_locations=valid_locations,
                    intent_logger=intent_logger,
                    agent_commands=agent_commands,
                    urgency_assessment=urgency_text,
                )
                cs.set_new_high_command(intent="evacuate_local_fire", cmd=new_cmd)
                agent_commands[agent_id] = new_cmd
                planned_paths[agent_id] = []  # clear any low-level segment
                agent.local_fire_smoke_seen = True
                print(
                    f"[LOCAL FIRE/SMOKE REPLAN] t={t} agent={agent.config.name} "
                    f"updates command to '{new_cmd}' (reason: {react_reason})"
                )


        # full_paths = []
        for agent_id in range(env.num_agents):
            cs = control_states[agent_id]
            cmd = agent_commands.get(agent_id, "stay")

            # If staying, no path
            if cmd.strip().lower() == "stay":
                planned_paths[agent_id] = []
                cs.clear_segment()
                cs.next_level = PlannerLevel.LOW
                continue

            # 1) If we already have a low-level segment, keep executing it
            if cs.has_segment():
                # low-level execution; nothing to re-plan this step
                cs.next_level = PlannerLevel.LOW
                continue

            # this is so that we can give the LLM the valid directions at intersections
            # Need a new path from current tile for this command
            if is_intersection(env, agent_id):
                cs.next_level = PlannerLevel.MID
            else:
                cs.next_level = PlannerLevel.LOW

            # 3) MID-level: only at intersections when there is no current segment
            if cs.next_level == PlannerLevel.MID:
                valid_moves = valid_move_actions(env, agent_id)
                valid_dirs = action_names(valid_moves)
                priors_for_agent = get_agent_route_priors(env.agents[agent_id].config, route_choice_priors)

                desc = describe_perception(env, agent_id, include_decision_info=True)
                print("\n[LLM MID-LEVEL GOAL]")
                print(f"agent={env.agents[agent_id].config.name}, perception:{desc}, high_level_goal:{cmd}, valid_dirs:{valid_dirs}")
                local_reply = llm_decide_local_direction(
                    conv=conv,
                    agent_cfg=env.agents[agent_id].config,
                    perception_desc=desc,
                    high_level_goal=cmd,
                    valid_dirs=valid_dirs,
                    route_priors=priors_for_agent,
                    t=t,
                )

                local_logger.debug(
                    f"t={t} agent={env.agents[agent_id].config.name} "
                    f"high_level_goal={cmd} "
                    f"perception={desc} "
                    f"valid_dirs={valid_dirs} "
                    f"direction={local_reply.get('direction')} "
                    f"reason={local_reply.get('reason')}"
                )

                print(f'[LLM MID-LEVEL GOAL REPLY] agent={env.agents[agent_id].config.name} chooses ={local_reply["direction"]} because {local_reply["reason"]}')

                try:
                    if local_reply == "stay":
                        planned_paths[agent_id] = []
                        cs.clear_segment()
                        cs.next_level = PlannerLevel.LOW
                        continue

                    chosen = local_reply["direction"]
                    cs.last_mid_direction = chosen
                    chosen_action = Action[chosen]

                    # force first move by stepping to neighbor as new start
                    sx, sy = get_agent_tile(env, agent_id)
                    dx, dy = DIR_TO_VEC[chosen_action]
                    forced_start = (sx + dx, sy + dy)

                    if env._can_move_to(*forced_start):
                        start, goal = high_level_planner(env, agent_id, cmd)
                        tail = astar(env, forced_start, goal)
                        full_path = [chosen_action] + tail
                    else:
                        # fall back to normal A*
                        start, goal = high_level_planner(env, agent_id, cmd)
                        full_path = astar(env, start, goal)
                except Exception as e:
                    print("[WARN] local decision parse error:", e)
                    start, goal = high_level_planner(env, agent_id, cmd)
                    full_path = astar(env, start, goal)

            else:
                # Not at intersection, A* from current tile
                start, goal = high_level_planner(env, agent_id=agent_id, command=cmd)
                print(
                    f"[LOW LEVEL PLANNER] t={t} Agent {agent_id} command='{cmd}' "
                    f"-> start={start}, goal={goal}"
                )
                planner_logger.debug(
                    f"t={t} agent={agent_id} command='{cmd}' "
                    f"start={start} goal={goal}"
                )
                full_path = astar(env, start, goal)


            # ---------------- Hazard avoidance on the new path, unsure if redundant -------------
            known_hazards = getattr(env.agents[agent_id], "known_hazard_cells", set())
            if full_path is not None and known_hazards:
                if any(cell in known_hazards for cell in full_path):
                    full_path = None
                    planner_logger.debug(
                        f"t={t} agent={agent_id} command='{cmd}' "
                        f"path_intersects_hazards={known_hazards}; cancelling_path"
                    )

            if full_path is None:
                print(
                    f"[WARN] No path for agent {agent_id} to '{cmd}'. "
                    f"Will stay and try again next step."
                )
                planner_logger.warning(
                    f"t={t} agent={agent_id} command='{cmd}' no_path_found"
                )
                planned_paths[agent_id] = []
                cs.clear_segment()
                continue

            fov_range = env.agents[agent_id].config.fov.range_cells
            segment_len = max(1, int(fov_range))
            segment = full_path[:segment_len]

            planned_paths[agent_id] = segment
            cs.segment_actions = list(segment)
            cs.next_level = PlannerLevel.LOW
            # ---------------------------------------------------------------



        # If everyone staying and no paths → skip env.step
        if all(not cs.has_segment() and agent_commands[aid].strip().lower() == "stay"
            for aid, cs in control_states.items()):
            print(f"t={t} all agents staying. Waiting for next stimulus.")
            continue

        # 4) EXECUTE ONE ACTION PER AGENT (single env.step)
        actions_this_step = []
        for agent_id in range(env.num_agents):
            cs = control_states[agent_id]

            if cs.has_segment():
                a = cs.pop_next_action()
            else:
                a = Action.STAY

            actions_this_step.append(a)

        action = np.array(actions_this_step, dtype=np.int64)
        obs, _, terminated, truncated, info = env.step(action)

        print(f"\nStep {t + 1}")


        # 5) Perception, social behavior, logging
        for agent_id in range(env.num_agents):
            desc = describe_perception(env, agent_id, include_decision_info=False)
            agent = env.agents[agent_id]

            # (a) social interactions
            hazards = get_local_hazards(env, agent_id)

            if not hasattr(agent, "social_ignored_friends"):
                agent.social_ignored_friends = set()

            nearby_ids = agents_in_fov(env, agent_id)

            friend_candidates = []
            for other_id in nearby_ids:
                other = env.agents[other_id]
                if other_id in agent.social_ignored_friends:
                    continue
                if is_friend(agent.config, other.config):
                    friend_candidates.append(other_id)

            if friend_candidates:
                other_id = friend_candidates[0]
                other = env.agents[other_id]

                current_cmd = agent_commands.get(agent_id, "stay")

                social_dec = llm_decide_social(
                    conv=conv,
                    ego_cfg=agent.config,
                    friend_cfg=other.config,
                    perception_desc=desc,
                    hazards=hazards,
                    current_command=current_cmd,
                    clock_time=clock_time,
                    t=t,
                )

                talk = bool(social_dec.get("talk", False))
                new_command = social_dec.get("new_command")

                if not talk:
                    agent.social_ignored_friends.add(other_id)
                else:
                    if new_command and isinstance(new_command, str):
                        agent_commands[agent_id] = new_command.strip()
                        planned_paths[agent_id] = []  # replan next step
                        print(
                            f"[SOCIAL] t={t} agent={agent.config.name} "
                            f"talks with {other.config.name}, "
                            f"new_command='{agent_commands[agent_id]}' "
                            f"reason={social_dec.get('reason')}"
                        )
                    else:
                        print(
                            f"[SOCIAL] t={t} agent={agent.config.name} "
                            f"talks with {other.config.name} but keeps command='{current_cmd}' "
                            f"reason={social_dec.get('reason')}"
                        )
                        add_social_memory(
                            agent,
                            info=(
                                f"Talked with {other.config.name} about: "
                                f"{', '.join(hazards) or 'no hazards'}"
                            ),
                        )

                    agent.social_ignored_friends.add(other_id)

                # hazards = get_local_hazards(env, agent_id)
                # if hazards:
                #     for other_id, other in enumerate(env.agents):
                #         if other_id == agent_id:
                #             continue
                        

                #         friends = getattr(agent.config, "friends_with", [])
                #         other_id_str = getattr(other.config, "id", None)
                #         other_name = getattr(other.config, "name", None)

                #         if (other_id_str in friends) or (other_name in friends):
                #             for hz in hazards:
                    
                #                 social_hazard_memory[other_id].add(hz)
                #                 add_social_memory(other, hz)

                #             print(
                #                 f"[SOCIAL] Agent {agent_id} shares {hazards} "
                #                 f"with {other.config.name}"
                #             )

                # heard = sorted(social_hazard_memory.get(agent_id, set()))

                # if heard:
                #     MAX_DIALOGUES_PER_STEP = 2
                #     heard = heard[:MAX_DIALOGUES_PER_STEP]

                #     dialogue_lines = []
                #     for hz in heard:
                #         spoken = hazard_to_dialogue(hz)

                #         dialogue_lines.append(f'A neighbor says: "{spoken}"')

                #     if dialogue_lines:
                #         desc = desc + " " + " ".join(dialogue_lines)
                #------------------------------------------------------------------

            sm = agent.config.spatial_memory
            spatial_info = None

            if sm:
                x, y = int(agent.x), int(agent.y)
                sm_info = sm.elements_at_position(env, x, y)
                contents = sm_info["contents"]

                if contents:
                    rooms_here = list(contents.keys())
                    spatial_info = (
                        f"cell='{sm_info['cell_name']}', "
                        f"lookup_key='{sm_info.get('lookup_key')}', "
                        f"rooms={rooms_here}"
                    )
                else:
                    spatial_info = f"cell='{sm_info['cell_name']}'"

            log_agent_step(
                log_file=agent_logs[agent_id],
                agent_id=agent_id,
                step=t + 1,
                agent=agent,
                action=int(action[agent_id]),
                desc=desc,
                spatial_info=spatial_info,
            )

        env.render()
        time.sleep(0.5)

        if terminated or truncated:
            break
            
    for f in agent_logs.values():
        f.close()
    env.close()
    
    # Write urgency data to CSV file
    urgency_file_path = os.path.join(run_dir, "agent_urgency.csv")
    if urgency_data:
        with open(urgency_file_path, 'w', newline='') as csvfile:
            fieldnames = ['time_step', 'clock_time', 'agent_id', 'agent_name', 'urgency_level', 
                         'urgency_score', 'safety_assessment', 'fire_proximity', 'smoke_proximity',
                         'visibility_impact', 'time_since_awareness', 'primary_factors']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(urgency_data)
        print(f"\n[Output] Urgency data written to {urgency_file_path}")
    else:
        print("\n[WARN] No urgency data collected")
    
    step_tokens = defaultdict(int)
    for rec in conv.call_log:
        if rec.t is not None:
            step_tokens[rec.t] += rec.total_tokens

    # Write token usage to file
    token_usage_file_path = os.path.join(run_dir, "token_usage.txt")
    with open(token_usage_file_path, 'w') as f:
        f.write("=== TOKEN USAGE PER STEP ===\n")
        for t in sorted(step_tokens.keys()):
            line = f"t={t}: {step_tokens[t]} tokens\n"
            f.write(line)

        f.write("\n=== TOTAL TOKEN USAGE ===\n")
        f.write(f"Total prompt tokens: {conv.total_prompt_tokens}\n")
        f.write(f"Total completion tokens: {conv.total_completion_tokens}\n")
        f.write(f"Total tokens: {conv.total_prompt_tokens + conv.total_completion_tokens}\n")
        f.write(f"Total LLM calls: {conv.total_calls}\n")
    
    print("\n=== TOKEN USAGE PER STEP ===")
    for t in sorted(step_tokens.keys()):
        print(f"t={t}: {step_tokens[t]} tokens")
    
    print(f"\n[Output] Token usage written to {token_usage_file_path}")
    print("\n=== TOTAL TOKEN USAGE ===")
    print(f"Total prompt tokens: {conv.total_prompt_tokens}")
    print(f"Total completion tokens: {conv.total_completion_tokens}")
    print(f"Total tokens: {conv.total_prompt_tokens + conv.total_completion_tokens}")
    print(f"Total LLM calls: {conv.total_calls}")
   

if __name__ == "__main__":
    run()

