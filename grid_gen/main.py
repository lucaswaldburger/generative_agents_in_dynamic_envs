from __future__ import annotations
import hydra
import sys
import numpy as np
import time
import json

from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from pathlib import Path
from typing import Dict, Any
import yaml

from env.load_map import load_map
from persona.load_personas import load_agent_configs
from env.grid import MultiHumanGridEnv
from env.constants import Action, DIR_TO_VEC 

# this might be temporary map until we move this high-level to the other cognitive models
from persona.cognitive.plan import get_accessible_locations, high_level_planner, astar, get_plan_for_time, normalize_command_for_planner, get_agent_tile
from env.constants import SemanticMap
from persona.cognitive.perceive import describe_perception, is_intersection, valid_move_actions, action_names, get_local_hazards

# load social memory
from persona.memory.associative_memory import add_social_memory, hazard_to_dialogue

from persona.prompt.gpt_structure import test_chat_completion, LLMConversation, llm_decide_intent, llm_decide_local_direction

# logging imports
from utils.logs import  create_agent_logs, log_agent_step, setup_debug_loggers, setup_sim_output_dir, sim_time_str
from utils.external_events import get_external_events_for_t

from utils.persona_utils import encode_persona





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
        print(f"[Persona] Loaded agent '{agent_cfg.name}' with encoding: {agent_cfg.persona_compact}")
    map_spec = load_map(map_path)

    route_choice_priors = None
    try:
        with open(priors_path, "r") as f:
            route_choice_priors = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load priors JSON at {priors_path}: {e}")

    env = MultiHumanGridEnv(
        map_spec=map_spec,
        agent_configs=agent_configs,
        max_steps=cfg.sim.steps,
        render_mode=cfg.sim.render_mode,
    )

    print("[Env] Resetting...")
    obs, info = env.reset()
    print("Initial state:")
    env.render()


    valid_locations = ['Home_A', 'Home_B', 'Park', 'Workplace_A', 'fire', 'park']


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
        "In emergencies, safety and survival are more important than routine preferences or habits. "
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

    

    # initializing variables for simulation loop
    last_external_events = None
    agent_commands = {aid: "stay" for aid in range(env.num_agents)}
    active_goal_cmd = {aid: None for aid in range(env.num_agents)} 
    social_hazard_memory = {i: set() for i in range(env.num_agents)}

    for t in range(cfg.sim.steps):
        clock_time = sim_time_str(cfg, t)

        external_events = get_external_events_for_t(t)
        stimulus_triggered = (external_events is not None and external_events != last_external_events)

        if stimulus_triggered:
            last_external_events = external_events

            # this will go to plannner eventially
            for agent_id, agent in enumerate(env.agents):
                plan_item = get_plan_for_time(agent.config, clock_time)
                if plan_item:
                    print(f"[PLAN] t={t} ({clock_time}) {agent.config.name} plan is", plan_item["activity"], "in", plan_item["location"])
                desc = describe_perception(env, agent_id)
                decision = llm_decide_intent(
                    conv=conv,
                    agent_cfg=agent.config,
                    plan_item=plan_item,
                    perception_desc=desc,
                    external_events=external_events,
                    clock_time=clock_time,
                    valid_locations=valid_locations,
                    current_location=plan_item["location"],
                )

                print(f"[LLM HIGH-LEVEL GOAL] t={t} ({clock_time}) {agent.config.name} intent {decision['intent']}, action {decision['action']}")

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

                # If LLM intent is stay or command includes stay → force stay
                if decision["intent"] == "stay" or "stay" in cmd:
                    agent_commands[agent_id] = "stay"
                else:
                    agent_commands[agent_id] = cmd



        full_paths = []
        for agent_id in range(env.num_agents):
            cmd = agent_commands.get(agent_id, "stay")

            if cmd.strip().lower() == "stay":
                full_paths.append([])  # no movement for this agent
                continue

            # this is so that we can give the LLM the valid directions at intersections
            if is_intersection(env, agent_id):
                # build valid dirs list for LLM
                valid_moves = valid_move_actions(env, agent_id)
                valid_dirs = action_names(valid_moves)

                desc = describe_perception(env, agent_id, include_decision_info=True)
                print("\n[LLM MID-LEVEL GOAL]")
                print(f"agent={env.agents[agent_id].config.name}, perception:{desc}, high_level_goal:{cmd}, valid_dirs:{valid_dirs}")
                local_reply = llm_decide_local_direction(
                    conv=conv,
                    agent_cfg=env.agents[agent_id].config,
                    perception_desc=desc,
                    high_level_goal=cmd,
                    valid_dirs=valid_dirs,
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
                    local_decision = local_reply
                    if local_reply == "stay":
                        full_paths.append([])  # no movement
                        continue
                    chosen = local_decision["direction"]
                    chosen_action = Action[chosen]  # "UP" -> Action.UP

                    # force first move by stepping to neighbor as new start
                    sx, sy = get_agent_tile(env, agent_id)
                    dx, dy = DIR_TO_VEC[chosen_action]
                    forced_start = (sx + dx, sy + dy)

                    # if forced cell not traversable, ignore and fall back to normal A*
                    if env._can_move_to(*forced_start):
                        start, goal = high_level_planner(env, agent_id, cmd)
                        # plan from forced_start to goal, and prepend chosen_action
                        tail = astar(env, forced_start, goal)
                        full_path = [chosen_action] + tail
                        full_paths.append(full_path)
                        continue
                except Exception as e:
                    print("[WARN] local decision parse error:", e)

            start, goal = high_level_planner(env, agent_id=agent_id, command=cmd)
            print(f"[LOW LEVEL PLANNER] t={t} Agent {agent_id} command='{cmd}' -> start={start}, goal={goal}")
            planner_logger.debug(
                f"t={t} agent={agent_id} command='{cmd}' "
                f"start={start} goal={goal}"
            )


            full_path = astar(env, start, goal)

            # ---------------------------------------------------------------
            # hazard avoidance
            # ---------------------------------------------------------------

            known_hazards = getattr(env.agents[agent_id], "known_hazard_cells", set())
            if full_path is not None and known_hazards:
                # if the planned path includes any known hazard cell, cancel it
                if any(cell in known_hazards for cell in full_path):
                    # print(
                    #     f"[HAZARD] Agent {agent_id} path to {cmd} intersects known hazards "
                    #     f"{known_hazards}. Cancelling path."
                    # )
                    
                    full_path = None  # invalidate the path

                    planner_logger.debug(
                        f"t={t} agent={agent_id} command='{cmd}' "
                        f"path_intersects_hazards={known_hazards}; cancelling_path"
                    )
            # ---------------------------------------------------------------


            if full_path is None:
                print(f"[WARN] No path for agent {agent_id} to '{cmd}'. Forcing replanning.")
                planner_logger.warning(
                    f"t={t} agent={agent_id} command='{cmd}' no_path_found; forcing_replan"
                )
                agent_commands[agent_id] = "stay"
                last_external_events = None # force replan next time
                full_paths.append([])
                continue

            full_paths.append(full_path)

        # Slice each agent path by their FOV
        paths = []
        for agent_id, agent in enumerate(env.agents):
            fov = agent.config.fov.range_cells
            paths.append(full_paths[agent_id][:fov])

        # If everyone staying, skip movement
        if all(len(p) == 0 for p in paths):
            print(f"t={t} all agents staying. Waiting for next stimulus.")
            continue


        ## This is A* planner
        max_substeps = max(len(p) for p in paths)
        for step in range(max_substeps):

            actions_this_step = []
            for agent_id in range(env.num_agents):
                a = paths[agent_id][step] if step < len(paths[agent_id]) else Action.STAY
                actions_this_step.append(a)


            action = np.array(actions_this_step, dtype=np.int64)
            obs, _, terminated, truncated, info = env.step(action)

            print(f"\nStep {t + 1}, Sub-step {step + 1}, action={action}")

            for agent_id in range(env.num_agents):
                desc = describe_perception(env, agent_id, include_decision_info=False)
                agent = env.agents[agent_id]

                #------------------------------------------------------------------
                # Planned social hazard sharing (inactive until hazard env integrated)
                # Social hazard sharing + memory integration
                #------------------------------------------------------------------

                hazards = get_local_hazards(env, agent_id)
                if hazards:
                    for other_id, other in enumerate(env.agents):
                        if other_id == agent_id:
                            continue
                        

                        friends = getattr(agent.config, "friends_with", [])
                        other_id_str = getattr(other.config, "id", None)
                        other_name = getattr(other.config, "name", None)

                        if (other_id_str in friends) or (other_name in friends):
                            for hz in hazards:
                    
                                social_hazard_memory[other_id].add(hz)
                                add_social_memory(other, hz)

                            print(
                                f"[SOCIAL] Agent {agent_id} shares {hazards} "
                                f"with {other.config.name}"
                            )

                heard = sorted(social_hazard_memory.get(agent_id, set()))

                if heard:
                    MAX_DIALOGUES_PER_STEP = 2
                    heard = heard[:MAX_DIALOGUES_PER_STEP]

                    dialogue_lines = []
                    for hz in heard:
                        spoken = hazard_to_dialogue(hz)

                        dialogue_lines.append(f'A neighbor says: "{spoken}"')

                    if dialogue_lines:
                        desc = desc + " " + " ".join(dialogue_lines)
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
                    substep=step + 1,
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
   

if __name__ == "__main__":
    run()

