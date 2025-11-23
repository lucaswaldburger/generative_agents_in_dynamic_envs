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

import numpy as np
import yaml

from env.load_map import load_map
from persona.load_personas import load_agent_configs
from env.grid import MultiHumanGridEnv
from env.constants import Action, DIR_TO_VEC 

# this might be temporary map until we move this high-level to the other cognitive models
from persona.cognitive.plan import get_accessible_locations, high_level_planner, astar, get_plan_for_time, normalize_command_for_planner, get_agent_tile
from env.constants import SemanticMap
from persona.cognitive.perceive import describe_perception, is_intersection, valid_move_actions, action_names
from persona.prompt.gpt_structure import test_chat_completion, LLMConversation, llm_decide_intent, llm_decide_local_direction

import os
import datetime




def sim_time_str(cfg, t: int) -> str:
    """Map sim timestep to clock time using cfg.sim.start_time and seconds_per_step."""
    start = datetime.datetime.strptime(cfg.sim.start_time, "%H:%M")
    curr = start + datetime.timedelta(seconds=t * cfg.sim.seconds_per_step)
    return curr.strftime("%H:%M")


# we can move these logger functions to other folder later
def setup_sim_output_dir():
    """
    Creates a run folder in sim_outputs/ with timestamp.
    Returns the folder path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "sim_outputs"
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def create_agent_logs(run_dir, env):
    """
    Creates one text file per agent using their name.
    Returns a dict: agent_id -> file_handle
    """
    logs = {}
    for agent_id, agent in enumerate(env.agents):
        safe_name = agent.config.name.replace(" ", "_")
        path = os.path.join(run_dir, f"{safe_name}.txt")
        logs[agent_id] = open(path, "w")
        logs[agent_id].write(f"Log for {agent.config.name}\n")
        logs[agent_id].write("=" * 40 + "\n\n")
    return logs

def log_agent_step(log_file, agent_id, step, substep, agent, action, desc):
    """

    """
    log_file.write(f"Step {step}, Sub-step {substep}\n")
    log_file.write(f" Position: ({agent.x}, {agent.y})\n")
    log_file.write(f" Action: {action}\n")
    log_file.write(f" Observation: {desc}\n")
    log_file.write("-" * 30 + "\n")


def get_external_events_for_t(t):
    if t == 0:
        return "humans receive an alert text: there is a fire, but no need to evacuate yet"
    if t == 10:
        return "the fire alarm sounds loudly, evacuation is now required. Isabella sees smoke outside of the building."
    return None






@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """
    Loads map and runs the MiniGrid environment defined in JSON.
    """
    map_path = to_absolute_path(cfg.map.file)
    personas_path = to_absolute_path(cfg.personas.file)
    priors_path = to_absolute_path(cfg.memory.route_choice_priors_file)

    agent_configs = load_agent_configs(personas_path, cfg.personas.personas_in_sim)
    map_spec = load_map(map_path)
    # print("Unique access codes in grid:", np.unique(map_spec.access_grid))

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
    # for r in env.map_spec.regions:
    #     valid_locations.append(r["name"])
    #     if "type" in r:
    #         valid_locations.append(r["type"])
    # valid_locations = sorted(set([str(x) for x in valid_locations]))
    print("[VALID LOCATIONS]", valid_locations)

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
    conv = LLMConversation(
        cfg,
        system_prompt=(
            "You are the cognitive model for humans in a fire evacuation simulation. "
            "You must combine empirical route-choice priors with each agent's persona "
            "and demographics to decide how they move."
        ),
    )

    persona_summary = []
    for a in agent_configs:
        persona_summary.append({
            "id": a.id,
            "name": a.name,
            "age": getattr(a, "age", None),
            "gender": getattr(a, "gender", None),
            "innate": getattr(a, "innate", None),
            "learned": getattr(a, "learned", None),
            "lifestyle": getattr(a, "lifestyle", None),
            "living_area": getattr(a, "living_area", None),
            "friends_with": getattr(a, "friends_with", []),
            "dependents": getattr(a, "dependents", []),
            "daily_plan": getattr(a, "daily_plan", []),
            "fov_range": a.fov.range_cells,
        })

    # Give priors + personas to LLM ONCE at the start
    # move this to the llm scripts later
    init_reply = conv.ask_llm(
        f"""
        Here are demographic-based route choice priors (JSON):

        {json.dumps(route_choice_priors['route_choice_priors'], indent=2)}

        Here are the agent personas (JSON):

        {json.dumps(persona_summary, indent=2)}

        You will later be asked to make LOCAL route-choice decisions at intersections 
        (e.g., left / right / forward / back) while pursuing a high-level goal. 
        Use the demographic priors together with each agent’s innate and learned traits 
        (risk-proneness vs. risk-aversion, exploration vs. familiarity preference, etc.) 
        to bias these local choices, especially when hazards (smoke, fire) or congestion are present.

        When asked for local choices, you MUST select only from the provided list of "Valid directions."

        These demographic priors are soft behavioral rules:
        - Age buckets: "<25", "25-34", "35-49", "50+" (choose the closest bucket when uncertain).
        - Missing values (null) mean you should rely on general reasoning.

        We will use this information throughout the simulation to determine how each agent moves.
        Briefly summarize how the agents’ demographics align with these priors:

        """
            )
    print("[LLM INIT SUMMARY]\n", init_reply)

    run_dir = setup_sim_output_dir()
    agent_logs = create_agent_logs(run_dir, env)

    last_external_events = None
    agent_commands = {aid: "stay" for aid in range(env.num_agents)}
    active_goal_cmd = {aid: None for aid in range(env.num_agents)}  # e.g. "go to Home_A"


    for t in range(cfg.sim.steps):
        clock_time = sim_time_str(cfg, t)

        external_events = get_external_events_for_t(t)
        stimulus_triggered = (external_events is not None and external_events != last_external_events)

        if stimulus_triggered:
            last_external_events = external_events
            # valid_locations_accessible = {}
            # for agent_id in range(env.num_agents):
            #     valid_locations_accessible[agent_id] = get_accessible_locations(env, agent_id)
            #     print(f"[ACCESSIBLE LOCS] agent {agent_id}: {valid_locations_accessible[agent_id]}")



            # this will go to plannner eventially
            # FOR NOW WE have two agents only so this hardcoding works but eventually change to env.agent_ids
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
                )

                print(f"[LLM INTENT] t={t} ({clock_time}) {agent.config.name}:")
                print("  intent =", decision["intent"])
                print("  command =", decision["command"])
                print("  reason =", decision["reason"])

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
                print("\n[LLM LOCAL PROMPT]")
                print(f"agent={env.agents[agent_id].config.name}")
                print("perception:", desc)
                print("high_level_goal:", cmd)
                print("valid_dirs:", valid_dirs)
                local_reply = llm_decide_local_direction(
                    conv=conv,
                    agent_cfg=env.agents[agent_id].config,
                    perception_desc=desc,
                    high_level_goal=cmd,
                    valid_dirs=valid_dirs,
                )

                print("[LLM REPLY LOCAL DIR]")
                print(local_reply)

                try:
                    local_decision = json.loads(local_reply)
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
            print(f"[PLANNER] t={t} Agent {agent_id} command='{cmd}' -> start={start}, goal={goal}")
            full_path = astar(env, start, goal)

            if full_path is None:
                print(f"[WARN] No path for agent {agent_id} to '{cmd}'. Forcing replanning.")
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

                log_agent_step(
                    log_file=agent_logs[agent_id],
                    agent_id=agent_id,
                    step=t + 1,
                    substep=step + 1,
                    agent=agent,
                    action=int(action[agent_id]),
                    desc=desc,
                )

                sm = agent.config.spatial_memory
                if sm:
                    x, y = int(agent.x), int(agent.y)
                    sm_info = sm.elements_at_position(env, x, y)
                    contents = sm_info["contents"]

                    if contents:
                        rooms_here = list(contents.keys())
                        print(
                            f"[SM] Agent {agent_id} at ({x}, {y}) -> "
                            f"cell='{sm_info['cell_name']}', lookup_key='{sm_info.get('lookup_key')}', rooms={rooms_here}"
                        )
                    else:
                        print(
                            f"[SM] Agent {agent_id} at ({x}, {y}) -> "
                            f"cell='{sm_info['cell_name']}'"
                        )

                print(f"[Agent {agent_id}] {desc}")

            env.render()
            time.sleep(0.5)

            if terminated or truncated:
                break
            
    for f in agent_logs.values():
        f.close()
    env.close()
   

if __name__ == "__main__":
    run()

