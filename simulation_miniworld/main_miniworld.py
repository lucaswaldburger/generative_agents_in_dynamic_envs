"""MiniWorld 3D Building Evacuation Simulation with LLM Agents.

Mirrors the structure of ``grid_gen/main.py`` but runs agents inside
a MiniWorld-based 3D environment with first-person rendering.

Usage
-----
    cd simulation_miniworld
    PYGLET_HEADLESS=true python main_miniworld.py
    PYGLET_HEADLESS=true python main_miniworld.py sim.steps=50 world.num_rooms=6
"""
from __future__ import annotations

import sys
import os
import csv
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import hydra
import numpy as np
from omegaconf import DictConfig

os.environ.setdefault("PYGLET_HEADLESS", "true")

# ---------------------------------------------------------------------------
# Make grid_gen importable (sibling directory)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GRID_GEN = _PROJECT_ROOT / "grid_gen"
sys.path.insert(0, str(_GRID_GEN))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import (
    SimulationLLM,
    test_connection,
    decide_intent,
    decide_local_direction,
    decide_social,
    SIMULATION_SYSTEM_PROMPT,
)
from persona.load_personas import load_agent_configs
from persona.memory.associative_memory import add_social_memory, is_friend
from utils.external_events import get_external_events_for_t
from utils.persona_utils import encode_persona
from utils.parse_route_choice_priors import get_agent_route_priors

from multi_agent_wrapper import MultiAgentMiniWorldEnv
from perception_miniworld import (
    describe_perception,
    get_local_hazards,
    agents_in_fov,
    assess_urgency,
    format_urgency_for_llm,
)
from planning_miniworld import (
    ControlStateMW,
    PlannerLevel,
    plan_waypoints,
    compute_action_to_waypoint,
    at_waypoint,
    normalize_command,
    resolve_goal_room,
    get_valid_directions,
)


# ======================================================================
# Logging helpers
# ======================================================================

def setup_sim_output_dir() -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = os.path.join("sim_outputs", f"run_mw_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def create_agent_logs(run_dir, env):
    logs = {}
    for aid, a in enumerate(env.agents):
        name = a.name
        folder = os.path.join(run_dir, name)
        os.makedirs(folder, exist_ok=True)
        fh = open(os.path.join(folder, "trajectory.txt"), "w")
        logs[aid] = fh
    return logs


def log_agent_step(log_file, aid, step, agent, action, desc):
    log_file.write(
        f"step={step} pos=({agent.pos[0]:.1f},{agent.pos[2]:.1f}) "
        f"dir={agent.direction:.2f} action={action} | {desc}\n"
    )
    log_file.flush()


def setup_debug_loggers(run_dir):
    def _make(name, fname):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(run_dir, fname))
        fh.setLevel(logging.DEBUG)
        lg.addHandler(fh)
        return lg
    return (
        _make("intent_mw", "high_level_intent.log"),
        _make("local_mw", "local_decisions.log"),
        _make("planner_mw", "planner.log"),
    )


def sim_time_str(cfg, t):
    h, m = map(int, cfg.sim.start_time.split(":"))
    total_sec = h * 3600 + m * 60 + t * cfg.sim.seconds_per_step
    hh = (total_sec // 3600) % 24
    mm = (total_sec % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def _save_video(frames, path, fps=5):
    if not frames:
        return
    import imageio.v3 as iio
    arr = np.stack(frames, axis=0).astype(np.uint8)
    iio.imwrite(path, arr, fps=fps, codec="libx264")
    print(f"[Output] Video saved to {path}")


# ======================================================================
# Main simulation loop
# ======================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config_miniworld")
def run(cfg: DictConfig):
    personas_path = str(_GRID_GEN / cfg.personas.file)
    priors_path = str(_GRID_GEN / cfg.data.route_choice_priors_file)

    agent_configs = load_agent_configs(personas_path, cfg.personas.personas_in_sim)
    for ac in agent_configs:
        ac.persona_compact = encode_persona(ac)

    route_choice_priors = None
    try:
        with open(priors_path) as f:
            route_choice_priors = json.load(f).get("route_choice_priors", {})
    except Exception as e:
        print(f"[WARN] Could not load priors: {e}")

    save_video = getattr(cfg.sim, "save_video", False)
    render_mode = "rgb_array" if save_video else cfg.sim.render_mode
    video_frames = []
    fpv_frames = {i: [] for i in range(len(agent_configs))}

    valid_locations = list(
        {"Lobby", "Office_A", "Office_B", "Office_C", "Office_D",
         "Hallway_Main", "Exit"}
    )

    env = MultiAgentMiniWorldEnv(
        agent_configs=agent_configs,
        num_rooms=cfg.world.num_rooms,
        building_scale=cfg.world.building_scale,
        fire_spread_rate=cfg.world.fire_spread_rate,
        max_steps=cfg.sim.steps,
        render_mode=render_mode,
    )

    print("[Env MiniWorld] Resetting...")
    obs, info = env.reset()
    frame = env.render()
    if save_video and frame is not None:
        video_frames.append(frame)
        for aid in range(env.num_agents):
            fpv = env.render_agent_view(aid)
            fpv_frames[aid].append(fpv)

    try:
        reply = test_connection(cfg)
        print(f'[LLM] Connection OK: "{reply}"')
    except Exception as e:
        print(f"[LLM] Connection error: {e}")

    sim_llm = SimulationLLM(cfg, system_prompt=SIMULATION_SYSTEM_PROMPT)

    run_dir = setup_sim_output_dir()
    agent_logs = create_agent_logs(run_dir, env)
    intent_logger, local_logger, planner_logger = setup_debug_loggers(run_dir)

    urgency_data = []
    last_external_events = None
    agent_commands = {aid: "stay" for aid in range(env.num_agents)}
    control_states = {aid: ControlStateMW() for aid in range(env.num_agents)}

    fire_replan_count = {aid: 0 for aid in range(env.num_agents)}
    smoke_replan_count = {aid: 0 for aid in range(env.num_agents)}

    for aid, a in enumerate(env.agents):
        a.local_fire_smoke_seen = False
        a.last_urgency_level = "low"

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    for t in range(cfg.sim.steps):
        clock_time = sim_time_str(cfg, t)
        external_events = get_external_events_for_t(t)
        stimulus_triggered = (
            external_events is not None and external_events != last_external_events
        )

        # 0) Urgency assessment
        for aid in range(env.num_agents):
            agent = env.agents[aid]
            if agent.reached_exit or agent.dead:
                continue
            ua = assess_urgency(
                env, aid, agent,
                fire_replan_count.get(aid, 0),
                smoke_replan_count.get(aid, 0),
            )
            agent.current_urgency_assessment = ua
            print(f"[URGENCY] t={t} {agent.name}: {ua.urgency_level.upper()} ({ua.urgency_score:.2f})")
            urgency_data.append({
                "time_step": t,
                "clock_time": clock_time,
                "agent_id": aid,
                "agent_name": agent.name,
                "urgency_level": ua.urgency_level,
                "urgency_score": ua.urgency_score,
                "nearest_fire": ua.nearest_fire_dist,
                "nearest_smoke": ua.nearest_smoke_dist,
            })

        # 1) High-level intent on external stimulus
        if stimulus_triggered:
            last_external_events = external_events
            for aid in range(env.num_agents):
                agent = env.agents[aid]
                if agent.reached_exit or agent.dead:
                    continue

                desc = describe_perception(env, aid)
                ua = agent.current_urgency_assessment
                urgency_text = format_urgency_for_llm(ua) if ua else ""

                decision = decide_intent(
                    sim_llm=sim_llm,
                    agent_cfg=agent.config,
                    plan_item=None,
                    perception_desc=desc,
                    external_events=external_events,
                    clock_time=clock_time,
                    valid_locations=valid_locations,
                    current_location=env.get_room_for_pos(agent.pos),
                    t=t,
                    urgency_assessment=urgency_text,
                )
                print(f"[HIGH-LEVEL] t={t} {agent.name} intent={decision['intent']} action={decision['action']}")

                intent_logger.debug(
                    f"t={t} agent={agent.name} decision={decision} "
                    f"perception={desc} events={external_events}"
                )

                cmd = normalize_command(decision, agent.config, env)
                agent_commands[aid] = cmd
                cs = control_states[aid]
                cs.high_intent = decision.get("intent", "")
                cs.high_command = cmd
                cs.waypoints = []
                cs.current_waypoint_idx = 0
                cs.next_level = PlannerLevel.HIGH

        # 2) Local fire/smoke perception and reaction
        for aid in range(env.num_agents):
            agent = env.agents[aid]
            if agent.reached_exit or agent.dead:
                continue
            hazards = get_local_hazards(env, aid)
            cs = control_states[aid]

            should_react = False
            if hazards["has_local_fire"] and not agent.local_fire_smoke_seen:
                should_react = True
            elif hazards["has_local_fire"] and agent.current_urgency_assessment:
                ua = agent.current_urgency_assessment
                if ua.urgency_level in ("medium", "high", "critical"):
                    current_cmd = agent_commands.get(aid, "stay")
                    if current_cmd == "stay":
                        should_react = True

            if should_react:
                agent.local_fire_smoke_seen = True
                fire_replan_count[aid] += 1

                desc = describe_perception(env, aid)
                ua = agent.current_urgency_assessment
                urgency_text = format_urgency_for_llm(ua) if ua else ""

                decision = decide_intent(
                    sim_llm=sim_llm,
                    agent_cfg=agent.config,
                    plan_item=None,
                    perception_desc=desc,
                    external_events="You see fire/smoke nearby!",
                    clock_time=clock_time,
                    valid_locations=valid_locations,
                    current_location=env.get_room_for_pos(agent.pos),
                    t=t,
                    urgency_assessment=urgency_text,
                )
                cmd = normalize_command(decision, agent.config, env)
                agent_commands[aid] = cmd
                cs.high_intent = "evacuate_local_fire"
                cs.high_command = cmd
                cs.waypoints = []
                cs.current_waypoint_idx = 0
                print(f"[LOCAL FIRE REPLAN] t={t} {agent.name} -> '{cmd}'")

        # 3) Path planning
        for aid in range(env.num_agents):
            agent = env.agents[aid]
            if agent.reached_exit or agent.dead:
                continue
            cs = control_states[aid]
            cmd = agent_commands.get(aid, "stay")

            if cmd == "stay":
                cs.waypoints = []
                cs.current_waypoint_idx = 0
                continue

            if cs.waypoints and cs.current_waypoint_idx < len(cs.waypoints):
                wp = cs.waypoints[cs.current_waypoint_idx]
                if at_waypoint(agent.pos, wp):
                    cs.current_waypoint_idx += 1

            if not cs.waypoints or cs.current_waypoint_idx >= len(cs.waypoints):
                goal_room = resolve_goal_room(cmd, env)
                if goal_room:
                    cs.waypoints = plan_waypoints(env, aid, goal_room)
                    cs.current_waypoint_idx = 0
                    planner_logger.debug(
                        f"t={t} agent={agent.name} cmd='{cmd}' "
                        f"goal_room={goal_room} waypoints={len(cs.waypoints)}"
                    )

        # 4) Execute actions
        action_arr = np.zeros(env.num_agents, dtype=np.int64)
        for aid in range(env.num_agents):
            agent = env.agents[aid]
            if agent.reached_exit or agent.dead:
                continue
            cs = control_states[aid]
            cmd = agent_commands.get(aid, "stay")

            if cmd == "stay":
                action_arr[aid] = 2  # move_forward (effectively no-op when staying)
                continue

            if cs.waypoints and cs.current_waypoint_idx < len(cs.waypoints):
                wp = cs.waypoints[cs.current_waypoint_idx]
                action_arr[aid] = compute_action_to_waypoint(
                    agent.pos, agent.direction, wp
                )
            else:
                exit_pos = env.get_exit_pos()
                action_arr[aid] = compute_action_to_waypoint(
                    agent.pos, agent.direction, (exit_pos[0], exit_pos[2])
                )

        obs, reward, terminated, truncated, step_info = env.step(action_arr)
        fire_n = step_info.get("fire_count", 0)
        smoke_n = step_info.get("smoke_count", 0)
        alive_n = step_info.get("agents_alive", 0)
        exited_n = step_info.get("agents_exited", 0)
        print(f"\nStep {t + 1}  [fire={fire_n} smoke={smoke_n} alive={alive_n} exited={exited_n}]")

        # 5) Social interactions and logging
        for aid in range(env.num_agents):
            agent = env.agents[aid]
            if agent.reached_exit or agent.dead:
                continue
            desc = describe_perception(env, aid)

            nearby = agents_in_fov(env, aid)
            for oid, oname, odist in nearby:
                if oid in agent.social_ignored_friends:
                    continue
                other = env.agents[oid]
                if is_friend(agent.config, other.config):
                    hazards_info = get_local_hazards(env, aid)
                    hazards_str = []
                    if hazards_info["has_local_fire"]:
                        hazards_str.append("fire nearby")
                    if hazards_info["has_local_smoke"]:
                        hazards_str.append("smoke nearby")

                    social_dec = decide_social(
                        sim_llm=sim_llm,
                        ego_cfg=agent.config,
                        friend_cfg=other.config,
                        perception_desc=desc,
                        hazards=hazards_str,
                        current_command=agent_commands.get(aid, "stay"),
                        clock_time=clock_time,
                        t=t,
                    )
                    talk = bool(social_dec.get("talk", False))
                    if talk:
                        new_cmd = social_dec.get("new_command")
                        if new_cmd and isinstance(new_cmd, str):
                            agent_commands[aid] = new_cmd.strip()
                            cs = control_states[aid]
                            cs.waypoints = []
                            cs.current_waypoint_idx = 0
                        add_social_memory(agent, info=f"Talked with {other.config.name}")
                    agent.social_ignored_friends.add(oid)

            log_agent_step(
                agent_logs[aid], aid, t + 1, agent,
                int(action_arr[aid]), desc,
            )

        frame = env.render()
        if save_video and frame is not None:
            video_frames.append(frame)
            for aid in range(env.num_agents):
                agent = env.agents[aid]
                if not agent.reached_exit and not agent.dead:
                    fpv = env.render_agent_view(aid)
                else:
                    fpv = np.zeros_like(fpv_frames[aid][0]) if fpv_frames[aid] else np.zeros((480, 640, 3), dtype=np.uint8)
                fpv_frames[aid].append(fpv)
        if not save_video:
            time.sleep(0.3)

        if terminated or truncated:
            break

    # ------------------------------------------------------------------
    # Clean-up and outputs
    # ------------------------------------------------------------------
    for fh in agent_logs.values():
        fh.close()
    env.close()

    if save_video and video_frames:
        from hydra.core.hydra_config import HydraConfig
        hydra_out = HydraConfig.get().runtime.output_dir
        video_fps = getattr(cfg.sim, "video_fps", 5)
        _save_video(video_frames, os.path.join(hydra_out, "simulation_mw.mp4"),
                     fps=video_fps)
        _save_video(video_frames, os.path.join(run_dir, "simulation_mw.mp4"),
                     fps=video_fps)

        for aid in range(env.num_agents):
            if fpv_frames[aid]:
                agent_name = env.agents[aid].name.replace(" ", "_")
                fpv_name = f"fpv_{agent_name}.mp4"
                _save_video(fpv_frames[aid], os.path.join(hydra_out, fpv_name), fps=video_fps)
                _save_video(fpv_frames[aid], os.path.join(run_dir, fpv_name), fps=video_fps)

    if urgency_data:
        urgency_path = os.path.join(run_dir, "agent_urgency.csv")
        with open(urgency_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(urgency_data[0].keys()))
            writer.writeheader()
            writer.writerows(urgency_data)
        print(f"[Output] Urgency CSV -> {urgency_path}")

    step_tokens = defaultdict(int)
    for rec in sim_llm.call_log:
        if rec.t is not None:
            step_tokens[rec.t] += rec.total_tokens

    token_path = os.path.join(run_dir, "token_usage.txt")
    with open(token_path, "w") as f:
        f.write("=== TOKEN USAGE PER STEP ===\n")
        for st in sorted(step_tokens):
            f.write(f"t={st}: {step_tokens[st]} tokens\n")
        f.write(f"\nTotal prompt tokens: {sim_llm.total_prompt_tokens}\n")
        f.write(f"Total completion tokens: {sim_llm.total_completion_tokens}\n")
        f.write(f"Total tokens: {sim_llm.total_prompt_tokens + sim_llm.total_completion_tokens}\n")
        f.write(f"Total LLM calls: {sim_llm.total_calls}\n")

    print(f"\n[Output] Token usage -> {token_path}")
    print(f"Total LLM calls: {sim_llm.total_calls}")
    print(f"Total tokens: {sim_llm.total_prompt_tokens + sim_llm.total_completion_tokens}")


if __name__ == "__main__":
    run()
