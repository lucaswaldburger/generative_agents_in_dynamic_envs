"""
Refactored SmallVille – Pygame-based simulation with LLM-driven agents.

Uses Llama 3.2 (via Ollama) to decide agent actions each cycle.
Records frames to an mp4 video.

Usage:
    conda activate cs294
    python main.py --steps 1000

Controls (in human mode):
    Arrow keys / mouse drag  – pan the map
    Scroll wheel / +/-       – zoom in/out
    Home                     – reset camera
    Q / Escape               – quit
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
from openai import OpenAI

from env.grid import SmallVilleEnv, Action
from env.path_finder import closest_coordinate

PERSONAS: Dict[str, Dict[str, Any]] = {
    "Isabella Rodriguez": {
        "innate": "friendly, outgoing, hospitable",
        "learned": "Isabella Rodriguez is a cafe owner of Hobbs Cafe who loves to make people feel welcome.",
        "currently": "Isabella is planning a Valentine's Day party at Hobbs Cafe.",
        "living_area": "the Ville:Isabella Rodriguez's apartment:main room",
        "daily_plan": "Opens Hobbs Cafe at 8am, works the counter until 8pm, then closes.",
        "lifestyle": "Goes to bed around 11pm, wakes up around 6am.",
    },
    "Klaus Mueller": {
        "innate": "kind, inquisitive, passionate",
        "learned": "Klaus Mueller is a student at Oak Hill College studying sociology, passionate about social justice.",
        "currently": "Klaus is writing a research paper on gentrification in low-income communities.",
        "living_area": "the Ville:Dorm for Oak Hill College:Klaus Mueller's room",
        "daily_plan": "Goes to the library early, writes all day, eats at Hobbs Cafe.",
        "lifestyle": "Student schedule, up early, studies late.",
    },
    "Maria Lopez": {
        "innate": "energetic, enthusiastic, inquisitive",
        "learned": "Maria Lopez is a physics student at Oak Hill College and part-time Twitch streamer.",
        "currently": "Maria is working on her physics degree and streaming games on Twitch.",
        "living_area": "the Ville:Dorm for Oak Hill College:Maria Lopez's room",
        "daily_plan": "Spends at least 6 hours a day streaming or gaming. Visits Hobbs Cafe daily.",
        "lifestyle": "College student, streams in the evening.",
    },
    "Abigail Chen": {
        "innate": "open-minded, curious, determined",
        "learned": "Abigail Chen is a digital artist and animator exploring art and technology.",
        "currently": "Abigail is working on an animation project and experimenting with interactive art.",
        "living_area": "the Ville:artist's co-living space:Abigail Chen's room",
        "daily_plan": "Works on art projects, visits the cafe and park for inspiration.",
        "lifestyle": "Creative schedule, works in bursts.",
    },
    "John Lin": {
        "innate": "patient, kind, organized",
        "learned": "John Lin runs the pharmacy at Willow Market and Pharmacy. Lives with wife Mei and son Eddy.",
        "currently": "John is shop keeping and asking around about the upcoming mayor election.",
        "living_area": "the Ville:Lin family's house:Mei and John Lin's bedroom",
        "daily_plan": "Opens pharmacy at 9am, works counter until 5pm, goes home.",
        "lifestyle": "Family man, steady routine.",
    },
}


def build_location_list(env: SmallVilleEnv) -> List[str]:
    """Get a deduplicated list of human-readable locations from the maze."""
    seen = set()
    locations = []
    for addr in sorted(env.maze.address_tiles.keys()):
        if addr.startswith("<spawn_loc>"):
            continue
        parts = addr.split(":")
        if len(parts) >= 2:
            short = " > ".join(parts[1:])
            if short not in seen:
                seen.add(short)
                locations.append(addr)
    return locations


def format_time(step: int, seconds_per_step: int = 60) -> str:
    base = datetime.datetime(2023, 2, 13, 6, 0)
    dt = base + datetime.timedelta(seconds=step * seconds_per_step)
    return dt.strftime("%I:%M %p")


def ask_llm_for_action(
    client: OpenAI,
    agent_name: str,
    persona: Dict[str, Any],
    current_location: str,
    nearby_people: List[str],
    available_locations: List[str],
    sim_time: str,
    model: str = "llama3.2",
) -> Optional[str]:
    """Ask the LLM where the agent should go next. Returns an address string or None."""

    loc_sample = random.sample(available_locations, min(15, len(available_locations)))
    loc_list = "\n".join(f"  - {a}" for a in loc_sample)

    nearby_str = ", ".join(nearby_people) if nearby_people else "nobody"

    prompt = f"""You are {agent_name} in the town of SmallVille.

Personality: {persona['innate']}
Background: {persona['learned']}
Current situation: {persona['currently']}
Daily routine: {persona['daily_plan']}

It is currently {sim_time}. You are at: {current_location}
People nearby: {nearby_str}

Some available locations in town:
{loc_list}

Based on your personality, routine, and the current time, where should you go next?
Reply with ONLY the location name from the list above (copy it exactly), or "STAY" if you want to remain.
Do not explain, just output the location."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.7,
        )
        answer = resp.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"  LLM error for {agent_name}: {e}")
        return None


def resolve_llm_answer(
    env: SmallVilleEnv,
    answer: str,
    agent_idx: int,
    available_locations: List[str],
) -> bool:
    """Try to match the LLM answer to a known address and move the agent. Returns True if moved."""
    if not answer or answer.upper() == "STAY":
        return False

    answer_clean = answer.strip().strip('"').strip("'").strip()

    if answer_clean in env.maze.address_tiles:
        env.move_agent_to_address(agent_idx, answer_clean)
        return True

    for addr in available_locations:
        parts = addr.split(":")
        short = " > ".join(parts[1:]) if len(parts) >= 2 else addr
        if answer_clean.lower() in addr.lower() or answer_clean.lower() in short.lower():
            env.move_agent_to_address(agent_idx, addr)
            return True

    return False


def get_agent_location_str(env: SmallVilleEnv, agent_idx: int) -> str:
    agent = env.agents[agent_idx]
    tile = env.maze.access_tile((agent.x, agent.y))
    parts = [tile.get("sector", ""), tile.get("arena", ""), tile.get("game_object", "")]
    parts = [p for p in parts if p]
    return " > ".join(parts) if parts else "outdoors"


def get_nearby_agents(env: SmallVilleEnv, agent_idx: int, radius: int = 5) -> List[str]:
    agent = env.agents[agent_idx]
    nearby = []
    for j, other in enumerate(env.agents):
        if j == agent_idx:
            continue
        if abs(other.x - agent.x) <= radius and abs(other.y - agent.y) <= radius:
            nearby.append(other.name)
    return nearby


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmallVille (Pygame + LLM)")
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=str(Path(__file__).parent / "assets"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--agents", nargs="*", default=None)
    parser.add_argument("--window-w", type=int, default=1280)
    parser.add_argument("--window-h", type=int, default=800)
    parser.add_argument("--output", type=str, default="simulation.mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--llm-model", type=str, default="llama3.2")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=30,
        help="How many steps between LLM decisions per agent",
    )
    parser.add_argument("--zoom", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()
    agent_names = args.agents if args.agents else list(PERSONAS.keys())

    client = OpenAI(base_url=args.ollama_url, api_key="ollama")

    env = SmallVilleEnv(
        assets_dir=args.assets_dir,
        agent_names=agent_names,
        render_mode=args.render_mode,
        window_w=args.window_w,
        window_h=args.window_h,
    )

    obs, info = env.reset()
    env._zoom = args.zoom
    env._camera_x = -(env.renderer.pixel_width * args.zoom - args.window_w) / 2
    env._camera_y = -(env.renderer.pixel_height * args.zoom - args.window_h) / 2

    available_locations = build_location_list(env)
    print(f"SmallVille loaded: {env.maze.maze_width}x{env.maze.maze_height} tiles")
    print(f"Agents: {[a.name for a in env.agents]}")
    print(f"Available locations: {len(available_locations)}")
    print(f"LLM model: {args.llm_model} @ {args.ollama_url}")
    print(f"Recording to: {args.output} ({args.fps} fps)")
    print(f"Decision interval: every {args.decision_interval} steps")
    print()

    w_out = args.window_w if args.window_w % 2 == 0 else args.window_w + 1
    h_out = args.window_h if args.window_h % 2 == 0 else args.window_h + 1
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    for i, agent in enumerate(env.agents):
        persona = PERSONAS.get(agent.name, {})
        if persona.get("living_area"):
            env.move_agent_to_address(i, persona["living_area"])
            agent.description = "at home"

    step = 0
    t_start = time.time()
    try:
        while step < args.steps:
            if args.render_mode == "human":
                if not env.handle_pygame_events():
                    break
                import pygame
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                    break

            for i, agent in enumerate(env.agents):
                if not agent.path and step % args.decision_interval == 0:
                    sim_time = format_time(step)
                    loc = get_agent_location_str(env, i)
                    nearby = get_nearby_agents(env, i)
                    persona = PERSONAS.get(agent.name, {})

                    print(
                        f"[step {step:4d} {sim_time}] {agent.name} at '{loc}' "
                        f"(nearby: {nearby or 'none'}) — asking LLM..."
                    )

                    answer = ask_llm_for_action(
                        client,
                        agent.name,
                        persona,
                        loc,
                        nearby,
                        available_locations,
                        sim_time,
                        model=args.llm_model,
                    )

                    if answer:
                        matched = resolve_llm_answer(env, answer, i, available_locations)
                        desc = answer[:60] if answer else "?"
                        if matched:
                            agent.description = f"→ {desc}"
                            print(f"           → heading to: {desc}")
                        else:
                            agent.description = f"staying ({desc})"
                            print(f"           → could not match: {desc}")

                    emoji_map = {
                        "cafe": "☕", "library": "📚", "pharmacy": "💊",
                        "apartment": "🏠", "room": "🛏️", "park": "🌳",
                        "store": "🏪", "dorm": "🏫", "pub": "🍺",
                        "kitchen": "🍳", "bathroom": "🚿", "garden": "🌻",
                    }
                    for key, emoji in emoji_map.items():
                        if key in loc.lower():
                            agent.pronunciatio = emoji
                            break
                    else:
                        agent.pronunciatio = "🚶"

            actions = [Action.STAY] * len(env.agents)
            obs, reward, terminated, truncated, info = env.step(actions)

            frame = env.render()
            if frame is not None:
                if frame.shape[0] != h_out or frame.shape[1] != w_out:
                    frame = np.pad(
                        frame,
                        (
                            (0, max(0, h_out - frame.shape[0])),
                            (0, max(0, w_out - frame.shape[1])),
                            (0, 0),
                        ),
                    )[:h_out, :w_out]
                writer.append_data(frame)

            step += 1

            if step % 100 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  --- step {step}/{args.steps} "
                    f"({elapsed:.0f}s elapsed, "
                    f"{step/elapsed:.1f} steps/s) ---"
                )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        writer.close()
        env.close()
        elapsed = time.time() - t_start
        print(f"\nDone: {step} steps in {elapsed:.1f}s")
        print(f"Video saved to: {args.output}")


if __name__ == "__main__":
    main()
