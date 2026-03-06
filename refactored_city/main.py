"""
Refactored City – Pygame-based urban simulation with LLM-driven agents.

Uses Llama 3.2 (via Ollama) to decide agent actions each cycle.
Records frames to an mp4 video.

Usage:
    conda activate cs294
    python main.py --steps 100

Controls (in human mode):
    Arrow keys / mouse drag  – pan the map
    Scroll wheel / +/-       – zoom in/out
    Home                     – reset camera
    Q / Escape               – quit
"""
from __future__ import annotations

import argparse
import datetime
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
from openai import OpenAI

from env.grid import CityEnv, Action

PERSONAS: Dict[str, Dict[str, Any]] = {
    "Alex Chen": {
        "innate": "analytical, focused, friendly",
        "learned": "Alex Chen is a software developer at TechHub Office. He spends most of his day coding and reviewing pull requests.",
        "currently": "Alex is working on a deadline for a product launch next week.",
        "living_area": "the City:Block A:Sunrise Apartments",
        "work_area": "the City:Block B:TechHub Office",
        "daily_plan": "Wakes up at 7am, grabs coffee at Bean & Leaf Cafe, works at TechHub until 6pm, hits the gym, then goes home.",
        "lifestyle": "Goes to bed around 11pm, wakes up at 7am.",
    },
    "Maya Johnson": {
        "innate": "warm, creative, sociable",
        "learned": "Maya Johnson is a barista and shift manager at Bean & Leaf Cafe. She knows everyone in the neighborhood.",
        "currently": "Maya is training a new hire and experimenting with seasonal drink recipes.",
        "living_area": "the City:Block F:Downtown Lofts",
        "work_area": "the City:Block B:Bean & Leaf Cafe",
        "daily_plan": "Opens the cafe at 6:30am, works until 2pm, then paints at Art Studio or relaxes in the park.",
        "lifestyle": "Early riser, in bed by 10pm.",
    },
    "David Kim": {
        "innate": "patient, inspiring, thoughtful",
        "learned": "David Kim is an art teacher at Greenfield School. He also volunteers at the Community Center on weekends.",
        "currently": "David is organizing a student art exhibition at the Community Center.",
        "living_area": "the City:Block E:Riverside Condos",
        "work_area": "the City:Block C:Greenfield School",
        "daily_plan": "Teaches from 8am to 3pm, visits the art studio, then walks through Central Park before heading home.",
        "lifestyle": "Steady routine, enjoys evenings reading at the library.",
    },
    "Sarah Torres": {
        "innate": "energetic, motivating, disciplined",
        "learned": "Sarah Torres is a fitness trainer at Quick Gym. She also teaches yoga at FitLife Yoga on weekends.",
        "currently": "Sarah is preparing for a charity fitness marathon she's organizing.",
        "living_area": "the City:Block I:Harbor View Apts",
        "work_area": "the City:Block A:Quick Gym",
        "daily_plan": "Morning run at 5:30am, trains clients 7am-4pm, grabs lunch at Metro Diner, evening at home.",
        "lifestyle": "Very early riser, in bed by 9:30pm.",
    },
    "Marcus Williams": {
        "innate": "quiet, knowledgeable, helpful",
        "learned": "Marcus Williams is the head librarian at City Library. He curates the reading programs and hosts community events.",
        "currently": "Marcus is setting up a new digital lending program at the library.",
        "living_area": "the City:Block A:Sunrise Apartments",
        "work_area": "the City:Block E:City Library",
        "daily_plan": "Opens library at 9am, works until 5pm, stops by City Bookstore, then dinner at Noodle House.",
        "lifestyle": "Bookworm, stays up late reading, wakes at 8am.",
    },
}


def build_location_list(env: CityEnv) -> List[str]:
    seen: set[str] = set()
    locations: List[str] = []
    for addr in sorted(env.maze.address_tiles.keys()):
        parts = addr.split(":")
        if len(parts) >= 2:
            short = " > ".join(parts[1:])
            if short not in seen:
                seen.add(short)
                locations.append(addr)
    return locations


def format_time(step: int, seconds_per_step: int = 60) -> str:
    base = datetime.datetime(2025, 6, 16, 6, 0)
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
    loc_sample = random.sample(available_locations, min(15, len(available_locations)))
    loc_list = "\n".join(f"  - {a}" for a in loc_sample)
    nearby_str = ", ".join(nearby_people) if nearby_people else "nobody"

    prompt = f"""You are {agent_name} in an urban city.

Personality: {persona['innate']}
Background: {persona['learned']}
Current situation: {persona['currently']}
Daily routine: {persona['daily_plan']}

It is currently {sim_time}. You are at: {current_location}
People nearby: {nearby_str}

Some available locations in the city:
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
    env: CityEnv,
    answer: str,
    agent_idx: int,
    available_locations: List[str],
) -> bool:
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


def get_agent_location_str(env: CityEnv, agent_idx: int) -> str:
    agent = env.agents[agent_idx]
    tile = env.maze.access_tile((agent.x, agent.y))
    parts = [tile.get("sector", ""), tile.get("arena", ""), tile.get("game_object", "")]
    parts = [p for p in parts if p]
    return " > ".join(parts) if parts else "on the street"


def get_nearby_agents(env: CityEnv, agent_idx: int, radius: int = 5) -> List[str]:
    agent = env.agents[agent_idx]
    nearby = []
    for j, other in enumerate(env.agents):
        if j == agent_idx:
            continue
        if abs(other.x - agent.x) <= radius and abs(other.y - agent.y) <= radius:
            nearby.append(other.name)
    return nearby


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urban City Simulation (Pygame + LLM)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "city_map.json"),
    )
    parser.add_argument("--steps", type=int, default=100)
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
    parser.add_argument("--zoom", type=float, default=1.5)
    parser.add_argument(
        "--sprites",
        action="store_true",
        default=False,
        help="Use character sprite sheets instead of colored dots (requires sprite assets)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=False,
        help="Use RPG pixel-art-style rendering inspired by SmallVille",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=str(Path(__file__).parent / "assets"),
        help="Path to assets directory containing character sprites",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    agent_names = args.agents if args.agents else list(PERSONAS.keys())

    client = OpenAI(base_url=args.ollama_url, api_key="ollama")

    env = CityEnv(
        config_path=args.config,
        agent_names=agent_names,
        render_mode=args.render_mode,
        window_w=args.window_w,
        window_h=args.window_h,
        use_sprites=args.sprites,
        use_detailed=args.detailed,
        assets_dir=args.assets_dir,
    )

    obs, info = env.reset()
    env._zoom = args.zoom
    env._camera_x = -(env.renderer.pixel_width * args.zoom - args.window_w) / 2
    env._camera_y = -(env.renderer.pixel_height * args.zoom - args.window_h) / 2

    available_locations = build_location_list(env)
    print(f"City loaded: {env.maze.maze_width}x{env.maze.maze_height} tiles")
    print(f"Agents: {[a.name for a in env.agents]}")
    print(f"Cars: {len(env.cars)}")
    print(f"Available locations: {len(available_locations)}")
    print(f"LLM model: {args.llm_model} @ {args.ollama_url}")
    print(f"Recording to: {args.output} ({args.fps} fps)")
    print(f"Decision interval: every {args.decision_interval} steps")
    print()

    for i, agent in enumerate(env.agents):
        persona = PERSONAS.get(agent.name, {})
        if persona.get("living_area"):
            env.move_agent_to_address(i, persona["living_area"])
            agent.description = "at home"

    w_out = args.window_w if args.window_w % 2 == 0 else args.window_w + 1
    h_out = args.window_h if args.window_h % 2 == 0 else args.window_h + 1
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

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
                        "cafe": "☕", "library": "📚", "gym": "🏋️",
                        "apartment": "🏠", "office": "💼", "park": "🌳",
                        "school": "🏫", "diner": "🍔", "bar": "🍺",
                        "hospital": "🏥", "hotel": "🏨", "station": "🚉",
                        "mall": "🛍️", "cinema": "🎬", "yoga": "🧘",
                        "pharmacy": "💊", "pizza": "🍕", "noodle": "🍜",
                        "bookstore": "📖", "studio": "🎨",
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

            if step % 50 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  --- step {step}/{args.steps} "
                    f"({elapsed:.0f}s elapsed, "
                    f"{step / elapsed:.1f} steps/s) ---"
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
