"""
Refactored Piedmont – Pygame-based simulation with LLM-driven agents
walking through an OSM-derived map of Piedmont, CA.

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
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
from openai import OpenAI

from env.grid import PiedmontEnv, Action

# ---------------------------------------------------------------------------
# Persona generation
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Isabella", "Klaus", "Maria", "Abigail", "John",
    "Sofia", "Liam", "Mei", "Carlos", "Priya",
    "Omar", "Yuki", "Elena", "Andre", "Fatima",
    "David", "Naomi", "Raj", "Chloe", "Marcus",
    "Lucia", "James", "Aisha", "Wei", "Olivia",
    "Noah", "Zara", "Ethan", "Leila", "Samuel",
    "Maya", "Daniel", "Sana", "Leo", "Iris",
    "Felix", "Nadia", "Hugo", "Amara", "Oscar",
    "Jasmine", "Victor", "Rosa", "Kai", "Hannah",
    "Diego", "Lily", "Ravi", "Carmen", "Elias",
    "Tanya", "Marco", "Hana", "Theo", "Nina",
    "Alex", "Serena", "Ivan", "Daphne", "Samir",
    "Julia", "Dante", "Mira", "Finn", "Esme",
    "Roman", "Lena", "Axel", "Vera", "Stefan",
    "Ingrid", "Rafael", "Freya", "Kofi", "Astrid",
    "Mateo", "Camille", "Arjun", "Sienna", "Nico",
    "Phoebe", "Tariq", "Stella", "Ezra", "Celeste",
    "Henrik", "Alma", "Soren", "Thea", "Idris",
    "Luna", "Emilio", "Beatrice", "Rowan", "Clara",
    "Xavier", "Maia", "Kieran", "Lina", "Bennett",
]

_LAST_NAMES = [
    "Rodriguez", "Mueller", "Lopez", "Chen", "Lin",
    "Patel", "Kim", "Nguyen", "Williams", "Garcia",
    "Johnson", "Brown", "Jones", "Davis", "Wilson",
    "Martinez", "Anderson", "Taylor", "Thomas", "Moore",
    "Jackson", "White", "Harris", "Clark", "Lewis",
    "Robinson", "Walker", "Young", "Hall", "Allen",
    "Wright", "King", "Scott", "Green", "Baker",
    "Adams", "Nelson", "Carter", "Mitchell", "Santos",
    "Tanaka", "Weber", "Singh", "Park", "Okafor",
    "Johansson", "Rossi", "Dubois", "Petrov", "Ali",
]

_TRAITS = [
    "friendly, outgoing, hospitable",
    "kind, inquisitive, passionate",
    "energetic, enthusiastic, inquisitive",
    "open-minded, curious, determined",
    "patient, kind, organized",
    "analytical, focused, calm",
    "creative, spontaneous, warm",
    "thoughtful, reliable, diligent",
    "witty, adventurous, sociable",
    "quiet, observant, empathetic",
    "bold, confident, generous",
    "gentle, artistic, contemplative",
    "pragmatic, resourceful, cheerful",
    "idealistic, caring, persistent",
    "humorous, easygoing, loyal",
    "ambitious, meticulous, fair",
    "compassionate, perceptive, grounded",
    "charismatic, decisive, playful",
    "introspective, principled, sincere",
    "spirited, adaptable, nurturing",
]

_OCCUPATIONS = [
    ("cafe owner", "runs a cozy cafe", "Opens the cafe at 7am, works the counter until 3pm, then relaxes."),
    ("pharmacist", "works at a local pharmacy", "Opens the pharmacy at 9am, works until 5pm, then heads home."),
    ("teacher", "teaches at a local school", "Teaches from 8am to 3pm, grades papers, then walks around town."),
    ("librarian", "works at the public library", "Opens the library at 9am, helps patrons until 5pm, then reads at home."),
    ("software developer", "works remotely as a software developer", "Codes from 9am to 5pm at a cafe or home, takes afternoon walks."),
    ("artist", "is a freelance painter and illustrator", "Paints outdoors in the morning, works in studio afternoons."),
    ("chef", "is a chef at a local restaurant", "Preps from 10am, cooks through dinner service, closes at 10pm."),
    ("yoga instructor", "teaches yoga and meditation", "Teaches morning classes at 6am and 8am, afternoons free for errands."),
    ("retired professor", "is a retired university professor", "Reads at the library mornings, walks the neighborhood afternoons."),
    ("nurse", "works at a nearby clinic", "Shifts from 7am to 3pm, then grocery shopping and errands."),
    ("musician", "plays guitar and performs at local venues", "Practices mornings, performs evenings, explores town in between."),
    ("architect", "designs residential buildings", "Works at a studio from 9am to 6pm, sketches buildings on walks."),
    ("gardener", "maintains community gardens", "Works in the garden from 7am to noon, visits neighbors afternoons."),
    ("journalist", "writes for a local newspaper", "Interviews people mornings, writes at a cafe afternoons."),
    ("baker", "runs a small bakery", "Bakes from 4am, opens at 7am, closes at 2pm, naps then errands."),
    ("personal trainer", "works as a personal trainer", "Trains clients 6am to noon, lunch at a cafe, afternoon jog."),
    ("social worker", "does community outreach", "Visits families mornings, office work afternoons, community events evenings."),
    ("bookshop owner", "runs an independent bookshop", "Opens shop at 10am, closes at 6pm, reads and walks evenings."),
    ("veterinarian", "runs a small animal clinic", "Sees patients 8am to 4pm, walks the dog, then dinner at home."),
    ("student", "is a graduate student studying urban planning", "Studies at the library all day, grabs coffee at a cafe."),
    ("photographer", "is a freelance photographer", "Shoots on location mornings, edits photos at a cafe afternoons."),
    ("mechanic", "runs an auto repair shop", "Opens shop at 8am, works until 5pm, grabs dinner in town."),
    ("florist", "owns a flower shop", "Arranges flowers from 7am, open until 5pm, walks the park evenings."),
    ("dentist", "has a dental practice", "Sees patients 8am to 4pm, then errands and family time."),
    ("barista", "works at a popular coffee shop", "Morning shift 6am to 2pm, paints or reads in the afternoon."),
]

_CURRENT_ACTIVITIES = [
    "planning a community event this weekend",
    "working on a big project with a tight deadline",
    "training for an upcoming charity run",
    "writing a blog about life in Piedmont",
    "learning to cook a new cuisine",
    "organizing a neighborhood cleanup",
    "preparing for a gallery exhibition",
    "mentoring a young person in the community",
    "researching local history for a presentation",
    "renovating their home kitchen",
    "starting a book club with neighbors",
    "volunteering at the community center",
    "experimenting with a new recipe",
    "learning to play a musical instrument",
    "planning a surprise birthday party for a friend",
]

_LIFESTYLES = [
    "Goes to bed around 11pm, wakes up around 6am.",
    "Early riser, up at 5:30am, asleep by 10pm.",
    "Night owl, sleeps at midnight, up at 7:30am.",
    "Steady routine, bed by 10:30pm, up at 6:30am.",
    "Flexible schedule, works in bursts.",
    "Very early riser, bed by 9:30pm, up at 5am.",
    "Active lifestyle, exercises daily.",
    "Family-oriented, steady routine.",
    "Social butterfly, often out in the evenings.",
    "Homebody, prefers quiet nights in.",
]


def generate_personas(
    n: int,
    available_streets: List[str],
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Generate *n* unique personas with diverse names, traits, and routines."""
    rng = random.Random(seed)
    personas: Dict[str, Dict[str, Any]] = {}

    used_names: set[str] = set()
    first_pool = list(_FIRST_NAMES)
    last_pool = list(_LAST_NAMES)

    streets = available_streets if available_streets else ["Piedmont:Streets"]

    for i in range(n):
        for _ in range(200):
            first = first_pool[i % len(first_pool)] if i < len(first_pool) else rng.choice(first_pool)
            last = last_pool[i % len(last_pool)] if i < len(last_pool) else rng.choice(last_pool)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                break
            last = rng.choice(last_pool)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                break
        used_names.add(full_name)

        traits = _TRAITS[i % len(_TRAITS)]
        occ_title, occ_desc, occ_plan = _OCCUPATIONS[i % len(_OCCUPATIONS)]
        activity = _CURRENT_ACTIVITIES[i % len(_CURRENT_ACTIVITIES)]
        lifestyle = _LIFESTYLES[i % len(_LIFESTYLES)]
        street = streets[i % len(streets)]

        personas[full_name] = {
            "innate": traits,
            "learned": f"{full_name} {occ_desc} in Piedmont.",
            "currently": f"{first} is {activity}.",
            "living_area": street,
            "daily_plan": occ_plan,
            "lifestyle": lifestyle,
        }

    return personas


def build_location_list(env: PiedmontEnv) -> List[str]:
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

    prompt = f"""You are {agent_name} in the city of Piedmont, California.

Personality: {persona['innate']}
Background: {persona['learned']}
Current situation: {persona['currently']}
Daily routine: {persona['daily_plan']}

It is currently {sim_time}. You are at: {current_location}
People nearby: {nearby_str}

Some available locations in Piedmont:
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


def ask_llm_for_emoji(
    client: OpenAI,
    action_description: str,
    model: str = "llama3.2",
) -> str:
    prompt = (
        "Convert the following action description to an emoji "
        "(important: respond with one or two emojis only, nothing else).\n\n"
        f"Action description: {action_description}\n"
        "Emoji:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip()
        emojis = re.findall(
            r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
            r"\U0000200D\U00002702-\U000027B0\U0001F900-\U0001F9FF"
            r"\U00002190-\U000021FF\U00002B05-\U00002B07\U00002934-\U00002935"
            r"\U000023CF\U000023E9-\U000023F3\U000023F8-\U000023FA"
            r"\U0000231A-\U0000231B\U00002328\U000025AA-\U000025AB"
            r"\U000025B6\U000025C0\U000025FB-\U000025FE]+",
            raw,
        )
        result = "".join(emojis)[:2] if emojis else ""
        if result:
            return result
    except Exception as e:
        print(f"  Emoji LLM error: {e}")
    return "\U0001F6B6"


def resolve_llm_answer(
    env: PiedmontEnv,
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


def get_agent_location_str(env: PiedmontEnv, agent_idx: int) -> str:
    agent = env.agents[agent_idx]
    tile = env.maze.access_tile((agent.x, agent.y))
    parts = [tile.get("sector", ""), tile.get("arena", ""), tile.get("game_object", "")]
    parts = [p for p in parts if p]
    return " > ".join(parts) if parts else "on the street"


def get_nearby_agents(env: PiedmontEnv, agent_idx: int, radius: int = 5) -> List[str]:
    agent = env.agents[agent_idx]
    nearby = []
    for j, other in enumerate(env.agents):
        if j == agent_idx:
            continue
        if abs(other.x - agent.x) <= radius and abs(other.y - agent.y) <= radius:
            nearby.append(other.name)
    return nearby


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Piedmont, CA Agent Simulation (Pygame + LLM)"
    )
    parser.add_argument("--place-name", type=str, default="Piedmont, California, USA")
    parser.add_argument("--grid-size", type=int, default=250)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--agents", nargs="*", default=None,
                        help="Explicit list of agent names (overrides --num-agents)")
    parser.add_argument("--num-agents", type=int, default=5,
                        help="Number of agents to generate (1-100, ignored if --agents is set)")
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
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for persona generation")
    parser.add_argument("--spawn-mode", type=str, default="street",
                        choices=["street", "building"],
                        help="Agent spawn mode: 'street' (random street tiles) "
                             "or 'building' (weighted by building population)")
    return parser.parse_args()


def main():
    args = parse_args()

    num = max(1, min(100, args.num_agents))

    from env.osm_map import OSMMap
    osm_map = OSMMap(
        place_name=args.place_name,
        grid_size=args.grid_size,
    )
    available_streets = sorted(
        addr for addr in osm_map.address_tiles.keys()
        if addr.startswith("Piedmont:") and addr.count(":") == 1
        and addr != "Piedmont:Streets"
    )

    personas = generate_personas(num, available_streets, seed=args.seed)

    if args.agents:
        agent_names = args.agents
    else:
        agent_names = list(personas.keys())

    client = OpenAI(base_url=args.ollama_url, api_key="ollama")

    env = PiedmontEnv(
        place_name=args.place_name,
        grid_size=args.grid_size,
        agent_names=agent_names,
        render_mode=args.render_mode,
        window_w=args.window_w,
        window_h=args.window_h,
        osm_map=osm_map,
        spawn_mode=args.spawn_mode,
    )

    obs, info = env.reset()
    env._zoom = args.zoom
    env.center_camera()

    available_locations = build_location_list(env)
    named = env.maze.named_buildings
    print(f"Piedmont loaded: {env.maze.maze_width}x{env.maze.maze_height} tiles")
    print(f"Agents ({len(env.agents)}): {[a.name for a in env.agents]}")
    print(f"Available locations: {len(available_locations)}")
    print(f"Named buildings from OSM: {len(named)}")
    if named:
        for b in named[:10]:
            print(f"  - {b['name']} ({b['type']}) on {b['sector']}")
        if len(named) > 10:
            print(f"  ... and {len(named) - 10} more")
    print(f"LLM model: {args.llm_model} @ {args.ollama_url}")
    print(f"Recording to: {args.output} ({args.fps} fps)")
    print(f"Decision interval: every {args.decision_interval} steps")
    print()

    emoji_map = {
        "cafe": "\u2615", "library": "\U0001F4DA", "pharmacy": "\U0001F48A",
        "apartment": "\U0001F3E0", "park": "\U0001F333", "school": "\U0001F3EB",
        "church": "\u26EA", "restaurant": "\U0001F37D\uFE0F", "shop": "\U0001F6D2",
        "garden": "\U0001F33B", "gym": "\U0001F3CB\uFE0F", "home": "\U0001F3E0",
    }

    for i, agent in enumerate(env.agents):
        persona = personas.get(agent.name, {})
        if persona.get("living_area"):
            env.move_agent_to_address(i, persona["living_area"])
            agent.description = "at home"
            loc = persona["living_area"].lower()
            for key, emoji in emoji_map.items():
                if key in loc:
                    agent.pronunciatio = emoji
                    break
            else:
                agent.pronunciatio = "\U0001F6B6"

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
                    persona = personas.get(agent.name, {})

                    print(
                        f"[step {step:4d} {sim_time}] {agent.name} at '{loc}' "
                        f"(nearby: {nearby or 'none'}) \u2014 asking LLM..."
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
                            agent.description = f"\u2192 {desc}"
                            print(f"           \u2192 heading to: {desc}")
                        else:
                            agent.description = f"staying ({desc})"
                            print(f"           \u2192 could not match: {desc}")

                    action_desc = (
                        f"{agent.name} is at {loc}"
                        + (
                            f", heading to {agent.description[2:]}"
                            if agent.description.startswith("\u2192")
                            else ""
                        )
                    )
                    emoji = ask_llm_for_emoji(
                        client, action_desc, model=args.llm_model
                    )
                    agent.pronunciatio = emoji
                    print(f"           emoji: {emoji}")

            actions = [Action.STAY] * len(env.agents)
            obs, reward, terminated, truncated, info = env.step(actions)

            env.sim_step = step
            env.sim_time = format_time(step)
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
