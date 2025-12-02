import os, json, datetime
from persona.cognitive.plan import get_agent_tile
from persona.cognitive.perceive  import classify_location, regions_containing_point


def safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def open_agent_memory_logs(run_dir, env):
    """
    Creates one memory log per agent.
    """
    logs = {}
    for agent_id, agent in enumerate(env.agents):
        safe = agent.config.name.replace(" ", "_")
        path = os.path.join(run_dir, f"{safe}_memory.txt")
        logs[agent_id] = open(path, "w")
        logs[agent_id].write(f"Memory log for {agent.config.name}\n")
        logs[agent_id].write("=" * 40 + "\n\n")
    return logs


def log_memory(log_file, t, clock_time, agent_cfg, task_state):
    """
    Logs memory state for debugging / future LLM input.
    """
    pending = task_state["pending"]
    done = task_state["done"][-3:]  # last 3

    log_file.write(f"[t={t} {clock_time}] {agent_cfg.name}\n")
    log_file.write(f"Pending:\n")
    for task in pending:
        log_file.write(f"  - {task['id']} ({task['kind']}, target={task['target']})\n")

    if done:
        log_file.write(f"Recently done:\n")
        for task in done:
            log_file.write(f"  - {task['id']}\n")

    log_file.write("\n")
    log_file.flush()

def add_pending_task(task_state, task):
    if task not in task_state["pending"] and task not in task_state["done"]:
        task_state["pending"].append(task)
        return True
    return False

def complete_task(task_state, task_id):

    for task in list(task_state["pending"]):
        if task["id"] == task_id:
            task_state["pending"].remove(task)
            task_state["done"].append({**task, "status": "done"})
            return True
    return False

def tasks_from_persona(agent_cfg):
    """derive stable tasks from persona info (pets, dependents)"""
    tasks = []
    for d in getattr(agent_cfg, "dependents", []) or []:
        if (d.get("type","").lower() == "pet"):
            name = d.get("name","pet")
            tasks.append(f"find_pet:{name}")
    return tasks




def add_persona_tasks(task_state, agent_cfg, t):
    """
    Generate tasks based on persona data: dependents, pets, valuables, etc.
    NOTHING is tied to names in main.py.
    """
    added = False

    # Example: dependents
    for dep in getattr(agent_cfg, "dependents", []):
        task = {
            "id": f"check:{dep['name']}",
            "kind": "check",
            "target": dep["name"],
            "where": [agent_cfg.living_area],      # home location
            "priority": 3,
            "created_t": t,
            "status": "pending",
            "notes": f"Ensure dependent {dep['name']} is safe"
        }
        added |= add_pending_task(task_state, task)

    # Example: pets (if agent_cfg has pets)
    for pet in getattr(agent_cfg, "pets", []):
        task = {
            "id": f"find:{pet['name']}",
            "kind": "find",
            "target": pet["name"],
            "where": [agent_cfg.living_area],
            "priority": 2,
            "created_t": t,
            "status": "pending",
            "notes": f"Find pet {pet['name']}"
        }
        added |= add_pending_task(task_state, task)

    # Example: personal belongings
    if getattr(agent_cfg, "has_valuables", False):
        task = {
            "id": f"gather:valuables",
            "kind": "gather",
            "target": "valuables",
            "where": [agent_cfg.living_area],
            "priority": 1,
            "created_t": t,
            "status": "pending",
            "notes": "Gather important belongings"
        }
        added |= add_pending_task(task_state, task)

    return added


def add_pending_task(task_state, task):
    # avoid duplicates
    if any(t["id"] == task["id"] for t in task_state["pending"]):
        return False

    task_state["pending"].append(task)
    return True


def add_evacuation_tasks(task_state, agent_cfg, decision, t):
    """
    Tasks added when the agent chooses to evacuate.
    Typically: gather dependents, notify family, secure belongings, etc.
    """
    added = False

    # notify family
    if getattr(agent_cfg, "dependents", []):
        task = {
            "id": "notify:family",
            "kind": "notify",
            "target": "family",
            "where": [agent_cfg.living_area],
            "priority": 2,
            "created_t": t,
            "status": "pending",
            "notes": "Notify family about evacuation"
        }
        added |= add_pending_task(task_state, task)

    # gather dependents if evacuating
    for dep in getattr(agent_cfg, "dependents", []):
        task = {
            "id": f"gather:{dep['name']}",
            "kind": "gather",
            "target": dep["name"],
            "where": [agent_cfg.living_area],
            "priority": 3,
            "created_t": t,
            "status": "pending",
            "notes": f"Gather {dep['name']} for evacuation"
        }
        added |= add_pending_task(task_state, task)

    # secure home (optional)
    task = {
        "id": "secure:home",
        "kind": "secure",
        "target": agent_cfg.living_area,
        "where": [agent_cfg.living_area],
        "priority": 1,
        "created_t": t,
        "status": "pending",
        "notes": "Secure home before evacuating"
    }
    added |= add_pending_task(task_state, task)

    return added


def add_dependent_tasks(task_state, agent_cfg, t):
    """
    When LLM explicitly chooses 'check_dependent'.
    """
    added = False

    for dep in getattr(agent_cfg, "dependents", []):
        task = {
            "id": f"check:{dep['name']}",
            "kind": "check",
            "target": dep["name"],
            "where": [agent_cfg.living_area],
            "priority": 3,
            "created_t": t,
            "status": "pending",
            "notes": f"Check on {dep['name']}"
        }
        added |= add_pending_task(task_state, task)

    return added

def maybe_add_tasks_from_decision(task_state, decision, agent_cfg, t):
    """
    The main task generator. Called once per LLM high-level decision.
    """
    added = False
    intent = (decision.get("intent") or "").lower()
    cmd = (decision.get("command") or "").lower()

    # STAY AND REACT → internal tasks
    if intent == "stay" and "react" in cmd:
        added |= add_persona_tasks(task_state, agent_cfg, t)

    # EVACUATE → family, dependents, secure home
    if intent == "evacuate":
        added |= add_evacuation_tasks(task_state, agent_cfg, decision, t)

    # DIRECT request: check dependent
    if intent == "check_dependent":
        added |= add_dependent_tasks(task_state, agent_cfg, t)

    return added

def is_agent_at_home(env, agent_id):
    x, y, *_ = get_agent_tile(env, agent_id)
    loc = classify_location(env, x, y)
    return loc in ["Home_A", "Home_B"]  # matches your map naming

def has_pending_pet_task(task_state):
    return any(t.startswith("find_pet:") for t in task_state["pending"])


def actionable_tasks_now(env, agent_id, task_state, t, agent_cfg):
    """
    Returns a list of tasks that are actionable now.
    'where' matches the agent's current semantic location.
    """
    from persona.cognitive.perceive import classify_location, get_agent_pose
    x, y, *_ = get_agent_pose(env, agent_id)
    loc = classify_location(env, x, y)

    actionable = []
    for task in task_state["pending"]:
        if loc in task["where"]:
            actionable.append(task)

    return actionable
