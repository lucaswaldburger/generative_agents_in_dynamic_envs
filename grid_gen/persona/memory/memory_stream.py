# persona/memory/memory_stream.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from persona.cognitive.plan import get_agent_tile
from persona.cognitive.perceive import classify_location, regions_containing_point


def safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


@dataclass
class Task:
    """Represents a single task in the agent's memory."""
    id: str
    kind: str
    target: str
    where: List[str]
    priority: int
    created_t: int
    status: str = "pending"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "where": self.where,
            "priority": self.priority,
            "created_t": self.created_t,
            "status": self.status,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            kind=data["kind"],
            target=data["target"],
            where=data.get("where", []),
            priority=data.get("priority", 1),
            created_t=data.get("created_t", 0),
            status=data.get("status", "pending"),
            notes=data.get("notes", ""),
        )


class TaskMemory:
    """
    Manages an agent's task memory, similar to SpatialMemory structure.
    
    This class encapsulates all task-related operations including:
    - Storing pending and completed tasks
    - Adding/removing tasks
    - Ensuring priority tasks are remembered
    - Querying actionable tasks
    """
    
    def __init__(self):
        """Initialize an empty task memory."""
        self.pending: List[Task] = []
        self.done: List[Task] = []
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMemory":
        """Create TaskMemory from dictionary format."""
        memory = cls()
        memory.pending = [Task.from_dict(t) for t in data.get("pending", [])]
        memory.done = [Task.from_dict(t) for t in data.get("done", [])]
        return memory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task memory to dictionary format."""
        return {
            "pending": [task.to_dict() for task in self.pending],
            "done": [task.to_dict() for task in self.done],
        }
    
    def add_task(self, task: Task) -> bool:
        """
        Add a task to pending list if it doesn't already exist.
        
        Args:
            task: Task object to add
            
        Returns:
            bool: True if task was added, False if it already exists
        """
        # Check if task with same ID already exists
        if any(t.id == task.id for t in self.pending):
            return False
        if any(t.id == task.id for t in self.done):
            return False
        
        self.pending.append(task)
        return True
    
    def complete_task(self, task_id: str) -> bool:
        """
        Mark a task as completed by moving it from pending to done.
        
        Args:
            task_id: ID of the task to complete
            
        Returns:
            bool: True if task was found and completed, False otherwise
        """
        for task in list(self.pending):
            if task.id == task_id:
                self.pending.remove(task)
                task.status = "done"
                self.done.append(task)
                return True
        return False
    
    def get_pending_task_ids(self) -> List[str]:
        """Get list of all pending task IDs."""
        return [task.id for task in self.pending]
    
    def get_done_task_ids(self) -> List[str]:
        """Get list of all done task IDs."""
        return [task.id for task in self.done]
    
    def has_task(self, task_id: str) -> bool:
        """Check if a task exists (pending or done)."""
        return any(t.id == task_id for t in self.pending) or any(t.id == task_id for t in self.done)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by ID from pending or done lists."""
        for task in self.pending:
            if task.id == task_id:
                return task
        for task in self.done:
            if task.id == task_id:
                return task
        return None
    
    def get_tasks_by_priority(self, min_priority: int = 1) -> List[Task]:
        """Get all pending tasks with priority >= min_priority."""
        return [task for task in self.pending if task.priority >= min_priority]
    
    def get_actionable_tasks(self, env, agent_id, agent_cfg) -> List[Task]:
        """
        Returns a list of tasks that are actionable now.
        'where' matches the agent's current semantic location.
        """
        from persona.cognitive.perceive import classify_location, get_agent_pose
        x, y, *_ = get_agent_pose(env, agent_id)
        loc = classify_location(env, x, y)
        
        actionable = []
        for task in self.pending:
            if loc in task.where:
                actionable.append(task)
        
        return actionable
    
    def ensure_priority_tasks_remembered(
        self, 
        agent_cfg, 
        t: int, 
        min_priority: int = 2, 
        max_age_steps: int = 50
    ) -> int:
        """
        Ensures that priority tasks are not forgotten by the agent.
        
        This method:
        1. Checks for high-priority tasks that should exist based on agent config
        2. Re-adds missing priority tasks if they're not in pending or done
        3. Optionally refreshes old priority tasks that have been pending too long
        
        Args:
            agent_cfg: Agent configuration object
            t: Current simulation step
            min_priority: Minimum priority level to consider (default 2)
            max_age_steps: Maximum steps a priority task can be pending before being refreshed (default 50)
        
        Returns:
            int: Number of tasks that were re-added or refreshed
        """
        re_added_count = 0
        
        pending_ids = set(self.get_pending_task_ids())
        done_ids = set(self.get_done_task_ids())
        
        # Check for dependents - these should always be remembered if they exist
        for dep in getattr(agent_cfg, "dependents", []):
            check_task_id = f"check:{dep['name']}"
            gather_task_id = f"gather:{dep['name']}"
            
            # If check task doesn't exist and hasn't been done, re-add it
            if check_task_id not in pending_ids and check_task_id not in done_ids:
                task = Task(
                    id=check_task_id,
                    kind="check",
                    target=dep["name"],
                    where=[agent_cfg.living_area],
                    priority=3,
                    created_t=t,
                    status="pending",
                    notes=f"Ensure dependent {dep['name']} is safe (remembered priority task)"
                )
                if self.add_task(task):
                    re_added_count += 1
            
            # Check if gather task should exist (for evacuation scenarios)
            has_evacuation_intent = any(
                "evacuate" in task.notes.lower() or task.kind == "gather"
                for task in self.pending
            )
            
            if has_evacuation_intent and gather_task_id not in pending_ids and gather_task_id not in done_ids:
                task = Task(
                    id=gather_task_id,
                    kind="gather",
                    target=dep["name"],
                    where=[agent_cfg.living_area],
                    priority=3,
                    created_t=t,
                    status="pending",
                    notes=f"Gather {dep['name']} for evacuation (remembered priority task)"
                )
                if self.add_task(task):
                    re_added_count += 1
        
        # Check for pets
        for pet in getattr(agent_cfg, "pets", []):
            pet_task_id = f"find:{pet['name']}"
            if pet_task_id not in pending_ids and pet_task_id not in done_ids:
                task = Task(
                    id=pet_task_id,
                    kind="find",
                    target=pet["name"],
                    where=[agent_cfg.living_area],
                    priority=2,
                    created_t=t,
                    status="pending",
                    notes=f"Find pet {pet['name']} (remembered priority task)"
                )
                if self.add_task(task):
                    re_added_count += 1
        
        # Refresh old priority tasks that have been pending too long
        current_time = t
        tasks_to_refresh = []
        for task in self.pending:
            if task.priority >= min_priority:
                age = current_time - task.created_t
                if age > max_age_steps:
                    tasks_to_refresh.append(task)
        
        # Refresh and move tasks to front
        for task in tasks_to_refresh:
            task.created_t = current_time
            if "remembered" not in task.notes:
                task.notes = task.notes + " (refreshed - high priority)"
            # Move to front of list to increase visibility
            self.pending.remove(task)
            self.pending.insert(0, task)
            re_added_count += 1
        
        return re_added_count
    
    def add_persona_tasks(self, agent_cfg, t: int) -> int:
        """
        Generate tasks based on persona data: dependents, pets, valuables, etc.
        
        Returns:
            int: Number of tasks added
        """
        added_count = 0
        
        # Example: dependents
        for dep in getattr(agent_cfg, "dependents", []):
            task = Task(
                id=f"check:{dep['name']}",
                kind="check",
                target=dep["name"],
                where=[agent_cfg.living_area],
                priority=3,
                created_t=t,
                status="pending",
                notes=f"Ensure dependent {dep['name']} is safe"
            )
            if self.add_task(task):
                added_count += 1
        
        # Example: pets (if agent_cfg has pets)
        for pet in getattr(agent_cfg, "pets", []):
            task = Task(
                id=f"find:{pet['name']}",
                kind="find",
                target=pet["name"],
                where=[agent_cfg.living_area],
                priority=2,
                created_t=t,
                status="pending",
                notes=f"Find pet {pet['name']}"
            )
            if self.add_task(task):
                added_count += 1
        
        # Example: personal belongings
        if getattr(agent_cfg, "has_valuables", False):
            task = Task(
                id="gather:valuables",
                kind="gather",
                target="valuables",
                where=[agent_cfg.living_area],
                priority=1,
                created_t=t,
                status="pending",
                notes="Gather important belongings"
            )
            if self.add_task(task):
                added_count += 1
        
        return added_count
    
    def add_evacuation_tasks(self, agent_cfg, decision: Dict[str, Any], t: int) -> int:
        """
        Tasks added when the agent chooses to evacuate.
        Typically: gather dependents, notify family, secure belongings, etc.
        
        Returns:
            int: Number of tasks added
        """
        added_count = 0
        
        # notify family
        if getattr(agent_cfg, "dependents", []):
            task = Task(
                id="notify:family",
                kind="notify",
                target="family",
                where=[agent_cfg.living_area],
                priority=2,
                created_t=t,
                status="pending",
                notes="Notify family about evacuation"
            )
            if self.add_task(task):
                added_count += 1
        
        # gather dependents if evacuating
        for dep in getattr(agent_cfg, "dependents", []):
            task = Task(
                id=f"gather:{dep['name']}",
                kind="gather",
                target=dep["name"],
                where=[agent_cfg.living_area],
                priority=3,
                created_t=t,
                status="pending",
                notes=f"Gather {dep['name']} for evacuation"
            )
            if self.add_task(task):
                added_count += 1
        
        # secure home (optional)
        task = Task(
            id="secure:home",
            kind="secure",
            target=agent_cfg.living_area,
            where=[agent_cfg.living_area],
            priority=1,
            created_t=t,
            status="pending",
            notes="Secure home before evacuating"
        )
        if self.add_task(task):
            added_count += 1
        
        return added_count
    
    def add_dependent_tasks(self, agent_cfg, t: int) -> int:
        """
        When LLM explicitly chooses 'check_dependent'.
        
        Returns:
            int: Number of tasks added
        """
        added_count = 0
        
        for dep in getattr(agent_cfg, "dependents", []):
            task = Task(
                id=f"check:{dep['name']}",
                kind="check",
                target=dep["name"],
                where=[agent_cfg.living_area],
                priority=3,
                created_t=t,
                status="pending",
                notes=f"Check on {dep['name']}"
            )
            if self.add_task(task):
                added_count += 1
        
        return added_count
    
    def maybe_add_tasks_from_decision(self, decision: Dict[str, Any], agent_cfg, t: int) -> int:
        """
        The main task generator. Called once per LLM high-level decision.
        
        Returns:
            int: Number of tasks added
        """
        added_count = 0
        intent = (decision.get("intent") or "").lower()
        cmd = (decision.get("command") or "").lower()
        
        # STAY AND REACT → internal tasks
        if intent == "stay" and "react" in cmd:
            added_count += self.add_persona_tasks(agent_cfg, t)
        
        # EVACUATE → family, dependents, secure home
        if intent == "evacuate":
            added_count += self.add_evacuation_tasks(agent_cfg, decision, t)
        
        # DIRECT request: check dependent
        if intent == "check_dependent":
            added_count += self.add_dependent_tasks(agent_cfg, t)
        
        return added_count


# ============================================================================
# Legacy function-based API for backward compatibility
# ============================================================================

def init_task_state() -> Dict[str, Any]:
    """
    Initialize an empty task state dictionary (legacy format).
    
    Returns:
        dict: A task state dict with "pending" and "done" lists
    """
    return {
        "pending": [],
        "done": []
    }


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
    Supports both TaskMemory objects and legacy dict format.
    """
    if isinstance(task_state, TaskMemory):
        pending = [task.to_dict() for task in task_state.pending]
        done = [task.to_dict() for task in task_state.done[-3:]]  # last 3
    else:
        pending = task_state.get("pending", [])
        done = task_state.get("done", [])[-3:]  # last 3
    
    log_file.write(f"[t={t} {clock_time}] {agent_cfg.name}\n")
    log_file.write(f"Pending:\n")
    for task in pending:
        task_dict = task if isinstance(task, dict) else task.to_dict()
        log_file.write(f"  - {task_dict['id']} ({task_dict['kind']}, target={task_dict['target']})\n")
    
    if done:
        log_file.write(f"Recently done:\n")
        for task in done:
            task_dict = task if isinstance(task, dict) else task.to_dict()
            log_file.write(f"  - {task_dict['id']}\n")
    
    log_file.write("\n")
    log_file.flush()


def add_pending_task(task_state, task) -> bool:
    """
    Legacy function: Add a task to pending list.
    Supports both TaskMemory objects and legacy dict format.
    """
    if isinstance(task_state, TaskMemory):
        task_obj = task if isinstance(task, Task) else Task.from_dict(task)
        return task_state.add_task(task_obj)
    else:
        # Legacy dict format
        if isinstance(task, Task):
            task = task.to_dict()
        if any(t["id"] == task["id"] for t in task_state["pending"]):
            return False
        task_state["pending"].append(task)
        return True


def complete_task(task_state, task_id: str) -> bool:
    """
    Legacy function: Mark a task as completed.
    Supports both TaskMemory objects and legacy dict format.
    """
    if isinstance(task_state, TaskMemory):
        return task_state.complete_task(task_id)
    else:
        # Legacy dict format
        for task in list(task_state["pending"]):
            if task["id"] == task_id:
                task_state["pending"].remove(task)
                task_state["done"].append({**task, "status": "done"})
                return True
        return False


def actionable_tasks_now(env, agent_id, task_state, t, agent_cfg):
    """
    Legacy function: Returns a list of tasks that are actionable now.
    Supports both TaskMemory objects and legacy dict format.
    """
    if isinstance(task_state, TaskMemory):
        tasks = task_state.get_actionable_tasks(env, agent_id, agent_cfg)
        return [task.to_dict() for task in tasks]
    else:
        # Legacy dict format
        from persona.cognitive.perceive import classify_location, get_agent_pose
        x, y, *_ = get_agent_pose(env, agent_id)
        loc = classify_location(env, x, y)
        
        actionable = []
        for task in task_state["pending"]:
            if loc in task["where"]:
                actionable.append(task)
        return actionable


def ensure_priority_tasks_remembered(task_state, agent_cfg, t, min_priority=2, max_age_steps=50):
    """
    Legacy function: Ensures that priority tasks are not forgotten.
    Supports both TaskMemory objects and legacy dict format.
    """
    if isinstance(task_state, TaskMemory):
        return task_state.ensure_priority_tasks_remembered(agent_cfg, t, min_priority, max_age_steps)
    else:
        # Legacy dict format - convert to TaskMemory, process, convert back
        memory = TaskMemory.from_dict(task_state)
        count = memory.ensure_priority_tasks_remembered(agent_cfg, t, min_priority, max_age_steps)
        # Update the original dict
        task_state["pending"] = [task.to_dict() for task in memory.pending]
        task_state["done"] = [task.to_dict() for task in memory.done]
        return count


def maybe_add_tasks_from_decision(task_state, decision, agent_cfg, t):
    """
    Legacy function: The main task generator.
    Supports both TaskMemory objects and legacy dict format.
    """
    if isinstance(task_state, TaskMemory):
        return task_state.maybe_add_tasks_from_decision(decision, agent_cfg, t)
    else:
        # Legacy dict format
        added = False
        intent = (decision.get("intent") or "").lower()
        cmd = (decision.get("command") or "").lower()
        
        # Convert to TaskMemory for processing
        memory = TaskMemory.from_dict(task_state)
        
        if intent == "stay" and "react" in cmd:
            added = memory.add_persona_tasks(agent_cfg, t) > 0
        if intent == "evacuate":
            added = memory.add_evacuation_tasks(agent_cfg, decision, t) > 0
        if intent == "check_dependent":
            added = memory.add_dependent_tasks(agent_cfg, t) > 0
        
        # Update original dict
        task_state["pending"] = [task.to_dict() for task in memory.pending]
        task_state["done"] = [task.to_dict() for task in memory.done]
        
        return added
