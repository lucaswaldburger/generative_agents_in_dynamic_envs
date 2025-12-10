import logging
import os
import datetime
from hydra.utils import get_original_cwd

def sim_time_str(cfg, t: int) -> str:
    """Map sim timestep to clock time using cfg.sim.start_time and seconds_per_step."""
    start = datetime.datetime.strptime(cfg.sim.start_time, "%H:%M")
    curr = start + datetime.timedelta(seconds=t * cfg.sim.seconds_per_step)
    return curr.strftime("%H:%M")


def setup_sim_output_dir():
    """
    Creates a run folder in <project_root>/sim_outputs/ with timestamp.
    Returns the folder path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    project_root = get_original_cwd()  
    base_dir = os.path.join(project_root, "sim_outputs")

    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Output] Created run directory at {run_dir}")
    return run_dir

def create_agent_logs(run_dir, env):
    """
    Creates one folder per agent under run_dir, and inside each folder
    a main trajectory log file: <run_dir>/<Agent_Name>/trajectory.txt

    Returns a dict: agent_id -> file_handle
    """
    logs = {}
    for agent_id, agent in enumerate(env.agents):
        safe_name = agent.config.name.replace(" ", "_")

        # Folder per agent
        agent_dir = os.path.join(run_dir, safe_name)
        os.makedirs(agent_dir, exist_ok=True)

        # Main per-agent log file
        path = os.path.join(agent_dir, "trajectory.txt")
        f = open(path, "w")
        f.write(f"Log for {agent.config.name}\n")
        f.write("=" * 40 + "\n\n")
        f.flush()  # just to be safe

        logs[agent_id] = f

        # Debug print so you can see exactly where it went
        print(f"[Output] Created agent log for {agent.config.name} at {path}")

    return logs

def log_agent_step(log_file, agent_id, step, agent, action, desc, spatial_info=None):
    """
    Log one step for an agent, including optional spatial memory info.
    """
    log_file.write(f"Step {step}\n")
    log_file.write(f" Agent id: {agent_id}\n")
    log_file.write(f" Position: ({agent.x}, {agent.y})\n")
    log_file.write(f" Action: {action}\n")
    log_file.write(f" Observation: {desc}\n")

    if spatial_info is not None:
        log_file.write(f" Spatial memory: {spatial_info}\n")

    log_file.write("-" * 30 + "\n")
    log_file.flush()






def setup_debug_loggers(run_dir: str):
    """
    Sets up three loggers that write to separate files under run_dir:
      - high_level_intent.log
      - local_decisions.log
      - planner_astar.log
    Returns (intent_logger, local_logger, planner_logger).
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    def make_logger(name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(os.path.join(run_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        # Avoid duplicate logs to stdout
        logger.propagate = False

        # Only add handler once (in case Hydra reloads, etc.)
        if not logger.handlers:
            logger.addHandler(fh)

        return logger

    intent_logger = make_logger("intent", "high_level_intent.log")
    local_logger = make_logger("local", "local_decisions.log")
    planner_logger = make_logger("planner", "planner_astar.log")

    return intent_logger, local_logger, planner_logger
