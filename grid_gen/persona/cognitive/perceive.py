# persona/cognitive/perceive.py
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, Tuple, List

import numpy as np

try:
    from env.constants import SEM_TO_ID, ID_TO_SEM
except Exception:
    SEM_TO_ID, ID_TO_SEM = {}, {}

def _cell_semantics(cell) -> Tuple[int, str, str]:
    """Return (sem_id, sem_name, mg_type) for a MiniGrid cell."""
    if cell is None:
        return SEM_TO_ID.get("street", 0), "street", "empty"
    if hasattr(cell, "sem_id"):
        sid = int(getattr(cell, "sem_id", 0))
        sname = getattr(cell, "sem_name", ID_TO_SEM.get(sid, "unknown"))
        return sid, sname, getattr(cell, "type", "floor")

    t = getattr(cell, "type", "empty")
    sid = int(SEM_TO_ID.get(t, SEM_TO_ID.get("street", 0)))
    sname = ID_TO_SEM.get(sid, t)
    return sid, sname, t

def get_obs(env, *, include_world_coords: bool = True, include_street: bool = True) -> Dict[str, Any]:
    """
    Use MiniGrid's agent-centric FOV and visibility mask.
    Returns ONLY the cells that are actually visible (not occluded).
    If include_street=False, 'street' cells are filtered out.
    Output:
      {
        'visible': [ { 'local':(lx,ly), 'world':(wx,wy), 'sem_id':int, 'sem_name':str, 'type':str }, ... ],
        'counts': { sem_name: count, ... },
        'agent_pos': (x,y),
        'agent_dir': int
      }
    """
    if not hasattr(env, "gen_obs_grid"):
        raise RuntimeError("env.gen_obs_grid() not available")

    local_grid, vis_mask = env.gen_obs_grid()  
    V = env.agent_view_size
    visible: List[Dict[str, Any]] = []

    topX = topY = None
    if include_world_coords and hasattr(env, "get_view_exts"):
        topX, topY, _, _ = env.get_view_exts()

    for ly in range(V):
        for lx in range(V):
            if vis_mask is not None and not bool(vis_mask[lx, ly]):
                continue  # occluded: skip entirely

            cell = local_grid.get(lx, ly)
            sid, sname, mg_type = _cell_semantics(cell)

            if sname in ("empty", "unknown"):
                sname = "street"
                sid = SEM_TO_ID.get("street", 0)
            if sname == "block":
                sname = "wall"

            if not include_street and sname == "street":
                continue  

            item = {
                "local": (lx, ly),
                "sem_id": sid,
                "sem_name": sname,
                "type": mg_type,  
            }
            if include_world_coords and topX is not None:
                item["world"] = (topX + lx, topY + ly)

            visible.append(item)

    counts = Counter(it["sem_name"] for it in visible)



    ### we need now here to store the observations in memeory

    return {
        "visible": visible,
        "counts": dict(sorted(counts.items())),
        "agent_pos": tuple(env.agent_pos),
        "agent_dir": int(env.agent_dir),
    }

def print_visible(obs: Dict[str, Any]):

    ax, ay = obs["agent_pos"]
    print(f"\nVisible (Agent @ {(ax, ay)}, dir={obs['agent_dir']}):")

    if not obs["visible"]:
        print("nothing visible")
        return

    # use counts already computed in get_obs()
    for sem_name, count in obs["counts"].items():
        print(f"  {sem_name:>8s} (x{count})")

    total = sum(obs["counts"].values())
    print(f"Total distinct: {len(obs['counts'])}, total visible cells: {total}")
