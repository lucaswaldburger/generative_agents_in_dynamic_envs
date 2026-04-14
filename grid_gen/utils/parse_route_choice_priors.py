from __future__ import annotations

def _age_to_bucket(age: int) -> str:
    if age < 25:
        return "<25"
    elif 25 <= age <= 34:
        return "25-34"
    elif 35 <= age <= 49:
        return "35-49"
    else:
        return "50+"


def _gender_to_key(gender: str) -> str:
    g = (gender or "").strip().lower()
    if g.startswith("m"):
        return "man"
    if g.startswith("f"):
        return "woman"
    return "x"


def get_agent_route_priors(agent_cfg, route_choice_priors: dict) -> dict | None:
    """
    Returns a compact dict of priors for this agent's age/gender, or None if missing.
    """
    if not route_choice_priors:
        return None

    try:
        age_bucket = _age_to_bucket(int(agent_cfg.age))
    except Exception:
        return None

    gender_key = _gender_to_key(getattr(agent_cfg, "gender", ""))

    def _safe_get(path, default=None):
        d = route_choice_priors
        for key in path:
            if d is None:
                return default
            d = d.get(key)
        return d if d is not None else default

    width_pref = _safe_get(["width_preference", "by_demographics", age_bucket, gender_key])
    trans_pref = _safe_get(["transition_cue_preference", "by_demographics", age_bucket, gender_key])
    conflict = _safe_get(["conflict_width_vs_transition", "by_demographics", age_bucket, gender_key], {})

    if width_pref is None and trans_pref is None and not conflict:
        return None

    return {
        "age_bucket": age_bucket,
        "gender_key": gender_key,
        "width_pref": width_pref,
        "transition_pref": trans_pref,
        "conflict_follow_width": conflict.get("follow_width"),
        "conflict_follow_transition": conflict.get("follow_transition"),
    }
