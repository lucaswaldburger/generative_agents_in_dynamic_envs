# persona/memory/empirical_analysis.py

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "grid_gen" / "persona" / "prompt" / "evacuation_data"
OUT_PATH = ROOT / "grid_gen" / "persona" / "memory" / "route_choice_priors.json"

def main():
    resp = pd.read_csv("grid_gen/persona/memory/evacuation_data/responses.csv")
    parts = pd.read_csv("grid_gen/persona/memory/evacuation_data/participants.csv")

    # Extract left/right codes
    pat = re.compile(r".*L-([^_]+)_R-([^_]+)")
    def extract_codes(tid):
        m = pat.match(str(tid))
        if not m:
            return (None, None)
        return m.group(1), m.group(2)

    resp[['left', 'right']] = resp['taskId'].apply(
        lambda x: pd.Series(extract_codes(x))
    )

    # Extract width & transition cue (stairs) from taskId components
    resp['left_width']  = resp['left'].str[0]
    resp['right_width'] = resp['right'].str[0]

    # last digit: 0/1 → interpreted as transition cue
    resp['left_transition']  = resp['left'].str[-1]
    resp['right_transition'] = resp['right'].str[-1]

    # demographics
    resp = resp.merge(parts[['userId','yearOfBirth','gender']], on='userId', how='left')
    resp['age'] = 2020 - resp['yearOfBirth']

    bins = [0, 24, 34, 49, 120]
    labels = ['<25','25-34','35-49','50+']
    resp['age_group'] = pd.cut(resp['age'], bins=bins, labels=labels)

    # Only corridor choices
    resp = resp[resp['AOI'].isin(['corridor_left','corridor_right'])]
    resp['choice'] = resp['AOI'].map({'corridor_left':'L', 'corridor_right':'R'})


    # 1) WIDTH PREFERENCE
    rank_width = {'W':3, 'N':2, 'E':1}
    resp['left_rank']  = resp['left_width'].map(rank_width)
    resp['right_rank'] = resp['right_width'].map(rank_width)

    resp['wider_side'] = np.where(
        resp['left_rank'] > resp['right_rank'], 'L',
        np.where(resp['left_rank'] < resp['right_rank'], 'R', 'equal')
    )

    align_w = resp[resp['wider_side'] != 'equal'].copy()
    align_w['aligned_width'] = (align_w['choice'] == align_w['wider_side'])

    width_pref = (
        align_w.groupby(['age_group','gender'])['aligned_width']
        .mean()
        .reset_index()
    )


    # 2) TRANSITION CUE PREFERENCE
    # transition cue: '1' = present, '0' = absent
    rank_transition = {'0': 1, '1': 2}

    resp['left_transition_rank']  = resp['left_transition'].map(rank_transition)
    resp['right_transition_rank'] = resp['right_transition'].map(rank_transition)

    resp['transition_side'] = np.where(
        resp['left_transition_rank'] > resp['right_transition_rank'], 'L',
        np.where(resp['left_transition_rank'] < resp['right_transition_rank'], 'R', 'equal')
    )

    align_t = resp[resp['transition_side'] != 'equal'].copy()
    align_t['aligned_transition'] = (align_t['choice'] == align_t['transition_side'])

    transition_pref = (
        align_t.groupby(['age_group','gender'])['aligned_transition']
        .mean()
        .reset_index()
    )


    # 3) TRADE-OFF: Width vs Transition
    trade = resp[
        (resp['wider_side'] != 'equal') &
        (resp['transition_side'] != 'equal') &
        (resp['wider_side'] != resp['transition_side'])
    ].copy()

    trade['follow_width'] = (trade['choice'] == trade['wider_side'])
    trade['follow_transition'] = (trade['choice'] == trade['transition_side'])

    trade_pref = (
        trade.groupby(['age_group','gender'])
        .agg(
            follow_width=('follow_width','mean'),
            follow_transition=('follow_transition','mean'),
            n=('userId','size')
        )
        .reset_index()
    )

    # ==========================
    # JSON OUTPUT
    # ==========================
    def get_val(df, age, gender, col):
        row = df[(df['age_group'] == age) & (df['gender'] == gender)]
        if row.empty or pd.isna(row.iloc[0][col]):
            return None
        return float(row.iloc[0][col])

    age_groups = ['<25','25-34','35-49','50+']
    genders = ['man','woman','x']

    priors = {
        "route_choice_priors": {
            "width_preference": {
                "description": "Probability of choosing the wider corridor when width differs.",
                "by_demographics": {}
            },
            "transition_cue_preference": {
                "description": "Probability of choosing the corridor containing a transition cue (stairs → generalized transition element).",
                "by_demographics": {}
            },
            "conflict_width_vs_transition": {
                "description": "When width and transition cues conflict, probability of following each cue.",
                "by_demographics": {}
            },
            "meta": {
                "notes": [
                    "Transition cue is a generalized signal representing spatial transitions (e.g., stairs, ramps, entrances).",
                    "Use these values as soft priors for behavioral models.",
                    "Derived from empirical evacuation dataset."
                ],
                "source": "evacuation_data/*.csv",
                "missing_value": "null"
            }
        }
    }

    wp = priors["route_choice_priors"]["width_preference"]["by_demographics"]
    tp = priors["route_choice_priors"]["transition_cue_preference"]["by_demographics"]
    cp = priors["route_choice_priors"]["conflict_width_vs_transition"]["by_demographics"]

    for age in age_groups:
        wp[age] = {}
        tp[age] = {}
        cp[age] = {}
        for g in genders:
            wp[age][g] = get_val(width_pref, age, g, "aligned_width")
            tp[age][g] = get_val(transition_pref, age, g, "aligned_transition")
            fw = get_val(trade_pref, age, g, "follow_width")
            ft = get_val(trade_pref, age, g, "follow_transition")
            cp[age][g] = {
                "follow_width": fw,
                "follow_transition": ft
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"[OK] route_choice_priors.json written to: {OUT_PATH}")

if __name__ == "__main__":
    main()
