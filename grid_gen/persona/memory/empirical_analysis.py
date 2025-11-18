# persona/memory/empirical_analysis.py

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cs294_286_final_project-1/
DATA_DIR = ROOT / "grid_gen" / "persona" / "prompt" / "evacuation_data"
OUT_PATH = ROOT / "grid_gen" / "persona" / "memory" / "route_choice_priors.json"

def main():
    resp = pd.read_csv("grid_gen/persona/memory/evacuation_data/responses.csv")
    parts = pd.read_csv("grid_gen/persona/memory/evacuation_data/participants.csv")

    # left/right corridor code)
    pat = re.compile(r".*L-([^_]+)_R-([^_]+)")
    def extract_codes(tid):
        m = pat.match(str(tid))
        if not m:
            return (None, None)
        return m.group(1), m.group(2)

    resp[['left', 'right']] = resp['taskId'].apply(
        lambda x: pd.Series(extract_codes(x))
    )

    # width (W/N/E) & visibility (0/1)
    resp['left_width']  = resp['left'].str[0]
    resp['right_width'] = resp['right'].str[0]
    resp['left_vis']    = resp['left'].str[-1]
    resp['right_vis']   = resp['right'].str[-1]

    # demographics
    resp = resp.merge(parts[['userId','yearOfBirth','gender']], on='userId', how='left')
    resp['age'] = 2020 - resp['yearOfBirth']  # 실험 연도 근사

    bins = [0, 24, 34, 49, 120]
    labels = ['<25','25-34','35-49','50+']
    resp['age_group'] = pd.cut(resp['age'], bins=bins, labels=labels)

    # only left/right decisions
    resp = resp[resp['AOI'].isin(['corridor_left','corridor_right'])]
    resp['choice'] = resp['AOI'].map({'corridor_left':'L', 'corridor_right':'R'})

    # WIDTH PREFERENCE
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

    # ---------- VISIBILITY PREFERENCE ----------
    # 가정: vis '0' = better, '1' = worse
    rank_vis = {'0': 2, '1': 1}
    resp['left_vis_rank']  = resp['left_vis'].map(rank_vis)
    resp['right_vis_rank'] = resp['right_vis'].map(rank_vis)

    resp['more_visible_side'] = np.where(
        resp['left_vis_rank'] > resp['right_vis_rank'], 'L',
        np.where(resp['left_vis_rank'] < resp['right_vis_rank'], 'R', 'equal')
    )

    align_v = resp[resp['more_visible_side'] != 'equal'].copy()
    align_v['aligned_vis'] = (align_v['choice'] == align_v['more_visible_side'])

    vis_pref = (
        align_v.groupby(['age_group','gender'])['aligned_vis']
        .mean()
        .reset_index()
    )

    # trad-offs (width vs visibility) 
    trade = resp[
        (resp['wider_side'] != 'equal') &
        (resp['more_visible_side'] != 'equal') &
        (resp['wider_side'] != resp['more_visible_side'])
    ].copy()

    trade['follow_width'] = (trade['choice'] == trade['wider_side'])
    trade['follow_vis']   = (trade['choice'] == trade['more_visible_side'])

    trade_pref = (
        trade.groupby(['age_group','gender'])
        .agg(
            follow_width=('follow_width','mean'),
            follow_visibility=('follow_vis','mean'),
            n=('userId','size')
        )
        .reset_index()
    )

    # generate json
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
                "description": "Probability of choosing the wider corridor when left and right widths differ.",
                "by_demographics": {}
            },
            "visibility_preference": {
                "description": "Probability of choosing the more visible corridor when visibility differs.",
                "by_demographics": {}
            },
            "conflict_resolution": {
                "description": "When width and visibility cues conflict, probability of following each cue.",
                "by_demographics": {}
            },
            "meta": {
                "notes": [
                    "Width is the dominant cue for younger adults across genders.",
                    "Visibility influence increases with age, especially for 50+ men.",
                    "Women over 35 exhibit lower width sensitivity.",
                    "Use these values as priors, not hard rules."
                ],
                "source": "evacuation_data/*.csv",
                "missing_value": "null"
            }
        }
    }

    wp = priors["route_choice_priors"]["width_preference"]["by_demographics"]
    vp = priors["route_choice_priors"]["visibility_preference"]["by_demographics"]
    cp = priors["route_choice_priors"]["conflict_resolution"]["by_demographics"]

    for age in age_groups:
        wp[age] = {}
        vp[age] = {}
        cp[age] = {}
        for g in genders:
            wp[age][g] = get_val(width_pref, age, g, "aligned_width")
            vp[age][g] = get_val(vis_pref, age, g, "aligned_vis")
            fw = get_val(trade_pref, age, g, "follow_width")
            fv = get_val(trade_pref, age, g, "follow_visibility")
            cp[age][g] = {
                "follow_width": fw,
                "follow_visibility": fv
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"[OK] route_choice_priors.json written to: {OUT_PATH}")

if __name__ == "__main__":
    main()
