"""
check_stimuli.py  –  Verify which stimuli.json file matches the human CSV data.

For each experiment in each stimuli.json, checks whether:
    len(times) * 3  ==  number of rows in the corresponding human CSV

Run from anywhere:
    python evaluation/check_stimuli.py
"""

import json, os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STIMULI_FILES = {
    "root  (domains/doors-keys-gems/stimuli.json)":
        os.path.join(ROOT, "domains", "doors-keys-gems", "stimuli.json"),
    "sub   (domains/doors-keys-gems/stimuli/stimuli.json)":
        os.path.join(ROOT, "domains", "doors-keys-gems", "stimuli", "stimuli.json"),
}

HUMAN_DIR = os.path.join(ROOT, "domains", "doors-keys-gems", "average_human_results_arrays")


def check_file(label, path):
    with open(path) as f:
        entries = json.load(f)

    matches   = []
    mismatches = []

    for entry in entries:
        name   = entry["name"]                          # e.g. "scenario_1_3"
        exp_id = name.replace("scenario_", "")          # e.g. "1_3"
        times  = entry["times"]
        n_times = len(times)

        csv_path = os.path.join(HUMAN_DIR, f"{exp_id}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [MISSING CSV] {exp_id}")
            continue

        with open(csv_path) as cf:
            n_rows = sum(1 for _ in cf)

        expected = n_times * 3
        ok = (expected == n_rows)

        row = {
            "exp_id":   exp_id,
            "n_times":  n_times,
            "expected": expected,
            "actual":   n_rows,
        }
        (matches if ok else mismatches).append(row)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Exp':<6}  {'len(times)':>10}  {'expected_rows':>13}  {'actual_rows':>11}  {'OK?':>4}")
    print(f"  {'-'*52}")
    for r in sorted(matches + mismatches, key=lambda x: x["exp_id"]):
        ok_str = "OK" if r["expected"] == r["actual"] else "FAIL"
        print(f"  {r['exp_id']:<6}  {r['n_times']:>10}  {r['expected']:>13}  {r['actual']:>11}  {ok_str:>4}")
    print(f"  {'-'*52}")
    print(f"  Matches: {len(matches)}/16    Mismatches: {len(mismatches)}/16")


for label, path in STIMULI_FILES.items():
    check_file(label, path)

print()
