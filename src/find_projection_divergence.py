#!/usr/bin/env python3
"""Compare the original closed-loop trajectories for projection ON vs OFF.

Reads Experiment 06 diagnostic CSV. Does not rerun training or MPC.
Run from the repository root:
    python src/find_projection_divergence.py --seeds 70002 70006
"""
import argparse
import csv
import math
from pathlib import Path

DEFAULT_CSV = Path("results/experiment_06/diagnostics/trajectories.csv")
OFF = "multistep_only"
ON = "multistep_plus_projection"


def read_rows(path, seed, variant):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"seed", "variant", "step", "action", "theta_before",
                    "omega_before", "theta_actual_next", "omega_actual_next",
                    "cumulative_return"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        selected = {
            int(row["step"]): row
            for row in reader
            if int(row["seed"]) == seed and row["variant"] == variant
        }
    return selected


def angular_distance(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def first_above(records, field, threshold):
    return next((r for r in records if r[field] > threshold), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seeds", type=int, nargs="+", default=[70002, 70006])
    parser.add_argument("--action-tol", type=float, default=1e-3)
    parser.add_argument("--state-tol", type=float, default=1e-4)
    parser.add_argument("--show", type=int, default=8,
                        help="Number of paired rows to show around first divergence.")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"File not found: {args.input}")

    for seed in args.seeds:
        a, b = read_rows(args.input, seed, OFF), read_rows(args.input, seed, ON)
        common = sorted(set(a) & set(b))
        if not common:
            print(f"Seed {seed}: no paired rows found. Check seed/variant names.")
            continue
        records = []
        for step in common:
            x, y = a[step], b[step]
            act_off, act_on = float(x["action"]), float(y["action"])
            theta_off = float(x["theta_before"])
            theta_on = float(y["theta_before"])
            omega_off = float(x["omega_before"])
            omega_on = float(y["omega_before"])
            records.append({
                "step": step,
                "a_off": act_off,
                "a_on": act_on,
                "action_diff": abs(act_off - act_on),
                "state_angle_diff": angular_distance(theta_off, theta_on),
                "state_omega_diff": abs(omega_off - omega_on),
                "return_diff": float(y["cumulative_return"]) - float(x["cumulative_return"]),
            })
        first_action = first_above(records, "action_diff", args.action_tol)
        first_angle = first_above(records, "state_angle_diff", args.state_tol)
        first_omega = first_above(records, "state_omega_diff", args.state_tol)
        last = records[-1]
        print(f"\n=== Seed {seed}: {len(records)} paired steps ===")
        print(f"First |action difference| > {args.action_tol}: "
              f"{first_action['step'] if first_action else 'not observed'}")
        print(f"First |angle difference| > {args.state_tol}: "
              f"{first_angle['step'] if first_angle else 'not observed'}")
        print(f"First |omega difference| > {args.state_tol}: "
              f"{first_omega['step'] if first_omega else 'not observed'}")
        print(f"Final return (ON minus OFF): {last['return_diff']:+.6f}")
        if first_action:
            idx = records.index(first_action)
            lo = max(0, idx - 2)
            hi = min(len(records), idx + args.show)
            print(" step | action OFF | action ON | abs delta | angle diff | omega diff | return ON-OFF")
            for row in records[lo:hi]:
                print(f"{row['step']:5d} | {row['a_off']:+10.5f} | "
                      f"{row['a_on']:+9.5f} | {row['action_diff']:9.5f} | "
                      f"{row['state_angle_diff']:10.6f} | {row['state_omega_diff']:10.6f} | "
                      f"{row['return_diff']:+13.5f}")
        else:
            print("No action difference above tolerance; inspect smaller tolerances if needed.")


if __name__ == "__main__":
    main()
