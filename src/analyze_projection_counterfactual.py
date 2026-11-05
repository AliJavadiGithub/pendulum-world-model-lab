"""Experiment 06: fixed-state, fixed-candidate projection counterfactual.

Run from repository root:
  python src/analyze_projection_counterfactual.py --seed 70006 --steps 30 55

Uses saved diagnostic trajectories and the multi-step checkpoint. No training.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment06 import load_base, predict, reward_cost


def load_states(path: Path, seed: int, variant: str, start: int, end: int):
    """Read *pre-action* observations from diagnostic CSV, one state per step."""
    out = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if int(row['seed']) != seed or row['variant'] != variant:
                continue
            step = int(row['step'])
            if start <= step <= end:
                theta = float(row['theta_before'])
                omega = float(row['omega_before'])
                out[step] = np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)
    return out


@torch.inference_mode()
def score_sequences(state, actions, model, norm, projected):
    """Roll out every identical action sequence, returning one cost per sequence."""
    n, horizon, one = actions.shape
    assert one == 1
    s = state[None, :].expand(n, -1).clone()
    cost = torch.zeros(n, device=state.device)
    for t in range(horizon):
        s = predict(model, norm, s, actions[:, t, :], constrained=projected)
        cost += reward_cost(s, actions[:, t, :])
    return cost


def rank_correlation(x, y):
    """Spearman correlation without scipy; ranks broken deterministically by index.

    This is adequate for continuous costs, where exact ties are very unlikely.
    """
    n = len(x)
    rx = np.empty(n, dtype=np.int64)
    ry = np.empty(n, dtype=np.int64)
    rx[np.argsort(x, kind='stable')] = np.arange(n)
    ry[np.argsort(y, kind='stable')] = np.arange(n)
    return float(np.corrcoef(rx, ry)[0, 1]) if n > 1 else 1.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=70006)
    p.add_argument('--steps', type=int, nargs=2, metavar=('FIRST', 'LAST'), default=[30, 55])
    p.add_argument('--source-variant', choices=['baseline', 'projection_only', 'multistep_only', 'multistep_plus_projection'], default='multistep_only', help='Source of fixed pre-action states, not model weights')
    p.add_argument('--diagnostics', type=Path, default=ROOT/'results/experiment_06/diagnostics/trajectories.csv')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'results/experiment_06/multistep_world_model.pt')
    p.add_argument('--output', type=Path, default=ROOT/'results/experiment_06/counterfactual')
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--candidates', type=int, default=256)
    p.add_argument('--candidate-seed', type=int, default=123456)
    args = p.parse_args()
    start, end = args.steps
    if start < 0 or end < start or args.horizon < 1 or args.candidates < 2:
        p.error('Require 0 <= FIRST <= LAST, horizon >= 1, candidates >= 2')
    if not args.diagnostics.is_file():
        p.error(f'Missing diagnostic CSV: {args.diagnostics}. Run diagnose_experiment06.py first.')
    if not args.checkpoint.is_file():
        p.error(f'Missing tuned checkpoint: {args.checkpoint}. Run experiment06.py first.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, norm = load_base(args.checkpoint, device)
    model.eval()
    states = load_states(args.diagnostics, args.seed, args.source_variant, start, end)
    if not states:
        p.error(f'No states for seed={args.seed}, variant={args.source_variant}, steps={start}..{end}')

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for step in sorted(states):
        state = torch.as_tensor(states[step], device=device)
        # Recreate the SAME candidate bank for both evaluations at this state.
        # Not the bank used in the original MPC: this is a controlled replay.
        gen = torch.Generator(device='cpu').manual_seed(args.candidate_seed + args.seed * 1000 + step)
        actions = (torch.rand((args.candidates, args.horizon, 1), generator=gen, device='cpu') * 4.0 - 2.0).to(device)
        cost_off = score_sequences(state, actions, model, norm, False).cpu().numpy()
        cost_on = score_sequences(state, actions, model, norm, True).cpu().numpy()
        a0 = actions[:, 0, 0].cpu().numpy()
        off_idx = int(np.argmin(cost_off))
        on_idx = int(np.argmin(cost_on))
        # Cross-evaluate both selected sequences under both scoring rules.
        row = {
            'seed': args.seed, 'step': step, 'source_variant': args.source_variant,
            'theta_before_rad': float(np.arctan2(states[step][1], states[step][0])),
            'omega_before_rad_s': float(states[step][2]),
            'winner_index_no_projection': off_idx,
            'winner_index_with_projection': on_idx,
            'winner_changed': int(off_idx != on_idx),
            'first_torque_no_projection': float(a0[off_idx]),
            'first_torque_with_projection': float(a0[on_idx]),
            'abs_first_torque_difference': float(abs(a0[off_idx] - a0[on_idx])),
            'cost_off_at_off_winner': float(cost_off[off_idx]),
            'cost_off_at_on_winner': float(cost_off[on_idx]),
            'cost_on_at_off_winner': float(cost_on[off_idx]),
            'cost_on_at_on_winner': float(cost_on[on_idx]),
            'spearman_cost_rank': rank_correlation(cost_off, cost_on),
            'top10_overlap': int(len(set(np.argsort(cost_off)[:min(10, args.candidates)]) &
                                     set(np.argsort(cost_on)[:min(10, args.candidates)]))),
            'mean_abs_cost_change': float(np.mean(np.abs(cost_on - cost_off))),
        }
        records.append(row)
        print(f'step={step:3d} winner_changed={bool(row["winner_changed"])} '
              f'a0_off={row["first_torque_no_projection"]:+.3f} '
              f'a0_on={row["first_torque_with_projection"]:+.3f} '
              f'rank_r={row["spearman_cost_rank"]:.4f}', flush=True)

    csv_file = args.output / f'seed_{args.seed}_ranking.csv'
    with csv_file.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)

    changed = sum(r['winner_changed'] for r in records)
    summary = {
        'experiment': '06-fixed-state-projection-counterfactual',
        'seed': args.seed, 'source_variant': args.source_variant,
        'step_range': [start, end], 'steps_found': len(records),
        'horizon': args.horizon, 'candidates_per_step': args.candidates,
        'candidate_seed': args.candidate_seed,
        'changed_winner_count': changed,
        'changed_winner_fraction': changed / len(records),
        'mean_abs_first_torque_difference': float(np.mean([r['abs_first_torque_difference'] for r in records])),
        'mean_spearman_cost_rank': float(np.mean([r['spearman_cost_rank'] for r in records])),
        'mean_top10_overlap_out_of_10': float(np.mean([r['top10_overlap'] for r in records])),
        'notes': [
            'Same fixed pre-action state and same randomly drawn candidate actions under both scoring modes.',
            'Both modes use the exact same multi-step-trained model parameters.',
            'Candidate banks are newly generated reproducibly, not the original MPC internal draws.',
            'This demonstrates scoring/selection sensitivity at fixed states, not causal episode-return changes.',
            'Even different candidate indices may share similar first actions; inspect torque differences too.',
        ],
    }
    (args.output/f'seed_{args.seed}_summary.json').write_text(json.dumps(summary, indent=2))

    steps = [r['step'] for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(steps, [r['first_torque_no_projection'] for r in records], label='No projection', marker='.', linewidth=1)
    axes[0].plot(steps, [r['first_torque_with_projection'] for r in records], label='Projection', marker='.', linewidth=1)
    axes[0].set_ylabel('First selected torque'); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(steps, [r['spearman_cost_rank'] for r in records], label='Spearman rank correlation')
    axes[1].set_ylim(-1.05, 1.05); axes[1].set_ylabel('Cost ranking agreement'); axes[1].set_xlabel('Environment step'); axes[1].grid(alpha=.25)
    fig.suptitle(f'Seed {args.seed} — identical states/candidates, projection on vs off')
    fig.tight_layout()
    fig.savefig(args.output/f'seed_{args.seed}_counterfactual.png', dpi=150)
    plt.close(fig)

    print(f'Winner changed: {changed}/{len(records)} ({100*changed/len(records):.1f}%)')
    print(f'Mean |first action difference|: {summary["mean_abs_first_torque_difference"]:.4f}')
    print(f'Mean cost-rank correlation: {summary["mean_spearman_cost_rank"]:.4f}')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
