"""Experiment 06 follow-up: paired closed-loop diagnostics without retraining.

Run from repository root:
  python src/diagnose_experiment06.py --seeds 70002 70006
Requires src/experiment06.py and the two checkpoints from Experiment 06.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment06 import load_base, predict, mpc_action

VARIANTS = {
    'baseline': ('base', False),
    'projection_only': ('base', True),
    'multistep_only': ('tuned', False),
    'multistep_plus_projection': ('tuned', True),
}


def get_models(base_path: Path, tuned_path: Path, device: torch.device):
    if not base_path.is_file():
        raise FileNotFoundError(f'Missing baseline checkpoint: {base_path}')
    if not tuned_path.is_file():
        raise FileNotFoundError(f'Missing multi-step checkpoint: {tuned_path}')
    base, norm = load_base(base_path, device)
    tuned, tuned_norm = load_base(tuned_path, device)
    # Check that both models use the original normalization statistics.
    for key in norm:
        if not torch.allclose(norm[key], tuned_norm[key], rtol=0, atol=0):
            raise ValueError(f'Normalization mismatch for {key}; cannot compare safely.')
    base.eval(); tuned.eval()
    return {'base': base, 'tuned': tuned}, norm


def angle(obs):
    return float(np.arctan2(float(obs[1]), float(obs[0])))


def wrapped_angle_diff(a, b):
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


@torch.no_grad()
def one_episode(seed, label, model, constrained, norm, device, horizon, candidates, steps):
    # This matches Experiment 06's initial-state seed and candidate generator.
    env = gym.make('Pendulum-v1')
    state, _ = env.reset(seed=seed)
    generator = torch.Generator(device='cpu').manual_seed(seed + 900000)
    rows = []
    total_return = 0.0
    try:
        for t in range(steps):
            before = np.asarray(state, dtype=np.float32).copy()
            st = torch.as_tensor(before, dtype=torch.float32, device=device)
            action = mpc_action(st, model, norm, horizon, candidates, generator, constrained)
            act_array = np.asarray([action], dtype=np.float32)
            # One-step prediction is diagnostic ONLY. The MPC action selection above
            # already performs its own internal candidate rollouts.
            predicted = predict(model, norm, st.unsqueeze(0),
                                torch.as_tensor(act_array, device=device).view(1, 1),
                                constrained=constrained).squeeze(0).cpu().numpy()
            after, reward, terminated, truncated, _ = env.step(act_array)
            after = np.asarray(after, dtype=np.float32).copy()
            total_return += float(reward)
            theta_actual = angle(after)
            theta_pred = angle(predicted)
            err = predicted - after
            rows.append({
                'seed': seed, 'variant': label, 'step': t,
                'theta_before': angle(before),
                'omega_before': float(before[2]),
                'action': float(action),
                'theta_actual_next': theta_actual,
                'omega_actual_next': float(after[2]),
                'theta_pred_next': theta_pred,
                'omega_pred_next': float(predicted[2]),
                'reward': float(reward),
                'cumulative_return': total_return,
                'one_step_state_mse': float(np.mean(err ** 2)),
                'one_step_angle_abs_error_rad': abs(wrapped_angle_diff(theta_pred, theta_actual)),
                'one_step_omega_abs_error': abs(float(predicted[2] - after[2])),
                'predicted_radius_error': abs(float(np.linalg.norm(predicted[:2])) - 1.0),
                'is_stable_next': int(abs(theta_actual) < 0.2 and abs(float(after[2])) < 0.5),
            })
            state = after
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows, float(total_return)


def episode_summary(rows, ret):
    stable = np.asarray([r['is_stable_next'] for r in rows], dtype=np.float32)
    return {
        'total_return': ret,
        'initial_theta_rad': rows[0]['theta_before'],
        'initial_omega_rad_s': rows[0]['omega_before'],
        'steps': len(rows),
        'tail_50_stable_fraction': float(stable[-50:].mean()),
        'mean_one_step_state_mse': float(np.mean([r['one_step_state_mse'] for r in rows])),
        'mean_one_step_angle_abs_error_rad': float(np.mean([r['one_step_angle_abs_error_rad'] for r in rows])),
        'mean_one_step_omega_abs_error': float(np.mean([r['one_step_omega_abs_error'] for r in rows])),
        'mean_predicted_radius_error': float(np.mean([r['predicted_radius_error'] for r in rows])),
        'first_10_consecutive_stable_step': first_stable_run(stable, 10),
    }


def first_stable_run(flags, window):
    if len(flags) < window:
        return None
    for i in range(len(flags)-window+1):
        if bool(np.all(flags[i:i+window])):
            return int(i)
    return None


def plot_seed(seed, by_variant, output):
    panels = [
        ('theta_actual_next', 'Wrapped angle (rad)'),
        ('omega_actual_next', 'Angular velocity (rad/s)'),
        ('action', 'Applied torque'),
        ('cumulative_return', 'Cumulative episode return'),
        ('one_step_angle_abs_error_rad', 'One-step angular abs error (rad)'),
        ('one_step_omega_abs_error', 'One-step omega abs error (rad/s)'),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, (field, ylabel) in zip(axs.flat, panels):
        for label, rows in by_variant.items():
            ax.plot([r['step'] for r in rows], [r[field] for r in rows], label=label, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.set_xlabel('Environment step')
    axs[0,0].legend(fontsize=8, loc='best')
    fig.suptitle(f'Experiment 06 closed-loop diagnostics — seed {seed}')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, default=ROOT/'runs/one_step/pendulum_world_model.pt')
    parser.add_argument('--multistep-checkpoint', type=Path, default=ROOT/'results/experiment_06/multistep_world_model.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'results/experiment_06/diagnostics')
    parser.add_argument('--seeds', type=int, nargs='+', default=[70002,70006])
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=256)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--reference-metrics', type=Path, default=ROOT/'results/experiment_06/metrics.json')
    args = parser.parse_args()
    if min(args.horizon, args.candidates, args.steps) < 1:
        parser.error('horizon, candidates and steps must be positive integers')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, norm = get_models(args.base_checkpoint, args.multistep_checkpoint, device)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = {}
    if args.reference_metrics.exists():
        original = json.loads(args.reference_metrics.read_text())
        for label, values in original.get('mpc', {}).items():
            reference[label] = {int(x['seed']): x['return'] for x in values.get('per_seed', [])}
    all_rows = []
    summaries = {}
    print(f'Device={device}; seeds={args.seeds}; H={args.horizon}; candidates={args.candidates}', flush=True)
    for seed in args.seeds:
        by_variant = {}
        summaries[str(seed)] = {}
        for label, (model_id, constrained) in VARIANTS.items():
            rows, ret = one_episode(seed, label, models[model_id], constrained,
                                    norm, device, args.horizon, args.candidates, args.steps)
            by_variant[label] = rows
            all_rows.extend(rows)
            s = episode_summary(rows, ret)
            summaries[str(seed)][label] = s
            ref = reference.get(label, {}).get(seed)
            extra = f' | original={ref:.3f} difference={ret-ref:+.6f}' if ref is not None else ''
            print(f'seed={seed} {label:26s} return={ret:10.3f} stable_tail={s["tail_50_stable_fraction"]:.2f}' + extra, flush=True)
        plot_seed(seed, by_variant, args.output/f'seed_{seed}_diagnostics.png')
    if not all_rows:
        raise RuntimeError('No transitions were collected')
    with (args.output/'trajectories.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader();w.writerows(all_rows)
    metadata = {
        'experiment': '06-closing-loop-diagnostics',
        'model_training': 'None; loaded saved weights only',
        'base_checkpoint': str(args.base_checkpoint),
        'multistep_checkpoint': str(args.multistep_checkpoint),
        'horizon':args.horizon,'candidates':args.candidates,'seeds':args.seeds,
        'notes': [
            'Matches Experiment 06 MPC generator seed+900000 and random-shooting implementation.',
            'One-step prediction errors are measured on policy-generated transitions, not random-action evaluation rollouts.',
            'Wrapped-angle discontinuities at +/-pi are representational, not physical jumps.',
            'Comparison to original return may differ if the local checkpoints or software have changed.',
            'Observed differences do not by themselves identify a causal mechanism.',
        ],
        'per_seed':summaries,
    }
    (args.output/'metrics.json').write_text(json.dumps(metadata, indent=2))
    print(f'Files saved to: {args.output}', flush=True)

if __name__=='__main__':
    main()
