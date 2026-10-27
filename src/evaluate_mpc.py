"""Experiment 03: random-shooting MPC with a learned Pendulum-v1 world model.

Run from repository root: python src/evaluate_mpc.py
Requires the checkpoint produced by src/train_one_step.py.
"""
import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def make_model():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))


def load_world_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    stats = [checkpoint[k].to(device) for k in ('x_mean', 'x_std', 'y_mean', 'y_std')]
    return model, stats


@torch.inference_mode()
def choose_mpc_action(state, model, stats, *, rng, device, horizon, candidates):
    """Vectorized random-shooting; use model for planning, simulator only for execution."""
    action_sequences = rng.uniform(-2., 2., (candidates, horizon, 1)).astype(np.float32)
    actions = torch.from_numpy(action_sequences).to(device)
    states = torch.as_tensor(state, dtype=torch.float32, device=device).expand(candidates, -1).clone()
    x_mean, x_std, y_mean, y_std = stats
    total_cost = torch.zeros(candidates, device=device)
    for t in range(horizon):
        act = actions[:, t, :]
        inp = torch.cat((states, act), dim=-1)
        delta = model((inp - x_mean) / x_std) * y_std + y_mean
        states = states + delta
        # This is the Gymnasium Pendulum cost, evaluated on predicted next state.
        angle = torch.atan2(states[:, 1], states[:, 0])
        velocity = states[:, 2]
        total_cost += angle.square() + 0.1 * velocity.square() + 0.001 * act[:, 0].square()
    best = int(total_cost.argmin().item())
    return action_sequences[best, 0].copy()


def run_episode(policy, seed, model, stats, args, device):
    env = gym.make('Pendulum-v1')
    state, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 123456)
    total_reward, records = 0.0, []
    for step in range(args.steps):
        if policy == 'mpc':
            action = choose_mpc_action(state, model, stats, rng=rng, device=device,
                                       horizon=args.horizon, candidates=args.candidates)
        elif policy == 'random':
            action = rng.uniform(-2., 2., (1,)).astype(np.float32)
        else:
            action = np.zeros(1, dtype=np.float32)
        next_state, reward, terminated, truncated, _ = env.step(action)
        theta = float(np.arctan2(next_state[1], next_state[0]))
        records.append({'policy': policy, 'seed': seed, 'step': step + 1,
                        'theta_rad': theta, 'angular_velocity': float(next_state[2]),
                        'action': float(action[0]), 'reward': float(reward)})
        total_reward += float(reward)
        state = next_state
        if terminated or truncated:
            break
    env.close()
    return total_reward, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/one_step/pendulum_world_model.pt')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/mpc')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--candidates', type=int, default=256)
    args = p.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f'Checkpoint not found: {args.checkpoint}\n'
                         'First run: python src/train_one_step.py\n'
                         'Or supply: --checkpoint /path/to/pendulum_world_model.pt')
    if min(args.episodes, args.steps, args.horizon, args.candidates) < 1:
        p.error('All counts must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; checkpoint: {args.checkpoint}', flush=True)
    model, stats = load_world_model(args.checkpoint, device)
    policies = ('zero', 'random', 'mpc')
    data, rows = [], []
    for seed in range(21000, 21000 + args.episodes):
        for policy in policies:
            reward, trace = run_episode(policy, seed, model, stats, args, device)
            data.append({'policy': policy, 'seed': seed, 'total_reward': reward})
            rows.extend(trace)
            print(f'{policy:6s} seed={seed} total_reward={reward:.2f}', flush=True)
    summary = {}
    for policy in policies:
        rewards = [d['total_reward'] for d in data if d['policy'] == policy]
        summary[policy] = {'mean_return': float(np.mean(rewards)),
                           'std_return': float(np.std(rewards)), 'returns': rewards}
    output = {'setup': {'episodes': args.episodes, 'steps': args.steps,
                        'horizon': args.horizon, 'candidates': args.candidates,
                        'seeds': list(range(21000, 21000 + args.episodes)),
                        'checkpoint': str(args.checkpoint), 'device': str(device)},
              'summary': summary}
    (args.output_dir / 'metrics.json').write_text(json.dumps(output, indent=2) + '\n')
    with (args.output_dir / 'trajectories.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['policy', 'seed', 'step', 'theta_rad',
                                              'angular_velocity', 'action', 'reward'])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(policies)
    means = [summary[x]['mean_return'] for x in names]
    errors = [summary[x]['std_return'] for x in names]
    ax.bar(names, means, yerr=errors, capsize=4)
    ax.set_ylabel('Episode return (higher is better)')
    ax.set_title('Pendulum-v1: baseline vs learned-model MPC')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'returns.png', dpi=150)
    plt.close(fig)
    seed = 21000
    fig, ax = plt.subplots(figsize=(10, 4))
    for policy in policies:
        subset = [r for r in rows if r['seed'] == seed and r['policy'] == policy]
        ax.plot([r['step'] for r in subset], [r['theta_rad'] for r in subset], label=policy)
    ax.set(xlabel='Time step', ylabel='Angle (rad)', title=f'Trajectories (seed {seed})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / 'angles.png', dpi=150)
    plt.close(fig)
    print('\nAverage returns (higher is better):', flush=True)
    for name in policies:
        print(f"{name}: {summary[name]['mean_return']:.2f} ± {summary[name]['std_return']:.2f}")
    print(f'Files saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
