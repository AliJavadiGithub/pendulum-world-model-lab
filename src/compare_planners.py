"""Experiment 05: random shooting versus CEM for learned-model MPC.
Run from repository root: python src/compare_planners.py --episodes 3
"""
import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3)).to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    stats = [ckpt[key].to(device) for key in ('x_mean', 'x_std', 'y_mean', 'y_std')]
    return net, stats


@torch.inference_mode()
def evaluate_sequences(state, seq, net, stats, device):
    """Return cost for B action sequences of shape [B,H,1]."""
    actions = torch.from_numpy(seq).to(device)
    batch, horizon, _ = actions.shape
    states = torch.as_tensor(state, device=device, dtype=torch.float32).expand(batch, -1).clone()
    x_mean, x_std, y_mean, y_std = stats
    costs = torch.zeros(batch, device=device)
    for t in range(horizon):
        action = actions[:, t, :]
        inp = torch.cat((states, action), dim=-1)
        delta = net((inp - x_mean) / x_std) * y_std + y_mean
        states = states + delta
        angle = torch.atan2(states[:, 1], states[:, 0])
        costs += angle.square() + 0.1 * states[:, 2].square() + 0.001 * action[:, 0].square()
    return costs.cpu().numpy()


def plan(state, method, net, stats, rng, device, horizon, budget, iterations, elite_frac):
    best_seq, best_cost = None, np.inf
    if method == 'random':
        seq = rng.uniform(-2., 2., (budget, horizon, 1)).astype(np.float32)
        costs = evaluate_sequences(state, seq, net, stats, device)
        index = int(np.argmin(costs))
        best_seq, best_cost = seq[index].copy(), float(costs[index])
    else:
        per_round = budget // iterations
        elite_count = max(2, int(np.ceil(per_round * elite_frac)))
        mean = np.zeros((horizon, 1), dtype=np.float32)
        std = np.full((horizon, 1), 2.0, dtype=np.float32)
        for round_idx in range(iterations):
            if round_idx == 0:
                # Same initial proposal distribution as Random Shooting.
                seq = rng.uniform(-2., 2., (per_round, horizon, 1)).astype(np.float32)
            else:
                seq = rng.normal(mean, std, (per_round, horizon, 1)).astype(np.float32)
                np.clip(seq, -2., 2., out=seq)
            costs = evaluate_sequences(state, seq, net, stats, device)
            rank = np.argsort(costs)
            idx = int(rank[0])
            if float(costs[idx]) < best_cost:
                best_cost, best_seq = float(costs[idx]), seq[idx].copy()
            elite = seq[rank[:elite_count]]
            mean = elite.mean(axis=0).astype(np.float32)
            std = np.maximum(elite.std(axis=0), 0.1).astype(np.float32)
    return float(best_seq[0, 0]), best_cost


def run_episode(seed, method, horizon, net, stats, device, args):
    env = gym.make('Pendulum-v1')
    state, _ = env.reset(seed=seed)
    init_theta = float(np.arctan2(state[1], state[0]))
    init_velocity = float(state[2])
    # Independent and reproducible RNG per (seed, method, horizon).
    method_code = 0 if method == 'random' else 1
    rng = np.random.default_rng(np.random.SeedSequence([args.planner_seed, seed, horizon, method_code]))
    rows = []
    total_reward = 0.0
    for step in range(args.steps):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        action_value, predicted_cost = plan(state, method, net, stats, rng, device,
                                            horizon, args.budget, args.iterations, args.elite_frac)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        planning_ms = (time.perf_counter() - start) * 1000.0
        action = np.array([action_value], dtype=np.float32)
        next_state, reward, terminated, truncated, _ = env.step(action)
        theta = float(np.arctan2(next_state[1], next_state[0]))
        velocity = float(next_state[2])
        rows.append(dict(seed=seed, method=method, horizon=horizon, step=step + 1,
                         theta_rad=theta, angular_velocity=velocity, action=action_value,
                         reward=float(reward), planning_ms=planning_ms, predicted_cost=predicted_cost))
        total_reward += float(reward)
        state = next_state
        if terminated or truncated:
            break
    env.close()
    tail = rows[-50:]
    stable_fraction = float(np.mean([abs(r['theta_rad']) < 0.2 and abs(r['angular_velocity']) < 0.5 for r in tail]))
    info = dict(seed=seed, method=method, horizon=horizon, initial_theta_rad=init_theta,
                initial_angular_velocity=init_velocity, total_return=total_reward,
                final_50_stable_fraction=stable_fraction,
                avg_planning_ms=float(np.mean([r['planning_ms'] for r in rows])))
    return info, rows


def write_outputs(args, records, trajectories):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for h in args.horizons:
        summary[str(h)] = {}
        for method in ('random', 'cem'):
            subset = [r for r in records if r['horizon'] == h and r['method'] == method]
            ret = np.array([r['total_return'] for r in subset])
            stable = np.array([r['final_50_stable_fraction'] for r in subset])
            summary[str(h)][method] = dict(mean_return=float(ret.mean()),
                                           population_sd_return=float(ret.std(ddof=0)),
                                           median_return=float(np.median(ret)),
                                           mean_tail_stable_fraction=float(stable.mean()),
                                           mean_planning_ms=float(np.mean([r['avg_planning_ms'] for r in subset])))
    payload = dict(protocol=dict(episodes=args.episodes, steps=args.steps,
                                 seeds=list(range(args.seed_start, args.seed_start + args.episodes)),
                                 horizons=args.horizons, candidates_per_decision=args.budget,
                                 cem_iterations=args.iterations, cem_elite_fraction=args.elite_frac,
                                 cem_min_std=0.1, checkpoint=str(args.checkpoint),
                                 planner_seed=args.planner_seed),
                   summary=summary, per_episode=records)
    (out / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (out / 'trajectories.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trajectories[0]))
        w.writeheader()
        w.writerows(trajectories)
    fig, axs = plt.subplots(1, len(args.horizons), figsize=(6 * len(args.horizons), 4), squeeze=False)
    for ax, h in zip(axs[0], args.horizons):
        for method in ('random', 'cem'):
            vals = [r['total_return'] for r in records if r['horizon'] == h and r['method'] == method]
            ax.scatter([method] * len(vals), vals, alpha=0.5)
            ax.plot(method, np.mean(vals), marker='D', markersize=9)
        ax.set_title(f'H={h}: return by planner')
        ax.set_ylabel('Episode return (higher is better)')
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'planner_returns.png', dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, len(args.horizons), figsize=(6 * len(args.horizons), 4), squeeze=False)
    for ax, h in zip(axs[0], args.horizons):
        for method in ('random', 'cem'):
            trace = [r for r in trajectories if r['seed'] == args.inspect_seed and r['horizon'] == h and r['method'] == method]
            if trace:
                ax.plot([r['step'] for r in trace], [r['theta_rad'] for r in trace], label=method)
        ax.set_title(f'Seed {args.inspect_seed}, H={h}')
        ax.set_xlabel('Environment step')
        ax.set_ylabel('Wrapped angle (rad)')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'seed_21002_angles.png' if args.inspect_seed == 21002 else out / f'seed_{args.inspect_seed}_angles.png', dpi=150)
    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/one_step/pendulum_world_model.pt')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/mpc_planner_comparison')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seed-start', type=int, default=21000)
    p.add_argument('--inspect-seed', type=int, default=21002)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--horizons', nargs='+', type=int, default=[10, 20])
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--elite-frac', type=float, default=0.1)
    p.add_argument('--planner-seed', type=int, default=2026)
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f'Checkpoint missing: {args.checkpoint}. Run python src/train_one_step.py first.')
    if args.episodes < 1 or args.steps < 1 or not args.horizons or min(args.horizons) < 1:
        p.error('Episode count, steps, and horizons must be positive')
    if args.budget < 2 or args.iterations < 2 or args.budget % args.iterations:
        p.error('--budget must be divisible by --iterations, both >= 2')
    if args.budget // args.iterations < 2 or not 0 < args.elite_frac <= 1:
        p.error('Invalid CEM sample count or elite fraction')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats = load_model(args.checkpoint, device)
    print(f'Device={device}; checkpoint={args.checkpoint}', flush=True)
    print(f'Equal budget: {args.budget} candidates/decision; CEM {args.iterations} x {args.budget // args.iterations}', flush=True)
    summaries, traces = [], []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        for horizon in args.horizons:
            for method in ('random', 'cem'):
                record, trace = run_episode(seed, method, horizon, model, stats, device, args)
                summaries.append(record)
                traces.extend(trace)
                print(f'seed={seed} H={horizon:2d} {method:6s} return={record["total_return"]:9.2f} '
                      f'tail={record["final_50_stable_fraction"]:.2f} plan_ms={record["avg_planning_ms"]:.1f}', flush=True)
    summary = write_outputs(args, summaries, traces)
    print('\nSummary:')
    for h, group in summary.items():
        for method, v in group.items():
            print(f'H={h:>2s} {method:6s} return={v["mean_return"]:.2f} ± {v["population_sd_return"]:.2f} '
                  f'median={v["median_return"]:.2f} tail={v["mean_tail_stable_fraction"]:.1%} '
                  f'plan={v["mean_planning_ms"]:.1f}ms')
    print(f'Saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
