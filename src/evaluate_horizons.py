"""Experiment 04: paired MPC horizon ablation for Pendulum-v1.

From repository root:
    python src/evaluate_horizons.py --episodes 10 --horizons 5 10 20

Uses Experiment 03's model loader; no retraining or modification of past results.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_mpc import load_world_model

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def plan(state, action_sequences, model, stats, device):
    """Evaluate given candidate sequences with learned dynamics, never simulator."""
    actions = torch.from_numpy(action_sequences).to(device)
    candidates, horizon, _ = actions.shape
    states = torch.as_tensor(state, dtype=torch.float32, device=device).expand(candidates, -1).clone()
    x_mean, x_std, y_mean, y_std = stats
    costs = torch.zeros(candidates, device=device)
    for t in range(horizon):
        act = actions[:, t, :]
        inp = torch.cat((states, act), dim=-1)
        delta = model((inp - x_mean) / x_std) * y_std + y_mean
        states = states + delta
        theta = torch.atan2(states[:, 1], states[:, 0])
        velocity = states[:, 2]
        costs += theta.square() + 0.1 * velocity.square() + 0.001 * act[:, 0].square()
    return action_sequences[int(costs.argmin().item()), 0].copy()


def evaluate(seed, horizon, candidate_bank, model, stats, device, steps):
    env = gym.make("Pendulum-v1")
    state, _ = env.reset(seed=seed)
    init_theta = float(np.arctan2(state[1], state[0]))
    init_vel = float(state[2])
    rows = []
    total_return = 0.0
    for step in range(steps):
        start = time.perf_counter()
        act = plan(state, candidate_bank[step, :, :horizon, :], model, stats, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        planning_ms = (time.perf_counter() - start) * 1000.0
        next_state, reward, terminated, truncated, _ = env.step(act)
        theta = float(np.arctan2(next_state[1], next_state[0]))
        vel = float(next_state[2])
        rows.append(dict(seed=seed, horizon=horizon, step=step+1,
                         theta_rad=theta, angular_velocity=vel,
                         action=float(act[0]), reward=float(reward),
                         planning_ms=planning_ms))
        total_return += float(reward)
        state = next_state
        if terminated or truncated:
            break
    env.close()
    tail = rows[-min(50,len(rows)):]
    fraction_stable = sum(abs(r['theta_rad']) < 0.2 and
                          abs(r['angular_velocity']) < 0.5 for r in tail) / len(tail)
    return dict(seed=seed, horizon=horizon, initial_theta_rad=init_theta,
                initial_angular_velocity=init_vel, total_return=total_return,
                final_50_stable_fraction=fraction_stable,
                avg_planning_ms=float(np.mean([r['planning_ms'] for r in rows]))), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/one_step/pendulum_world_model.pt')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/mpc_horizon_ablation')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=21000)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--candidates', type=int, default=256)
    p.add_argument('--horizons', type=int, nargs='+', default=[5,10,20])
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f'Checkpoint not found: {args.checkpoint}')
    horizons = sorted(set(args.horizons))
    if min(args.episodes,args.steps,args.candidates,*horizons) < 1:
        p.error('Counts and horizons must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,stats=load_world_model(args.checkpoint,device)
    print(f'Using {device}; checkpoint={args.checkpoint}', flush=True)
    print('Paired seeds:', list(range(args.seed_start,args.seed_start+args.episodes)), flush=True)
    records,trajectories=[],[]
    # Each horizon receives identical candidate prefixes at every decision
    # and matching initial states. Actual state trajectories subsequently differ.
    for seed in range(args.seed_start,args.seed_start+args.episodes):
        rng=np.random.default_rng(seed+98765)
        bank=rng.uniform(-2,2,size=(args.steps,args.candidates,max(horizons),1)).astype(np.float32)
        for horizon in horizons:
            result,rows=evaluate(seed,horizon,bank,model,stats,device,args.steps)
            records.append(result)
            trajectories.extend(rows)
            print(f'seed={seed} H={horizon:2d} return={result["total_return"]:8.2f} '
                  f'stable_tail={result["final_50_stable_fraction"]:.2f} '
                  f'planning_ms={result["avg_planning_ms"]:.1f}',flush=True)
    summary={}
    for h in horizons:
        r=[x for x in records if x['horizon']==h]
        returns=np.array([x['total_return'] for x in r])
        summary[str(h)]={'mean_return':float(returns.mean()),
                         'std_return_population':float(returns.std()),
                         'median_return':float(np.median(returns)),
                         'mean_final_50_stable_fraction':float(np.mean([x['final_50_stable_fraction'] for x in r])),
                         'mean_planning_ms':float(np.mean([x['avg_planning_ms'] for x in r]))}
    out={'setup':{'episodes':args.episodes,'steps':args.steps,'candidates':args.candidates,
                  'horizons':horizons,'seed_start':args.seed_start,
                  'checkpoint':str(args.checkpoint),'device':str(device),
                  'stable_condition':'abs(theta)<0.2 rad and abs(omega)<0.5 rad/s; last 50 steps',
                  'cost':'sum(theta_next^2+0.1*omega_next^2+0.001*action^2)'},
         'summary':summary,'per_episode':records}
    (args.output_dir/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    with (args.output_dir/'trajectories.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','horizon','step','theta_rad','angular_velocity','action','reward','planning_ms'])
        w.writeheader(); w.writerows(trajectories)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for h in horizons:
        r=[x for x in records if x['horizon']==h]
        ax.scatter([h]*len(r),[x['total_return'] for x in r],alpha=.45)
    ax.plot(horizons,[summary[str(h)]['mean_return'] for h in horizons],marker='o',label='Mean return')
    ax.set(xlabel='Planning horizon (steps)',ylabel='Episode return (higher is better)',title='Learned-model MPC: horizon ablation')
    ax.set_xticks(horizons); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(args.output_dir/'horizon_returns.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for h in horizons:
        trace=[r for r in trajectories if r['seed']==21002 and r['horizon']==h]
        if trace: ax.plot([r['step'] for r in trace],[r['theta_rad'] for r in trace],label=f'H={h}')
    ax.set(xlabel='Step',ylabel='Wrapped angle (rad)',title='Seed 21002: trajectory by horizon')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(args.output_dir/'seed_21002_angles.png',dpi=150);plt.close(fig)
    print('\nSummary (higher return is better):')
    for h in horizons:
        s=summary[str(h)]
        print(f'H={h:2d}: return={s["mean_return"]:.2f} ± {s["std_return_population"]:.2f}; '
              f'median={s["median_return"]:.2f}; tail stable={s["mean_final_50_stable_fraction"]:.2%}; '
              f'planning={s["mean_planning_ms"]:.1f} ms')
    print(f'Saved to {args.output_dir}')

if __name__=='__main__':
    main()
