"""Experiment 06: one-step vs multi-step training and unit-circle projection.
Run from repository root. Tested for torch >=2.1, Gymnasium Pendulum-v1.
"""
import argparse
import json
import math
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


def model_factory(device):
    return nn.Sequential(nn.Linear(4,128), nn.ReLU(), nn.Linear(128,128),nn.ReLU(),nn.Linear(128,3)).to(device)


def load_base(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = model_factory(device)
    model.load_state_dict(ck['model'])
    norm = {k: ck[k].to(device) for k in ('x_mean','x_std','y_mean','y_std')}
    return model, norm


def project(s):
    xy = s[..., :2]
    radius = torch.linalg.vector_norm(xy, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((xy / radius, s[..., 2:]), dim=-1)


def predict(model, norm, state, action, constrained=False):
    x = torch.cat((state, action), dim=-1)
    delta = model((x - norm['x_mean']) / norm['x_std']) * norm['y_std'] + norm['y_mean']
    next_state = state + delta
    return project(next_state) if constrained else next_state


def collect_trajectories(episodes, horizon, seed):
    """Independent episode seeds, explicit action RNG; state/action pairs per episode."""
    env = gym.make('Pendulum-v1')
    rng = np.random.default_rng(seed + 500_000)
    obs_list, act_list = [], []
    for e in range(episodes):
        s, _ = env.reset(seed=seed + e)
        obs = [s.copy()]
        actions = []
        for _ in range(horizon):
            a = rng.uniform(-2.,2.,size=(1,)).astype(np.float32)
            ns, _, term, trunc, _ = env.step(a)
            actions.append(a)
            obs.append(ns.copy())
            s = ns
            if term or trunc:
                break
        if len(actions) != horizon:
            raise RuntimeError('Unexpected early episode end; decrease horizon.')
        obs_list.append(np.stack(obs)); act_list.append(np.stack(actions))
    env.close()
    return np.stack(obs_list).astype(np.float32), np.stack(act_list).astype(np.float32)


def train_multistep(baseline, norm, device, seed, args, output):
    """Finetune exact same network from one-step checkpoint with multi-step unroll loss."""
    torch.manual_seed(seed)
    states_np, acts_np = collect_trajectories(args.train_episodes, args.trajectory_steps, seed=30000)
    states = torch.from_numpy(states_np).to(device)
    acts = torch.from_numpy(acts_np).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    history = []
    h = args.train_horizon
    baseline.train()
    for epoch in range(1, args.epochs+1):
        batch_losses = []
        for _ in range(args.batches_per_epoch):
            idx = torch.randint(0, states.shape[0], (args.batch_size,), device=device)
            starts = torch.randint(0, args.trajectory_steps-h+1, (args.batch_size,), device=device)
            current = states[idx, starts]
            loss = torch.zeros((), device=device)
            for k in range(h):
                current = predict(baseline, norm, current, acts[idx, starts+k], constrained=args.train_projection)
                target = states[idx, starts+k+1]
                # normalize residual for balanced physical state dimensions, like original training
                loss = loss + torch.mean(((current-target) / norm['y_std'])**2)
            loss = loss / h
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 10.)
            optimizer.step()
            batch_losses.append(loss.item())
        avg = float(np.mean(batch_losses)); history.append({'epoch':epoch, 'loss':avg})
        print(f'Multi-step epoch {epoch:02d}/{args.epochs}: loss={avg:.6f}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':baseline.state_dict(), **{k:v.cpu() for k,v in norm.items()},
                'train_horizon':h, 'train_projection':args.train_projection}, output)
    return history


@torch.no_grad()
def evaluate_rollouts(models, norm, device, args):
    # shared evaluation data, strictly separate seeds from finetuning
    states_np, actions_np = collect_trajectories(args.eval_episodes, args.rollout_horizon, seed=60000)
    states = torch.from_numpy(states_np).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    detailed = {}
    for label, (model, constrained) in models.items():
        model.eval()
        current = states[:, 0]
        step_mse, radius_err, per_component = [], [], []
        for t in range(args.rollout_horizon):
            current = predict(model,norm,current,actions[:,t],constrained)
            error = current-states[:,t+1]
            per_component.append(error.square().mean(dim=0).cpu().tolist())
            step_mse.append(error.square().mean().item())
            radius_err.append((torch.linalg.vector_norm(current[:,:2],dim=1)-1).abs().mean().item())
        detailed[label] = {'rollout_mse':step_mse,'mean_radius_error':radius_err,'component_mse':per_component}
        checkpoints = [1,5,10,20,50,100]
        print(label, ' '.join(f'H{t}={step_mse[t-1]:.5f}' for t in checkpoints if t <= args.rollout_horizon),flush=True)
    return detailed


def reward_cost(next_states, actions):
    theta = torch.atan2(next_states[...,1],next_states[...,0])
    return theta.square() + 0.1*next_states[...,2].square()+0.001*actions[...,0].square()


@torch.no_grad()
def mpc_action(state, model, norm, horizon, candidates, generator, constrained):
    # Random shooting with fixed budget, same candidate draws for each model at each step.
    acts = (torch.rand((candidates,horizon,1),generator=generator,device='cpu')*4-2).to(state.device)
    s = state.unsqueeze(0).expand(candidates,-1)
    costs = torch.zeros(candidates,device=state.device)
    for t in range(horizon):
        s = predict(model,norm,s,acts[:,t],constrained)
        costs += reward_cost(s,acts[:,t])
    return float(acts[costs.argmin(),0,0].item())


def evaluate_control(models, norm, device, args):
    results = {}
    for label,(model,projected) in models.items():
        model.eval()
        episodes = []
        for index in range(args.mpc_episodes):
            seed = 70000+index
            env = gym.make('Pendulum-v1')
            state_np,_=env.reset(seed=seed)
            # reproducible random sequences, matched across models by fixed seed
            generator = torch.Generator(device='cpu').manual_seed(seed+900000)
            total=0.0
            for step in range(200):
                st = torch.as_tensor(state_np,dtype=torch.float32,device=device)
                act = mpc_action(st,model,norm,args.mpc_horizon,args.mpc_candidates,generator,projected)
                state_np,r,terminated,truncated,_=env.step(np.array([act],dtype=np.float32))
                total+=float(r)
                if terminated or truncated: break
            env.close()
            episodes.append({'seed':seed,'return':total})
        values=np.asarray([r['return'] for r in episodes])
        results[label]={'mean_return':float(values.mean()),'population_sd':float(values.std()),'per_seed':episodes}
        print(f'MPC {label}: {values.mean():.3f} ± {values.std():.3f}',flush=True)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/one_step/pendulum_world_model.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'results/experiment_06')
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--train-episodes',type=int,default=100)
    parser.add_argument('--trajectory-steps',type=int,default=100)
    parser.add_argument('--train-horizon',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--batches-per-epoch',type=int,default=40)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--eval-episodes',type=int,default=20)
    parser.add_argument('--rollout-horizon',type=int,default=100)
    parser.add_argument('--mpc-episodes',type=int,default=10)
    parser.add_argument('--mpc-horizon',type=int,default=10)
    parser.add_argument('--mpc-candidates',type=int,default=256)
    parser.add_argument('--skip-mpc',action='store_true')
    parser.add_argument('--train-projection',action='store_true',help='Project during multi-step finetuning; default is off to isolate training and inference projection.')
    args=parser.parse_args()
    torch.manual_seed(123); np.random.seed(123)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device={device}; checkpoint={args.checkpoint}',flush=True)
    base,norm=load_base(args.checkpoint,device)
    tuned,_=load_base(args.checkpoint,device)
    # One shared finetuned model; projection is a separate inference-only intervention
    ckpath=args.output/'multistep_world_model.pt'
    hist=train_multistep(tuned,norm,device,123,args,ckpath)
    variants={'baseline':(base,False),'projection_only':(base,True),
              'multistep_only':(tuned,False),'multistep_plus_projection':(tuned,True)}
    detailed=evaluate_rollouts(variants,norm,device,args)
    control={} if args.skip_mpc else evaluate_control(variants,norm,device,args)
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    with (args.output/'metrics.json').open('w') as f:
        json.dump({'config':config,'training_history':hist,'rollouts':detailed,'mpc':control},f,indent=2)
    plt.figure(figsize=(9,5))
    for label,d in detailed.items():
        plt.plot(np.arange(1,args.rollout_horizon+1),d['rollout_mse'],label=label)
    plt.xlabel('Rollout step');plt.ylabel('Mean state MSE')
    plt.title('Experiment 06 — Open-loop prediction error')
    plt.legend();plt.grid(alpha=.2);plt.tight_layout()
    plt.savefig(args.output/'rollout_mse.png',dpi=150);plt.close()
    plt.figure(figsize=(9,5))
    for label,d in detailed.items():
        plt.plot(np.arange(1,args.rollout_horizon+1),d['mean_radius_error'],label=label)
    plt.xlabel('Rollout step');plt.ylabel('Mean abs unit-circle radius error')
    plt.legend();plt.grid(alpha=.2);plt.tight_layout()
    plt.savefig(args.output/'radius_error.png',dpi=150);plt.close()
    print('Saved to',args.output)

if __name__=='__main__': main()
