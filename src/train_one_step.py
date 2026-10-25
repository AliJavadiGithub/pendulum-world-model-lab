"""Train/evaluate a one-step neural dynamics model on Pendulum-v1."""
import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def collect_data(episodes: int, seed: int, horizon: int = 200):
    env = gym.make("Pendulum-v1")
    inputs, targets = [], []
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        for _ in range(horizon):
            action = rng.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)
            next_state, _, terminated, truncated, _ = env.step(action)
            inputs.append(np.concatenate((state, action)))
            targets.append(next_state - state)
            state = next_state
            if terminated or truncated:
                break
    env.close()
    return (torch.tensor(np.asarray(inputs), dtype=torch.float32),
            torch.tensor(np.asarray(targets), dtype=torch.float32))


def make_model():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))


def evaluate(model, X, Y, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        pred_norm = model(((X - x_mean) / x_std).to(device)).cpu()
        y_norm = (Y - y_mean) / y_std
        normalized_mse = ((pred_norm - y_norm) ** 2).mean().item()
        pred_delta = pred_norm * y_std + y_mean
        per_dim_mse = ((pred_delta - Y) ** 2).mean(dim=0).tolist()
        mse = ((pred_delta - Y) ** 2).mean().item()
        baseline_mse = (Y ** 2).mean().item()
    return dict(normalized_mse=normalized_mse, original_scale_mse=mse,
                per_component_mse=dict(zip(['cos_theta', 'sin_theta', 'angular_velocity'], per_dim_mse)),
                no_change_baseline_mse=baseline_mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-episodes', type=int, default=100)
    parser.add_argument('--test-episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/one_step'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    print('Collecting data...', flush=True)
    X_train, Y_train = collect_data(args.train_episodes, args.seed)
    X_test, Y_test = collect_data(args.test_episodes, 10000)
    x_mean, x_std = X_train.mean(0), X_train.std(0).clamp_min(1e-6)
    y_mean, y_std = Y_train.mean(0), Y_train.std(0).clamp_min(1e-6)
    Xn, Yn = (X_train - x_mean) / x_std, (Y_train - y_mean) / y_std
    loader = DataLoader(TensorDataset(Xn, Yn), batch_size=256, shuffle=True)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        train_loss = total / len(X_train)
        history.append({'epoch': epoch, 'train_normalized_mse': train_loss})
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch}/{args.epochs} - train normalized MSE: {train_loss:.6f}', flush=True)
    metrics = {
        'model': 'MLP(4,128,128,3), ReLU; predicts delta_state',
        'environment': 'Pendulum-v1',
        'seed': args.seed,
        'device': str(device),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'train': evaluate(model, X_train, Y_train, x_mean, x_std, y_mean, y_std, device),
        'test': evaluate(model, X_test, Y_test, x_mean, x_std, y_mean, y_std, device),
    }
    torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std,
                'y_mean': y_mean, 'y_std': y_std}, args.output_dir / 'pendulum_world_model.pt')
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    print(json.dumps(metrics['test'], indent=2))
    print(f'Artifacts saved to {args.output_dir}')


if __name__ == '__main__':
    main()
