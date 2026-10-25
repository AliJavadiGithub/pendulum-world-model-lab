# Pendulum World Model Lab

A reproducible, beginner-to-research learning log for **one-step neural dynamics prediction** using PyTorch, Gymnasium `Pendulum-v1`, and an NVIDIA GPU. The longer-term roadmap is multi-step rollouts, model predictive control (MPC), latent world models, and vision-language-action (VLA) policies.

## Progress

- [x] Set up Ubuntu / virtual environment and CUDA-enabled PyTorch
- [x] Train a neural one-step world model (first experimental run)
- [x] Evaluate one-step MSE against a no-change baseline
- [ ] Evaluate multi-step, open-loop rollouts
- [ ] Add an MPC controller
- [ ] Explore visual/latent world models
- [ ] Explore small VLAs

## Environment

Initial experiment machine: Lenovo LOQ 15IRH8; Ubuntu 22.04.5 LTS; Intel Core i5-13420H; ~16 GB RAM; NVIDIA GeForce RTX 2050 4 GB. Verified with PyTorch `2.11.0+cu126`, CUDA accessible. Python 3.10.12.

## Problem formulation

The Pendulum observation is `s_t = [cos(theta), sin(theta), angular_velocity]` and the continuous torque action is `a_t in [-2, 2]`. The network consumes `[s_t, a_t]` (4 numbers) and predicts the state difference `delta_s_t = s_(t+1) - s_t` (3 numbers). The next-state estimate is `s_hat_(t+1) = s_t + delta_s_hat_t`.

The model is an MLP `4 -> 128 -> 128 -> 3` with ReLU, trained using Adam (lr `1e-3`), batch size 256 and MSE on normalized state deltas. Training normalization statistics come **only** from the training split; test episodes use different environment reset seeds.

## Setup (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch version appropriate to your OS/GPU from:
# https://pytorch.org/get-started/locally/
# The original machine successfully ran torch==2.11.0+cu126.
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

If CUDA wheels for a particular PyTorch version are no longer offered for your Python version, use the current official install selector rather than changing NVIDIA drivers unnecessarily. CPU training works as a fallback.

## Reproduce the current pipeline

From the repository root:

```bash
python src/train_one_step.py
```

Outputs (excluded from Git as generated artifacts):

- `runs/one_step/pendulum_world_model.pt` — weights + normalization statistics
- `runs/one_step/metrics.json` — train/test metrics on both scales
- `runs/one_step/history.json` — per-epoch training loss

Plot the **historical, user-reported** first run:

```bash
python src/plot_first_run.py
```

It generates `results/first_run_training_curve.png` from six recorded training-loss points. This is **not** a reconstruction of every epoch.

## Recorded first run (2026-10-25)

The first successful run used the original `train_world_model.py` tutorial script, preserved as [`src/original_train_world_model.py`](src/original_train_world_model.py). These are actual terminal outputs transcribed from that run; they are **not claims of re-execution by this repository**:

| Metric | Observed value |
|---|---:|
| Training MSE at epoch 5 (normalized delta scale) | 0.001486 |
| Training MSE at epoch 10 (normalized delta scale) | 0.000817 |
| Training MSE at epoch 15 (normalized delta scale) | 0.000762 |
| Training MSE at epoch 20 (normalized delta scale) | 0.000629 |
| Training MSE at epoch 25 (normalized delta scale) | 0.000560 |
| Training MSE at epoch 30 (normalized delta scale) | 0.000557 |
| Test one-step MSE (original state scale) | 0.0001419463 |
| No-change baseline MSE (original state scale) | 0.1061588004 |

The model's test one-step MSE was about 748x **lower** than the no-change baseline MSE on the reported dataset. This is a useful sanity check, **not** proof of long-horizon accuracy, safety or superiority to strong physics-informed baselines.

### Understanding the metrics

- **Train normalized MSE** compares normalized predicted state deltas to normalized true deltas.
- **Test original-scale MSE** compares predicted next states to true next states in the environment's state representation. This is algebraically identical to original-scale delta MSE.
- These two MSE values use **different scales**, so do not compare their magnitudes as if they were the same metric. The new training script calculates both train and test errors in the same spaces for fair comparison.
- **No-change baseline** predicts `s_hat_(t+1) = s_t`, or equivalently `delta_s_hat_t = 0`; its MSE is `mean((0 - delta_s_t)^2) = mean(Y_test^2)`. No learned network is required for this prediction.
- State components differ physically (`cos(theta)`, `sin(theta)`, and `rad/s`), so also inspect per-component MSE; aggregate MSE can hide a large error in one component.

## Experimental protocol and limitations

1. Training and testing use distinct Pendulum episode seeds. The original historical run relied on Gymnasium `action_space.sample()` without explicitly seeding the action sampler, so its exact results may not be reproducible.
2. The maintained `src/train_one_step.py` uses an explicitly seeded NumPy generator to sample actions, so it is more reproducible; exact metrics may differ from the historical run.
3. The model predicts one step only. Multi-step open-loop performance remains **untested** in this repository.
4. The predicted cosine and sine are not explicitly projected back onto the unit circle.
5. Random-action data coverage and the weak no-change baseline are limited evaluation choices. Future versions should add fixed evaluation trajectories, stronger baselines, rollout tests and uncertainty estimates.
6. GPU training may still vary slightly across environments/hardware despite fixed seeds.

## Repo layout

```text
pendulum-world-model-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train_one_step.py
│   ├── original_train_world_model.py
│   └── plot_first_run.py
├── results/
│   └── first_run_observed.json
└── docs/
    └── roadmap.md
```

`results/first_run_training_curve.png` will be created when you run the plotting command. `runs/` checkpoints are intentionally ignored by Git because binary model files can grow quickly.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for milestones and suggested experiment tracking.

## License

MIT (see [LICENSE](LICENSE)).
