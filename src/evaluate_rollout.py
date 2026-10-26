import numpy as np
import torch
from torch import nn
import gymnasium as gym
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Same model architecture as training
model = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 3)
).to(device)

checkpoint = torch.load(
    "runs/one_step/pendulum_world_model.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model"])
model.eval()

x_mean = checkpoint["x_mean"].to(device)
x_std = checkpoint["x_std"].to(device)
y_mean = checkpoint["y_mean"].to(device)
y_std = checkpoint["y_std"].to(device)

@torch.no_grad()
def predict_next(state, action):
    x = torch.cat([state, action])
    x_normalized = (x - x_mean) / x_std

    delta_normalized = model(x_normalized)
    delta = delta_normalized * y_std + y_mean

    return state + delta

# Evaluate on 10 independent trajectories
horizon = 100
num_episodes = 10
all_errors = []
example_real = None
example_pred = None

env = gym.make("Pendulum-v1")
rng = np.random.default_rng(123)

for episode in range(num_episodes):
    state, _ = env.reset(seed=20000 + episode)

    predicted_state = torch.tensor(
        state, dtype=torch.float32, device=device
    )

    real_states = [state.copy()]
    pred_states = [state.copy()]

    for step in range(horizon):
        # Same action applied to simulator and world model
        action_np = rng.uniform(-2, 2, size=1).astype(
            np.float32
        )
        action = torch.tensor(action_np, device=device)
        
        next_state, _, terminated, truncated, _ = env.step(
            action_np
        )

        predicted_state = predict_next(
            predicted_state, action
        )

        real_states.append(next_state.copy())
        pred_states.append(
            predicted_state.cpu().numpy().copy()
        )

        if terminated or truncated:
            break

    real_states = np.asarray(real_states)
    pred_states = np.asarray(pred_states)

    predicted_radius = np.sqrt(
        pred_states[:, 0] ** 2 +
        pred_states[:, 1] ** 2
    )

    radius_error = np.abs(predicted_radius - 1.0)

    print(
        f"Episode {episode}: "
        f"Mean radius error = {radius_error.mean():.6f}, "
        f"Max radius error = {radius_error.max():.6f}"
     )

    errors = np.mean(
        (real_states - pred_states) ** 2, axis=1
    )
    all_errors.append(errors)

    if episode == 0:
        example_real = real_states
        example_pred = pred_states

env.close()

all_errors = np.stack(all_errors)
mean_error = all_errors.mean(axis=0)

print("Mean rollout MSE:")
for step in [1, 5, 10, 20, 50, 100]:
    print(f"Step {step}: {mean_error[step]:.6f}")

# Plot predicted vs actual angular velocity
plt.figure(figsize=(10, 5))
plt.plot(
    example_real[:, 2],
    label="Actual angular velocity"
)
plt.plot(
    example_pred[:, 2],
    "--",
    label="Predicted angular velocity"
)
plt.xlabel("Time step")
plt.ylabel("Angular velocity (rad/s)")
plt.title("World Model: 100-Step Open-Loop Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/multi_step/rollout_comparison.png", dpi=150)
plt.close()

# Plot error accumulation
plt.figure(figsize=(10, 5))
plt.plot(mean_error)
plt.xlabel("Prediction horizon")
plt.ylabel("Mean state MSE")
plt.title("Compounding Prediction Error")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/multi_step/rollout_error.png", dpi=150)
plt.close()

print("Saved: rollout_comparison.png")
print("Saved: rollout_error.png")

