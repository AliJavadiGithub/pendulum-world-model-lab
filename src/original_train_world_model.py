"""Original tutorial training script (historical version; less reproducible action sampling)."""
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def collect_data(episodes, seed):
    env = gym.make("Pendulum-v1")
    inputs, targets = [], []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        for step in range(200):
            action = env.action_space.sample()
            next_state, _, terminated, truncated, _ = env.step(action)
            x = np.concatenate((state, action))
            delta = next_state - state
            inputs.append(x)
            targets.append(delta)
            state = next_state
            if terminated or truncated:
                break
    env.close()
    return (
        torch.tensor(np.array(inputs), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
    )

print("Collecting training data...")
X_train, Y_train = collect_data(100, 42)
X_test, Y_test = collect_data(20, 10000)
x_mean = X_train.mean(dim=0)
x_std = X_train.std(dim=0).clamp_min(1e-6)
y_mean = Y_train.mean(dim=0)
y_std = Y_train.std(dim=0).clamp_min(1e-6)
Xn = (X_train - x_mean) / x_std
Yn = (Y_train - y_mean) / y_std
loader = DataLoader(TensorDataset(Xn, Yn), batch_size=256, shuffle=True)
model = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
print("Training started...")
for epoch in range(30):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30 - Train MSE: {total_loss / len(X_train):.6f}")
model.eval()
with torch.no_grad():
    test_inputs = ((X_test - x_mean) / x_std).to(device)
    predicted_normalized = model(test_inputs)
    predicted_delta = predicted_normalized.cpu() * y_std + y_mean
    predicted_next = X_test[:, :3] + predicted_delta
    actual_next = X_test[:, :3] + Y_test
    test_mse = torch.mean((predicted_next - actual_next) ** 2)
    baseline_mse = torch.mean(Y_test ** 2)
print("\nEvaluation:")
print("One-step test MSE:", test_mse.item())
print("No-change baseline MSE:", baseline_mse.item())
print("Actual next state:", actual_next[0].numpy())
print("Predicted next state:", predicted_next[0].numpy())
torch.save({"model": model.state_dict(), "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std}, "pendulum_world_model.pt")
print("\nSaved: pendulum_world_model.pt")
