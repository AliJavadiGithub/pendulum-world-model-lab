"""Create plots using recorded real results from the first experiment."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
record = json.loads((root / 'results' / 'first_run_observed.json').read_text())
points = record['train_normalized_mse_by_epoch']
x = list(map(int, points.keys()))
y = list(points.values())
plt.figure(figsize=(7, 4))
plt.plot(x, y, marker='o')
plt.title('First run: training loss (normalized target space)')
plt.xlabel('Epoch')
plt.ylabel('Training MSE')
plt.grid(alpha=.25)
plt.tight_layout()
output = root / 'results' / 'first_run_training_curve.png'
plt.savefig(output, dpi=160)
print(f'Saved {output}')
