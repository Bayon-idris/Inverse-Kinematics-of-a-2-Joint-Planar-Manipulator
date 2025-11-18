# data_gen.py
import os
import numpy as np
import pandas as pd

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

L1 = 10.0
L2 = 7.0
NUM_POINTS = 299
np.random.seed(42)

theta1 = np.random.uniform(0, np.pi/2, NUM_POINTS)
theta2 = np.random.uniform(0, np.pi, NUM_POINTS)

x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

dist = np.sqrt(x**2 + y**2)
reachable = (dist >= abs(L1 - L2)) & (dist <= (L1 + L2))

theta1_out = theta1.copy()
theta2_out = theta2.copy()
theta1_out[~reachable] = np.nan
theta2_out[~reachable] = np.nan

mask = (x >= 0) & (y >= 0)
df = pd.DataFrame({
    "x": x[mask],
    "y": y[mask],
    "theta1": theta1_out[mask],
    "theta2": theta2_out[mask],
    "reachable": reachable[mask]
})

out_path = os.path.join(OUT_DIR, "dataset_inverse_kinematics.csv")
df.to_csv(out_path, index=False)
print(f"Saved dataset: {len(df)} rows -> {out_path}")
