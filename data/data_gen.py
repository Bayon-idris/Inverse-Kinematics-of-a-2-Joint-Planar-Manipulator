# data_gen.py
import numpy as np
import pandas as pd

# Robot arm parameters
l1 = 10.0
l2 = 7.0

# Exact number of points to generate (as specified in the experiment)
num_points = 299

# Random generation of theta1 and theta2 in allowed domain
np.random.seed(42)  # for reproducibility
theta1 = np.random.uniform(0, np.pi/2, num_points)  # θ1 ∈ [0, π/2] - first quadrant
theta2 = np.random.uniform(0, np.pi, num_points)    # θ2 ∈ [0, π] - as specified

# Forward kinematics: calculate end effector positions (x, y)
x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

# Calculate distance from origin to check reachability
dist = np.sqrt(x**2 + y**2)
reachable = (dist >= abs(l1 - l2)) & (dist <= (l1 + l2))

# Set theta1 and theta2 to NaN for unreachable points
theta1[~reachable] = np.nan
theta2[~reachable] = np.nan

# Create DataFrame
df = pd.DataFrame({
    'x': x,
    'y': y,
    'theta1': theta1,
    'theta2': theta2,
    'reachable': reachable
})

# Keep only first quadrant points (x ≥ 0, y ≥ 0)
df = df[(df['x'] >= 0) & (df['y'] >= 0)]

# Save to CSV
df.to_csv('data/dataset_inverse_kinematics.csv', index=False)
print(f"Dataset generated: {len(df)} points saved to 'data/dataset_inverse_kinematics.csv'")