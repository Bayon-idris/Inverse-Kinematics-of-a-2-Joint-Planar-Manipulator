# explore_dataset.py
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/dataset_inverse_kinematics.csv')

# Basic info
print("Dataset info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Count reachable vs unreachable points
reachable_count = df['reachable'].sum()
total_count = len(df)
print(f"\nReachable points: {reachable_count}/{total_count}")

# Scatter plot: x vs y colored by reachability
plt.figure(figsize=(8, 6))
plt.scatter(df['x'], df['y'], c=df['reachable'], cmap='coolwarm', edgecolors='k')
plt.title("Scatter plot of manipulator end-effector positions")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label='Reachable (True=1, False=0)')
plt.grid(True)
plt.show()
