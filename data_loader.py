# data/data_loader.py
import pandas as pd
import numpy as np

DEFAULT_PATH = "data/dataset_inverse_kinematics.csv"

def load_and_prepare(path=DEFAULT_PATH, seed=42):
    df = pd.read_csv(path)
    # Drop unreachable / NaN rows
    df = df.dropna(subset=["theta1", "theta2"])
    # Keep only first quadrant per consigne
    df = df[(df["x"] >= 0) & (df["y"] >= 0)].reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    X = df[["x", "y"]].values.astype(np.float32)
    y1 = df["theta1"].values.astype(np.float32).reshape(-1, 1)
    y2 = df["theta2"].values.astype(np.float32).reshape(-1, 1)
    return X, y1, y2, df

if __name__ == "__main__":
    X, y1, y2, df = load_and_prepare()
    print("Samples:", len(X))
