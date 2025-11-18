# utils/normalizer.py
import numpy as np

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

    def inverse(self, Xn: np.ndarray):
        return Xn * self.std + self.mean
