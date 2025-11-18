# predict.py
import sys
import torch
import numpy as np
from fuzzy_defs import AnfisNet
from data_loader import load_and_prepare
from utils.normalizer import Normalizer
import os

MODELS_DIR = "models"

def load_models():
    m1 = AnfisNet(); m2 = AnfisNet()
    m1.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_theta1.pth")))
    m2.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_theta2.pth")))
    m1.eval(); m2.eval()
    return m1, m2

def main(x, y):
    X, y1, y2, df = load_and_prepare()
    norm = Normalizer(); norm.fit(X)
    m1, m2 = load_models()
    pt = norm.transform(np.array([[x,y]], dtype=np.float32))
    with torch.no_grad():
        t = torch.tensor(pt, dtype=torch.float32)
        th1 = m1(t).cpu().numpy().reshape(-1)[0]
        th2 = m2(t).cpu().numpy().reshape(-1)[0]
    print(f"Input x={x:.3f}, y={y:.3f} -> theta1={th1:.6f}, theta2={th2:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <x> <y>")
    else:
        x = float(sys.argv[1]); y = float(sys.argv[2])
        main(x, y)
