# evaluate.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from fuzzy_defs import AnfisNet
from data_loader import load_and_prepare
from utils.normalizer import Normalizer
from math import atan2, sqrt

MODELS_DIR = "models"
FIGS_DIR = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)

def analytic_inverse(x, y, l1=10.0, l2=7.0):
    d2 = x*x + y*y
    if d2 < (l1-l2)**2 or d2 > (l1+l2)**2:
        return float('nan'), float('nan')
    c2 = (d2 - l1**2 - l2**2) / (2*l1*l2)
    c2 = max(-1.0, min(1.0, c2))
    s2 = sqrt(max(0.0, 1-c2*c2))
    theta2 = atan2(s2, c2)
    k1 = l1 + l2*c2
    k2 = l2*s2
    theta1 = atan2(y, x) - atan2(k2, k1)
    return theta1, theta2

def main():
    X, y1, y2, df = load_and_prepare()
    norm = Normalizer(); norm.fit(X)
    xmin, xmax = X[:,0].min(), X[:,0].max()
    ymin, ymax = X[:,1].min(), X[:,1].max()

    gx = np.linspace(xmin, xmax, 80)
    gy = np.linspace(ymin, ymax, 80)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    # filter reachable points
    mask = [not np.isnan(analytic_inverse(px,py)[0]) for px,py in pts]
    pts = pts[np.array(mask)]

    # normalize
    pts_n = norm.transform(pts)
    pts_t = torch.tensor(pts_n, dtype=torch.float32)

    m1 = AnfisNet(); m2 = AnfisNet()
    m1.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_theta1.pth")))
    m2.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_theta2.pth")))
    m1.eval(); m2.eval()

    with torch.no_grad():
        out1 = m1(pts_t).cpu().numpy().reshape(-1)
        out2 = m2(pts_t).cpu().numpy().reshape(-1)

    plt.figure(figsize=(7,5))
    sc = plt.scatter(pts[:,0], pts[:,1], c=out1, s=8)
    plt.colorbar(sc); plt.xlabel('x'); plt.ylabel('y'); plt.title('Inverse surface - theta1 (ANFIS)')
    plt.savefig(os.path.join(FIGS_DIR, "inv_surface_theta1.png")); plt.close()

    plt.figure(figsize=(7,5))
    sc = plt.scatter(pts[:,0], pts[:,1], c=out2, s=8)
    plt.colorbar(sc); plt.xlabel('x'); plt.ylabel('y'); plt.title('Inverse surface - theta2 (ANFIS)')
    plt.savefig(os.path.join(FIGS_DIR, "inv_surface_theta2.png")); plt.close()

    print("Evaluation figures saved to", FIGS_DIR)

if __name__ == "__main__":
    main()
