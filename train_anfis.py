# train_anfis.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

from data_loader import load_and_prepare
from utils.normalizer import Normalizer
from fuzzy_defs import AnfisNet

# config
TEST_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 80
SEED = 42
MODELS_DIR = "models"
FIGS_DIR = "figs"
LR = 1e-3

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    train_losses, val_losses = [], []
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            tr += loss.item() * xb.size(0)
        train_loss = tr / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        vr = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vr += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = vr / len(val_loader.dataset)
        val_losses.append(val_loss)

        if ep == 1 or ep % 10 == 0:
            print(f"Epoch {ep}/{epochs} Train={train_loss:.6f} Val={val_loss:.6f}")

    return model, train_losses, val_losses

def main():
    X, y1, y2, df = load_and_prepare()
    normalizer = Normalizer(); normalizer.fit(X)
    Xn = normalizer.transform(X)

    Xt = torch.tensor(Xn, dtype=torch.float32)
    y1t = torch.tensor(y1, dtype=torch.float32)
    y2t = torch.tensor(y2, dtype=torch.float32)

    ds1 = TensorDataset(Xt, y1t)
    ds2 = TensorDataset(Xt, y2t)
    n_test = int(len(ds1) * TEST_RATIO)
    n_train = len(ds1) - n_test
    train1, test1 = random_split(ds1, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))
    train2, test2 = random_split(ds2, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))

    loader_train1 = DataLoader(train1, batch_size=BATCH_SIZE, shuffle=True)
    loader_val1 = DataLoader(test1, batch_size=BATCH_SIZE, shuffle=False)
    loader_train2 = DataLoader(train2, batch_size=BATCH_SIZE, shuffle=True)
    loader_val2 = DataLoader(test2, batch_size=BATCH_SIZE, shuffle=False)

    model1 = AnfisNet()
    model2 = AnfisNet()

    print("\nTraining theta1...")
    model1, t1_tr, t1_val = train_one(model1, loader_train1, loader_val1)
    torch.save(model1.state_dict(), os.path.join(MODELS_DIR, "model_theta1.pth"))

    print("\nTraining theta2...")
    model2, t2_tr, t2_val = train_one(model2, loader_train2, loader_val2)
    torch.save(model2.state_dict(), os.path.join(MODELS_DIR, "model_theta2.pth"))

    # save loss plots
    plt.figure(); plt.plot(t1_tr, label="train"); plt.plot(t1_val, label="val")
    plt.title("theta1 loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(FIGS_DIR, "loss_theta1.png")); plt.close()

    plt.figure(); plt.plot(t2_tr, label="train"); plt.plot(t2_val, label="val")
    plt.title("theta2 loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(FIGS_DIR, "loss_theta2.png")); plt.close()

    # quick RMS evaluation vs dataset ground truth
    model1.eval(); model2.eval()
    with torch.no_grad():
        preds1 = model1(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy().reshape(-1)
        preds2 = model2(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy().reshape(-1)
    rms1 = np.sqrt(np.mean((preds1 - y1.reshape(-1)) ** 2))
    rms2 = np.sqrt(np.mean((preds2 - y2.reshape(-1)) ** 2))
    print(f"RMS vs dataset — θ1: {rms1:.6f}, θ2: {rms2:.6f}")

if __name__ == "__main__":
    main()
