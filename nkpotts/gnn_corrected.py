"""
gnn_corrected.py
================
Inverse regression of h from single snapshots, framed correctly for the
FIRST-ORDER transition. Key additions vs the original gnn.py:

  1. CORRECT dE_glob (factor-2 fix) in the snapshot generator.
  2. Three regressors, not two:
       (a) GNN                      : GCN + mean/max pooling  (topology)
       (b) MLP-global               : occupation fractions only (Curie-Weiss sufficient stat)
       (c) Ridge on 2 hand features : [max occupation fraction, domain-wall density]
     Control (c) is the crucial one: if a 2-number physically-motivated feature
     already matches the GNN, then "GNN exploits topology" is NOT supported.
     The honest question the paper answers is: does the GNN beat the
     domain-wall density, which is the leading spatial correlator?
  3. Multi-seed runs -> report mean +/- std of R^2 and RMSE so the
     GNN-vs-baseline gap has error bars (the original reported point values).
  4. Emphasis on the coexistence region (near h_t) where inference is
     genuinely hard because P(phi) is bimodal: a single snapshot can come
     from either coexisting phase, imposing an information-theoretic floor.

Run: python3 gnn_corrected.py --seeds 3 --samples 3000 --epochs 120
"""
import numpy as np
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from multiprocessing import Pool, cpu_count
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

Q = 3
J = 1.0
T = 1.0
H_T = 1.2530467500626585


@njit(cache=True)
def generate_snapshot(L, h, q, J, burn, seed):
    np.random.seed(seed)
    spins = np.random.randint(0, q, L)
    counts = np.zeros(q, dtype=np.int64)
    for s in spins:
        counts[s] += 1
    for _ in range(burn):
        for _ in range(L):
            idx = np.random.randint(0, L)
            old_s = spins[idx]
            new_s = np.random.randint(0, q)
            if old_s == new_s:
                continue
            left = spins[L-1] if idx == 0 else spins[idx-1]
            right = spins[(idx+1) % L]
            dE_loc = J*((old_s == left)+(old_s == right)-(new_s == left)-(new_s == right))
            dE_glob = -(h/L)*(counts[new_s]-counts[old_s]+1)   # CORRECTED
            dE = dE_loc + dE_glob
            if dE <= 0.0 or np.random.rand() < np.exp(-dE):
                spins[idx] = new_s
                counts[old_s] -= 1
                counts[new_s] += 1
    return spins


def hand_features(spins, q=Q):
    L = len(spins)
    counts = np.bincount(spins, minlength=q)
    frac_max = counts.max()/L
    dw = np.mean(spins != np.roll(spins, 1))   # domain-wall density
    return np.array([frac_max, dw])


def _gen(args):
    L, h, burn, seed = args
    s = generate_snapshot(L, h, Q, J, burn, seed)
    return h, s


def make_dataset(L, n, burn, seed0):
    rng = np.random.default_rng(seed0)
    hs = rng.uniform(0.4, 2.0, n)
    tasks = [(L, float(h), burn, int(seed0 + i + 1)) for i, h in enumerate(hs)]
    with Pool(max(1, cpu_count()-1)) as p:
        res = p.map(_gen, tasks)
    return res


def run_torch_models(train, test, L, epochs, seed):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    torch.manual_seed(seed); np.random.seed(seed)

    edges = torch.tensor([[i, (i+1) % L] for i in range(L)]
                         + [[(i+1) % L, i] for i in range(L)], dtype=torch.long).t()

    def to_data(sample):
        h, s = sample
        x = torch.zeros((L, Q)); x[np.arange(L), s] = 1.0
        return Data(x=x, edge_index=edges, y=torch.tensor([[h]], dtype=torch.float))

    train_ds = [to_data(z) for z in train]
    test_ds = [to_data(z) for z in test]
    trl = DataLoader(train_ds, batch_size=64, shuffle=True)
    tel = DataLoader(test_ds, batch_size=128)

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(Q, 64); self.c2 = GCNConv(64, 128); self.c3 = GCNConv(128, 64)
            self.f1 = nn.Linear(128, 64); self.f2 = nn.Linear(64, 1); self.do = nn.Dropout(0.2)
        def forward(self, d):
            x, ei, b = d.x, d.edge_index, d.batch
            x = F.relu(self.c1(x, ei)); x = F.relu(self.c2(x, ei)); x = F.relu(self.c3(x, ei))
            x = torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], 1)
            x = F.relu(self.f1(x)); x = self.do(x)
            return self.f2(x)

    class MLPg(nn.Module):
        """MLP on the 3 global occupation fractions only (CW sufficient statistic)."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(Q, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, d):
            oh = d.x.view(d.num_graphs, L, Q).mean(dim=1)
            return self.net(oh)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for name, Model in [("GNN", GNN), ("MLPg", MLPg)]:
        m = Model().to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.HuberLoss()
        m.train()
        for _ in range(epochs):
            for batch in trl:
                batch = batch.to(dev); opt.zero_grad()
                loss = crit(m(batch), batch.y); loss.backward(); opt.step()
        m.eval(); preds = []; ys = []
        import torch as T_
        with T_.no_grad():
            for batch in tel:
                preds.extend(m(batch.to(dev)).cpu().numpy().ravel())
                ys.extend(batch.y.numpy().ravel())
        out[name] = (np.array(ys), np.array(preds))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--L", type=int, default=400)
    ap.add_argument("--burn", type=int, default=5000)
    args = ap.parse_args()

    agg = {k: {"r2": [], "rmse": []} for k in ["GNN", "MLPg", "Ridge-hand"]}
    last_preds = None
    for si in range(args.seeds):
        data = make_dataset(args.L, args.samples, args.burn, seed0=1000*(si+1))
        ntr = int(0.8*len(data))
        train, test = data[:ntr], data[ntr:]

        # hand-feature ridge
        Xtr = np.array([hand_features(s) for _, s in train]); ytr = np.array([h for h, _ in train])
        Xte = np.array([hand_features(s) for _, s in test]);  yte = np.array([h for h, _ in test])
        ridge = Ridge(alpha=1.0).fit(Xtr, ytr); pr = ridge.predict(Xte)
        agg["Ridge-hand"]["r2"].append(r2_score(yte, pr))
        agg["Ridge-hand"]["rmse"].append(np.sqrt(mean_squared_error(yte, pr)))

        try:
            res = run_torch_models(train, test, args.L, args.epochs, seed=si)
            for name in ["GNN", "MLPg"]:
                y, p = res[name]
                agg[name]["r2"].append(r2_score(y, p))
                agg[name]["rmse"].append(np.sqrt(mean_squared_error(y, p)))
            last_preds = {"GNN": res["GNN"], "MLPg": res["MLPg"], "Ridge": (yte, pr)}
        except ImportError:
            print("torch/pyg not available; ran Ridge control only.")

    summary = {}
    for name, d in agg.items():
        if d["r2"]:
            summary[name] = {"R2_mean": float(np.mean(d["r2"])), "R2_std": float(np.std(d["r2"])),
                             "RMSE_mean": float(np.mean(d["rmse"])), "RMSE_std": float(np.std(d["rmse"]))}
    with open("gnn_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if last_preds is not None and "GNN" in last_preds:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        bins = np.linspace(0.4, 2.0, 17); cen = 0.5*(bins[:-1]+bins[1:])
        raw_rows = []; binned_rows = []
        for name, (y, p), col, mk in [("GNN", last_preds["GNN"], "#1f77b4", "o"),
                                      ("MLPg", last_preds["MLPg"], "#d62728", "s"),
                                      ("Ridge-hand", last_preds["Ridge"], "#2ca02c", "^")]:
            idx = np.digitize(y, bins)
            mean = [p[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(bins))]
            std = [p[idx == k].std() if np.any(idx == k) else np.nan for k in range(1, len(bins))]
            mean = np.array(mean); std = np.array(std)
            ax.errorbar(cen, mean, yerr=std, fmt="-"+mk, ms=4, color=col, capsize=2,
                        label=f"{name} ($R^2$={np.mean(agg[name]['r2']):.3f})")
            ax.fill_between(cen, mean-std, mean+std, color=col, alpha=0.12)
            for yt, pp in zip(y, p):
                raw_rows.append(dict(model=name, true_h=float(yt), pred_h=float(pp)))
            for c, mn, sd in zip(cen, mean, std):
                binned_rows.append(dict(model=name, true_h_bin=c, pred_mean=mn, pred_std=sd))
        ax.plot([0.4, 2.0], [0.4, 2.0], "k--", lw=1.2, label="ideal")
        ax.axvspan(1.15, 1.40, color="gray", alpha=0.15)
        ax.axvline(H_T, color="0.4", ls=":", lw=1)
        ax.set_xlabel(r"true $h$"); ax.set_ylabel(r"predicted $\hat h$")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        fig.tight_layout(); fig.savefig("fig_gnn.pdf"); fig.savefig("fig_gnn.png", dpi=200)
        import pandas as _pd
        _pd.DataFrame(raw_rows).to_csv("data_gnn_predictions.csv", index=False)     # raw (true,pred)
        _pd.DataFrame(binned_rows).to_csv("data_gnn_binned.csv", index=False)       # binned mean/std


if __name__ == "__main__":
    main()
