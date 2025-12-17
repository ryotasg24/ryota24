#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import random

# 既存クラスを再利用（推論側と同一の出力変換：alpha=softmax、beta=softplus）
from AMA_SHAP_for_PointNeXt import AdaptiveGateMLP

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser("Fit Gate-MLP (X=[N,var_reg,var_pnt] -> Y=[alpha,beta])")
    ap.add_argument("--train_npz", type=str, default="result/_gate_train_dump/gate_mlp_train_data.npz", help="npz with arrays X:(M,3), Y:(M,2)")
    ap.add_argument("--out_pth", type=str, default="result/_gate_train_dump/gate_mlp.pth")
    ap.add_argument("--out_scaler", type=str, default="result/_gate_train_dump/gate_scaler.npz")
    ap.add_argument("--n_list", type=int, nargs="+", default=None, help="学習に使う N のみを選択。指定があれば出力ファイル名に _n... のサフィックスを自動付与")
    ap.add_argument("--no_auto_suffix", action="store_true", help="サフィックス自動付与を無効化（明示的にファイル名を指定する場合など）")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load training tuples -----
    npz = np.load(args.train_npz)
    X = npz["X"].astype(np.float32)  # (M,3)  [N, var_reg, var_pnt]
    Y = npz["Y"].astype(np.float32)  # (M,2)  [alpha, beta]

    # sanitize
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y = X[ok], Y[ok]
    if len(X) == 0:
        raise RuntimeError("No valid samples in NPZ.")

# ----- N でフィルタ（指定があれば）-----
    if args.n_list is not None and len(args.n_list) > 0:
        wants = sorted(set(int(n) for n in args.n_list))
        keep = np.isin(X[:, 0].round().astype(int), wants)
        X, Y = X[keep], Y[keep]
        if len(X) == 0:
            raise RuntimeError(f"No samples left after filtering by n_list={wants}.")
        # 出力ファイル名にサフィックスを自動付与
        if not args.no_auto_suffix:
            if len(wants) == 2:
                suf = f"n{wants[0]}_{wants[1]}"
            else:
                suf = "n" + "_".join(str(n) for n in wants)
            def _with_suffix(path: str, s: str) -> str:
                root, ext = os.path.splitext(path)
                return f"{root}_{s}{ext}"
            args.out_pth    = _with_suffix(args.out_pth, suf)
            args.out_scaler = _with_suffix(args.out_scaler, suf)
            print(f"[info] n_list filter: {wants} -> save as:\n"
                    f"       model : {args.out_pth}\n"
                    f"       scaler: {args.out_scaler}")

    # ----- feature transform to match inference -----
    eps = 1e-9
    Xtr = np.stack([X[:,0] / 1024.0,
                    np.log(X[:,1] + eps),
                    np.log(X[:,2] + eps)], axis=1).astype(np.float32)
    # ----- scaler (z-score) for transformed features -----
    mu = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-9
    Xz = (Xtr - mu) / std
    os.makedirs(os.path.dirname(args.out_scaler) or ".", exist_ok=True)
    np.savez_compressed(args.out_scaler, mu=mu.astype(np.float32), std=std.astype(np.float32))
    print(f"[scaler] saved -> {args.out_scaler} (mu={mu}, std={std})")

    # ----- dataset split -----
    ds = TensorDataset(torch.from_numpy(Xz), torch.from_numpy(Y))
    n_val = int(len(ds) * args.val_split)
    n_trn = len(ds) - n_val
    trn_ds, val_ds = random_split(ds, [n_trn, n_val], generator=torch.Generator().manual_seed(args.seed))
    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False) if n_val>0 else None

    # ----- model -----
    model = AdaptiveGateMLP(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for ep in range(1, args.epochs+1):
        # train
        model.train()
        trn_loss = 0.0
        for xb, yb in trn_loader:
            xb = xb.to(device); yb = yb.to(device)          # yb: (B,2) [alpha,beta]
            opt.zero_grad()
            alpha, beta = model(xb)                         # alpha,beta: (B,)
            pred = torch.stack([alpha, beta], dim=1)        # (B,2)
            loss = crit(pred, yb)
            loss.backward()
            if args.clip_grad is not None and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            trn_loss += loss.item() * xb.size(0)
        trn_loss /= n_trn

        # val
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    a,b = model(xb)
                    pred = torch.stack([a,b], dim=1)
                    loss = crit(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_ds)
        else:
            val_loss = trn_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"[ep {ep:03d}] train={trn_loss:.6f}  val={val_loss:.6f}  best_val={best_val:.6f}")

    # save best
    if best_state is None:
        best_state = model.state_dict()
    os.makedirs(os.path.dirname(args.out_pth) or ".", exist_ok=True)
    torch.save(best_state, args.out_pth)
    print(f"[model] saved -> {args.out_pth}")

if __name__ == "__main__":
    main()
