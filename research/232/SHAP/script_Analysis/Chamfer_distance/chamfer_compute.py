#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コマンド:
python script_Analysis/Chamfer_distance/chamfer_compute.py \
--shap_root /workspace/PointNeXt/result/Scheduling-mlp-scheduling_kr32_kp64_p10000_dsSCHED_PointNeXt_h5

出力:
    === Averaged Chamfer (mean loss) per ds_points ===
    mean_loss: 平均Chamfer距離（Chamfer Loss）

    === Detailed per-method (mean±std; x->y, y->x) ===
    mean_cd_x2y	オリジナル点群 → ダウンサンプリング点群（X→Y）の平均距離    (求めていた指標)
    mean_cd_y2x	ダウンサンプリング点群 → オリジナル点群（Y→X）の平均距離
    mean_loss	mean_cd_x2y + mean_cd_y2x（双方向の合計=通常のChamfer Loss）
"""

import os
import h5py
import argparse
import numpy as np
import torch
import pandas as pd


def load_h5_points(h5_path, key="data"):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    with h5py.File(h5_path, "r") as f:
        if key in f:
            arr = f[key][:]
        else:
            # Fallback: first dataset
            ds_name = list(f.keys())[0]
            arr = f[ds_name][:]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected shape in {h5_path}: {arr.shape}")
    return arr


@torch.no_grad()
def chamfer_components(x_np: np.ndarray,
                       y_np: np.ndarray,
                       device: str = "cpu",
                       pair_chunk: int = 1024):
    """
    x_np: (N,3), y_np: (M,3)
    Returns mean squared distances x->y, y->x, and their sum.
    Uses torch.cdist in chunks to avoid OOM.
    """
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)  # (N,3)
    y = torch.from_numpy(y_np.astype(np.float32)).to(device)  # (M,3)

    def one_direction(a, b):
        # average over points in a of min_{b} ||a-b||^2
        total = 0.0
        n = a.shape[0]
        for s in range(0, n, pair_chunk):
            e = min(n, s + pair_chunk)
            a_chunk = a[s:e].unsqueeze(0)         # (1,c,3)
            b_batch = b.unsqueeze(0)              # (1,M,3)
            d = torch.cdist(a_chunk, b_batch, p=2)  # (1,c,M)
            d2_min = (d ** 2).min(dim=2).values    # (1,c)
            total += d2_min.sum().item()
            del d, d2_min, a_chunk
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return total / max(1, n)

    cd_x2y = one_direction(x, y)
    cd_y2x = one_direction(y, x)
    return cd_x2y, cd_y2x, cd_x2y + cd_y2x


def build_paths(orig_root, method_root, ds_points, split, file_id):
    fname = f"ply_data_{split}{file_id}.h5"
    orig_file = os.path.join(orig_root, fname)
    method_file = os.path.join(method_root, str(ds_points),
                               "modelnet40_ply_hdf5_2048", fname)
    return orig_file, method_file


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def init_agg():
    return {"count": 0,
            "sum_x2y": 0.0,
            "sum_y2x": 0.0,
            "sum_loss": 0.0,
            "sum_loss2": 0.0}


def update_agg(agg, cd_x2y, cd_y2x, loss):
    agg["count"] += 1
    agg["sum_x2y"] += cd_x2y
    agg["sum_y2x"] += cd_y2x
    agg["sum_loss"] += loss
    agg["sum_loss2"] += loss * loss


def finalize_row(method, ds_points, agg):
    n = max(1, agg["count"])
    mean_x2y = agg["sum_x2y"] / n
    mean_y2x = agg["sum_y2x"] / n
    mean_loss = agg["sum_loss"] / n
    # Unbiased std if count > 1
    if agg["count"] > 1:
        var = (agg["sum_loss2"] - (agg["sum_loss"] ** 2) / agg["count"]) / (agg["count"] - 1)
        std_loss = float(np.sqrt(max(0.0, var)))
    else:
        std_loss = 0.0
    return [method, ds_points, agg["count"], mean_loss, std_loss, mean_x2y, mean_y2x]


def print_console_table(df_sum):
    if df_sum.empty:
        print("[INFO] No results to summarize.")
        return
    # Pivot for easy side-by-side view
    methods = sorted(df_sum["method"].unique())
    ds_list = sorted(df_sum["ds_points"].unique())
    print("\n=== Averaged Chamfer (mean loss) per ds_points ===")
    header = ["ds_points"] + [f"{m}:mean_loss" for m in methods] + [f"{m}:count" for m in methods]
    print("\t".join(header))
    for ds in ds_list:
        row = [str(ds)]
        for m in methods:
            sub = df_sum[(df_sum["method"] == m) & (df_sum["ds_points"] == ds)]
            row.append(f"{float(sub['mean_loss'].iloc[0]):.8f}" if not sub.empty else "NA")
        for m in methods:
            sub = df_sum[(df_sum["method"] == m) & (df_sum["ds_points"] == ds)]
            row.append(str(int(sub["count"].iloc[0])) if not sub.empty else "0")
        print("\t".join(row))

    print("\n=== Detailed per-method (mean±std; x->y, y->x) ===")
    print("method\tds_points\tcount\tmean_loss\tstd_loss\tmean_cd_x2y\tmean_cd_y2x")
    for _, r in df_sum.sort_values(["method", "ds_points"]).iterrows():
        print(f"{r['method']}\t{int(r['ds_points'])}\t{int(r['count'])}\t"
              f"{r['mean_loss']:.8f}\t{r['std_loss']:.8f}\t"
              f"{r['mean_cd_x2y']:.8f}\t{r['mean_cd_y2x']:.8f}")


def main():
    ap = argparse.ArgumentParser()
    # Roots
    ap.add_argument("--orig_root",
                    default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048")
    ap.add_argument("--shap_root",
                    default="/workspace/PointNeXt/result/AMA-mlp-gate_mlp_n300_400_500_600_700_800_900_1000_kr32_kp64_p10000_dsSHAP_PointNeXt_h5")
    ap.add_argument("--fps_root",
                    default="/workspace/PointNeXt/result/FPS")
    # Settings
    ap.add_argument("--ds_list", nargs="+", type=int,
                    default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["test", "train"])
    ap.add_argument("--test_ids", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--train_ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_dir", default="/workspace/PointNeXt/script_Analysis/Chamfer_distance/chamfer_outputs")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--pair_chunk", type=int, default=512,
                    help="points per cdist chunk (reduce if OOM)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Aggregators: dict[(method, ds)] -> agg
    aggs = {}
    def get_agg(method, ds):
        key = (method, ds)
        if key not in aggs:
            aggs[key] = init_agg()
        return aggs[key]

    for ds in args.ds_list:
        for split in args.splits:
            file_ids = args.test_ids if split == "test" else args.train_ids
            for fid in file_ids:
                try:
                    orig_file, shap_file = build_paths(args.orig_root, args.shap_root, ds, split, fid)
                    _, fps_file = build_paths(args.orig_root, args.fps_root, ds, split, fid)

                    orig = load_h5_points(orig_file)  # (S,N0,3)
                    shap = load_h5_points(shap_file)  # (S,ds,3)
                    fps  = load_h5_points(fps_file)   # (S,ds,3)
                except FileNotFoundError as e:
                    print(f"[WARN] missing file for ds={ds}, {split}{fid}: {e}")
                    continue

                S = min(len(orig), len(shap), len(fps))
                for i in range(S):
                    p_orig = orig[i, :, :3]
                    p_shap = shap[i, :, :3]
                    p_fps  = fps[i,  :, :3]

                    # SHAP vs ORIG
                    cd_xy, cd_yx, loss = chamfer_components(
                        p_orig, p_shap, device=args.device, pair_chunk=args.pair_chunk
                    )
                    update_agg(get_agg("SHAP", ds), cd_xy, cd_yx, loss)

                    # FPS vs ORIG
                    cd_xy_f, cd_yx_f, loss_f = chamfer_components(
                        p_orig, p_fps, device=args.device, pair_chunk=args.pair_chunk
                    )
                    update_agg(get_agg("FPS", ds), cd_xy_f, cd_yx_f, loss_f)

                print(f"[OK] ds={ds} {split}{fid}: compared {S} samples")

    # Build summary dataframe
    sum_rows = []
    for (method, ds_points), agg in sorted(aggs.items(), key=lambda x: (x[0][0], x[0][1])):
        sum_rows.append(finalize_row(method, ds_points, agg))

    sum_cols = ["method", "ds_points", "count", "mean_loss", "std_loss",
                "mean_cd_x2y", "mean_cd_y2x"]
    df_sum = pd.DataFrame(sum_rows, columns=sum_cols).sort_values(["method", "ds_points"])

    # Save summary
    sum_tsv = os.path.join(args.out_dir, "summary.tsv")
    df_sum.to_csv(sum_tsv, sep="\t", index=False, float_format="%.8f")
    print(f"[SAVE] {sum_tsv}")

    # Console print
    print_console_table(df_sum)


if __name__ == "__main__":
    main()
