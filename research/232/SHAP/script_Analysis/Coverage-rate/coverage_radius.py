#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
coverage_radius.py
  ・元点群 P（ModelNet40 test h5）→ サブセット S（SD-GLM / FPS）への directed Hausdorff 距離
    r_cov = max_{x in P} min_{y in S} ||x - y||
  ・各サンプルごとに r_cov をバウンディングボックス対角で正規化: r_norm = r_cov / (||bbox_max - bbox_min|| + eps)
  ・直感的な “被覆率” として cov_score = 1 - r_norm も併記（0〜1想定、1が良い）
  ・各ダウンサンプリング点数 N ごとに、SD-GLM / FPS の平均・標準偏差・件数を出力（CSV保存＆標準出力）

前提：
  - SD-GLM の h5 は「元 test h5 の順序と件数に対応」していることを想定（不一致は自動スキップ）
  - FPS 側も同様
  - ラベル不一致や長さ不一致のサンプルは安全に除外します

使い方（例）：
  python script_Analysis/Coverage-rate/coverage_radius.py \
    --orig_root /workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048 \
    --fps_root   /workspace/PointNeXt/result/FPS \
    --n_list 100 200 300 400 500 600 700 800 900 1000 \
    --out_csv /workspace/PointNeXt/script_Analysis/Coverage-rate/coverage_radius_summary.csv \
    --sdglm_root /workspace/PointNeXt/result/NdepSDCS2-mlp_kr32_kp64_p10000_dsSDCS_PointNeXt_h5

出力:
    r_norm(mean) SD-GLM / FPS:
        元の点群P→サブセットSの directed Hausdorff 半径を、Pのバウンディングボックス対角で割った“正規化半径”の全体平均。小さいほど“穴が少ない＝よく覆えている”。
    cov(mean) SD-GLM / FPS: (求めていた被覆率)
        簡易の被覆率スコア＝ 1 - r_norm(mean)。大きいほど良い（1に近いほど隙間が少ない）。直感的に“どれだけ覆えているか”を見る。
    Δr = r_norm_mean(SD-GLM) - r_norm_mean(FPS):
        負ならSD-GLMのほうが半径が小さく優勢、正ならFPSが優勢。今は正なのでFPSが広く均一に覆えている傾向。
    Δcov = cov_mean(SD-GLM) - cov_mean(FPS):
        正ならSD-GLMの被覆率が高い、負ならFPSが高い。今は負なのでFPSの被覆率が高い。

"""

import os
import sys
import argparse
import numpy as np
import h5py
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict


def load_h5(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data, label). data: (M,N,3) or (M,3)?? here expected (M,N,3). label: (M,)"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        data = f["data"][:]
        label = f["label"][:]
    # labels may be (M,1)
    label = label.reshape(-1)
    return data, label


def directed_hausdorff_radius(P: np.ndarray, S: np.ndarray) -> float:
    """ Compute r_cov(P->S) = max_{x in P} min_{y in S} ||x - y||_2
        P: (Np,3), S: (Ns,3)
    """
    if P.ndim != 2 or S.ndim != 2 or P.shape[1] != 3 or S.shape[1] != 3:
        raise ValueError("P and S must be (N,3)")
    if len(S) == 0 or len(P) == 0:
        return float("inf")
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(S)                         # build NN on S
    dists, _ = nn.kneighbors(P)       # (Np,1)
    return float(dists.max())


def bbox_diag(P: np.ndarray, eps: float = 1e-12) -> float:
    """Bounding box diagonal length of P (N,3)."""
    d = np.linalg.norm(P.max(axis=0) - P.min(axis=0))
    return float(d + eps)


def summarize(vals: List[float]) -> Tuple[float, float, int]:
    """Return (mean, std, count) with safe defaults."""
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), int(len(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_root", type=str,
                    default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048",
                    help="Original ModelNet40 test h5 directory (contains ply_data_test0.h5, ply_data_test1.h5)")
    ap.add_argument("--sdglm_root", type=str,
                    default="/workspace/PointNeXt/result/AMA-mlp-stage2_fps500_gate_mlp_n100_200_300_400-stage2_fps500_kr32_kp64_p10000_dsSHAP_PointNeXt_h5",
                    help="SD-GLM base directory that has {N}/modelnet40_ply_hdf5_2048/ply_data_test{0,1}.h5")
    ap.add_argument("--fps_root", type=str,
                    default="/workspace/PointNeXt/result/FPS",
                    help="FPS base directory that has {N}/modelnet40_ply_hdf5_2048/ply_data_test{0,1}.h5")
    ap.add_argument("--n_list", type=int, nargs="+", default=[100,200,300,400,500,600,700,800,900,1000],
                    help="Downsampled sizes to evaluate")
    ap.add_argument("--tests", type=int, nargs="+", default=[0,1], choices=[0,1],
                    help="Which test files to include (0 => test0, 1 => test1)")
    ap.add_argument("--out_csv", type=str, default="/workspace/PointNeXt/script_Analysis/Coverage-rate/coverage_radius_summary.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    # Prepare originals
    orig_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for t in args.tests:
        opath = os.path.join(args.orig_root, f"ply_data_test{t}.h5")
        data, label = load_h5(opath)  # data: (M,2048,3), label: (M,)
        orig_cache[t] = (data, label)

    # CSV header
    rows = []
    header = [
        "N",
        "r_norm_mean_sdglm", "r_norm_std_sdglm", "count_sdglm",
        "r_norm_mean_fps",   "r_norm_std_fps",   "count_fps",
        "cov_score_mean_sdglm", "cov_score_mean_fps",
        "delta_r_norm_sdglm_minus_fps",          # <0 なら SD-GLM が良（半径が小）
        "delta_cov_score_sdglm_minus_fps"        # >0 なら SD-GLM が良（被覆率が高）
    ]

    for N in args.n_list:
        rnorms_sdglm: List[float] = []
        rnorms_fps:   List[float] = []

        for t in args.tests:
            # Paths
            sd_path = os.path.join(args.sdglm_root, str(N), "modelnet40_ply_hdf5_2048", f"ply_data_test{t}.h5")
            fps_path = os.path.join(args.fps_root,   str(N), "modelnet40_ply_hdf5_2048", f"ply_data_test{t}.h5")

            if not (os.path.exists(sd_path) and os.path.exists(fps_path)):
                # どちらか無ければこの test はスキップ
                continue

            sd_data, sd_lab = load_h5(sd_path)   # (K,N,3), (K,)
            fp_data, fp_lab = load_h5(fps_path)  # (K,N,3), (K,)

            # Originals
            P_all, L_all = orig_cache[t]         # (M,2048,3), (M,)

            # 安全整合：長さの最小値まで
            m = min(len(P_all), len(sd_data), len(fp_data), len(sd_lab), len(fp_lab), len(L_all))
            if m == 0:
                continue

            for i in range(m):
                # ラベル整合を確認（ズレがあると危険）
                li = int(L_all[i])
                if li != int(sd_lab[i]) or li != int(fp_lab[i]):
                    # ラベル不一致サンプルはスキップ（生成時の欠損等に備える）
                    continue

                P = P_all[i]           # (2048,3) expected
                S_sd = sd_data[i]      # (N,3)
                S_fp = fp_data[i]      # (N,3)

                # directed Hausdorff radius & normalization
                diag = bbox_diag(P)
                r_sd = directed_hausdorff_radius(P, S_sd) / diag
                r_fp = directed_hausdorff_radius(P, S_fp) / diag

                rnorms_sdglm.append(r_sd)
                rnorms_fps.append(r_fp)

        mean_sd, std_sd, cnt_sd = summarize(rnorms_sdglm)
        mean_fp, std_fp, cnt_fp = summarize(rnorms_fps)
        cov_sd = (1.0 - mean_sd) if np.isfinite(mean_sd) else np.nan
        cov_fp = (1.0 - mean_fp) if np.isfinite(mean_fp) else np.nan
        delta_r = (mean_sd - mean_fp) if (np.isfinite(mean_sd) and np.isfinite(mean_fp)) else np.nan
        delta_cov = (cov_sd - cov_fp) if (np.isfinite(cov_sd) and np.isfinite(cov_fp)) else np.nan

        rows.append([
            N,
            mean_sd, std_sd, cnt_sd,
            mean_fp, std_fp, cnt_fp,
            cov_sd, cov_fp,
            delta_r, delta_cov
        ])

        # Console quick view
        print(f"[N={N:4d}]  r_norm(mean)  SD-GLM: {mean_sd:.6f} (n={cnt_sd})   FPS: {mean_fp:.6f} (n={cnt_fp})   "
                f"Δr={delta_r:.6f} | cov(mean)  SD-GLM: {cov_sd:.6f}  FPS: {cov_fp:.6f}  Δcov={delta_cov:.6f}")

    # Save CSV
    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
