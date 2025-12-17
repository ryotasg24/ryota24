#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
αは0.0〜1.0を0.1刻み、βは{0.5,1.0,2.0}、Nは 300/500/800/1000
Usage:
$ CUDA_VISIBLE_DEVICES=0 python train_gate_AMAmlp.py --division_region 32 --division_point 64 --pattern 800 --n_list 300 500 800 1000 --alpha_grid 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --beta_grid 0.5 1.0 2.0 --max_files 10 --max_samples_per_file 200 --out_npz gate_mlp_train_data.npz
"""

"""
学習データ作成スクリプト（複数 N 対応）
・役割：各サンプル × 各 N ごとに
    1. Region-SHAP / Point-SHAP を計算
    2. L¹合わせ後の分散 σ²_R, σ²_P を計算
    3. α, β の候補をグリッド探索し、下流モデルの正解クラス確率が最大になる (α*, β*) を “擬似教師” として保存
    4. 入力特徴 = [N, σ²_R, σ²_P]、ターゲット = [α*, β*] を 1 行としてダンプ
・出力：gate_mlp_train_data.npz（features: (M,3), targets: (M,2)）
    → この NPZ を使って Gate-MLP を学習できます（学習コードは前に出したものをそのまま流用）。

既存の AMA_SHAP_for_PointNeXt.py などと同じディレクトリに保存し、
そこにある関数（hierarchical_kmeans, compute_background_baseline, …）を import して使います。
"""

import os, sys, time, math, warnings, argparse, gc
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
import shap
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import h5py
from sklearn.cluster import KMeans
import re
import matplotlib.pyplot as plt
from typing import Optional

# ---- 既存実装から関数を再利用 ------------------------------------------------
# 同フォルダのあなたの実装ファイル名に合わせて import してください
from AMA_SHAP_for_PointNeXt import (
    get_class_names, hierarchical_kmeans, compute_background_baseline, compute_point_baseline,
    shap_predict_block_mask, compute_point_level_shap_values, l1_match_point,
    AdaptiveGateMLP, compute_attention_fusion_features,  # 使わないものもOK
    load_pointnext_model, collect_background_point_clouds
)
import provider
from openpoints.utils import EasyConfig

def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def score_prob_of_trueclass(logits_np, true_idx):
    """ロジットから正解クラスの確率を返す（平均）"""
    prob = softmax_np(logits_np)
    return float(prob[:, true_idx].mean())

def build_hybrid(region_per_pt, point_adj, alpha, beta):
    return alpha * region_per_pt + (1.0 - alpha) * beta * point_adj

def downsample_by_hybrid(pc, hybrid_vals, N):
    idx = np.argsort(hybrid_vals)[::-1][:N]
    return pc[idx, :], idx

def predict_logits(model, pc, device):
    with torch.no_grad():
        t = torch.tensor(pc, dtype=torch.float32, device=device).unsqueeze(0)
        out = model(t).cpu().numpy()  # (1, C)
    return out

def _pat_tag_from_dump_or_args(dump_dir: str, pattern: Optional[int]):
        if dump_dir:
            m = re.search(r"p(\d+)", dump_dir)
            if m:
                return f"p{m.group(1)}"
        return f"p{pattern}" if pattern is not None else "pNA"

def main():
    p = argparse.ArgumentParser("Dump Gate-MLP training tuples for multiple N")
    # データ/モデル
    p.add_argument("--data_root", type=str, default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048")
    p.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml")
    p.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth", help="Path to the PointNeXt checkpoint file")
    # SHAP/クラスタ設定
    p.add_argument("--division_region", type=int, default=32)
    p.add_argument("--division_point", type=int, default=64, help="k_point = fanout * k_region")
    p.add_argument("--pattern", type=int, default=10000, help="KernelSHAP nsamples（学習用は少し抑え目でもOK）")
    p.add_argument("--ama_eps", type=float, default=1e-6)
    # N の複数指定
    p.add_argument("--n_list", type=int, nargs="+", default=[300, 500, 800, 1000])
    # グリッド探索の範囲
    p.add_argument("--alpha_grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--beta_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    # 何サンプル処理するか（時間短縮用）
    p.add_argument("--max_files", type=int, default=9999)
    p.add_argument("--max_samples_per_file", type=int, default=999999)
    p.add_argument("--bg_per_class", type=int, default=50, help="背景点群の収集数")
    # 入力（オプション）: AMA_SHAP_for_PointNeXt.py --gate_dump_train のダンプを読む場合のディレクトリ
    p.add_argument("--dump_dir", type=str, default="", help="Directory containing dumped npz (region_per_pt, point_adj, pc, label)")
    # 出力
    p.add_argument("--out_npz", type=str, default="result/_gate_train_dump/gate_mlp_train_data.npz")
    # αβプロット
    p.add_argument("--plot_alpha_beta", action="store_true", help="各Nにおける推定最適(α*,β*)の平均を可視化（αとβを別画像で保存）し、Nを1刻みでテキスト出力する")
    p.add_argument("--plot_n_min", type=int, default=300, help="可視化・テキスト出力のN下限（未指定なら n_list の最小）")
    p.add_argument("--plot_n_max", type=int, default=1000, help="可視化・テキスト出力のN上限（未指定なら n_list の最大）")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 事前チェック
    k_region = args.division_region
    k_point = args.division_point
    assert k_point % k_region == 0, "--division_point は --division_region の整数倍にしてください"
    fanout = k_point // k_region

    # モデル読込
    model, _ = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)

    # データファイル
    TRAIN_FILES = provider.getDataFiles(os.path.join(args.data_root, "train_files.txt"))
    TEST_FILES = provider.getDataFiles(os.path.join(args.data_root, "test_files.txt"))
    class_names = get_class_names(os.path.join(args.data_root, "shape_names.txt"))

    # dump_dir が与えられ、かつディレクトリが存在する場合のみ「ダンプ読込モード」
    use_dump = bool(args.dump_dir) and os.path.isdir(args.dump_dir)
    if not use_dump:
        # 背景点群（全クラス分）を先に用意（自給式モードのみ）
        print("Collecting class-wise backgrounds...")
        backgrounds = {}
        for cls_id in range(len(class_names)):
            bg = collect_background_point_clouds(cls_id, TRAIN_FILES, num_point_clouds=args.bg_per_class)
            backgrounds[cls_id] = bg

    X_list, Y_list = [], []  # X=[N, var_reg, var_pnt], Y=[alpha*, beta*]

    # テストファイルを順に処理（trainでもOK、重複でもOK）
    if use_dump:
        # --- ダンプ読込モード: --dump_dir 内の *.npz を読む ---
        files = sorted(glob(os.path.join(args.dump_dir, "*.npz")))
        # 必須キーを持つ NPZ のみ採用（旧形式や学習成果物 *.npz を除外）
        required_keys = {"region_per_pt", "point_adj", "pc", "label"}
        valid_files, skipped = [], 0
        for fp in files:
            try:
                with np.load(fp) as z:
                    if required_keys.issubset(set(z.files)):
                        valid_files.append(fp)
                    else:
                        skipped += 1
                        print(f"[SKIP] {os.path.basename(fp)} missing keys; have={set(z.files)}")
            except Exception as e:
                skipped += 1
                print(f"[SKIP] failed to load {fp}: {e}")
        files = valid_files
        if len(files) == 0:
            raise RuntimeError(f"No valid npz found in --dump_dir: {args.dump_dir}")
        # 件数制御：max_files は無視し、max_samples_per_file を総件数上限として扱う
        if args.max_samples_per_file < len(files):
            files = files[:args.max_samples_per_file]
        print(f"[dump_dir] use {len(files)} files from {args.dump_dir} (skipped {skipped})")

        for fp in files:
            with np.load(fp) as z:
                # AMA_SHAP_for_PointNeXt.py --gate_dump_train のフォーマットに準拠
                region_per_pt = np.asarray(z["region_per_pt"], dtype=np.float32).reshape(-1)
                point_adj     = np.asarray(z["point_adj"],     dtype=np.float32).reshape(-1)
                pc            = np.asarray(z["pc"],            dtype=np.float32)
                lab           = np.asarray(z["label"]).reshape(-1)
                cls           = int(lab[0])
            # 形状チェック（不一致はスキップ）
            if pc.ndim != 2 or pc.shape[1] != 3 or \
                pc.shape[0] != region_per_pt.shape[0] or pc.shape[0] != point_adj.shape[0]:
                print(f"[SKIP] shape mismatch in {os.path.basename(fp)}: "
                        f"region={region_per_pt.shape}, point={point_adj.shape}, pc={pc.shape}")
                continue

            # 分散（Gate 入力）
            var_reg = float(np.var(region_per_pt))
            var_pnt = float(np.var(point_adj))

            # N を振って (alpha*, beta*) を探索
            for N in args.n_list:
                best_score, best_alpha, best_beta = -1.0, 0.5, 1.0
                for alpha in args.alpha_grid:
                    for beta in args.beta_grid:
                        hybrid = build_hybrid(region_per_pt, point_adj, alpha, beta)
                        ds_pc, _ = downsample_by_hybrid(pc, hybrid, N)
                        logits = predict_logits(model, ds_pc, device)
                        score = score_prob_of_trueclass(logits, true_idx=cls)
                        if score > best_score:
                            best_score, best_alpha, best_beta = score, alpha, beta
                X_list.append([float(N), var_reg, var_pnt])
                Y_list.append([best_alpha, best_beta])
            gc.collect()
    else:
        # --- 自給式モード（従来のロジック；SHAP を都度計算） ---
        file_cnt = 0
        for test_file in TEST_FILES:
            file_cnt += 1
            if file_cnt > args.max_files:
                break
            data, labels = provider.loadDataFile(test_file)
            n_samples = min(len(data), args.max_samples_per_file)
            print(f"[{os.path.basename(test_file)}] use {n_samples} samples")

            for i in range(n_samples):
                pc = data[i]
                cls = int(labels[i])
                bg_pc = backgrounds.get(cls, None)
                if bg_pc is None or bg_pc.size == 0:
                    continue
                if pc.shape[0] > 1024:
                    pc = pc[:1024]
                elif pc.shape[0] < 1024:
                    pc = np.pad(pc, ((0,1024-pc.shape[0]), (0,0)), 'constant')
                reg_blk, pt_blk, km_reg, km_sub = hierarchical_kmeans(pc, k_region, fanout)
                baseline_reg = compute_background_baseline(bg_pc, km_reg)
                baseline_pt  = compute_point_baseline(bg_pc, km_reg, km_sub, fanout)
                preds_blk = []
                def _shap_predict(mv):
                    out = shap_predict_block_mask(mv, reg_blk, pc, baseline_reg, model, device)
                    preds_blk.append(out); return out
                explainer = shap.KernelExplainer(_shap_predict, np.zeros((1, len(reg_blk))))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                    block_vals = (explainer.shap_values(
                        np.ones((1, len(reg_blk))), nsamples=args.pattern, l1_reg="num_features(10)"
                    )[cls].reshape(-1))
                block_vals_pt = np.repeat(block_vals, fanout)
                pt_shap = compute_point_level_shap_values(
                    pc, pt_blk, block_vals_pt, baseline_pt, model, cls, device, scaling_factor=1.0
                )
                region_per_pt = np.zeros_like(pt_shap)
                region_exp = np.repeat(block_vals, fanout)
                for b_idx, idxs in enumerate(pt_blk):
                    region_per_pt[idxs] = region_exp[b_idx]
                point_adj, _, _ = l1_match_point(pt_shap, region_per_pt, args.ama_eps)
                for N in args.n_list:
                    var_reg = float(np.var(region_per_pt))
                    var_pnt = float(np.var(point_adj))
                    best_score, best_alpha, best_beta = -1.0, 0.5, 1.0
                    for alpha in args.alpha_grid:
                        for beta in args.beta_grid:
                            hybrid = build_hybrid(region_per_pt, point_adj, alpha, beta)
                            ds_pc, _ = downsample_by_hybrid(pc, hybrid, N)
                            logits = predict_logits(model, ds_pc, device)
                            score = score_prob_of_trueclass(logits, true_idx=cls)
                            if score > best_score:
                                best_score, best_alpha, best_beta = score, alpha, beta
                    X_list.append([float(N), var_reg, var_pnt])
                    Y_list.append([best_alpha, best_beta])
                gc.collect()

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, X=X, Y=Y)
    print(f"Saved training tuples: X.shape={X.shape}, Y.shape={Y.shape} -> {args.out_npz}")

    # === Plot α*, β* vs N（別グラフ） & テキスト出力（Nを1刻み） ===
    if args.plot_alpha_beta and len(X) > 0:
        # 1) 集計：Nごとに α*, β* をまとめる
        #   X[:,0] は N、Y[:,0]=alpha*, Y[:,1]=beta*
        n_vals_all = [int(round(n)) for n in X[:, 0].tolist()]
        alpha_all  = Y[:, 0].astype(np.float32)
        beta_all   = Y[:, 1].astype(np.float32)
        # 非有限値を除外
        keep = np.isfinite(alpha_all) & np.isfinite(beta_all) & np.isfinite(X[:,0])
        n_vals_all = np.array(n_vals_all)[keep]
        alpha_all  = alpha_all[keep]
        beta_all   = beta_all[keep]
        # 2) プロット対象のNを決める
        n_unique_sorted = sorted(set(int(n) for n in args.n_list))
        n_min = args.plot_n_min if args.plot_n_min is not None else min(n_unique_sorted)
        n_max = args.plot_n_max if args.plot_n_max is not None else max(n_unique_sorted)
        Ns = [n for n in n_unique_sorted if n_min <= n <= n_max]
        if len(Ns) == 0:
            print("[plot] no N in range -> skip plotting")
        else:
            # 3) Nごと平均を計算
            alpha_mean, beta_mean = [], []
            for N in Ns:
                msk = (n_vals_all == N)
                if not np.any(msk):
                    alpha_mean.append(np.nan)
                    beta_mean.append(np.nan)
                else:
                    alpha_mean.append(float(np.mean(alpha_all[msk])))
                    beta_mean.append(float(np.mean(beta_all[msk])))
            alpha_mean = np.array(alpha_mean, dtype=np.float32)
            beta_mean  = np.array(beta_mean , dtype=np.float32)
            # 欠損を除外（万一のため）
            valid = np.isfinite(alpha_mean) & np.isfinite(beta_mean)
            Ns_plot = [Ns[i] for i,v in enumerate(valid) if v]
            alpha_mean = alpha_mean[valid]
            beta_mean  = beta_mean [valid]
            if len(Ns_plot) == 0:
                print("[plot] nothing to plot after filtering")
            else:
                # 4) 出力ディレクトリの決定
                pat_tag = _pat_tag_from_dump_or_args(args.dump_dir, args.pattern if "pattern" in args else None)
                n_tag   = "_".join(str(n) for n in n_unique_sorted)
                out_dir = os.path.join("result", f"AMA-mlp-gate_mlp_n{n_tag}_kr{args.division_region}_kp{args.division_point}_{pat_tag}_dsSHAP_PointNeXt_h5")
                os.makedirs(out_dir, exist_ok=True)
                # 5) αグラフ（ライン＋強調点）
                plt.figure(figsize=(6.2, 3.9))
                # ライン（小さめマーカー）
                plt.plot(Ns_plot, alpha_mean, marker="o", markersize=4, linewidth=1.5, color="tab:blue", label=r"$\alpha^*$ (mean)")
                # 強調点：真に推定したN（=Ns_plot）を色・サイズで強調
                plt.scatter(Ns_plot, alpha_mean, s=64, color="tab:orange",
                            edgecolors="black", linewidths=0.5, zorder=3, label="estimated N")
                plt.xlabel("Downsampled points N")
                plt.ylabel(r"$\alpha^*$")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                out_png_a = os.path.join(out_dir, "alpha_vs_N.png")
                plt.tight_layout()
                plt.savefig(out_png_a, dpi=200)
                plt.close()
                print(f"[plot] saved -> {out_png_a}")

                # 6) βグラフ（ライン＋強調点）
                plt.figure(figsize=(6.2, 3.9))
                plt.plot(Ns_plot, beta_mean, marker="s", markersize=4, linewidth=1.5, color="tab:green", label=r"$\beta^*$ (mean)")
                plt.scatter(Ns_plot, beta_mean, s=64, color="tab:red",
                            edgecolors="black", linewidths=0.5, zorder=3, label="estimated N")
                plt.xlabel("Downsampled points N")
                plt.ylabel(r"$\beta^*$")
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.legend()
                out_png_b = os.path.join(out_dir, "beta_vs_N.png")
                plt.tight_layout()
                plt.savefig(out_png_b, dpi=200)
                plt.close()
                print(f"[plot] saved -> {out_png_b}")

                # 7) テキスト出力（Nを1刻み；線形補間で全Nを推定；数値のみ）
                N_full = np.arange(int(n_min), int(n_max) + 1, dtype=int)

                def _interp_series(xs_int, ys, X_full):
                    xs = np.asarray(xs_int, dtype=float)
                    ys = np.asarray(ys, dtype=float)
                    Xf = np.asarray(X_full, dtype=float)
                    if xs.size == 0:
                        return np.full_like(Xf, np.nan, dtype=float)
                    if xs.size == 1:
                        return np.full_like(Xf, ys[0], dtype=float)
                    # 端点は最近傍保持、内部は線形補間
                    vals = np.interp(Xf, xs, ys)
                    return vals

                alpha_full = _interp_series(Ns_plot, alpha_mean, N_full)
                beta_full  = _interp_series(Ns_plot, beta_mean , N_full)

                # αのみ数値を1行ずつ
                alpha_txt = os.path.join(out_dir, "alpha_values.txt")
                with open(alpha_txt, "w", encoding="utf-8") as fa:
                    for v in alpha_full:
                        fa.write(f"{float(v):.6f}\n")
                print(f"[plot] text saved -> {alpha_txt}  (lines = N {n_min}..{n_max})")

                # βのみ数値を1行ずつ
                beta_txt = os.path.join(out_dir, "beta_values.txt")
                with open(beta_txt, "w", encoding="utf-8") as fb:
                    for v in beta_full:
                        fb.write(f"{float(v):.6f}\n")
                print(f"[plot] text saved -> {beta_txt}   (lines = N {n_min}..{n_max})")

if __name__ == "__main__":
    main()