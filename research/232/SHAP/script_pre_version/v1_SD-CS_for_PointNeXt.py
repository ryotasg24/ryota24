#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SD-CS_for_PointNeXt.py

新たな提案手法：
  SD-CS : SHAP Downsampling with Coverage Scheduling

目的：
- SD-GLM(AMA_SHAP_for_PointNeXt.py) と同様に Region-SHAP / Point-SHAP を計算し、
  Hybrid-SHAP = α * Region-SHAP + (1 - α) * Point_adj
  を各点に持たせる（β は完全に廃止、あるいは 1.0 とみなして無視）。
- 点の選択は「点レベルのスケジューリング」：
    S(t) : t ステップ目までに選ばれた点集合
    d_i(t) : 点 i の「これまでの S(t) からの最短距離」を
             バウンディングボックス対角長で [0,1] に正規化した FPS 距離成分
    coverage(t) : 元の 2048 点 P に対する directed Hausdorff 半径に基づく被覆率
    coverage_target(N) : N 点サンプル時に目標とする被覆率（FPS/SD-GLM 実測値に基づく）
    gap_cov(t, N) = max(0, coverage_target(N) - coverage(t))

    スコア：
        score_i(t, N) = Hybrid[i] + λ_cov(N, t, gap_cov(t, N)) · d_i(t)

    λ_cov(N, t, gap) は、
        λ_cov(N, t, gap) = Λ_max(N) · f_t(t/N) · g_gap(gap)
    として設計し、
        - Λ_max(N): N に応じた距離成分の最大スケール（低Nほど大きく、高Nでは小さい）
        - f_t(t/N): t の進行に応じて前半重視で減衰させる関数
        - g_gap(gap): coverage が不足しているときだけ距離成分を強く効かせる on/off スケール
    とする。

アルゴリズム概要：
  1. Region-SHAP / Point-SHAP を計算し、L1 正規化でスケールを揃えた後、
     AMA Gate (weight-Estimator) により最適 α を推定して Hybrid-SHAP を構成する。
  2. ダウンサンプリングで N 点取りたいとき、
        「まだ選ばれていない点の中から 1 点選ぶ」
     ステップを t = 1〜N 回繰り返す。
  3. t=1 では Hybrid-SHAP のみで 1 点選択（FPS も1点目はランダムであることに対応）。
  4. t>=2 では、
        coverage(t-1) と gap_cov(t-1,N) を計算し、
        λ_cov(N, t-1, gap_cov) を用いて score_i を評価し、
        score_i が最大の点を 1 点だけ追加する。
     d_i(t-1) は S(t-1) からの最近傍距離をバウンディングボックス対角で割って正規化した値。
  5. coverage の計算は、全点に対する最近傍距離 dist_to_S をインクリメンタルに更新することで、
     directed Hausdorff 半径を高速（O(N) 更新）に評価する。

特徴：
- RegionScheduling-SD のような「Region 単位のタスクスケジューリング」ではなく、
  「点レベルの Hybrid-SHAP とグローバル coverage（directed HD）を組み合わせた
   Weighted Coverage + Hybrid-SHAP スケジューリング」となっている。
- coverage_target(N) は coverage_radius.py に基づく固定テーブルをコード内に埋め込み、
  N がテーブル値以外の場合には線形補間／最近傍による近似を行う。
- 既存の MID(N 非依存)キャッシュ（result/_shap_cache_mid/...）は読み取り専用の
  レガシーキャッシュとして利用し、新規の MID キャッシュは
      result/_shap_cache_mid_SDCS/...
  以下に保存して、既存キャッシュを上書きしない。
- SD-CS 固有の Scheduling(N 依存)キャッシュは
      result/_sdcs_cache_global/...
  とし、従来の Scheduling-SD の SCHED キャッシュとは分離する。

使い方（例）：
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
CUDA_VISIBLE_DEVICES=0 python SD-CS_for_PointNeXt.py \
    --ds_points 300 \
    --cache_mode auto \
    --pattern 10000 \
    --division_region 32 --division_point 64 \
    --mask_chunk 512 --min_mask_chunk 16 \
    --ama \
    --ama_mode heuristic \
    --no_l1

出力：
- result/SDCS-...(略)_dsSDCS_PointNeXt_h5/{ds_points}/<dataset_subdir>/<test_h5>.h5
  data: (num_samples, ds_points, 3), label: (num_samples,1)
  （attention_fusion 有効時は feat も含まれる）
- processing_times_by_class.txt, Monte_Carlo.txt

"""

import sys
import os
import gc
import numpy as np
import argparse
import shap
import h5py
from sklearn.cluster import KMeans
import time
import torch
import torch.nn.functional as F
import warnings
import math
from typing import Optional
from pathlib import Path

# 可視化（任意：現状未使用だが、将来のデバッグ用途として残す）
import matplotlib.pyplot as plt

# --- PointNeXt 関連 ---
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig

# provider.py
import provider

# ======================================================
# 保存用関数
# ======================================================
def save_h5_data(h5_filename, data, label, feat=None):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)
        if feat is not None:
            f.create_dataset('feat', data=feat)


# ===============================
# 各種関数の定義（AMA/RRから必要部分を踏襲）
# ===============================

def get_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def hierarchical_kmeans(points, k_region, fanout):
    """
    点群 points を
      - 上位 k_region 個の Region に分割
      - 各 Region を fanout 個のサブクラスタに分割
    する。

    戻り値：
        region_blocks : list[ list[int] ]
            各 Region が保持する点インデックス
        point_blocks  : list[ list[int] ]
            各 Region を fanout 分割したサブクラスタ（全体で k_region * fanout 個）
        km_reg        : KMeans
            Region クラスタリングの KMeans インスタンス
        subkm_dict    : dict[int, KMeans]
            各 Region に対するサブクラスタリングの KMeans
    """
    km_reg = KMeans(n_clusters=k_region, random_state=42).fit(points)
    reg_lbl = km_reg.labels_
    region_blocks = [np.where(reg_lbl == r)[0].tolist() for r in range(k_region)]

    point_blocks = []
    subkm_dict   = {}
    for r, idxs in enumerate(region_blocks):
        sub_pts = points[idxs]
        n_sub = fanout if len(sub_pts) >= fanout and fanout > 1 else max(1, len(sub_pts))
        km_sub = KMeans(n_clusters=n_sub, random_state=0).fit(sub_pts)
        subkm_dict[r] = km_sub
        for s in range(fanout):
            if s < n_sub:
                sub_idx = np.array(idxs)[km_sub.labels_ == s]
            else:
                sub_idx = np.array([], dtype=int)
            point_blocks.append(sub_idx.tolist())
    return region_blocks, point_blocks, km_reg, subkm_dict


def collect_background_point_clouds(target_label, train_files, num_point_clouds=50):
    """
    AMA と同様に、指定クラスの背景点群（1024点）を収集する。
    """
    background_point_clouds = []
    for file_path in train_files:
        data, labels = provider.loadDataFile(file_path)
        matching_indices = np.where(labels == target_label)[0]
        for idx in matching_indices:
            point_cloud = data[idx]
            if point_cloud.shape != (1024, 3):
                if point_cloud.shape[0] > 1024:
                    sampled_indices = np.random.choice(point_cloud.shape[0], 1024, replace=False)
                    point_cloud = point_cloud[sampled_indices, :]
                elif point_cloud.shape[0] < 1024:
                    continue
            if point_cloud.shape != (1024, 3):
                continue
            background_point_clouds.append(point_cloud)
            if len(background_point_clouds) >= num_point_clouds:
                return np.array(background_point_clouds)
    return np.array(background_point_clouds)


def load_pointnext_model(cfg_file, checkpoint_path, device):
    cfg = EasyConfig()
    cfg.load(cfg_file)
    model_cfg = cfg["model"]
    model = build_model_from_cfg(model_cfg)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def compute_background_baseline(background_point_clouds, kmeans):
    """
    Region-SHAP 用の背景ベースライン（各クラスタの平均座標）。
    """
    if background_point_clouds.ndim != 3:
        background_point_clouds = np.asarray(background_point_clouds)
    B, N, _ = background_point_clouds.shape
    X = background_point_clouds.reshape(B * N, 3)
    lbl = kmeans.predict(X)
    K = kmeans.n_clusters
    out = []
    for i in range(K):
        sel = (lbl == i)
        if np.any(sel):
            out.append(X[sel].mean(axis=0))
        else:
            out.append(np.zeros(3, dtype=X.dtype))
    return out


def compute_point_baseline(bg_pcs, km_reg, subkm_dict, fanout):
    """
    Point-SHAP 用の背景ベースライン（各サブクラスタの平均）。
    """
    k_region = km_reg.n_clusters
    baselines = []
    B, N, _ = bg_pcs.shape
    X = bg_pcs.reshape(B * N, 3)
    reg_lbl_all = km_reg.predict(X)
    for r in range(k_region):
        km_sub = subkm_dict[r]
        mask_r_all = (reg_lbl_all == r)
        if not np.any(mask_r_all):
            baselines.extend([np.zeros(3)] * fanout)
            continue
        X_r = X[mask_r_all]
        sub_lbl_all = km_sub.predict(X_r)
        for s in range(fanout):
            pts = X_r[sub_lbl_all == s]
            baselines.append(pts.mean(0) if len(pts) else np.zeros(3))
    return baselines


# 前計算：各点の block ID (N,) と per-point baseline (N,3)
def _precompute_pointwise_maps(blocks, baseline_reg_Blocks):
    N = sum(len(b) for b in blocks)
    block_ids = np.empty(N, dtype=np.int64)
    for b, idxs in enumerate(blocks):
        block_ids[np.array(idxs, dtype=np.int64)] = b
    base_pt = np.stack([baseline_reg_Blocks[b] for b in block_ids], axis=0).astype(np.float32)
    return block_ids, base_pt


def _predict_masks_in_chunks(mask_vectors, blocks, pc, baseline_reg_Blocks,
                             model, device, chunk_size: int = 512,
                             pin_mem: bool = True, do_empty_cache: bool = False,
                             min_chunk_size: int = 8):
    """
    SHAP KernelExplainer 用の予測ラッパ。
    mask_vectors: (M,K) の 0/1 マスク
    """
    model.eval()
    M = len(mask_vectors)
    if M == 0:
        return np.array([])

    block_ids, base_pt = _precompute_pointwise_maps(blocks, baseline_reg_Blocks)
    pc_t = torch.from_numpy(pc.astype(np.float32)).to(device).unsqueeze(0)
    base_t = torch.from_numpy(base_pt).to(device).unsqueeze(0)

    if pin_mem:
        pc_t = pc_t.contiguous()
        base_t = base_t.contiguous()

    out_list = []
    bid_t = torch.from_numpy(block_ids).to(device)
    step = int(chunk_size)
    s = 0
    while s < M:
        e = min(M, s + step)
        mv = np.asarray(mask_vectors[s:e], dtype=np.float32)
        try:
            if pin_mem and device.type == "cuda":
                mv_t_cpu = torch.from_numpy(mv).pin_memory()
                mv_t = mv_t_cpu.to(device, non_blocking=True)
            else:
                mv_t = torch.from_numpy(mv).to(device)

            keep = mv_t[:, bid_t].unsqueeze(-1)
            pcB   = pc_t.expand(keep.shape[0], -1, -1)
            baseB = base_t.expand(keep.shape[0], -1, -1)
            masked = keep * pcB + (1.0 - keep) * baseB

            with torch.no_grad():
                logits = model(masked)

            out_list.append(logits.detach().cpu().numpy())

            del mv_t, keep, pcB, baseB, masked, logits
            if do_empty_cache:
                torch.cuda.empty_cache()
            s = e
        except RuntimeError as err:
            oom = ("CUDA out of memory" in str(err)) or ("CUDA error: out of memory" in str(err))
            if not oom or step <= min_chunk_size:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            new_step = max(min_chunk_size, step // 2)
            print(f"[WARN] OOM detected. Reduce mask_chunk: {step} -> {new_step}")
            step = new_step

    return np.concatenate(out_list, axis=0)


def shap_predict_block_mask(mask_vectors, blocks, original_point_cloud,
                            baseline_reg_Blocks, model, device,
                            chunk_size=512, pin_mem=True, min_chunk_size=8):
    return _predict_masks_in_chunks(mask_vectors, blocks, original_point_cloud,
                                    baseline_reg_Blocks, model, device,
                                    chunk_size=chunk_size, pin_mem=pin_mem,
                                    min_chunk_size=min_chunk_size)


def compute_point_level_contributions(point_cloud, blocks, baseline_reg_Blocks,
                                      model, target_class_index, device):
    """
    各点の局所寄与度 (grad ⋅ (x - baseline)) を計算。
    """
    model.eval()
    pc_tensor = torch.tensor(point_cloud, dtype=torch.float32,
                             device=device, requires_grad=True)
    pc_tensor = pc_tensor.unsqueeze(0)
    pc_tensor.retain_grad()

    output = model(pc_tensor)
    target_output = output[0, target_class_index]
    target_output.backward()

    grad_val = pc_tensor.grad.squeeze(0).cpu().numpy()

    point_contrib = np.zeros(point_cloud.shape[0], dtype=np.float32)
    for i, block in enumerate(blocks):
        baseline = baseline_reg_Blocks[i]
        for idx in block:
            diff = point_cloud[idx] - baseline
            point_contrib[idx] = np.dot(grad_val[idx], diff)
    return point_contrib


def compute_point_level_shap_values(point_cloud, blocks, block_shap_values,
                                    baseline_reg_Blocks, model,
                                    target_class_index, device,
                                    scaling_factor=1.0):
    """
    Region-SHAP と勾配ベースの Point 寄与度を組み合わせた Point-SHAP を計算。
    定義:
        point_shap[idx] = block_shap_values[block] + scaling_factor * zscore(contrib_in_block)
    """
    point_contrib = compute_point_level_contributions(
        point_cloud, blocks, baseline_reg_Blocks,
        model, target_class_index, device
    )
    point_shap = np.zeros(point_cloud.shape[0], dtype=np.float32)
    for i, block in enumerate(blocks):
        block_contribs = point_contrib[block]
        mean_val = np.mean(block_contribs)
        std_val = np.std(block_contribs)
        if std_val < 1e-6:
            normalized_contrib = np.zeros_like(block_contribs)
        else:
            normalized_contrib = (block_contribs - mean_val) / std_val
        for j, idx in enumerate(block):
            point_shap[idx] = block_shap_values[i] + scaling_factor * normalized_contrib[j]
    return point_shap


def l1_match_point(point_shap: np.ndarray, region_per_pt: np.ndarray, eps: float):
    """
    Region-SHAP と Point-SHAP の L¹ノルムを揃えるためのスケーリング。
    Point-SHAP 全体に一定スケールを掛けるだけなので、
    点同士の順位は変わらない。
    """
    L1_reg = float(np.sum(np.abs(region_per_pt))) + eps
    L1_pnt = float(np.sum(np.abs(point_shap))) + eps
    scale  = L1_reg / L1_pnt
    return point_shap * scale, L1_reg, L1_pnt


# 参考：Attention Fusion 特徴生成（任意）
def _knn_idx(x: np.ndarray, k: int) -> np.ndarray:
    K = x.shape[0]
    if K <= 1:
        return np.zeros((K, 1), dtype=int)
    k = min(k, K-1)
    d = np.linalg.norm(x[None,:,:] - x[:,None,:], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def compute_attention_fusion_features(points: np.ndarray, alpha: float,
                                      feat_dim: int = 32, k_local: int = 16) -> np.ndarray:
    """
    AMA と同様の Attention Fusion 特徴（任意）。
    Global/Local の幾何特徴を α で線形結合し、feat_dim 次元に再構成する。
    """
    K = points.shape[0]
    c = points.mean(axis=0, keepdims=True)
    gstd = points.std(axis=0, keepdims=True) + 1e-9
    r = np.linalg.norm(points - c, axis=1, keepdims=True)
    gfeat = np.concatenate([np.repeat(c, K, axis=0) - points,
                            np.repeat(gstd, K, axis=0),
                            r], axis=1)
    idx = _knn_idx(points, k_local)
    nbr = points[idx]
    off = nbr - points[:,None,:]
    lfeat = np.concatenate([off.mean(axis=1),
                            off.std(axis=1),
                            np.linalg.norm(off, axis=2).max(axis=1, keepdims=True)], axis=1)
    Ff = alpha * gfeat + (1.0 - alpha) * lfeat
    rep = int(math.ceil(float(feat_dim) / Ff.shape[1]))
    Ff = np.tile(Ff, (1, rep))[:, :feat_dim].astype(np.float32)
    return Ff


# ===============================
# AMA Gate (α推定) 関連（β は廃止）
# ===============================

class AdaptiveGateMLP(torch.nn.Module):
    """
    AMA 用の Gate-MLP。
    出力は (alpha, beta) だが、SD-CS では alpha のみ使用し、
    beta（スケール）は無視する。
    ネットワーク構造自体は AMA_SHAP と同じで、既存 ckpt 互換。
    """
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3)
        )

    def forward(self, x: torch.Tensor):
        y = self.net(x)                 # (B,3)
        lg, ll, b = y[:,0:1], y[:,1:2], y[:,2:3]
        a2 = torch.softmax(torch.cat([lg, ll], dim=1), dim=1)  # (B,2)
        alpha = a2[:,0]                 # global の比率
        beta  = F.softplus(b).squeeze(1)  # 互換性のため残すが SD-CS では使用しない
        return alpha, beta


def gate_forward(N: int, var_reg: float, var_pnt: float,
                 args, gate_model: Optional[AdaptiveGateMLP], device,
                 scaler: Optional[dict] = None) -> float:
    """
    α を決定する Gate。
    - ama_mode == "mlp" かつ gate_model が与えられていれば、MLP で推定。
    - それ以外は heuristic で決定。
    戻り値は α（0〜1）。
    """
    if args.ama_mode == "mlp" and gate_model is not None:
        eps = 1e-9
        feat = np.array([float(N)/1024.0,
                         math.log(var_reg + eps),
                         math.log(var_pnt + eps)], dtype=np.float32)
        if scaler is not None:
            mu = scaler["mu"]; std = scaler["std"]
            feat = (feat - mu) / (std + 1e-9)
        x = torch.tensor(feat[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            a, b = gate_model(x)
        alpha = float(a.item())
        return alpha

    # heuristic: N が小さいほど α↑、分散比も考慮
    Nmid = max(1, args.ama_Nmid)
    k    = float(args.ama_k)
    termN = -k * ((float(N) - Nmid) / Nmid)
    logr  = math.log((var_reg + 1e-9) / (var_pnt + 1e-9))
    logit = termN + 0.5 * logr
    alpha = 1.0 / (1.0 + math.exp(-logit))
    return alpha


# ===============================
# Coverage Scheduling (点レベル) 関連
# ===============================

# coverage_target(N) テーブル（coverage_radius.py の実測値に基づく）
COVERAGE_TARGET_TABLE = {
    100: 0.927338,
    200: 0.949036,
    300: 0.958705,
    400: 0.964581,
    500: 0.913272,
    600: 0.919857,
    700: 0.925192,
    800: 0.930477,
    900: 0.935687,
    1000: 0.940567,
}


def get_coverage_target(N: int) -> float:
    """
    coverage_target(N) を返す。
    N がテーブルに存在しない場合は、近傍のキーで線形補間／最近傍近似を行う。
    """
    if N in COVERAGE_TARGET_TABLE:
        return float(COVERAGE_TARGET_TABLE[N])

    ks = sorted(COVERAGE_TARGET_TABLE.keys())
    if N <= ks[0]:
        return float(COVERAGE_TARGET_TABLE[ks[0]])
    if N >= ks[-1]:
        return float(COVERAGE_TARGET_TABLE[ks[-1]])

    # 線形補間
    for i in range(len(ks) - 1):
        k0, k1 = ks[i], ks[i+1]
        if k0 <= N <= k1:
            v0 = COVERAGE_TARGET_TABLE[k0]
            v1 = COVERAGE_TARGET_TABLE[k1]
            ratio = (float(N) - k0) / float(k1 - k0)
            return float(v0 + (v1 - v0) * ratio)

    # フォールバック（理論上ここには来ない）
    return float(COVERAGE_TARGET_TABLE[ks[-1]])


class GlobalCoverageState:
    """
    全点群 P に対する directed Hausdorff 半径ベースの被覆率を
    インクリメンタルに管理するクラス。

    定義：
        P : 元点群 points (N,3)
        S : これまでに選択された点の集合（インデックス集合）

    更新：
        新しい点 y_idx を S に追加するたびに、
            d_new[x] = || P[x] - P[y_idx] ||_2
        dist_to_S = min(dist_to_S, d_new)

    被覆率：
        r_cov = max_x dist_to_S[x]
        r_norm = r_cov / diag
        cov    = 1 - r_norm

    ここで diag は P のバウンディングボックス対角長 + eps。
    """

    def __init__(self, points: np.ndarray, eps: float = 1e-12):
        self.points = points.astype(np.float32, copy=False)
        self.N = self.points.shape[0]
        self.eps = float(eps)
        if self.N > 0:
            bbox_min = np.min(self.points, axis=0)
            bbox_max = np.max(self.points, axis=0)
            self.diag = float(np.linalg.norm(bbox_max - bbox_min) + self.eps)
        else:
            self.diag = self.eps

        self.dist_to_S = np.full(self.N, np.inf, dtype=np.float32)
        self.has_selected = False

    def update_with_point(self, idx: int):
        if self.N == 0:
            return
        idx = int(idx)
        y = self.points[idx]
        diff = self.points - y
        d_new = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
        if not self.has_selected:
            self.dist_to_S = d_new
        else:
            self.dist_to_S = np.minimum(self.dist_to_S, d_new)
        self.has_selected = True

    def get_coverage(self) -> float:
        if not self.has_selected or self.diag <= 0.0 or self.N == 0:
            return 0.0
        r_cov = float(np.max(self.dist_to_S))
        r_norm = r_cov / self.diag
        score = 1.0 - r_norm
        if not np.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))

    def get_normalized_distances(self) -> np.ndarray:
        """
        各点の「S からの最近傍距離 / diag」を返す。
        S が空の間は 0 ベクトルを返す。
        """
        if not self.has_selected or self.diag <= 0.0 or self.N == 0:
            return np.zeros(self.N, dtype=np.float32)
        d_norm = self.dist_to_S / self.diag
        d_norm = np.clip(d_norm, 0.0, 1.0).astype(np.float32)
        return d_norm


def compute_lambda_cov(N: int,
                       step_idx: int,
                       coverage: float,
                       coverage_target: float,
                       lambda_scale: float = 1.0) -> float:
    """
    λ_cov(N, t, gap) を計算する。
      t = step_idx + 1 とし、
      gap = max(0, coverage_target - coverage)
    とする。

    ・Λ_max(N):
        N<=200  : 1.0
        200<N<=400 : 0.8
        400<N<=600 : 0.5
        600<N<=800 : 0.3
        800<N     : 0.2
      に設定し、lambda_scale で全体スケールを調整可能。

    ・f_t(t/N) = max(0, 1 - t/N):
        ステップが進むほど距離成分を弱める。

    ・g_gap(gap):
        gap >= 0.05 で 1.0 に飽和、それ未満では gap/0.05 で線形スケール。
    """
    if N <= 0:
        return 0.0

    # Λ_max(N)
    if N <= 200:
        Lmax = 1.0
    elif N <= 400:
        Lmax = 0.8
    elif N <= 600:
        Lmax = 0.5
    elif N <= 800:
        Lmax = 0.3
    else:
        Lmax = 0.2
    Lmax *= float(lambda_scale)

    t = step_idx + 1
    s = float(t) / float(N)
    f_t = max(0.0, 1.0 - s)

    gap = max(0.0, coverage_target - coverage)
    gap_scale = 0.05
    if gap >= gap_scale:
        g_gap = 1.0
    else:
        g_gap = gap / gap_scale if gap_scale > 0 else 0.0

    return float(Lmax * f_t * g_gap)


def sdcs_select(points: np.ndarray,
                hybrid_scores: np.ndarray,
                target_N: int,
                coverage_target: float,
                lambda_scale: float = 1.0,
                debug: bool = False) -> np.ndarray:
    """
    SD-CS（点レベル Coverage Scheduling）による点選択アルゴリズム。

    引数：
        points        : (N,3) 点群
        hybrid_scores : (N,) 各点の Hybrid-SHAP
        target_N      : 選択する点数
        coverage_target : coverage_target(N)
        lambda_scale  : λ_cov の全体スケール
        debug         : True の場合、先頭数ステップのログを出力

    アルゴリズム：
        S(0) = ∅
        step=0:
            Hybrid-SHAP のみで 1 点選択（score_i = Hybrid[i]）。
        step>=1:
            coverage(t) と d_i(t) を GlobalCoverageState から取得し、
            λ_cov を計算して
                score_i = Hybrid[i] + λ_cov * d_i
            が最大の点を 1 点選ぶ。
    """
    N_all = points.shape[0]
    if N_all == 0:
        return np.zeros(0, dtype=np.int32)

    target_N = int(min(max(target_N, 1), N_all))
    hybrid_scores = hybrid_scores.astype(np.float32, copy=False)

    cov_state = GlobalCoverageState(points)
    selected = []
    selected_flag = np.zeros(N_all, dtype=bool)

    for step in range(target_N):
        # step=0: Hybrid のみで 1 点選択
        if step == 0:
            scores = hybrid_scores.copy()
            scores[selected_flag] = -np.inf
            chosen_idx = int(np.argmax(scores))
            selected.append(chosen_idx)
            selected_flag[chosen_idx] = True
            cov_state.update_with_point(chosen_idx)
            if debug:
                cov_now = cov_state.get_coverage()
                print(f"[DEBUG_SDCS] step={step:03d}, pid={chosen_idx:04d}, "
                      f"hybrid={float(hybrid_scores[chosen_idx]):.6f}, "
                      f"coverage={cov_now:.4f}, lambda=0.0000, d_norm=0.0000")
            continue

        # coverage と d_norm を取得
        coverage = cov_state.get_coverage()
        d_norm = cov_state.get_normalized_distances()  # (N,)

        lam = compute_lambda_cov(
            N=target_N,
            step_idx=step,
            coverage=coverage,
            coverage_target=coverage_target,
            lambda_scale=lambda_scale
        )

        scores = hybrid_scores + lam * d_norm
        scores[selected_flag] = -np.inf

        # すべて -inf になってしまった場合は Hybrid のみで fallback
        if not np.isfinite(scores).any():
            scores = hybrid_scores.copy()
            scores[selected_flag] = -np.inf

        chosen_idx = int(np.argmax(scores))
        selected.append(chosen_idx)
        selected_flag[chosen_idx] = True
        cov_state.update_with_point(chosen_idx)

        if debug and step < 40:
            cov_now = cov_state.get_coverage()
            d_chosen = float(d_norm[chosen_idx]) if d_norm.size > 0 else 0.0
            print(f"[DEBUG_SDCS] step={step:03d}, pid={chosen_idx:04d}, "
                  f"hybrid={float(hybrid_scores[chosen_idx]):.6f}, "
                  f"coverage={cov_now:.4f}, lambda={lam:.4f}, d_norm={d_chosen:.4f}")

    return np.asarray(selected[:target_N], dtype=np.int32)


# ===============================
# main
# ===============================

def main():
    parser = argparse.ArgumentParser(
        description="SD-CS (SHAP Downsampling with Coverage Scheduling) for ModelNet40 test files using PointNeXt."
    )
    parser.add_argument("--ds_points", type=int, default=500,
                        help="ダウンサンプリング後の点数")
    parser.add_argument("--pattern", type=int, default=10000,
                        help="Region-SHAP の近似マスクパターン数")
    parser.add_argument("--division_region", type=int, default=32,
                        help="Region SHAP 用 k (Region 数)")
    parser.add_argument("--division_point", type=int, default=64,
                        help="Point SHAP 用 k (Region の整数倍にすること)")
    parser.add_argument("--cache_mode", type=str, default="auto",
                        choices=["auto", "save", "load"],
                        help="auto=存在すればload/なければsave, save=常に保存, load=常に読み込み")
    parser.add_argument("--num_groups", type=int, default=1,
                        help="40クラスを何グループに分割するか")
    parser.add_argument("--group_index", type=int, default=0,
                        help="処理するグループのインデックス (0-indexed)")
    parser.add_argument("--cfg_file", type=str,
                        default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth")
    parser.add_argument("--dataset", type=str,
                        default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048")
    parser.add_argument("--run_tag", type=str, default="",
                        help="任意のタグ。出力ディレクトリ名やキャッシュ名に付与される")
    # L1正規化
    parser.add_argument("--no_l1", action="store_true",
                        help="True のとき Point-SHAP の L1 正規化を行わない")
    parser.add_argument("--ama_eps", type=float, default=1e-6,
                        help="L1 正規化時の ε")
    # AMA(Gate) 関連
    parser.add_argument("--ama", action="store_true",
                        help="Adaptive Multi-Scale Attention (Gate) を用いて α を推定する")
    parser.add_argument("--ama_mode", choices=["heuristic", "mlp"], default="heuristic",
                        help="α の決定を heuristic か MLP で行うか")
    parser.add_argument("--ama_Nmid", type=int, default=300,
                        help="heuristic Gate 用の N の基準点")
    parser.add_argument("--ama_k", type=float, default=4.0,
                        help="heuristic Gate 用の傾きパラメータ")
    parser.add_argument("--gate_ckpt", type=str, default="",
                        help="学習済み Gate-MLP (.pth)。ama_mode=mlp のときに使用")
    parser.add_argument("--gate_hidden", type=int, default=8,
                        help="Gate-MLP の隠れ次元")
    parser.add_argument("--gate_scaler", type=str, default="",
                        help="Gate-MLP 入力の前処理スケーラ (.npz: mu,std)")
    # 推論チャンク
    parser.add_argument("--mask_chunk", type=int, default=512)
    parser.add_argument("--min_mask_chunk", type=int, default=8)
    parser.add_argument("--no_pinmem", action="store_true",
                        help="pinned memory を使用しない")
    parser.add_argument("--force_mid_build", action="store_true",
                        help="MIDキャッシュがあっても Region/Point-SHAP を再計算する")
    # Attention Fusion（任意）
    parser.add_argument("--attention_fusion", action="store_true",
                        help="ダウンサンプル後の点群に対して AF 特徴 'feat' を生成して保存する")
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--knn_k", type=int, default=16)
    # Coverage 関連（λ_cov の全体スケール）
    parser.add_argument("--lambda_scale", type=float, default=1.0,
                        help="Coverage 距離成分 λ_cov の全体スケール")
    # 互換性のためのダミー引数（RegionScheduling-SD で使用していたが SD-CS では未使用）
    parser.add_argument("--cov_thresh", type=float, default=0.90,
                        help="(SD-CS では未使用。互換性のために残している)")
    # デバッグ
    parser.add_argument("--debug_sched", action="store_true",
                        help="SD-CS Scheduling の最初の取り出しログを表示する")

    args = parser.parse_args()
    ds_points = args.ds_points
    pattern = args.pattern
    k_region = args.division_region
    k_point  = args.division_point
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = args.dataset

    if k_point % k_region != 0:
        sys.exit("--division_point は --division_region の整数倍にして下さい")
    fanout = k_point // k_region
    print(f"[INFO] Region k={k_region}, Point k={k_point}, fan-out={fanout}")

    # AMA Gate モデル読み込み（必要なら）
    gate_model = None
    gate_scaler = None
    if args.ama and args.ama_mode == "mlp" and args.gate_ckpt:
        gate_model = AdaptiveGateMLP(hidden=args.gate_hidden)
        gate_model.load_state_dict(torch.load(args.gate_ckpt, map_location="cpu"))
        gate_model.to(device).eval()
        if args.gate_scaler and os.path.exists(args.gate_scaler):
            _sc = np.load(args.gate_scaler)
            gate_scaler = {
                "mu": _sc["mu"].astype(np.float32),
                "std": _sc["std"].astype(np.float32)
            }
        else:
            print("[WARN] Gate-MLP scaler not found. Using raw features for gate_forward().")

    DATA_DIR_TRAIN = "/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048"
    DATA_DIR_TEST  = args.dataset

    # クラス名読み込み
    class_names_file = os.path.join(DATA_DIR_TRAIN, "shape_names.txt")
    class_names = get_class_names(class_names_file)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.")

    # グループ分割（AMA互換）
    num_groups = args.num_groups
    group_index = args.group_index
    groups = []
    base = num_classes // num_groups
    remainder = num_classes % num_groups
    start = 0
    for g in range(num_groups):
        extra = 1 if g < remainder else 0
        end = start + base + extra
        groups.append(list(range(start, end)))
        start = end
    if group_index < 0 or group_index >= num_groups:
        sys.exit(f"Invalid group_index {group_index}.")
    selected_classes = groups[group_index]
    print(f"Processing group {group_index} with classes: {selected_classes}")

    # 出力先
    l1_tag = "-nol1" if args.no_l1 else ""
    ama_tag = "noAMA"
    if args.ama:
        ama_tag = args.ama_mode
    _out_tag_base = f"SDCS{l1_tag}-{ama_tag}"
    out_tag = _out_tag_base if args.run_tag == "" else f"{_out_tag_base}-{args.run_tag}"
    output_folder = os.path.join(
        "result", f"{out_tag}_kr{k_region}_kp{k_point}_p{pattern}_dsSDCS_PointNeXt_h5", str(ds_points)
    )
    os.makedirs(output_folder, exist_ok=True)

    # SD-CS Scheduling(N依存)キャッシュ
    BASE_SDCS_CACHE_DIR = os.path.join("result", "_sdcs_cache_global")
    os.makedirs(BASE_SDCS_CACHE_DIR, exist_ok=True)
    sched_cache_tag = f"sdcs_p{pattern}_kr{k_region}_kp{k_point}_N{ds_points}{('_noL1' if args.no_l1 else '')}_{ama_tag}"
    if args.run_tag:
        sched_cache_tag += f"_{args.run_tag}"
    SDCS_CACHE_DIR = os.path.join(BASE_SDCS_CACHE_DIR, sched_cache_tag)
    os.makedirs(SDCS_CACHE_DIR, exist_ok=True)

    # MID(N非依存)キャッシュ（SD-CS 用に新ディレクトリを用意）
    MID_CACHE_BASE_NEW = os.path.join("result", "_shap_cache_mid_SDCS")
    os.makedirs(MID_CACHE_BASE_NEW, exist_ok=True)
    # 既存の MID キャッシュ（AMA/RegionScheduling-SD）をレガシー読み取り専用として利用
    MID_CACHE_BASE_OLD = os.path.join("result", "_shap_cache_mid")
    ckpt_name_for_tag = Path(args.checkpoint_path).stem
    mid_tag = f"p{pattern}_kr{k_region}_kp{k_point}_ckpt-{ckpt_name_for_tag}{('_noL1' if args.no_l1 else '')}"
    if args.run_tag:
        mid_tag += f"_{args.run_tag}"
    MID_CACHE_DIR_NEW = os.path.join(MID_CACHE_BASE_NEW, mid_tag)
    os.makedirs(MID_CACHE_DIR_NEW, exist_ok=True)
    MID_CACHE_DIR_OLD = os.path.join(MID_CACHE_BASE_OLD, mid_tag)  # 存在するかどうかはサンプルごとに確認

    # train/test file list
    TRAIN_FILES = provider.getDataFiles(os.path.join(DATA_DIR_TRAIN, "train_files.txt"))
    TEST_FILES  = provider.getDataFiles(os.path.join(DATA_DIR_TEST,  "test_files.txt"))

    # 背景点群収集（対象クラスのみ）
    backgrounds = {}
    for cls in selected_classes:
        print(f"Collecting background for class '{class_names[cls]}' (idx={cls})...")
        bg = collect_background_point_clouds(cls, TRAIN_FILES, num_point_clouds=50)
        backgrounds[cls] = bg

    # モデル読み込み
    model, cfg = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)

    # 時間統計
    class_time_stats = {
        cls: {
            'sum_total': 0.0,
            'sum_down': 0.0,
            'sum_region': 0.0,
            'sum_point': 0.0,
            'sum_sched': 0.0,
            'sum_shap': 0.0,
            'count': 0
        }
        for cls in selected_classes
    }
    overall_total = overall_down = overall_region = overall_point = overall_sched = overall_shap = 0.0
    overall_count = 0

    # Monte-Carlo用
    all_deltas = []
    all_s      = []
    all_se     = []

    cov_target_for_N = get_coverage_target(ds_points)
    print(f"[INFO] coverage_target(N={ds_points}) = {cov_target_for_N:.6f}")

    for test_file in TEST_FILES:
        print(f"\n----- Processing test file: {test_file} -----")
        data, labels = provider.loadDataFile(test_file)
        num_samples = len(data)
        print(f"Total samples in {os.path.basename(test_file)}: {num_samples}")

        base_name = os.path.basename(test_file)
        name_without_ext  = os.path.splitext(base_name)[0]
        subdir = os.path.basename(DATA_DIR_TEST)
        output_filename = os.path.join(output_folder, subdir, base_name)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if os.path.exists(output_filename):
            print(f"[CLEAN] Remove existing output: {output_filename}")
            os.remove(output_filename)

        sampled_data_list = []
        sampled_label_list = []
        sampled_feat_list  = []

        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}")
            cls = int(labels[i])
            if cls not in selected_classes:
                continue

            gc.collect()

            bg_pc = backgrounds.get(cls, None)
            if bg_pc is None or bg_pc.size == 0:
                print(f"Skipping sample {i} (class {cls}): No background.")
                continue

            pc = data[i].astype(np.float32)

            # キャッシュID
            sample_id  = f"{name_without_ext}_{i:05d}_r{k_region}_p{k_point}"
            sched_cache_path = os.path.join(SDCS_CACHE_DIR, sample_id + ".npz")
            mid_cache_path_new = os.path.join(MID_CACHE_DIR_NEW, sample_id + ".npz")
            mid_cache_path_old = os.path.join(MID_CACHE_DIR_OLD, sample_id + ".npz")

            mid_hit_new = os.path.exists(mid_cache_path_new)
            mid_hit_old = os.path.exists(mid_cache_path_old)
            if mid_hit_new:
                mid_cache_path_read = mid_cache_path_new
                mid_hit = True
            elif mid_hit_old:
                mid_cache_path_read = mid_cache_path_old
                mid_hit = True
            else:
                mid_cache_path_read = None
                mid_hit = False

            use_sched_cache  = (args.cache_mode in ["auto", "load"]) and os.path.exists(sched_cache_path) and (not args.force_mid_build)
            save_sched_cache = (args.cache_mode in ["auto", "save"]) and (not use_sched_cache)

            # timing init
            sample_region_time = 0.0
            sample_point_time  = 0.0
            sample_sched_time  = 0.0
            sample_total_time  = 0.0
            sample_down_time   = 0.0
            sample_shap_time   = 0.0

            # =======================
            # 1) Scheduling cache hit
            # =======================
            if use_sched_cache:
                print(f"  ↳ SD-CS scheduling cache hit: {sched_cache_path}")
                z = np.load(sched_cache_path)
                top_indices = z["top_idx"].astype(np.int32).reshape(-1)
                ds_pc = pc[top_indices, :]
                # 全時間は 0 のまま（キャッシュヒット）
            else:
                # ==========================
                # 2) MID cache hit: Region/Point-SHAP は再利用
                # ==========================
                if mid_hit and (not args.force_mid_build):
                    print(f"  ↳ MID cache hit: {mid_cache_path_read}")
                    _mid = np.load(mid_cache_path_read)
                    region_per_pt = _mid["region_per_pt"].astype(np.float32).reshape(-1)
                    point_adj     = _mid["point_adj"].astype(np.float32).reshape(-1)
                    if "pc" in _mid.files:
                        pc = _mid["pc"].astype(np.float32)

                    t0 = time.time()

                    var_reg = float(np.var(region_per_pt))
                    var_pnt = float(np.var(point_adj))
                    if args.ama:
                        alpha = gate_forward(ds_points, var_reg, var_pnt,
                                             args, gate_model, device, gate_scaler)
                    else:
                        alpha = 0.5
                    hybrid = alpha * region_per_pt + (1.0 - alpha) * point_adj

                    sched0 = time.time()
                    top_indices = sdcs_select(
                        pc, hybrid, ds_points, cov_target_for_N,
                        lambda_scale=args.lambda_scale,
                        debug=args.debug_sched
                    )
                    sched1 = time.time()
                    sample_sched_time = sched1 - sched0

                    ds_pc = pc[top_indices, :]
                    t_final = time.time()
                    sample_total_time = t_final - t0
                    sample_down_time = sample_total_time - sample_sched_time
                    sample_region_time = 0.0
                    sample_point_time  = 0.0
                    sample_shap_time   = 0.0

                # ==========================
                # 3) MID cache なし: Region/Point-SHAP をフル計算
                # ==========================
                else:
                    if pc.shape[0] == 2048:
                        # ---------- heavy (2048 → A/B 1024) ----------
                        pc_A = pc[:1024]
                        pc_B = pc[1024:]
                        reg_A, pt_A, kmA_reg, kmA_sub = hierarchical_kmeans(pc_A, k_region, fanout)
                        reg_B, pt_B, kmB_reg, kmB_sub = hierarchical_kmeans(pc_B, k_region, fanout)
                        baseline_reg_A = compute_background_baseline(bg_pc, kmA_reg)
                        baseline_reg_B = compute_background_baseline(bg_pc, kmB_reg)
                        baseline_pt_A  = compute_point_baseline(bg_pc, kmA_reg, kmA_sub, fanout)
                        baseline_pt_B  = compute_point_baseline(bg_pc, kmB_reg, kmB_sub, fanout)
                        target_explain_class_index = cls

                        t0 = time.time()

                        # --- Region SHAP A ---
                        preds_A = []
                        def shap_predict_A(mv):
                            out = shap_predict_block_mask(
                                mv, reg_A, pc_A, baseline_reg_A, model, device,
                                chunk_size=args.mask_chunk,
                                pin_mem=(not args.no_pinmem),
                                min_chunk_size=args.min_mask_chunk
                            )
                            preds_A.append(out)
                            return out
                        explainer_A = shap.KernelExplainer(
                            shap_predict_A,
                            np.zeros((1, len(reg_A)))
                        )
                        rA0 = time.time()
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                            block_A = (
                                explainer_A.shap_values(
                                    np.ones((1, len(reg_A))),
                                    nsamples=pattern,
                                    l1_reg="num_features(10)"
                                )[target_explain_class_index].reshape(-1)
                            )
                            preds_A_np = np.concatenate(preds_A, axis=0)
                            delta_A = preds_A_np[:, target_explain_class_index] - \
                                      explainer_A.expected_value[target_explain_class_index]
                            s_A  = float(delta_A.std(ddof=1))
                            M_A  = delta_A.shape[0]
                            SE_A = s_A / np.sqrt(M_A)
                            all_deltas.append(delta_A); all_s.append(s_A); all_se.append(SE_A)
                        rA1 = time.time()

                        # --- Point SHAP A ---
                        pA0 = time.time()
                        block_A_pt = np.repeat(block_A, fanout)
                        pt_shap_A  = compute_point_level_shap_values(
                            pc_A, pt_A, block_A_pt, baseline_pt_A,
                            model, target_explain_class_index, device,
                            scaling_factor=1.0
                        )
                        pA1 = time.time()

                        # --- Region SHAP B ---
                        preds_B = []
                        def shap_predict_B(mv):
                            out = shap_predict_block_mask(
                                mv, reg_B, pc_B, baseline_reg_B, model, device,
                                chunk_size=args.mask_chunk,
                                pin_mem=(not args.no_pinmem),
                                min_chunk_size=args.min_mask_chunk
                            )
                            preds_B.append(out)
                            return out
                        explainer_B = shap.KernelExplainer(
                            shap_predict_B,
                            np.zeros((1, len(reg_B)))
                        )
                        rB0 = time.time()
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                            block_B = (
                                explainer_B.shap_values(
                                    np.ones((1, len(reg_B))),
                                    nsamples=pattern,
                                    l1_reg="num_features(10)"
                                )[target_explain_class_index].reshape(-1)
                            )
                            preds_B_np = np.concatenate(preds_B, axis=0)
                            delta_B = preds_B_np[:, target_explain_class_index] - \
                                      explainer_B.expected_value[target_explain_class_index]
                            s_B  = float(delta_B.std(ddof=1))
                            M_B  = delta_B.shape[0]
                            SE_B = s_B / np.sqrt(M_B)
                            all_deltas.append(delta_B); all_s.append(s_B); all_se.append(SE_B)
                        rB1 = time.time()

                        # --- Point SHAP B ---
                        pB0 = time.time()
                        block_B_pt = np.repeat(block_B, fanout)
                        pt_shap_B  = compute_point_level_shap_values(
                            pc_B, pt_B, block_B_pt, baseline_pt_B,
                            model, target_explain_class_index, device,
                            scaling_factor=1.0
                        )
                        pB1 = time.time()

                        # --- Region per-point (A/B) ---
                        region_A_exp = np.repeat(block_A, fanout)
                        region_A_per_pt = np.zeros_like(pt_shap_A)
                        for b_idx, idxs in enumerate(pt_A):
                            region_A_per_pt[idxs] = region_A_exp[b_idx]

                        region_B_exp = np.repeat(block_B, fanout)
                        region_B_per_pt = np.zeros_like(pt_shap_B)
                        for b_idx, idxs in enumerate(pt_B):
                            region_B_per_pt[idxs] = region_B_exp[b_idx]

                        # --- Point-SHAP L1 正規化 ---
                        if args.no_l1:
                            pt_adj_A = pt_shap_A
                            pt_adj_B = pt_shap_B
                        else:
                            pt_adj_A, _, _ = l1_match_point(pt_shap_A, region_A_per_pt, args.ama_eps)
                            pt_adj_B, _, _ = l1_match_point(pt_shap_B, region_B_per_pt, args.ama_eps)

                        # --- グローバル per-point 配列 ---
                        region_per_pt = np.concatenate([region_A_per_pt, region_B_per_pt], axis=0)
                        point_adj     = np.concatenate([pt_adj_A,      pt_adj_B],      axis=0)

                        # Gate & Hybrid-SHAP
                        var_reg = float(np.var(region_per_pt))
                        var_pnt = float(np.var(point_adj))
                        if args.ama:
                            alpha = gate_forward(ds_points, var_reg, var_pnt,
                                                 args, gate_model, device, gate_scaler)
                        else:
                            alpha = 0.5
                        hybrid = alpha * region_per_pt + (1.0 - alpha) * point_adj

                        # SD-CS Scheduling selection
                        sched0 = time.time()
                        top_indices = sdcs_select(
                            pc, hybrid, ds_points, cov_target_for_N,
                            lambda_scale=args.lambda_scale,
                            debug=args.debug_sched
                        )
                        sched1 = time.time()
                        sample_sched_time = sched1 - sched0

                        ds_pc = pc[top_indices, :]

                        # 時間まとめ
                        sample_region_time = (rA1 - rA0) + (rB1 - rB0)
                        sample_point_time  = (pA1 - pA0) + (pB1 - pB0)
                        sample_shap_time   = sample_region_time + sample_point_time
                        t_final            = time.time()
                        sample_total_time  = t_final - t0
                        sample_down_time   = sample_total_time - (sample_shap_time + sample_sched_time)

                        # MIDキャッシュ保存（SD-CS 用新パスにのみ保存）
                        try:
                            region_full = region_per_pt.astype(np.float32)
                            point_full  = point_adj.astype(np.float32)
                            pc_full     = pc.astype(np.float32)
                            np.savez_compressed(
                                mid_cache_path_new,
                                region_per_pt=region_full,
                                point_adj=point_full,
                                pc=pc_full,
                                label=np.array([cls], dtype=np.int32)
                            )
                        except Exception as e:
                            print(f"[WARN] failed to save MID cache (SDCS): {e}")

                    else:
                        # ---------- heavy (<=1024) ----------
                        if pc.shape[0] > 1024:
                            pc = pc[:1024]
                        elif pc.shape[0] < 1024:
                            pad = 1024 - pc.shape[0]
                            pc = np.pad(pc, ((0, pad), (0, 0)), 'constant')

                        reg_blk, pt_blk, km_reg, km_sub = hierarchical_kmeans(pc, k_region, fanout)
                        baseline_reg_blk = compute_background_baseline(bg_pc, km_reg)
                        baseline_pt_blk  = compute_point_baseline(bg_pc, km_reg, km_sub, fanout)
                        target_explain_class_index = cls

                        t0 = time.time()

                        preds_blk = []
                        def shap_predict_blk(mv):
                            out = shap_predict_block_mask(
                                mv, reg_blk, pc, baseline_reg_blk, model, device,
                                chunk_size=args.mask_chunk,
                                pin_mem=(not args.no_pinmem),
                                min_chunk_size=args.min_mask_chunk
                            )
                            preds_blk.append(out)
                            return out

                        explainer = shap.KernelExplainer(
                            shap_predict_blk,
                            np.zeros((1, len(reg_blk)))
                        )
                        r0 = time.time()
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                            block_vals = (
                                explainer.shap_values(
                                    np.ones((1, len(reg_blk))),
                                    nsamples=pattern,
                                    l1_reg="num_features(10)"
                                )[target_explain_class_index].reshape(-1)
                            )
                            preds_blk_np = np.concatenate(preds_blk, axis=0)
                            delta_blk = preds_blk_np[:, target_explain_class_index] - \
                                        explainer.expected_value[target_explain_class_index]
                            s_blk  = float(delta_blk.std(ddof=1))
                            M_blk  = delta_blk.shape[0]
                            SE_blk = s_blk / np.sqrt(M_blk)
                            all_deltas.append(delta_blk); all_s.append(s_blk); all_se.append(SE_blk)
                        r1 = time.time()

                        p0 = time.time()
                        block_vals_pt = np.repeat(block_vals, fanout)
                        pt_shap_blk   = compute_point_level_shap_values(
                            pc, pt_blk, block_vals_pt, baseline_pt_blk,
                            model, target_explain_class_index, device,
                            scaling_factor=1.0
                        )
                        p1 = time.time()

                        # Region per-point
                        region_blk_exp = np.repeat(block_vals, fanout)
                        region_blk_per_pt = np.zeros_like(pt_shap_blk)
                        for b_idx, idxs in enumerate(pt_blk):
                            region_blk_per_pt[idxs] = region_blk_exp[b_idx]

                        # Point-SHAP L1 正規化
                        if args.no_l1:
                            pt_adj_blk = pt_shap_blk
                        else:
                            pt_adj_blk, _, _ = l1_match_point(
                                pt_shap_blk, region_blk_per_pt, args.ama_eps
                            )

                        region_per_pt = region_blk_per_pt.astype(np.float32)
                        point_adj     = pt_adj_blk.astype(np.float32)

                        # Gate & Hybrid-SHAP
                        var_reg = float(np.var(region_per_pt))
                        var_pnt = float(np.var(point_adj))
                        if args.ama:
                            alpha = gate_forward(ds_points, var_reg, var_pnt,
                                                 args, gate_model, device, gate_scaler)
                        else:
                            alpha = 0.5
                        hybrid = alpha * region_per_pt + (1.0 - alpha) * point_adj

                        # SD-CS Scheduling selection
                        sched0 = time.time()
                        top_indices = sdcs_select(
                            pc, hybrid, ds_points, cov_target_for_N,
                            lambda_scale=args.lambda_scale,
                            debug=args.debug_sched
                        )
                        sched1 = time.time()
                        sample_sched_time = sched1 - sched0

                        ds_pc = pc[top_indices, :]

                        sample_region_time = (r1 - r0)
                        sample_point_time  = (p1 - p0)
                        sample_shap_time   = sample_region_time + sample_point_time
                        t_final            = time.time()
                        sample_total_time  = t_final - t0
                        sample_down_time   = sample_total_time - (sample_shap_time + sample_sched_time)

                        # MID保存（新パスのみ）
                        try:
                            np.savez_compressed(
                                mid_cache_path_new,
                                region_per_pt=region_per_pt.astype(np.float32),
                                point_adj=point_adj.astype(np.float32),
                                pc=pc.astype(np.float32),
                                label=np.array([cls], dtype=np.int32)
                            )
                        except Exception as e:
                            print(f"[WARN] failed to save MID cache (SDCS <=1024): {e}")

                # SD-CS Scheduling結果のキャッシュ保存
                if save_sched_cache and (top_indices is not None):
                    np.savez_compressed(
                        sched_cache_path,
                        top_idx=top_indices.astype(np.int32)
                    )

            # Attention Fusion（任意）
            if args.attention_fusion:
                if 'alpha' in locals():
                    alpha_af = float(alpha)
                else:
                    alpha_af = 0.5
                F_feat = compute_attention_fusion_features(
                    ds_pc, alpha_af, feat_dim=args.feat_dim, k_local=args.knn_k
                )
                sampled_feat_list.append(F_feat.astype('float32'))

            sampled_data_list.append(ds_pc.astype('float32'))
            sampled_label_list.append(cls)
            print(f"Processed sample {i+1}/{num_samples}: final downsampled points {ds_pc.shape[0]}")

            # 集計
            class_time_stats[cls]['sum_total']  += sample_total_time
            class_time_stats[cls]['sum_down']   += sample_down_time
            class_time_stats[cls]['sum_region'] += sample_region_time
            class_time_stats[cls]['sum_point']  += sample_point_time
            class_time_stats[cls]['sum_sched']  += sample_sched_time
            class_time_stats[cls]['sum_shap']   += sample_shap_time
            class_time_stats[cls]['count']      += 1

            overall_total  += sample_total_time
            overall_down   += sample_down_time
            overall_region += sample_region_time
            overall_point  += sample_point_time
            overall_sched  += sample_sched_time
            overall_shap   += sample_shap_time
            overall_count  += 1

        print(f"Finished processing {len(sampled_data_list)} samples from {test_file}.")

        # ファイル保存（マージ方式は AMA/RR と同様）
        if os.path.exists(output_filename):
            with h5py.File(output_filename, "r") as f:
                existing_data  = f["data"][:]
                existing_label = f["label"][:]
                existing_feat  = f["feat"][:] if "feat" in f else None

            new_data  = np.concatenate([existing_data, np.array(sampled_data_list)], axis=0)
            new_label = np.concatenate([existing_label, np.array(sampled_label_list).reshape(-1, 1)], axis=0)

            if args.attention_fusion:
                add_feat = np.array(sampled_feat_list) if len(sampled_feat_list) > 0 else None
                if existing_feat is not None and add_feat is not None:
                    new_feat = np.concatenate([existing_feat, add_feat], axis=0)
                elif existing_feat is None:
                    new_feat = add_feat
                else:
                    new_feat = existing_feat
                save_h5_data(output_filename, new_data, new_label, feat=new_feat)
            else:
                save_h5_data(output_filename, new_data, new_label)
        else:
            if args.attention_fusion and len(sampled_feat_list) > 0:
                save_h5_data(
                    output_filename,
                    np.array(sampled_data_list),
                    np.array(sampled_label_list).reshape(-1, 1),
                    feat=np.array(sampled_feat_list)
                )
            else:
                save_h5_data(
                    output_filename,
                    np.array(sampled_data_list),
                    np.array(sampled_label_list).reshape(-1, 1)
                )
        print(f"Saved SD-CS downsampled point clouds for {test_file} to {output_filename}")

    # --- time stats ---
    time_output = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(time_output, "w") as time_f:
        time_f.write("Class\tDS_Points\tAvg_Region(sec)\tAvg_Point(sec)\tAvg_SDCS_Sched(sec)\tAvg_Down(sec)\tAvg_Total(sec)\tR_region(%)\tR_point(%)\tR_sdcs_sched(%)\tSample_Count\n")
        for cls in sorted(class_time_stats.keys()):
            stats = class_time_stats[cls]
            if stats['count'] > 0:
                avg_region = stats['sum_region'] / stats['count']
                avg_point  = stats['sum_point']  / stats['count']
                avg_sched  = stats['sum_sched']  / stats['count']
                avg_down   = stats['sum_down']   / stats['count']
                avg_total  = stats['sum_total']  / stats['count']
                if avg_total == 0:
                    r_region = r_point = r_sched = 0.0
                else:
                    r_region = 100 * avg_region / avg_total
                    r_point  = 100 * avg_point  / avg_total
                    r_sched  = 100 * avg_sched  / avg_total
                time_f.write(
                    f"{class_names[cls]}\t{ds_points}\t"
                    f"{avg_region:.6f}\t{avg_point:.6f}\t{avg_sched:.6f}\t"
                    f"{avg_down:.6f}\t{avg_total:.6f}\t"
                    f"{r_region:.2f}\t{r_point:.2f}\t{r_sched:.2f}\t"
                    f"{stats['count']}\n"
                )
        if overall_count > 0:
            overall_avg_region = overall_region / overall_count
            overall_avg_point  = overall_point  / overall_count
            overall_avg_sched  = overall_sched  / overall_count
            overall_avg_down   = overall_down   / overall_count
            overall_avg_total  = overall_total  / overall_count
            if overall_avg_total == 0:
                Rr = Rp = Rs = 0.0
            else:
                Rr = 100 * overall_avg_region / overall_avg_total
                Rp = 100 * overall_avg_point  / overall_avg_total
                Rs = 100 * overall_avg_sched  / overall_avg_total
            time_f.write(
                f"ALL\t{ds_points}\t"
                f"{overall_avg_region:.6f}\t{overall_avg_point:.6f}\t{overall_avg_sched:.6f}\t"
                f"{overall_avg_down:.6f}\t{overall_avg_total:.6f}\t"
                f"{Rr:.2f}\t{Rp:.2f}\t{Rs:.2f}\t"
                f"{overall_count}\n"
            )
    print(f"Processing times by class saved to {time_output}")


if __name__ == "__main__":
    main()
