#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scheduling_SHAP_for_PointNeXt.py

目的：
- SD-GLM(AMA_SHAP_for_PointNeXt.py) と同様に Region-SHAP / Point-SHAP を計算し、
  Hybrid-SHAP = α * Region-SHAP + (1 - α) * Point-SHAP（βは廃止）
  を用いてダウンサンプリングを行う。
- 点の選択は「タスクスケジューリング型」のアルゴリズムで行う：
    前半フェーズ：
        各 Region r に対して、大域形状保持度（被覆率） cov_r を計算し、
        cov_r < τ の Region だけを「未被覆 Region」とみなす。
        未被覆 Region の中から、
            (1) 被覆率 cov_r が最も低いものを優先し、
            (2) その中で Region-SHAP が高い Region を優先
        というルールで Region を選び、
        その Region 内の Hybrid-SHAP が最大の点を1点ずつ選択する。
    後半フェーズ：
        すべての Region について cov_r ≥ τ を満たしたら
        （あるいは未被覆 Region に候補点がなくなったら）、
        残りの点は「領域条件なしで Hybrid-SHAP の大きい順」に
        グローバルに選択する。

被覆率（Coverage）の定義：
- 各 Region r について、
    P_r : Region r に属する元の点集合
    S_r : これまでに選択済みの点のうち Region r に属するもの
  とすると、
    r_cov_r = max_{x in P_r} min_{y in S_r} || x - y ||_2
  を「Region r における directed Hausdorff 半径」とし、
    diag_r = || bbox_max(P_r) - bbox_min(P_r) ||_2
  としたとき、
    r_norm_r = r_cov_r / (diag_r + eps)
    cov_r    = 1 - r_norm_r
  を 0〜1 にクリップしたものを「被覆率」として用いる。
- S_r が空の間は cov_r = 0 とする。

Hybrid-SHAP の定義：
- region_per_pt[i] : 点 i に対する Region-SHAP（各 Region の SHAP 値を
  その Region に属するすべての点にブロードキャストしたもの）
- point_adj[i]     : 点 i に対する Point-SHAP。Region-SHAP と L¹ノルムを
  揃える L1正規化付き（--no_l1 を指定した場合は正規化なし）。
- var_reg = Var[region_per_pt], var_pnt = Var[point_adj] を計算し、
  AMA(Gate)が有効な場合は、
    α = gate_forward(ds_points, var_reg, var_pnt, ...)
  により α を決定する（β は計算しない／使わない）。
  AMA を使わない場合は α = 0.5 とする。
- 最終的に、
    hybrid[i] = α * region_per_pt[i] + (1 - α) * point_adj[i]
  を Hybrid-SHAP として用いる。

キャッシュ構成：
- MID(N非依存)キャッシュ（AMA互換）:
    result/_shap_cache_mid/p{pattern}_kr{k_region}_kp{k_point}_ckpt-.../
    内に、各サンプルごとに
        region_per_pt : (N,) float32
        point_adj     : (N,) float32
        pc            : (N,3) float32
        label         : (1,) int32
  を保存。ds_points が変わっても再利用可能。
- Scheduling(N依存)キャッシュ:
    result/_sched_cache_global/.../
    内に、各サンプル・各 ds_points ごとに
        top_idx : (ds_points,) int32
  を保存。再実行時に --cache_mode auto/load で読み込み可能。

使い方例：
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
CUDA_VISIBLE_DEVICES=0 python scheduling_SHAP_for_PointNeXt.py \
    --ds_points 300 \
    --cache_mode auto \
    --pattern 10000 \
    --division_region 32 --division_point 64 \
    --mask_chunk 512 --min_mask_chunk 16 \
    --cov_thresh 0.90 \
    --ama \
    --ama_mode heuristic \
    --no_l1  # （任意：Point-SHAP の L1合わせを無効化）

出力：
- result/Scheduling(...)/{ds_points}/<dataset_subdir>/<test_h5>.h5
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
# AMA Gate (α推定) 関連
# ===============================

class AdaptiveGateMLP(torch.nn.Module):
    """
    AMA 用の Gate-MLP。
    出力は (alpha, beta) だが、本スクリプトでは alpha のみ使用し、
    beta（スケール）は無視する。
    既存の ckpt 互換のため、ネットワーク構造自体は AMA_SHAP と同じ。
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
        beta  = F.softplus(b).squeeze(1)  # 本スクリプトでは使わないが、ckpt互換のため残す
        return alpha, beta


def gate_forward(N: int, var_reg: float, var_pnt: float,
                 args, gate_model: Optional[AdaptiveGateMLP], device,
                 scaler: Optional[dict] = None) -> float:
    """
    α を決定する Gate。
    - ama_mode == "mlp" かつ gate_model が与えられていれば、MLP で推定。
    - それ以外は heuristic で決定。
    戻り値は α（0〜1）で、βは本スクリプトでは使用しない。
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
# Coverage 管理クラス & スケジューラ
# ===============================

class RegionCoverageState:
    """
    各 Region r に対する被覆率 cov_r をインクリメンタルに管理するクラス。

    定義：
        P_r : region_blocks[r] に属する元点群 points[idxs]
        S_r : これまでに選択された点のうち、region_blocks[r] に属するもの
    更新：
        新しい点 y（global_idx）を Region r に追加するたびに、
            d_new[x_in_P_r] = || x - y ||_2
        best_dists_r = min(best_dists_r, d_new)
    被覆率：
        r_cov_r = max(best_dists_r)
        r_norm_r = r_cov_r / diag_r
        cov_r    = 1 - r_norm_r
    """
    def __init__(self, points: np.ndarray, region_blocks, eps: float = 1e-12):
        self.points = points.astype(np.float32, copy=False)
        self.region_blocks = [np.asarray(b, dtype=np.int32) for b in region_blocks]
        self.R = len(self.region_blocks)
        self.eps = float(eps)

        self.diag = np.zeros(self.R, dtype=np.float32)
        self.best_dists = []
        self.has_selected = np.zeros(self.R, dtype=bool)

        for r, idxs in enumerate(self.region_blocks):
            if len(idxs) == 0:
                self.diag[r] = self.eps
                self.best_dists.append(np.zeros(0, dtype=np.float32))
                self.has_selected[r] = False
                continue
            P_r = self.points[idxs]  # (N_r,3)
            bbox_min = np.min(P_r, axis=0)
            bbox_max = np.max(P_r, axis=0)
            diag_r = np.linalg.norm(bbox_max - bbox_min).astype(np.float32)
            self.diag[r] = diag_r + self.eps
            self.best_dists.append(np.full(len(idxs), np.inf, dtype=np.float32))

    def update_with_point(self, global_idx: int, region_idx: int):
        """
        Region region_idx に属する新しい選択点 global_idx を用いて
        best_dists を更新する。
        """
        if region_idx < 0 or region_idx >= self.R:
            return
        idxs = self.region_blocks[region_idx]
        if len(idxs) == 0:
            return
        P_r = self.points[idxs]  # (N_r,3)
        y = self.points[int(global_idx)]  # (3,)
        d_new = np.linalg.norm(P_r - y, axis=1).astype(np.float32)
        bd = self.best_dists[region_idx]
        self.best_dists[region_idx] = np.minimum(bd, d_new)
        self.has_selected[region_idx] = True

    def get_coverages(self) -> np.ndarray:
        """
        各 Region の被覆率 cov_r を返す。
        S_r が空のときは cov_r = 0。
        """
        cov = np.zeros(self.R, dtype=np.float32)
        for r in range(self.R):
            if not self.has_selected[r] or len(self.best_dists[r]) == 0:
                cov[r] = 0.0
                continue
            r_cov = float(np.max(self.best_dists[r]))
            if self.diag[r] <= 0:
                cov[r] = 0.0
                continue
            r_norm = r_cov / self.diag[r]
            score = 1.0 - r_norm
            if not np.isfinite(score):
                score = 0.0
            cov[r] = max(0.0, min(1.0, score))
        return cov

    def coverage_of_region(self, region_idx: int) -> float:
        if region_idx < 0 or region_idx >= self.R:
            return 0.0
        if not self.has_selected[region_idx] or len(self.best_dists[region_idx]) == 0:
            return 0.0
        r_cov = float(np.max(self.best_dists[region_idx]))
        if self.diag[region_idx] <= 0:
            return 0.0
        r_norm = r_cov / self.diag[region_idx]
        score = 1.0 - r_norm
        if not np.isfinite(score):
            score = 0.0
        return max(0.0, min(1.0, score))


def scheduling_select(points: np.ndarray,
                      hybrid_scores: np.ndarray,
                      region_scores: np.ndarray,
                      region_blocks,
                      target_N: int,
                      cov_thresh: float = 0.9,
                      debug: bool = False) -> np.ndarray:
    """
    タスクスケジューリング型の点選択アルゴリズム。

    引数：
        points        : (N,3) 点群
        hybrid_scores : (N,) 各点の Hybrid-SHAP
        region_scores : (R,) 各 Region の Region-SHAP（あるいは region_per_pt の平均）
        region_blocks : list[list[int]] 各 Region 内の点インデックス
        target_N      : 選択する点数
        cov_thresh    : 被覆率のしきい値 τ
        debug         : True の場合、先頭数ステップのログを出力

    アルゴリズム概要：
        while |Selected| < N:
            coverage = cov_state.get_coverages()
            under = { r | coverage[r] < τ かつ region_active[r] = True }

            if under ≠ ∅:
                - under の中から coverage[r] が最小のものを集め、
                  その中で region_scores[r] が最大の Region r* を選ぶ
                - r* 内の Hybrid-SHAP が最大の未選択点を1つ選ぶ
            else:
                - 全点の Hybrid-SHAP 降順リストから、未選択点を1つ選ぶ

            - RegionCoverageState を更新し、Selected に追加する
    """
    N = points.shape[0]
    R = len(region_blocks)
    hybrid_scores = hybrid_scores.astype(np.float32, copy=False)
    region_scores = region_scores.astype(np.float32, copy=False)

    # 点 -> Region の対応（各点は高々1つの Region に属する前提）
    point_region = np.full(N, -1, dtype=np.int32)
    for r, idxs in enumerate(region_blocks):
        idxs_arr = np.asarray(idxs, dtype=np.int32)
        point_region[idxs_arr] = r

    # Region が空でないかどうか
    region_active = np.array([len(region_blocks[r]) > 0 for r in range(R)],
                             dtype=bool)

    # Region ごとの候補リスト（Hybrid降順ソート）
    region_candidates = []
    region_ptr = []
    for r in range(R):
        idxs_arr = np.asarray(region_blocks[r], dtype=np.int32)
        if idxs_arr.size == 0:
            region_candidates.append(idxs_arr)
            region_ptr.append(0)
            continue
        order = np.argsort(-hybrid_scores[idxs_arr])
        region_candidates.append(idxs_arr[order])
        region_ptr.append(0)

    # 全体の Hybrid 降順リスト
    global_order = np.argsort(-hybrid_scores)
    global_ptr = 0

    # Coverage 状態
    cov_state = RegionCoverageState(points, region_blocks)
    selected = []
    selected_flag = np.zeros(N, dtype=bool)

    step = 0
    while len(selected) < target_N:
        coverage = cov_state.get_coverages()
        under_mask = (coverage < cov_thresh) & region_active
        under = np.where(under_mask)[0]

        chosen_idx = None

        if under.size > 0:
            # 1) coverage が最小の Region 群
            cov_under = coverage[under]
            min_cov = float(np.min(cov_under))
            idx_min_cov = np.where(cov_under == min_cov)[0]
            cand_regions = under[idx_min_cov]

            # 2) その中から region_scores が最大の Region を選択
            best_region = cand_regions[np.argmax(region_scores[cand_regions])]

            # 3) 選ばれた Region 内の Hybrid 順候補から未選択点を探す
            cand_arr = region_candidates[best_region]
            ptr = region_ptr[best_region]
            while ptr < len(cand_arr) and selected_flag[cand_arr[ptr]]:
                ptr += 1
            region_ptr[best_region] = ptr
            if ptr < len(cand_arr):
                chosen_idx = int(cand_arr[ptr])
            else:
                # この Region にはもう候補点がないので非アクティブ化
                region_active[best_region] = False
                if debug:
                    print(f"[DEBUG_SCHED] region {int(best_region)} has no remaining "
                          f"candidates. Deactivate.")
                # このループでは global fallback に任せる

        if chosen_idx is None:
            # グローバル fallback：Hybrid 降順で未選択点を取る
            while global_ptr < N and selected_flag[global_order[global_ptr]]:
                global_ptr += 1
            if global_ptr >= N:
                print(f"[WARN] scheduling_select: ran out of candidates at "
                      f"step={len(selected)} / target={target_N}")
                break
            chosen_idx = int(global_order[global_ptr])

        # 選択確定
        selected_flag[chosen_idx] = True
        selected.append(chosen_idx)
        r = int(point_region[chosen_idx])
        if r >= 0:
            cov_state.update_with_point(chosen_idx, r)

        if debug and len(selected) <= 40:
            cov_r = cov_state.coverage_of_region(r) if r >= 0 else 0.0
            print(f"[DEBUG_SCHED] step={step:03d}, pid={chosen_idx:04d}, "
                  f"region={r:02d}, hybrid={float(hybrid_scores[chosen_idx]):.6f}, "
                  f"cov_r={cov_r:.4f}")

        step += 1

    return np.asarray(selected[:target_N], dtype=np.int32)


# ===============================
# main
# ===============================

def main():
    parser = argparse.ArgumentParser(
        description="Scheduling SHAP downsampling for ModelNet40 test files using PointNeXt."
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
    # Coverage 関連
    parser.add_argument("--cov_thresh", type=float, default=0.90,
                        help="Coverage のしきい値 τ（0〜1）")
    # デバッグ
    parser.add_argument("--debug_sched", action="store_true",
                        help="Scheduling の最初の取り出しログを表示する")

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
    _out_tag_base = f"Scheduling{l1_tag}-{ama_tag}"
    out_tag = _out_tag_base if args.run_tag == "" else f"{_out_tag_base}-{args.run_tag}"
    output_folder = os.path.join(
        "result", f"{out_tag}_kr{k_region}_kp{k_point}_p{pattern}_dsSCHED_PointNeXt_h5", str(ds_points)
    )
    os.makedirs(output_folder, exist_ok=True)

    # Scheduling(N依存)キャッシュ
    BASE_SCHED_CACHE_DIR = os.path.join("result", "_sched_cache_global")
    os.makedirs(BASE_SCHED_CACHE_DIR, exist_ok=True)
    sched_cache_tag = f"p{pattern}_kr{k_region}_kp{k_point}_N{ds_points}{('_noL1' if args.no_l1 else '')}_{ama_tag}"
    if args.run_tag:
        sched_cache_tag += f"_{args.run_tag}"
    SCHED_CACHE_DIR = os.path.join(BASE_SCHED_CACHE_DIR, sched_cache_tag)
    os.makedirs(SCHED_CACHE_DIR, exist_ok=True)

    # MID(N非依存)キャッシュ（AMA/RR互換）
    MID_CACHE_BASE = os.path.join("result", "_shap_cache_mid")
    ckpt_name_for_tag = Path(args.checkpoint_path).stem
    mid_tag = f"p{pattern}_kr{k_region}_kp{k_point}_ckpt-{ckpt_name_for_tag}{('_noL1' if args.no_l1 else '')}"
    if args.run_tag:
        mid_tag += f"_{args.run_tag}"
    MID_CACHE_DIR = os.path.join(MID_CACHE_BASE, mid_tag)
    os.makedirs(MID_CACHE_DIR, exist_ok=True)

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
            sched_cache_path = os.path.join(SCHED_CACHE_DIR, sample_id + ".npz")
            mid_cache_path   = os.path.join(MID_CACHE_DIR, sample_id + ".npz")

            use_sched_cache  = (args.cache_mode in ["auto", "load"]) and os.path.exists(sched_cache_path) and (not args.force_mid_build)
            save_sched_cache = (args.cache_mode in ["auto", "save"]) and (not use_sched_cache)

            # timing init
            sample_region_time = 0.0
            sample_point_time  = 0.0
            sample_sched_time  = 0.0
            sample_total_time  = 0.0
            sample_down_time   = 0.0
            sample_shap_time   = 0.0

            if use_sched_cache:
                print(f"  ↳ Scheduling cache hit: {sched_cache_path}")
                z = np.load(sched_cache_path)
                top_indices = z["top_idx"].astype(np.int32).reshape(-1)
                ds_pc = pc[top_indices, :]
                # キャッシュヒット時は timing は 0 のまま
            else:
                # まず MID があれば mid material を読む
                mid_hit = os.path.exists(mid_cache_path)
                region_per_pt = None
                point_adj = None
                region_scores_all = None
                region_blocks_all = None

                # ===== MID cache HIT: Region/Point-SHAP は再利用、クラスタだけ再計算 =====
                if mid_hit and (not args.force_mid_build):
                    print(f"  ↳ MID cache hit: {mid_cache_path}")
                    _mid = np.load(mid_cache_path)
                    region_per_pt = _mid["region_per_pt"].astype(np.float32).reshape(-1)
                    point_adj     = _mid["point_adj"].astype(np.float32).reshape(-1)
                    if "pc" in _mid.files:
                        pc = _mid["pc"].astype(np.float32)
                    # ここでは Region/Point SHAP の計算時間は 0 とする

                    t0 = time.time()

                    # 2048点かどうかでクラスタリング方針を分ける（AMA/RRと同様）
                    if pc.shape[0] == 2048:
                        pc_A = pc[:1024]
                        pc_B = pc[1024:]
                        reg_A, pt_A, _, _ = hierarchical_kmeans(pc_A, k_region, fanout)
                        reg_B, pt_B, _, _ = hierarchical_kmeans(pc_B, k_region, fanout)

                        # Region SHAP を region_per_pt の平均から復元
                        region_scores_A = np.zeros((k_region,), dtype=np.float32)
                        for rid, idxs in enumerate(reg_A):
                            arr = np.asarray(idxs, dtype=np.int32)
                            region_scores_A[rid] = float(np.mean(region_per_pt[arr]))
                        region_scores_B = np.zeros((k_region,), dtype=np.float32)
                        for rid, idxs in enumerate(reg_B):
                            gidx = (np.asarray(idxs, dtype=np.int32) + 1024)
                            region_scores_B[rid] = float(np.mean(region_per_pt[gidx]))

                        region_scores_all = np.concatenate([region_scores_A, region_scores_B]).astype(np.float32)

                        # Region blocks (global indices)
                        region_blocks_all = []
                        for idxs in reg_A:
                            region_blocks_all.append(np.asarray(idxs, dtype=np.int32))
                        for idxs in reg_B:
                            gidx = (np.asarray(idxs, dtype=np.int32) + 1024)
                            region_blocks_all.append(gidx)

                    else:
                        # <= 1024 の場合
                        if pc.shape[0] > 1024:
                            pc = pc[:1024]
                            region_per_pt = region_per_pt[:1024]
                            point_adj     = point_adj[:1024]
                        elif pc.shape[0] < 1024:
                            pad = 1024 - pc.shape[0]
                            pc = np.pad(pc, ((0, pad), (0, 0)), 'constant')
                            region_per_pt = np.pad(region_per_pt, (0, pad), 'constant')
                            point_adj     = np.pad(point_adj,     (0, pad), 'constant')

                        reg_blk, pt_blk, _, _ = hierarchical_kmeans(pc, k_region, fanout)
                        region_blocks_all = reg_blk
                        region_scores_all = np.zeros((k_region,), dtype=np.float32)
                        for rid, idxs in enumerate(reg_blk):
                            arr = np.asarray(idxs, dtype=np.int32)
                            region_scores_all[rid] = float(np.mean(region_per_pt[arr]))

                    # Gate & Hybrid-SHAP
                    var_reg = float(np.var(region_per_pt))
                    var_pnt = float(np.var(point_adj))
                    if args.ama:
                        alpha = gate_forward(ds_points, var_reg, var_pnt,
                                             args, gate_model, device, gate_scaler)
                    else:
                        alpha = 0.5
                    hybrid = alpha * region_per_pt + (1.0 - alpha) * point_adj

                    # Scheduling selection
                    sched0 = time.time()
                    top_indices = scheduling_select(
                        pc, hybrid, region_scores_all, region_blocks_all,
                        ds_points, cov_thresh=args.cov_thresh,
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

                # ===== MID cache NOT HIT: Region/Point-SHAP をフル計算 =====
                else:
                    # ここからは AMA_SHAP / RoundRobin の heavy 部分をベースに、
                    # Hybrid-SHAP + Scheduling を組み込む。

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

                        # --- Region blocks (global indices) & Region scores ---
                        region_blocks_all = []
                        region_scores_all = []

                        for rid, idxs in enumerate(reg_A):
                            arr = np.asarray(idxs, dtype=np.int32)
                            region_blocks_all.append(arr)
                            region_scores_all.append(float(block_A[rid]))

                        for rid, idxs in enumerate(reg_B):
                            arr = (np.asarray(idxs, dtype=np.int32) + 1024)
                            region_blocks_all.append(arr)
                            region_scores_all.append(float(block_B[rid]))

                        region_scores_all = np.asarray(region_scores_all, dtype=np.float32)

                        # Gate & Hybrid
                        var_reg = float(np.var(region_per_pt))
                        var_pnt = float(np.var(point_adj))
                        if args.ama:
                            alpha = gate_forward(ds_points, var_reg, var_pnt,
                                                 args, gate_model, device, gate_scaler)
                        else:
                            alpha = 0.5
                        hybrid = alpha * region_per_pt + (1.0 - alpha) * point_adj

                        # Scheduling selection
                        sched0 = time.time()
                        top_indices = scheduling_select(
                            pc, hybrid, region_scores_all, region_blocks_all,
                            ds_points, cov_thresh=args.cov_thresh,
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

                        # MIDキャッシュ保存
                        try:
                            region_full = region_per_pt.astype(np.float32)
                            point_full  = point_adj.astype(np.float32)
                            pc_full     = pc.astype(np.float32)
                            np.savez_compressed(
                                mid_cache_path,
                                region_per_pt=region_full,
                                point_adj=point_full,
                                pc=pc_full,
                                label=np.array([cls], dtype=np.int32)
                            )
                        except Exception as e:
                            print(f"[WARN] failed to save MID cache: {e}")

                    else:
                        # ---------- heavy (<=1024) ----------
                        if pc.shape[0] > 1024:
                            pc = pc[np.random.choice(pc.shape[0], 1024, replace=False)]
                        elif pc.shape[0] < 1024:
                            pc = np.pad(pc, ((0, 1024 - pc.shape[0]), (0, 0)), 'constant')

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

                        region_blocks_all = reg_blk
                        region_scores_all = block_vals.astype(np.float32)

                        # Gate & Hybrid
                        var_reg = float(np.var(region_per_pt))
                        var_pnt = float(np.var(point_adj))
                        if args.ama:
                            alpha = gate_forward(ds_points, var_reg, var_pnt,
                                                 args, gate_model, device, gate_scaler)
                        else:
                            alpha = 0.5
                        hybrid = alpha * region_per_pt + (1.0 - alpha) * point_adj

                        # Scheduling selection
                        sched0 = time.time()
                        top_indices = scheduling_select(
                            pc, hybrid, region_scores_all, region_blocks_all,
                            ds_points, cov_thresh=args.cov_thresh,
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

                        # MID保存
                        try:
                            np.savez_compressed(
                                mid_cache_path,
                                region_per_pt=region_per_pt.astype(np.float32),
                                point_adj=point_adj.astype(np.float32),
                                pc=pc.astype(np.float32),
                                label=np.array([cls], dtype=np.int32)
                            )
                        except Exception as e:
                            print(f"[WARN] failed to save MID cache: {e}")

                # Scheduling結果のキャッシュ保存
                if save_sched_cache and top_indices is not None:
                    np.savez_compressed(
                        sched_cache_path,
                        top_idx=top_indices.astype(np.int32)
                    )

            # Attention Fusion（任意）
            if args.attention_fusion:
                # Scheduling では α はサンプル毎に scalar なので、そのまま AF に流用。
                # AMA を使わない場合は α=0.5 としたが、AF でも同じ値を使う。
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
        print(f"Saved Scheduling downsampled point clouds for {test_file} to {output_filename}")

    # --- time stats ---
    time_output = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(time_output, "w") as time_f:
        time_f.write("Class\tDS_Points\tAvg_Region(sec)\tAvg_Point(sec)\tAvg_Sched(sec)\tAvg_Down(sec)\tAvg_Total(sec)\tR_region(%)\tR_point(%)\tR_sched(%)\tSample_Count\n")
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
