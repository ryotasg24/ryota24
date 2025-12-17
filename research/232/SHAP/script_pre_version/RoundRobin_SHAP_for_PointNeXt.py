#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoundRobin_SHAP_for_PointNeXt.py

目的：
- SD-GLM(AMA_SHAP_for_PointNeXt.py) の Hybrid(αβ) を完全に除去し、
  Region-SHAP の重要度順位に従って Region をラウンドロビンしながら、
  各 Region 内ではその Region に内包される fanout 個 (=k_point/k_region) の
  Point-SHAP サブ領域から Point-SHAP の高い点を順に取り出すダウンサンプリングを行う。

重要な設計：
- k_region = 32, k_point = 64 をデフォルトとし、fanout=2 に対応。
  「1つの Region(32分割) に対して 2つの Point-SHAP サブ領域(64分割) がネスト」
  されている前提を hierarchical_kmeans の返す point_blocks の並び規則を用いて満たす。
- ラウンドロビンは Region-SHAP のランキング順（降順）で回す。
- 2048点入力の場合は AMA と同じく A/B(1024ずつ) に分割して独立に Region/Point を作り、
  グローバルでは Region が 64 (=2*k_region)、Pointブロックが 128 (=2*k_point) になる。

使い方例：
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
CUDA_VISIBLE_DEVICES=0 python RoundRobin_SHAP_for_PointNeXt.py \
    --ds_points 300 \
    --cache_mode auto \
    --pattern 10000 \
    --division_region 32 --division_point 64 \
    --mask_chunk 512 --min_mask_chunk 16 \
    --no_l1  #（任意：Point-SHAPのL1合わせ無効）

出力：
- result/RoundRobin(...)/{ds_points}/<dataset_subdir>/<test_h5>.h5
  data: (num_samples, ds_points, 3), label: (num_samples,1)
- processing_times_by_class.txt, Monte_Carlo.txt（AMAと互換の簡易ログ）

検証用ログ：
--debug_rr を付けると、各サンプルで
  (1) Region のrank順
  (2) 各Regionからの取り出し順（周回番号）
  (3) 取り出した点のPoint-SHAP値
を先頭数点だけ表示し、実装が意図通りになっているかを確認できる。
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

# 可視化（任意）
import matplotlib.pyplot as plt

# --- PointNeXt 関連 ---
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig

# provider.py
import provider

# ======================================================
# 保存用関数
def save_h5_data(h5_filename, data, label, feat=None):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)
        if feat is not None:
            f.create_dataset('feat', data=feat)
# ======================================================

# ===============================
# 各種関数の定義（AMAから必要部分を踏襲）
# ===============================

def get_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def hierarchical_kmeans(points, k_region, fanout):
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
    L1_reg = float(np.sum(np.abs(region_per_pt))) + eps
    L1_pnt = float(np.sum(np.abs(point_shap))) + eps
    scale  = L1_reg / L1_pnt
    return point_shap * scale, L1_reg, L1_pnt

# 参考：AF特徴生成（使う場合のみ）
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
# Round-Robin 用の関数群
# ===============================

def _merge_fanout_lists(sub_lists, point_scores):
    """
    fanout 個のサブ領域を Point-SHAP 降順でマージする。
    sub_lists: list of np.ndarray(indices already sorted desc within each sub-block)
    point_scores: (num_points,) float
    """
    fanout = len(sub_lists)
    ptrs = [0] * fanout
    merged = []

    while True:
        best_j = -1
        best_sc = -float("inf")
        for j in range(fanout):
            arr = sub_lists[j]
            p = ptrs[j]
            if p < len(arr):
                sc = float(point_scores[arr[p]])
                if sc > best_sc:
                    best_sc = sc
                    best_j = j
        if best_j < 0:
            break
        merged.append(int(sub_lists[best_j][ptrs[best_j]]))
        ptrs[best_j] += 1

    return np.asarray(merged, dtype=np.int32)

def build_region_point_lists(region_blocks, point_blocks, point_scores,
                             k_region: int, fanout: int):
    """
    各Region rについて、
      - そのRegionに対応する fanout 個の Pointサブ領域（point_blocks）を取り出し
      - 各サブ領域内で Point-SHAP 降順ソート
      - fanout 個を Point-SHAP 降順でマージ
    した「Regionごとの候補列」を作る。
    """
    R = len(region_blocks)
    assert R == k_region, f"region_blocks len mismatch: {R} vs k_region={k_region}"
    assert len(point_blocks) == k_region * fanout

    region_point_lists = []
    for r in range(k_region):
        sub_lists_sorted = []
        for s in range(fanout):
            pb_idx = r * fanout + s
            pts = np.asarray(point_blocks[pb_idx], dtype=np.int32)
            if len(pts) == 0:
                sub_lists_sorted.append(np.asarray([], dtype=np.int32))
                continue
            sc = point_scores[pts]
            order = np.argsort(-sc)
            sub_lists_sorted.append(pts[order])
        merged = _merge_fanout_lists(sub_lists_sorted, point_scores)
        region_point_lists.append(merged)
    return region_point_lists  # list of (<=|region|,) arrays

def roundrobin_select(region_scores, region_point_lists, target_N: int,
                      debug: bool = False, point_scores: Optional[np.ndarray] = None):
    """
    Region-SHAP 降順の region_order に従いラウンドロビンで選択。
    各Regionからは「region_point_lists[r] の k-th 点」を取る（k=周回番号）。
    """
    R = len(region_scores)
    region_order = np.argsort(-region_scores)  # descending
    selected = []
    k = 0

    if debug:
        print("[DEBUG_RR] region_order (desc by Region-SHAP):")
        print(region_order.tolist())

    while len(selected) < target_N:
        added_any = False
        for r in region_order:
            pts = region_point_lists[r]
            if k < len(pts):
                pid = int(pts[k])
                selected.append(pid)
                added_any = True
                if debug and point_scores is not None and len(selected) <= 40:
                    print(f"[DEBUG_RR] cycle={k:02d}, region={int(r):02d}, "
                          f"pid={pid}, point_score={float(point_scores[pid]):.6f}")
                if len(selected) >= target_N:
                    break
        if not added_any:
            break
        k += 1

    return np.asarray(selected[:target_N], dtype=np.int32)

# ===============================
# main
# ===============================

def main():
    parser = argparse.ArgumentParser(
        description="RoundRobin SHAP downsampling for ModelNet40 test files using PointNeXt."
    )
    parser.add_argument("--ds_points", type=int, default=500)
    parser.add_argument("--pattern", type=int, default=10000)
    parser.add_argument("--division_region", type=int, default=32)
    parser.add_argument("--division_point", type=int, default=64)
    parser.add_argument("--cache_mode", type=str, default="auto",
                        choices=["auto", "save", "load"])
    parser.add_argument("--num_groups", type=int, default=1)
    parser.add_argument("--group_index", type=int, default=0)
    parser.add_argument("--cfg_file", type=str,
                        default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth")
    parser.add_argument("--dataset", type=str,
                        default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048")
    parser.add_argument("--run_tag", type=str, default="")
    # L1正規化（Pointのスケール合わせ）を切るオプションは残す
    parser.add_argument("--no_l1", action="store_true")
    parser.add_argument("--ama_eps", type=float, default=1e-6)
    # 推論チャンク
    parser.add_argument("--mask_chunk", type=int, default=512)
    parser.add_argument("--min_mask_chunk", type=int, default=8)
    parser.add_argument("--no_pinmem", action="store_true")
    parser.add_argument("--force_mid_build", action="store_true",
                        help="RRキャッシュがありそうでもSHAPを再計算する（通常不要）")
    # Attention Fusion（任意）
    parser.add_argument("--attention_fusion", action="store_true")
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--knn_k", type=int, default=16)
    # 検証デバッグ
    parser.add_argument("--debug_rr", action="store_true",
                        help="RoundRobinの最初の取り出し順を表示する")

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
    _out_tag_base = f"RoundRobin{l1_tag}"
    out_tag = _out_tag_base if args.run_tag == "" else f"{_out_tag_base}-{args.run_tag}"
    output_folder = os.path.join(
        "result", f"{out_tag}_kr{k_region}_kp{k_point}_p{pattern}_dsRR_PointNeXt_h5", str(ds_points)
    )
    os.makedirs(output_folder, exist_ok=True)

    # RR キャッシュ（N依存：選択インデックスだけ保存）
    BASE_RR_CACHE_DIR = os.path.join("result", "_rr_cache_global")
    os.makedirs(BASE_RR_CACHE_DIR, exist_ok=True)
    rr_cache_tag = f"p{pattern}_kr{k_region}_kp{k_point}_N{ds_points}{('_noL1' if args.no_l1 else '')}"
    if args.run_tag:
        rr_cache_tag += f"_{args.run_tag}"
    RR_CACHE_DIR = os.path.join(BASE_RR_CACHE_DIR, rr_cache_tag)
    os.makedirs(RR_CACHE_DIR, exist_ok=True)

    # MID(N非依存)キャッシュ（AMAと共通。あれば利用）
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
    class_time_stats = {cls: {'sum_total':0,'sum_down':0, 'sum_region':0,'sum_point':0,
                              'sum_rr':0, 'sum_shap':0, 'count':0}
                        for cls in selected_classes}
    overall_total = overall_down = overall_region = overall_point = overall_rr = overall_shap = 0.0
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
            rr_cache_path  = os.path.join(RR_CACHE_DIR, sample_id + ".npz")
            mid_cache_path = os.path.join(MID_CACHE_DIR, sample_id + ".npz")

            use_rr_cache  = (args.cache_mode in ["auto","load"]) and os.path.exists(rr_cache_path) and (not args.force_mid_build)
            save_rr_cache = (args.cache_mode in ["auto","save"]) and (not use_rr_cache)

            # timing init
            sample_region_time = sample_point_time = sample_rr_time = 0.0
            sample_total_time  = sample_down_time = sample_shap_time = 0.0

            if use_rr_cache:
                print(f"  ↳ RR cache hit: {rr_cache_path}")
                z = np.load(rr_cache_path)
                top_indices = z["top_idx"].astype(np.int32).reshape(-1)
                ds_pc = pc[top_indices, :]
                # all time =0 when cache hit
            else:
                # まず MID があれば mid material を読む
                mid_hit = os.path.exists(mid_cache_path)
                region_per_pt = None
                point_scores_all = None
                region_scores_all = None
                region_blocks_all = None

                if mid_hit:
                    print(f"  ↳ MID cache hit: {mid_cache_path}")
                    _mid = np.load(mid_cache_path)
                    region_per_pt = _mid["region_per_pt"].astype(np.float32).reshape(-1)
                    point_adj     = _mid["point_adj"].astype(np.float32).reshape(-1)
                    if "pc" in _mid.files:
                        pc = _mid["pc"].astype(np.float32)
                    # point scores for ordering (no_l1なら生、そうでなければpoint_adj)
                    point_scores_all = point_adj if (not args.no_l1) else point_adj  # MIDは既に no_l1反映済み
                    # region scoresは per-pointから復元（各Regionで一定）
                    # ※ 後で blocks を再構築し平均を取る
                # MIDが無ければ heavy 計算
                if (not mid_hit):
                    if pc.shape[0] == 2048:
                        # ---------- heavy (2048 A/B) ----------
                        pc_A = pc[:1024]; pc_B = pc[1024:]
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
                                chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem),
                                min_chunk_size=args.min_mask_chunk
                            )
                            preds_A.append(out)
                            return out
                        explainer_A = shap.KernelExplainer(shap_predict_A, np.zeros((1,len(reg_A))))
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
                            delta_A    = preds_A_np[:, target_explain_class_index] - explainer_A.expected_value[target_explain_class_index]
                            s_A  = float(delta_A.std(ddof=1))
                            M_A  = delta_A.shape[0]
                            SE_A = s_A / np.sqrt(M_A)
                            all_deltas.append(delta_A); all_s.append(s_A); all_se.append(SE_A)
                        rA1 = time.time()

                        # --- Point SHAP A ---
                        pA0 = time.time()
                        block_A_pt = np.repeat(block_A, fanout)  # (k_point,)
                        pt_shap_A  = compute_point_level_shap_values(
                            pc_A, pt_A, block_A_pt, baseline_pt_A,
                            model, target_explain_class_index, device, scaling_factor=1.0
                        )
                        pA1 = time.time()

                        # --- Region SHAP B ---
                        preds_B = []
                        def shap_predict_B(mv):
                            out = shap_predict_block_mask(
                                mv, reg_B, pc_B, baseline_reg_B, model, device,
                                chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem),
                                min_chunk_size=args.min_mask_chunk
                            )
                            preds_B.append(out)
                            return out
                        explainer_B = shap.KernelExplainer(shap_predict_B, np.zeros((1,len(reg_B))))
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
                            delta_B    = preds_B_np[:, target_explain_class_index] - explainer_B.expected_value[target_explain_class_index]
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
                            model, target_explain_class_index, device, scaling_factor=1.0
                        )
                        pB1 = time.time()

                        # --- Region per-point broadcast (A/B) ---
                        region_A_exp = np.repeat(block_A, fanout)
                        region_A_per_pt = np.zeros_like(pt_shap_A)
                        for b_idx, idxs in enumerate(pt_A):
                            region_A_per_pt[idxs] = region_A_exp[b_idx]

                        region_B_exp = np.repeat(block_B, fanout)
                        region_B_per_pt = np.zeros_like(pt_shap_B)
                        for b_idx, idxs in enumerate(pt_B):
                            region_B_per_pt[idxs] = region_B_exp[b_idx]

                        # --- Point score (no_l1なら生pt_shap, そうでなければL1合わせpt_adj) ---
                        if args.no_l1:
                            pt_adj_A = pt_shap_A
                            pt_adj_B = pt_shap_B
                        else:
                            pt_adj_A, _, _ = l1_match_point(pt_shap_A, region_A_per_pt, args.ama_eps)
                            pt_adj_B, _, _ = l1_match_point(pt_shap_B, region_B_per_pt, args.ama_eps)

                        # --- グローバルまとめ ---
                        # Region blocks (global indices)
                        region_blocks_all = []
                        region_scores_all = []

                        for rid, idxs in enumerate(reg_A):
                            region_blocks_all.append(np.asarray(idxs, dtype=np.int32))
                            region_scores_all.append(float(block_A[rid]))

                        for rid, idxs in enumerate(reg_B):
                            gidx = np.asarray(idxs, dtype=np.int32) + 1024
                            region_blocks_all.append(gidx)
                            region_scores_all.append(float(block_B[rid]))

                        region_scores_all = np.asarray(region_scores_all, dtype=np.float32)

                        # Point blocks（global indices, ordered region-wise）
                        point_blocks_all = []
                        for r in range(k_region):
                            for s in range(fanout):
                                pb = np.asarray(pt_A[r*fanout+s], dtype=np.int32)
                                point_blocks_all.append(pb.tolist())
                        for r in range(k_region):
                            for s in range(fanout):
                                pb = np.asarray(pt_B[r*fanout+s], dtype=np.int32) + 1024
                                point_blocks_all.append(pb.tolist())

                        point_scores_all = np.concatenate([pt_adj_A, pt_adj_B]).astype(np.float32)

                        # RR 選択
                        rr0 = time.time()
                        # Region A(0..31), Region B(32..63) なので region_point_listsを2段で作る
                        # まず A側
                        region_point_lists_A = build_region_point_lists(
                            reg_A, pt_A, pt_adj_A, k_region, fanout
                        )
                        # 次に B側（indicesはglobalに直して）
                        reg_B_global = [ (np.asarray(idxs, dtype=np.int32)+1024).tolist() for idxs in reg_B ]
                        pt_B_global  = [ (np.asarray(idxs, dtype=np.int32)+1024).tolist() for idxs in pt_B ]
                        region_point_lists_B = build_region_point_lists(
                            reg_B_global, pt_B_global, point_scores_all, k_region, fanout
                        )
                        region_point_lists_all = region_point_lists_A + region_point_lists_B

                        top_indices = roundrobin_select(
                            region_scores_all, region_point_lists_all, ds_points,
                            debug=args.debug_rr, point_scores=point_scores_all
                        )
                        rr1 = time.time()

                        ds_pc = pc[top_indices, :]

                        # timing
                        sample_region_time = (rA1-rA0) + (rB1-rB0)
                        sample_point_time  = (pA1-pA0) + (pB1-pB0)
                        sample_rr_time     = rr1-rr0
                        t_final            = time.time()
                        sample_total_time  = t_final - t0
                        sample_down_time   = sample_total_time - (sample_region_time + sample_point_time + sample_rr_time)
                        sample_shap_time   = sample_region_time + sample_point_time

                        # MID保存（N非依存材料、AMAと同形式に合わせる）
                        try:
                            region_full = np.concatenate([region_A_per_pt, region_B_per_pt], axis=0)
                            point_full  = np.concatenate([pt_adj_A,       pt_adj_B      ], axis=0)
                            pc_full     = np.concatenate([pc_A, pc_B], axis=0)
                            np.savez_compressed(
                                mid_cache_path,
                                region_per_pt=region_full.astype(np.float32),
                                point_adj=point_full.astype(np.float32),
                                pc=pc_full.astype(np.float32),
                                label=np.array([cls], dtype=np.int32)
                            )
                        except Exception as e:
                            print(f"[WARN] failed to save MID cache: {e}")

                    else:
                        # ---------- heavy (<=1024) ----------
                        if pc.shape[0] > 1024:
                            pc = pc[np.random.choice(pc.shape[0],1024,replace=False)]
                        elif pc.shape[0] < 1024:
                            pc = np.pad(pc,((0,1024-pc.shape[0]),(0,0)),'constant')

                        reg_blk, pt_blk, km_reg, km_sub = hierarchical_kmeans(pc, k_region, fanout)
                        baseline_reg_blk = compute_background_baseline(bg_pc, km_reg)
                        baseline_pt_blk  = compute_point_baseline(bg_pc, km_reg, km_sub, fanout)
                        target_explain_class_index = cls

                        t0 = time.time()
                        preds_blk = []
                        def shap_predict_blk(mv):
                            out = shap_predict_block_mask(
                                mv, reg_blk, pc, baseline_reg_blk, model, device,
                                chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem),
                                min_chunk_size=args.min_mask_chunk
                            )
                            preds_blk.append(out)
                            return out

                        explainer = shap.KernelExplainer(shap_predict_blk, np.zeros((1, len(reg_blk))))
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
                            delta_blk    = preds_blk_np[:, target_explain_class_index] - explainer.expected_value[target_explain_class_index]
                            s_blk  = float(delta_blk.std(ddof=1))
                            M_blk  = delta_blk.shape[0]
                            SE_blk = s_blk / np.sqrt(M_blk)
                            all_deltas.append(delta_blk); all_s.append(s_blk); all_se.append(SE_blk)
                        r1 = time.time()

                        p0 = time.time()
                        block_vals_pt = np.repeat(block_vals, fanout)
                        pt_shap_blk   = compute_point_level_shap_values(
                            pc, pt_blk, block_vals_pt, baseline_pt_blk,
                            model, target_explain_class_index, device, scaling_factor=1.0
                        )
                        p1 = time.time()

                        # Region per-point
                        region_blk_exp = np.repeat(block_vals, fanout)
                        region_blk_per_pt = np.zeros_like(pt_shap_blk)
                        for b_idx, idxs in enumerate(pt_blk):
                            region_blk_per_pt[idxs] = region_blk_exp[b_idx]

                        if args.no_l1:
                            pt_adj_blk = pt_shap_blk
                        else:
                            pt_adj_blk, _, _ = l1_match_point(pt_shap_blk, region_blk_per_pt, args.ama_eps)

                        # RR選択（Region=32）
                        rr0 = time.time()
                        region_scores_all = block_vals.astype(np.float32)
                        point_scores_all  = pt_adj_blk.astype(np.float32)

                        region_point_lists = build_region_point_lists(
                            reg_blk, pt_blk, point_scores_all, k_region, fanout
                        )
                        top_indices = roundrobin_select(
                            region_scores_all, region_point_lists, ds_points,
                            debug=args.debug_rr, point_scores=point_scores_all
                        )
                        rr1 = time.time()

                        ds_pc = pc[top_indices, :]

                        # timing
                        sample_region_time = r1-r0
                        sample_point_time  = p1-p0
                        sample_rr_time     = rr1-rr0
                        t_final            = time.time()
                        sample_total_time  = t_final - t0
                        sample_down_time   = sample_total_time - (sample_region_time + sample_point_time + sample_rr_time)
                        sample_shap_time   = sample_region_time + sample_point_time

                        # MID保存
                        try:
                            np.savez_compressed(
                                mid_cache_path,
                                region_per_pt=region_blk_per_pt.astype(np.float32),
                                point_adj=pt_adj_blk.astype(np.float32),
                                pc=pc.astype(np.float32),
                                label=np.array([cls], dtype=np.int32)
                            )
                        except Exception as e:
                            print(f"[WARN] failed to save MID cache: {e}")

                # MIDヒットだった場合の RR（blocks 再構築）
                if mid_hit:
                    # A/B判定
                    if pc.shape[0] == 2048:
                        pc_A = pc[:1024]; pc_B = pc[1024:]
                        reg_A, pt_A, _, _ = hierarchical_kmeans(pc_A, k_region, fanout)
                        reg_B, pt_B, _, _ = hierarchical_kmeans(pc_B, k_region, fanout)

                        # region_scoresを blocks から復元
                        # Region A/Bは per-point region_per_pt に応じて一定値のはずなので平均で取る
                        region_scores_A = np.zeros((k_region,), dtype=np.float32)
                        for rid, idxs in enumerate(reg_A):
                            region_scores_A[rid] = float(np.mean(region_per_pt[np.asarray(idxs, dtype=np.int32)]))
                        region_scores_B = np.zeros((k_region,), dtype=np.float32)
                        for rid, idxs in enumerate(reg_B):
                            gidx = np.asarray(idxs, dtype=np.int32) + 1024
                            region_scores_B[rid] = float(np.mean(region_per_pt[gidx]))
                        region_scores_all = np.concatenate([region_scores_A, region_scores_B]).astype(np.float32)

                        # point_scoresはMIDのpoint_adj（pc全体と同長）
                        point_scores_all = point_scores_all.astype(np.float32)

                        # Region point lists
                        region_point_lists_A = build_region_point_lists(
                            reg_A, pt_A, point_scores_all[:1024], k_region, fanout
                        )
                        reg_B_global = [ (np.asarray(idxs, dtype=np.int32)+1024).tolist() for idxs in reg_B ]
                        pt_B_global  = [ (np.asarray(idxs, dtype=np.int32)+1024).tolist() for idxs in pt_B ]
                        region_point_lists_B = build_region_point_lists(
                            reg_B_global, pt_B_global, point_scores_all, k_region, fanout
                        )
                        region_point_lists_all = region_point_lists_A + region_point_lists_B

                        rr0 = time.time()
                        top_indices = roundrobin_select(
                            region_scores_all, region_point_lists_all, ds_points,
                            debug=args.debug_rr, point_scores=point_scores_all
                        )
                        rr1 = time.time()
                        sample_rr_time = rr1-rr0

                        ds_pc = pc[top_indices, :]

                    else:
                        if pc.shape[0] > 1024:
                            pc_use = pc[:1024]
                        elif pc.shape[0] < 1024:
                            pc_use = np.pad(pc,((0,1024-pc.shape[0]),(0,0)),'constant')
                        else:
                            pc_use = pc

                        reg_blk, pt_blk, _, _ = hierarchical_kmeans(pc_use, k_region, fanout)

                        region_scores_all = np.zeros((k_region,), dtype=np.float32)
                        for rid, idxs in enumerate(reg_blk):
                            region_scores_all[rid] = float(np.mean(region_per_pt[np.asarray(idxs, dtype=np.int32)]))

                        point_scores_all = point_scores_all.astype(np.float32)

                        region_point_lists = build_region_point_lists(
                            reg_blk, pt_blk, point_scores_all, k_region, fanout
                        )
                        rr0 = time.time()
                        top_indices = roundrobin_select(
                            region_scores_all, region_point_lists, ds_points,
                            debug=args.debug_rr, point_scores=point_scores_all
                        )
                        rr1 = time.time()
                        sample_rr_time = rr1-rr0

                        ds_pc = pc_use[top_indices, :]
                        pc = pc_use  # 以降の保存整合のため

                if save_rr_cache and top_indices is not None:
                    np.savez_compressed(rr_cache_path, top_idx=top_indices.astype(np.int32))

            # Attention Fusion（任意）
            if args.attention_fusion:
                alpha_af = 0.5  # RRではαは無いので固定（必要なら別途設計）
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
            class_time_stats[cls]['sum_rr']     += sample_rr_time
            class_time_stats[cls]['sum_shap']   += sample_shap_time
            class_time_stats[cls]['count'] += 1

            overall_total  += sample_total_time
            overall_down   += sample_down_time
            overall_region += sample_region_time
            overall_point  += sample_point_time
            overall_rr     += sample_rr_time
            overall_shap   += sample_shap_time
            overall_count  += 1

        print(f"Finished processing {len(sampled_data_list)} samples from {test_file}.")

        # ファイル保存（マージ方式はAMAと同じ）
        if os.path.exists(output_filename):
            with h5py.File(output_filename, "r") as f:
                existing_data  = f["data"][:]
                existing_label = f["label"][:]
                existing_feat  = f["feat"][:] if "feat" in f else None

            new_data  = np.concatenate([existing_data, np.array(sampled_data_list)], axis=0)
            new_label = np.concatenate([existing_label, np.array(sampled_label_list).reshape(-1, 1)], axis=0)

            if args.attention_fusion:
                add_feat = np.array(sampled_feat_list) if len(sampled_feat_list)>0 else None
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
            if args.attention_fusion and len(sampled_feat_list)>0:
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
        print(f"Saved RoundRobin downsampled point clouds for {test_file} to {output_filename}")

    # --- time stats ---
    time_output = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(time_output, "w") as time_f:
        time_f.write("Class\tDS_Points\tAvg_Region(sec)\tAvg_Point(sec)\tAvg_RR(sec)\tAvg_Down(sec)\tAvg_Total(sec)\tR_region(%)\tR_point(%)\tR_rr(%)\tSample_Count\n")
        for cls in sorted(class_time_stats.keys()):
            stats = class_time_stats[cls]
            if stats['count'] > 0:
                avg_region = stats['sum_region'] / stats['count']
                avg_point  = stats['sum_point']  / stats['count']
                avg_rr     = stats['sum_rr']     / stats['count']
                avg_down   = stats['sum_down']   / stats['count']
                avg_total  = stats['sum_total']  / stats['count']
                if avg_total == 0:
                    r_region = r_point = r_rr = 0.0
                else:
                    r_region = 100 * avg_region / avg_total
                    r_point  = 100 * avg_point  / avg_total
                    r_rr     = 100 * avg_rr     / avg_total
                time_f.write(f"{class_names[cls]}\t{ds_points}\t"
                             f"{avg_region:.6f}\t{avg_point:.6f}\t{avg_rr:.6f}\t"
                             f"{avg_down:.6f}\t{avg_total:.6f}\t"
                             f"{r_region:.2f}\t{r_point:.2f}\t{r_rr:.2f}\t"
                             f"{stats['count']}\n")
        if overall_count > 0:
            overall_avg_region = overall_region / overall_count
            overall_avg_point  = overall_point  / overall_count
            overall_avg_rr     = overall_rr     / overall_count
            overall_avg_down   = overall_down   / overall_count
            overall_avg_total  = overall_total  / overall_count
            if overall_avg_total == 0:
                Rr = Rp = Rrr = 0.0
            else:
                Rr  = 100 * overall_avg_region / overall_avg_total
                Rp  = 100 * overall_avg_point  / overall_avg_total
                Rrr = 100 * overall_avg_rr     / overall_avg_total
            time_f.write(f"ALL\t{ds_points}\t"
                         f"{overall_avg_region:.6f}\t{overall_avg_point:.6f}\t{overall_avg_rr:.6f}\t"
                         f"{overall_avg_down:.6f}\t{overall_avg_total:.6f}\t"
                         f"{Rr:.2f}\t{Rp:.2f}\t{Rrr:.2f}\t"
                         f"{overall_count}\n")
    print(f"Processing times by class saved to {time_output}")

    # --- MonteCarlo stats ---
    monte_output = os.path.join(output_folder, "Monte_Carlo.txt")
    with open(monte_output, "w") as f_mc:
        if all_deltas:
            concat     = np.concatenate(all_deltas)
            mean_delta = concat.mean()
            min_delta  = concat.min()
            max_delta  = concat.max()
            f_mc.write("Metric\tmean\tmin\tmax\n")
            f_mc.write("Delta\t{:.6f}\t{:.6f}\t{:.6f}\n".format(mean_delta, min_delta, max_delta))
            if all_s:
                f_mc.write("s\t{:.6f}\t{:.6f}\t{:.6f}\n".format(np.mean(all_s), np.min(all_s), np.max(all_s)))
            if all_se:
                f_mc.write("SE\t{:.6f}\t{:.6f}\t{:.6f}\n".format(np.mean(all_se), np.min(all_se), np.max(all_se)))
        else:
            f_mc.write("No Monte-Carlo data were collected.\n")
    print(f"Monte-Carlo statistics saved to {monte_output}")

if __name__ == "__main__":
    main()
