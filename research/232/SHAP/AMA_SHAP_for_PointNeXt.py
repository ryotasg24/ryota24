#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# 可視化用ライブラリ（今回は出力には使用しない）
#import open3d as o3d
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap

# --- PointNeXt 関連 ---
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg  # モデル構築用関数
from openpoints.utils import EasyConfig  # 設定ファイル読み込み用

# --- provider.py （PointNet時代のもの） ---
# （同じディレクトリまたはPYTHONPATH上に配置しておく前提）
import provider

# 保存用関数（後で読み込みと連結後、'w'モードで再作成する）
def save_h5_data(h5_filename, data, label, feat=None):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)
        if feat is not None:
            f.create_dataset('feat', data=feat)

# 各種関数の定義
def get_class_names(file_path):
    # クラス名リストをファイルから読み込む関数
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def spatial_divide_into_blocks(points, num_blocks, kmeans=None):
    # 点群を空間的に領域に分割（K-Means）する関数
    if kmeans is None:
        kmeans = KMeans(n_clusters=num_blocks, random_state=42)
        labels = kmeans.fit_predict(points)
    else:
        labels = kmeans.predict(points)
    blocks = []
    for i in range(num_blocks):
        block_indices = np.where(labels == i)[0].tolist()
        blocks.append(block_indices)
    return blocks, kmeans

def hierarchical_kmeans(points, k_region, fanout):
    # 上位 k_region → 各クラスタ fanout 分割
    km_reg = KMeans(n_clusters=k_region, random_state=42).fit(points)
    reg_lbl = km_reg.labels_
    region_blocks = [np.where(reg_lbl == r)[0].tolist() for r in range(k_region)]

    point_blocks = []
    subkm_dict   = {}
    for r, idxs in enumerate(region_blocks):
        sub_pts = points[idxs]
        # region に含まれる点が fanout 未満なら、その点数でクラスタリング（最低 1）
        # それでも fanout 個の “子クラスタ枠” は用意しておき、足りないぶんは空リストで埋める
        n_sub = fanout if len(sub_pts) >= fanout and fanout > 1 else max(1, len(sub_pts))
        km_sub = KMeans(n_clusters=n_sub, random_state=0).fit(sub_pts)
        subkm_dict[r] = km_sub
        # fanout 個ぶん回して point ブロックを必ず fanout*k_region そろえる
        for s in range(fanout):
            if s < n_sub:
                sub_idx = np.array(idxs)[km_sub.labels_ == s]
            else:
                sub_idx = np.array([], dtype=int)     # 空クラスターでパディング
            point_blocks.append(sub_idx.tolist())
    return region_blocks, point_blocks, km_reg, subkm_dict

def collect_background_point_clouds(target_label, train_files, num_point_clouds=50):
    # 指定したクラスラベルに属する点群をトレーニングファイルから収集する関数
    # ※ 各点群は1024点に統一される前提

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
    # 背景ベースラインの計算をベクトル化
    # background_point_clouds: (B,1024,3)

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
    # fanout*k_region 個の baseline (各 Point クラスタ平均)
    k_region = km_reg.n_clusters
    baselines = []
    # まず全背景を1つにまとめて region を一括予測
    B, N, _ = bg_pcs.shape
    X = bg_pcs.reshape(B * N, 3)
    reg_lbl_all = km_reg.predict(X)                               # (B*N,)
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
    return baselines   # len = fanout*k_region

def apply_block_mask_fixed_baseline(point_cloud, blocks, mask_vector, baseline_reg_Blocks):
    # 指定したマスク(mask_vector)に従い、マスクが0の領域の点を背景基準値に置換する

    masked_point_cloud = point_cloud.copy()
    for block_idx, mask in enumerate(mask_vector):
        if mask == 0:
            masked_point_cloud[blocks[block_idx]] = baseline_reg_Blocks[block_idx]
    return masked_point_cloud

# 前計算：各点の block ID (N,) と per-point baseline (N,3)
def _precompute_pointwise_maps(blocks, baseline_reg_Blocks):
    N = sum(len(b) for b in blocks)
    block_ids = np.empty(N, dtype=np.int64)
    for b, idxs in enumerate(blocks):
        block_ids[np.array(idxs, dtype=np.int64)] = b
    # baseline を各点に対応付け（正確に従来と等価）
    base_pt = np.stack([baseline_reg_Blocks[b] for b in block_ids], axis=0).astype(np.float32)  # (N,3)
    return block_ids, base_pt

def _predict_masks_in_chunks(mask_vectors, blocks, pc, baseline_reg_Blocks, model, device, chunk_size: int = 512, pin_mem: bool = True, do_empty_cache: bool = False, min_chunk_size: int = 8):
    # mask_vectors: (M,K) 0/1
    # pc: (N,3)
    # blocks: K blocks

    model.eval()
    M = len(mask_vectors)
    if M == 0: return np.array([])

    # 前計算
    block_ids, base_pt = _precompute_pointwise_maps(blocks, baseline_reg_Blocks)  # (N,), (N,3)
    N = pc.shape[0]
    pc_t = torch.from_numpy(pc.astype(np.float32)).to(device).unsqueeze(0)        # (1,N,3)
    base_t = torch.from_numpy(base_pt).to(device).unsqueeze(0)                    # (1,N,3)

    # 転送効率向上（任意）
    if pin_mem:
        pc_t = pc_t.contiguous()
        base_t = base_t.contiguous()

    out_list = []
    bid_t = torch.from_numpy(block_ids).to(device)                                 # (N,)
    step = int(chunk_size)
    s = 0
    while s < M:
        e = min(M, s + step)
        mv = np.asarray(mask_vectors[s:e], dtype=np.float32)                       # (B,K)
        try:
            if pin_mem and device.type == "cuda":
                mv_t_cpu = torch.from_numpy(mv).pin_memory()
                mv_t = mv_t_cpu.to(device, non_blocking=True)
            else:
                mv_t = torch.from_numpy(mv).to(device)
            keep = mv_t[:, bid_t].unsqueeze(-1)                                    # (B,N,1)
            pcB   = pc_t.expand(keep.shape[0], -1, -1)                             # (B,N,3)
            baseB = base_t.expand(keep.shape[0], -1, -1)                           # (B,N,3)
            masked = keep * pcB + (1.0 - keep) * baseB
            with torch.no_grad():
                logits = model(masked)                                             # (B,C)
            out_list.append(logits.detach().cpu().numpy())
            del mv_t, keep, pcB, baseB, masked, logits
            if do_empty_cache:
                torch.cuda.empty_cache()
            s = e  # 成功したので次へ進む
        except RuntimeError as err:
            oom = ("CUDA out of memory" in str(err)) or ("CUDA error: out of memory" in str(err))
            if not oom or step <= min_chunk_size:
                raise
            # 自動縮小
            if device.type == "cuda":
                torch.cuda.empty_cache()
            new_step = max(min_chunk_size, step // 2)
            print(f"[WARN] OOM detected. Reduce mask_chunk: {step} -> {new_step}")
            step = new_step
    return np.concatenate(out_list, axis=0)

def pointnext_predict_with_block_mask_fixed(mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks, model, device, chunk_size=512, pin_mem=True, min_chunk_size=8):
    # チャンク分割を内部で行い、GPU上でベクトル化
    return _predict_masks_in_chunks(mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks,
                                    model, device, chunk_size=chunk_size, pin_mem=pin_mem,
                                    min_chunk_size=min_chunk_size)

def shap_predict_block_mask(mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks, model, device, chunk_size=512, pin_mem=True, min_chunk_size=8):
    # SHAPのKernelExplainer用ラッパー関数
    return pointnext_predict_with_block_mask_fixed(
        mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks,
        model, device, chunk_size=chunk_size, pin_mem=pin_mem,
        min_chunk_size=min_chunk_size
    )

def compute_point_level_contributions(point_cloud, blocks, baseline_reg_Blocks, model, target_class_index, device):
    # 各点の寄与度を、入力点群の勾配と背景との差の内積から算出する

    model.eval()
    pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device, requires_grad=True)
    pc_tensor = pc_tensor.unsqueeze(0)
    pc_tensor.retain_grad()
    output = model(pc_tensor)
    target_output = output[0, target_class_index]
    target_output.backward()
    grad_val = pc_tensor.grad.squeeze(0).cpu().numpy()

    point_contrib = np.zeros(point_cloud.shape[0])
    for i, block in enumerate(blocks):
        baseline = baseline_reg_Blocks[i]
        for idx in block:
            diff = point_cloud[idx] - baseline
            point_contrib[idx] = np.dot(grad_val[idx], diff)
    return point_contrib

def compute_point_level_shap_values(point_cloud, blocks, block_shap_values, baseline_reg_Blocks, model, target_class_index, device, scaling_factor=1.0):
    # 統合SHAP値を、領域単位のSHAP値と各点の寄与度（Zスコア正規化した値）から算出する
    # 定義: Hybrid_SHAP = (Region_SHAP) + (scaling_factor × Point_SHAP)

    point_contrib = compute_point_level_contributions(point_cloud, blocks, baseline_reg_Blocks, model, target_class_index, device)
    point_shap = np.zeros(point_cloud.shape[0])
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


# AMA 用ヘルパ（L¹合わせ、Gate、AF 特徴生成）
class AdaptiveGateMLP(torch.nn.Module):
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
        beta  = F.softplus(b).squeeze(1)
        return alpha, beta

def l1_match_point(point_shap: np.ndarray, region_per_pt: np.ndarray, eps: float):
    L1_reg = float(np.sum(np.abs(region_per_pt))) + eps
    L1_pnt = float(np.sum(np.abs(point_shap))) + eps
    scale  = L1_reg / L1_pnt
    return point_shap * scale, L1_reg, L1_pnt

def gate_forward(N: int, var_reg: float, var_pnt: float,
                args, gate_model: Optional[AdaptiveGateMLP], device,
                scaler: Optional[dict] = None):
    if args.ama_mode == "mlp" and gate_model is not None:
        # feature: [N/1024, log(var_reg+eps), log(var_pnt+eps)] -> z-score
        eps = 1e-9
        feat = np.array([float(N)/1024.0, math.log(var_reg + eps), math.log(var_pnt + eps)], dtype=np.float32)
        if scaler is not None:
            mu = scaler["mu"]; std = scaler["std"]
            feat = (feat - mu) / (std + 1e-9)
        x = torch.tensor(feat[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            a, b = gate_model(x)
        return float(a.item()), float(b.item())
    # heuristic: N が小さいほど α↑、分散比も考慮
    Nmid = max(1, args.ama_Nmid)
    k    = float(args.ama_k)
    termN = -k * ((float(N) - Nmid) / Nmid)
    logr  = math.log((var_reg + 1e-9) / (var_pnt + 1e-9))
    logit = termN + 0.5 * logr
    alpha = 1.0 / (1.0 + math.exp(-logit))
    beta  = 1.0
    return alpha, beta

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
    # Global: セントロイド差分・全体標準偏差・半径
    K = points.shape[0]
    c = points.mean(axis=0, keepdims=True)
    gstd = points.std(axis=0, keepdims=True) + 1e-9
    r = np.linalg.norm(points - c, axis=1, keepdims=True)
    gfeat = np.concatenate([np.repeat(c, K, axis=0) - points,
                            np.repeat(gstd, K, axis=0),
                            r], axis=1)  # (K,7)
    # Local: kNN のオフセット統計
    idx = _knn_idx(points, k_local)
    nbr = points[idx]                                # (K,k,3)
    off = nbr - points[:,None,:]
    lfeat = np.concatenate([off.mean(axis=1),
                            off.std(axis=1),
                            np.linalg.norm(off, axis=2).max(axis=1, keepdims=True)], axis=1)  # (K,7)
    F = alpha * gfeat + (1.0 - alpha) * lfeat        # (K,7)
    rep = int(math.ceil(float(feat_dim) / F.shape[1]))
    F = np.tile(F, (1, rep))[:, :feat_dim].astype(np.float32)
    return F

# ===============================
# main() 関数
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Downsample SHAP point clouds for ModelNet40 test files using PointNeXt."
    )
    parser.add_argument("--ds_points", type=int, default=500, help="Number of points to downsample each point cloud (default 500)")
    parser.add_argument("--pattern", type=int, default=10000, help="Region-SHAP の 近似マスキングパターン数 (default 100)")
    parser.add_argument("--division_region", type=int, default=32, help="Region-SHAP 用 k (default 32)")
    parser.add_argument("--division_point",  type=int, default=32, help="Point-SHAP 用 k (Region の整数倍にして下さい)")
    parser.add_argument("--cache_mode", type=str, default="auto", choices=["auto", "save", "load"], help="Hybrid-SHAP キャッシュ: auto=あればload/無ければsave, save=毎回再計算, load=常に読込")
    parser.add_argument("--num_groups", type=int, default=1, help="Number of groups to divide the 40 classes (default 1)")
    parser.add_argument("--group_index", type=int, default=0, help="Index of the group to process (0-indexed, default 0)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml", help="Path to the PointNeXt config file")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth", help="Path to the PointNeXt checkpoint file")
    parser.add_argument("--dataset", type=str, default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048", help="テスト用データセットのルート (test_files.txt を含む)。背景(train)はModelNet40を使用")
    parser.add_argument("--run_tag", type=str, default="", help="任意タグ。指定時のみ 出力/MID/Global キャッシュ名に付与して別ツリーに保存（未指定なら従来名のまま）")
    # AMA関連
    parser.add_argument("--ama", action="store_true", help="Adaptive Multi-Scale Attention を有効化（α, β を自動決定）")
    parser.add_argument("--ama_mode", choices=["heuristic","mlp"], default="heuristic", help="α,β の決定を heuristic か小型MLPで行う（学習時に後者を使用）")
    parser.add_argument("--ama_eps", type=float, default=1e-6, help="L1正規化の ε")
    parser.add_argument("--ama_Nmid", type=int, default=300, help="N の基準点（heuristic 用）")
    parser.add_argument("--ama_k", type=float, default=4.0, help="N ロジット傾き（heuristic 用）")
    parser.add_argument("--attention_fusion", action="store_true", help="ダウンサンプル後に α を再利用して Global/Local を特徴融合し、'feat' を保存")
    parser.add_argument("--feat_dim", type=int, default=32, help="Attention Fusion で出力する特徴次元")
    parser.add_argument("--knn_k", type=int, default=16, help="Attention Fusion の局所 kNN 数")
    # L1 正規化オフ
    parser.add_argument("--no_l1", action="store_true", help="Point-SHAP の L1 正規化を無効化（生の point SHAP を使用）")
    # AMAmlp用
    parser.add_argument("--gate_ckpt", type=str, default="", help="学習済みGate MLP(.pth)。指定時 ama_mode=mlp を推奨")
    parser.add_argument("--gate_hidden", type=int, default=8, help="Gate-MLP の隠れ次元（学習時と同じにする）")
    parser.add_argument("--gate_scaler", type=str, default="", help="Gate-MLP の前処理スケーラ(.npz: mu,std)")
    parser.add_argument("--gate_dump_train", action="store_true", help="学習用に per-point Region / Point_adj をダンプ")
    parser.add_argument("--gate_dump_dir", type=str, default="result/_gate_train_dump", help="ダンプ出力ディレクトリ")
    # 予測バッチ最適化
    parser.add_argument("--mask_chunk", type=int, default=512, help="SHAP用マスクの推論チャンクサイズ")
    parser.add_argument("--min_mask_chunk", type=int, default=8, help="自動縮小時の最小チャンクサイズ")
    parser.add_argument("--no_pinmem", action="store_true", help="CPU→GPU転送でpinned memoryを使わない（デフォルトは使う）")
    parser.add_argument("--force_mid_build", action="store_true", help="global(N依存)キャッシュがあっても再計算して MID(N非依存) を生成する")

    args = parser.parse_args()
    ds_points = args.ds_points
    num_groups = args.num_groups
    group_index = args.group_index
    pattern = args.pattern
    k_region = args.division_region
    k_point  = args.division_point
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_model = None
    gate_scaler = None
    gate_tag = ""
    DATA_DIR = args.dataset

    if args.ama and args.ama_mode == "mlp" and args.gate_ckpt:
        gate_model = AdaptiveGateMLP(hidden=args.gate_hidden)
        gate_model.load_state_dict(torch.load(args.gate_ckpt, map_location="cpu"))
        gate_model.to(device).eval()
        # ckpt ファイル名（拡張子除く）をタグ化。例: gate_mlp_n300_500_800_1000
        try:
            gate_tag = Path(args.gate_ckpt).stem
        except Exception:
            gate_tag = os.path.splitext(os.path.basename(args.gate_ckpt))[0]
        if args.gate_scaler and os.path.exists(args.gate_scaler):
            _sc = np.load(args.gate_scaler)
            gate_scaler = {"mu": _sc["mu"].astype(np.float32), "std": _sc["std"].astype(np.float32)}
        else:
            print("[WARN] MLP mode without scaler: falling back to raw features")

    # モード名を追加 (ama 無効なら "noama", 有効なら "heuristic" または "mlp")
    ama_tag = "noAMA"
    if args.ama:
        ama_tag = f"{args.ama_mode}"
    # Gate-MLP を使う場合は ckpt のベース名をタグに含めてキャッシュ衝突を防ぐ
    ckpt_tag = ""
    if args.ama and args.ama_mode == "mlp" and args.gate_ckpt:
        ckpt_tag = f"_{os.path.splitext(os.path.basename(args.gate_ckpt))[0]}"
        if args.ama_mode == "mlp" and not args.gate_ckpt:
            print("[WARN] --ama_mode=mlp だが --gate_ckpt が未指定。heuristic 相当の挙動になる。")

    # fan-out 動的判定
    if k_point % k_region != 0:
        sys.exit("--division_point は --division_region の整数倍にして下さい")
    fanout = k_point // k_region
    print(f"[INFO] Region k={k_region}, Point k={k_point}, fan-out={fanout}")


    # データセットディレクトリ
    DATA_DIR_TRAIN = "/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048"
    DATA_DIR_TEST  = args.dataset

    # クラス名の読み込み
    class_names_file = os.path.join(DATA_DIR_TRAIN, "shape_names.txt")
    class_names = get_class_names(class_names_file)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.")

    # 40クラスを40グループに分割（各グループは1クラスまたは複数クラス）
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
        sys.exit(f"Invalid group_index {group_index}. Must be between 0 and {num_groups-1}.")
    selected_classes = groups[group_index]
    print(f"Processing group {group_index} with classes: {selected_classes}")

    # 出力先フォルダ（ds_points毎に出力ファイルは同一フォルダ内でマージ）
    l1_tag = "-nol1" if args.no_l1 else ""
    _out_tag_base = f"AMA{l1_tag}-{ama_tag}" + (f"-{gate_tag}" if (ama_tag == "mlp" and gate_tag) else "")
    out_tag = _out_tag_base if (args.run_tag == "") else f"{_out_tag_base}-{args.run_tag}"
    output_folder = os.path.join(
        "result", f"{out_tag}_kr{k_region}_kp{k_point}_p{pattern}_dsSHAP_PointNeXt_h5", str(ds_points)
        )                    ####出力ファイルパスの修正
    os.makedirs(output_folder, exist_ok=True)
    # Gate-MLP 学習データダンプ先
    dump_dir_effective = args.gate_dump_dir
    if args.gate_dump_train:
        # 設定で分離（例: result/_gate_train_dump/p10000_kr32_kp64_heuristic_nol1/）
        dump_tag = f"p{pattern}_kr{k_region}_kp{k_point}_{('mlp' if args.ama_mode=='mlp' else 'heuristic')}{('_nol1' if args.no_l1 else '')}"
        if args.run_tag:
            dump_tag += f"_{args.run_tag}"
        dump_dir_effective = os.path.join(args.gate_dump_dir, dump_tag)
        Path(dump_dir_effective).mkdir(parents=True, exist_ok=True)
    # ★ キャッシュ（N も含めて分離：α,β が N 依存のため）
    BASE_CACHE_DIR = os.path.join("result", "_shap_cache_global")
    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    #   （モデル/重み/モード/N でサブディレクトリを分離）
    cache_tag = f"p{pattern}_kr{k_region}_kp{k_point}_ama-{ama_tag}_N{ds_points}{('_noL1' if args.no_l1 else '')}"
    # ckpt/gate の識別子も付与（ある場合）
    if ckpt_tag:
        cache_tag += f"_{ckpt_tag}"
    if ama_tag == "mlp" and gate_tag:
        cache_tag += f"-{gate_tag}"
    if args.run_tag:
        cache_tag += f"_{args.run_tag}"
    CACHE_DIR = os.path.join(BASE_CACHE_DIR, cache_tag)        ###キャッシュファイルの指定(有:キャッシュ使用, 無：新規計算)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # === MID(N非依存)キャッシュ (Region/Point per-point を保存) ===
    MID_CACHE_BASE = os.path.join("result", "_shap_cache_mid")
    ckpt_name_for_tag = Path(args.checkpoint_path).stem
    mid_tag = f"p{pattern}_kr{k_region}_kp{k_point}_ckpt-{ckpt_name_for_tag}{('_noL1' if args.no_l1 else '')}"
    if args.run_tag:
        mid_tag += f"_{args.run_tag}"
    MID_CACHE_DIR = os.path.join(MID_CACHE_BASE, mid_tag)
    os.makedirs(MID_CACHE_DIR, exist_ok=True)

    # テストファイルとトレーニングファイルのパス取得
    TRAIN_FILES = provider.getDataFiles(os.path.join(DATA_DIR_TRAIN, "train_files.txt"))
    TEST_FILES  = provider.getDataFiles(os.path.join(DATA_DIR_TEST,  "test_files.txt"))

    # 選択された各クラスの背景点群を収集
    backgrounds = {}
    for cls in selected_classes:
        print(f"Collecting background point clouds for class '{class_names[cls]}' (class index {cls})...")
        bg = collect_background_point_clouds(cls, TRAIN_FILES, num_point_clouds=50)
        backgrounds[cls] = bg

    # PointNeXtモデルのロード
    model, cfg = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)

    # 各クラスごとの処理時間を集計する辞書
    class_time_stats = {cls: {'sum_total':0,'sum_down':0, 'sum_region':0,'sum_point':0,'sum_hybrid':0, 'sum_shap':0, 'count':0} for cls in selected_classes}
    overall_total = 0.0
    overall_down = 0.0
    overall_region  = 0.0
    overall_point   = 0.0
    overall_hybrid  = 0.0
    overall_shap = 0.0
    overall_count   = 0


    # 各テストファイルを処理（結果は同一出力ファイルに追記してマージ）
    for test_file in TEST_FILES:
        print(f"\n----- Processing test file: {test_file} -----")
        data, labels = provider.loadDataFile(test_file)
        num_samples = len(data)
        print(f"Total samples in {os.path.basename(test_file)}: {num_samples}")
        base_name = os.path.basename(test_file) # キャッシュ判定
        name_without_ext  = os.path.splitext(base_name)[0]
        subdir    = os.path.basename(DATA_DIR_TEST)
        output_filename = os.path.join(output_folder, subdir, base_name)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if os.path.exists(output_filename):
            print(f"[CLEAN] Remove existing output to avoid accumulation: {output_filename}")
            os.remove(output_filename)

        sampled_data_list = []
        sampled_label_list = []
        sampled_feat_list  = []  # attention_fusion 有効時のみ使用

        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}")
            cls = int(labels[i])
            # --------- キャッシュ判定 ----------
            sample_id  = f"{name_without_ext}_{i:05d}_r{k_region}_p{k_point}"
            cache_path     = os.path.join(CACHE_DIR, sample_id + ".npz")
            mid_cache_path = os.path.join(MID_CACHE_DIR, sample_id + ".npz")
            print(f"[DEBUG] Using cache file: {cache_path}")
            use_cache   = (args.cache_mode in ["auto","load"]) and os.path.exists(cache_path) and (not args.force_mid_build)
            save_cache  = (args.cache_mode in ["auto","save"]) and (not use_cache)
            mid_hit     = os.path.exists(mid_cache_path)
            hybrid_shap = None

            if cls not in selected_classes:
                continue
            gc.collect()
            bg_pc = backgrounds.get(cls, None)
            if bg_pc is None or bg_pc.size == 0:
                print(f"Skipping sample {i} (class {cls}): No background data available.")
                continue
            pc = data[i]

            # 0)  キャッシュヒットなら即読み込み
            if use_cache:
                print(f"  ↳ cache hit: {cache_path}")
                _cache = np.load(cache_path)
                hybrid_shap = _cache["shap"]
                alpha_cached = float(_cache["alpha"][0]) if "alpha" in _cache.files else None
                # 速度計測は 0 扱い
                sample_region_time = sample_point_time = sample_hybrid_time = \
                sample_down_time   = sample_total_time = sample_shap_time   = 0.0

                # --- ダウンサンプリング用インデックス (cache hit 時) ---
                sorted_indices = np.argsort(hybrid_shap)[::-1]
                top_indices    = sorted_indices[:ds_points]
                ds_pc          = pc[top_indices, :]
            elif mid_hit:
                print(f"  ↳ MID cache hit: {mid_cache_path}")
                _mid = np.load(mid_cache_path)
                region_per_pt = _mid["region_per_pt"].astype(np.float32).reshape(-1)
                point_adj     = _mid["point_adj"].astype(np.float32).reshape(-1)
                if "pc" in _mid.files:
                    pc = _mid["pc"].astype(np.float32)
                # N依存は Gate のみ。ここでα,βを決定し再合成
                var_reg = float(np.var(region_per_pt))
                var_pnt = float(np.var(point_adj))
                if args.ama:
                    alpha, beta = gate_forward(ds_points, var_reg, var_pnt, args, gate_model, device, gate_scaler)
                else:
                    alpha, beta = 0.5, 1.0
                hybrid_shap = alpha * region_per_pt + (1.0 - alpha) * beta * point_adj
                # SHAP時間は回していないので 0 扱い
                sample_region_time = sample_point_time = sample_hybrid_time = \
                sample_down_time   = sample_total_time = sample_shap_time   = 0.0
                # 上位N抽出
                sorted_indices = np.argsort(hybrid_shap)[::-1]
                top_indices    = sorted_indices[:ds_points]
                ds_pc          = pc[top_indices, :]
                # ついでに global(N依存) も書いておくと次回更に速い
                if save_cache:
                    np.savez_compressed(cache_path, shap=hybrid_shap, alpha=np.array([alpha], dtype=np.float32))

            # 1)  キャッシュが無い場合はヘビー計算して保存
            else:
                if pc.shape[0] == 2048:
                    # ---------- heavy 計算 (A/B) ----------
                    pc_A = pc[:1024];  pc_B = pc[1024:]
                    (reg_A, pt_A, kmA_reg, kmA_sub) = hierarchical_kmeans(pc_A, k_region, fanout)
                    (reg_B, pt_B, kmB_reg, kmB_sub) = hierarchical_kmeans(pc_B, k_region, fanout)
                    baseline_reg_A = compute_background_baseline(bg_pc, kmA_reg)
                    baseline_reg_B = compute_background_baseline(bg_pc, kmB_reg)
                    baseline_pt_A  = compute_point_baseline(bg_pc, kmA_reg, kmA_sub, fanout)
                    baseline_pt_B  = compute_point_baseline(bg_pc, kmB_reg, kmB_sub, fanout)
                    target_explain_class_index = cls

                    t0 = time.time()
                    # --- Region / Point for A ---
                    def shap_predict_A(mv):
                        return shap_predict_block_mask(
                            mv, reg_A, pc_A, baseline_reg_A, model, device,
                            chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem), min_chunk_size=args.min_mask_chunk
                        )

                    explainer_A = shap.KernelExplainer(shap_predict_A, np.zeros((1, len(reg_A))))

                    rA0 = time.time()
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                        block_A = (
                            explainer_A.shap_values(
                                np.ones((1, len(reg_A))), nsamples=pattern, l1_reg="num_features(10)"
                            )[target_explain_class_index].reshape(-1)
                        )
                    rA1 = time.time()
                    pA0 = time.time()
                    block_A_pt = np.repeat(block_A, fanout)
                    pt_shap_A  = compute_point_level_shap_values(
                                        pc_A, pt_A, block_A_pt, baseline_pt_A,
                                        model, target_explain_class_index,
                                        device, scaling_factor=1.0)
                    pA1 = time.time()

                    # --- Region / Point for B ---
                    def shap_predict_B(mv):
                        out = shap_predict_block_mask(
                            mv, reg_B, pc_B, baseline_reg_B,
                            model, device,
                            chunk_size=args.mask_chunk,
                            pin_mem=(not args.no_pinmem),
                            min_chunk_size=args.min_mask_chunk
                        )
                        return out
                    explainer_B = shap.KernelExplainer(shap_predict_B, np.zeros((1,len(reg_B))))
                    rB0 = time.time()
                    with warnings.catch_warnings():                 # ★ 同様に抑制
                        warnings.filterwarnings("ignore",
                                                message=".*active set degenerate.*")
                        block_B = (
                            explainer_B.shap_values(np.ones((1, len(reg_B))), nsamples=pattern, l1_reg="num_features(10)")
                            [target_explain_class_index].reshape(-1)
                        )
                    rB1 = time.time()
                    pB0 = time.time()
                    block_B_pt = np.repeat(block_B, fanout)
                    pt_shap_B  = compute_point_level_shap_values(
                        pc_B, pt_B, block_B_pt, baseline_pt_B,
                        model, target_explain_class_index,
                        device, scaling_factor=1.0)
                    pB1 = time.time()

                    # --- Hybrid (AMA) ---
                    h0 = time.time()
                    # Region SHAP を子クラスタへブロードキャスト（per-point へ展開）
                    region_A_exp = np.repeat(block_A, fanout)
                    region_A_per_pt = np.zeros_like(pt_shap_A)
                    for b_idx, idxs in enumerate(pt_A):
                        region_A_per_pt[idxs] = region_A_exp[b_idx]
                    region_B_exp = np.repeat(block_B, fanout)
                    region_B_per_pt = np.zeros_like(pt_shap_B)
                    for b_idx, idxs in enumerate(pt_B):
                        region_B_per_pt[idxs] = region_B_exp[b_idx]
                    # L1 正規化でスケール合わせ（線形性維持）
                    if args.no_l1:
                        pt_adj_A = pt_shap_A
                        pt_adj_B = pt_shap_B
                    else:
                        pt_adj_A, L1rA, L1pA = l1_match_point(pt_shap_A, region_A_per_pt, args.ama_eps)
                        pt_adj_B, L1rB, L1pB = l1_match_point(pt_shap_B, region_B_per_pt, args.ama_eps)
                    # 分散統計（Gate 入力用）を全体で評価
                    reg_all = np.concatenate([region_A_per_pt, region_B_per_pt])
                    pnt_all = np.concatenate([pt_adj_A,        pt_adj_B])
                    var_reg = float(np.var(reg_all))
                    var_pnt = float(np.var(pnt_all))
                    # α, β の決定
                    if args.ama:
                        alpha, beta = gate_forward(ds_points, var_reg, var_pnt, args, gate_model, device, gate_scaler)
                    else:
                        alpha, beta = 0.5, 1.0
                    # (任意) Gate-MLP 学習データをダンプ
                    if args.gate_dump_train:
                        region_full = np.concatenate([region_A_per_pt, region_B_per_pt], axis=0)
                        point_full  = np.concatenate([pt_adj_A,       pt_adj_B], axis=0)
                        pc_full     = np.concatenate([pc_A, pc_B], axis=0)
                        dump_id = f"{name_without_ext}_{i:05d}.npz"
                        np.savez_compressed(
                            os.path.join(dump_dir_effective, dump_id),
                            region_per_pt=region_full.astype(np.float32),
                            point_adj=point_full.astype(np.float32),
                            pc=pc_full.astype(np.float32),
                            label=np.array([cls], dtype=np.int32)
                        )

                    # Hybrid-SHAP = α·Region + (1-α)·β·Point_adj
                    hybrid_A = alpha * region_A_per_pt + (1.0 - alpha) * beta * pt_adj_A
                    hybrid_B = alpha * region_B_per_pt + (1.0 - alpha) * beta * pt_adj_B
                    hybrid_shap = np.concatenate([hybrid_A, hybrid_B])
                    h1 = time.time()

                    # === MIDキャッシュ保存（必ず保存；Nに依存しない材料）===
                    try:
                        region_full = np.concatenate([region_A_per_pt, region_B_per_pt], axis=0)
                        point_full  = np.concatenate([pt_adj_A,       pt_adj_B],       axis=0)
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
                    # 時間まとめ
                    sample_region_time = (rA1-rA0) + (rB1-rB0)
                    sample_point_time  = (pA1-pA0) + (pB1-pB0)
                    sample_hybrid_time = h1-h0
                    t_final            = time.time()
                    sample_total_time  = t_final - t0
                    sample_down_time   = sample_total_time - (sample_region_time + sample_point_time + sample_hybrid_time)
                    sample_shap_time   = sample_region_time + sample_point_time

                else:
                    # ---------- heavy 計算 (≤1024) ----------
                    if pc.shape[0] > 1024:
                        pc = pc[np.random.choice(pc.shape[0],1024,replace=False)]
                    elif pc.shape[0] < 1024:
                        pc = np.pad(pc,((0,1024-pc.shape[0]),(0,0)),'constant')

                    (reg_blk, pt_blk, km_reg, km_sub) = hierarchical_kmeans(pc, k_region, fanout)
                    baseline_reg_blk = compute_background_baseline(bg_pc, km_reg)
                    baseline_pt_blk  = compute_point_baseline(bg_pc, km_reg, km_sub, fanout)
                    target_explain_class_index = cls

                    t0 = time.time()
                    def shap_predict_blk(mv):
                        out = shap_predict_block_mask(
                            mv, reg_blk, pc, baseline_reg_blk,
                            model, device,
                            chunk_size=args.mask_chunk,
                            pin_mem=(not args.no_pinmem),
                            min_chunk_size=args.min_mask_chunk
                        )
                        return out
                    explainer = shap.KernelExplainer(
                                    shap_predict_blk,
                                    np.zeros((1, len(reg_blk))))
                    r0 = time.time()
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                message=".*active set degenerate.*")
                        block_vals = (
                            explainer.shap_values(np.ones((1, len(reg_blk))), nsamples=pattern, l1_reg="num_features(10)")
                            [target_explain_class_index].reshape(-1)
                        )
                    r1 = time.time()
                    p0 = time.time()
                    block_vals_pt = np.repeat(block_vals, fanout)
                    pt_shap_blk   = compute_point_level_shap_values(
                        pc, pt_blk, block_vals_pt, baseline_pt_blk,
                        model, target_explain_class_index, device,
                        scaling_factor=1.0)
                    p1 = time.time()
                    h0 = time.time()
                    region_blk_exp = np.repeat(block_vals, fanout)
                    region_blk_per_pt = np.zeros_like(pt_shap_blk)
                    for b_idx, idxs in enumerate(pt_blk):
                        region_blk_per_pt[idxs] = region_blk_exp[b_idx]
                    if args.no_l1:
                        pt_adj_blk = pt_shap_blk
                    else:
                        pt_adj_blk, L1r, L1p = l1_match_point(pt_shap_blk, region_blk_per_pt, args.ama_eps)
                    var_reg = float(np.var(region_blk_per_pt))
                    var_pnt = float(np.var(pt_adj_blk))
                    if args.ama:
                        alpha, beta = gate_forward(ds_points, var_reg, var_pnt, args, gate_model, device, gate_scaler)
                    else:
                        alpha, beta = 0.5, 1.0
                    if args.gate_dump_train:
                        dump_id = f"{name_without_ext}_{i:05d}.npz"
                        np.savez_compressed(
                            os.path.join(dump_dir_effective, dump_id),
                            region_per_pt=region_blk_per_pt.astype(np.float32),
                            point_adj=pt_adj_blk.astype(np.float32),
                            pc=pc.astype(np.float32),
                            label=np.array([cls], dtype=np.int32)
                        )
                    hybrid_shap = alpha * region_blk_per_pt + (1.0 - alpha) * beta * pt_adj_blk
                    h1 = time.time()
                    sample_region_time = r1-r0
                    sample_point_time  = p1-p0
                    sample_hybrid_time = h1 - h0
                    t_final            = time.time()
                    sample_total_time  = t_final - t0
                    sample_down_time   = sample_total_time - (sample_region_time + sample_point_time + sample_hybrid_time)
                    sample_shap_time   = sample_region_time + sample_point_time

                # --- キャッシュ保存 ---
                if save_cache:
                    np.savez_compressed(cache_path, shap=hybrid_shap, alpha=np.array([alpha], dtype=np.float32))

                # --- ダウンサンプリング用インデックス ---
                sorted_indices = np.argsort(hybrid_shap)[::-1]
                top_indices    = sorted_indices[:ds_points]
                ds_pc          = pc[top_indices, :]

                # Attention Fusion（任意設定）: どちらの分岐でも分岐外の共通ブロックで統一実行
                if args.attention_fusion:
                    if use_cache and 'alpha_cached' in locals() and alpha_cached is not None:
                        alpha_af = alpha_cached
                    elif 'alpha' in locals():
                        alpha_af = alpha
                    else:
                        alpha_af, _ = gate_forward(ds_points, 0.5, 0.5, args, None, device)  # フォールバック
                    F_feat = compute_attention_fusion_features(
                        ds_pc, alpha_af, feat_dim=args.feat_dim, k_local=args.knn_k
                    )
                    sampled_feat_list.append(F_feat.astype('float32'))

            # 集計
            class_time_stats[cls]['sum_total']  += sample_total_time
            class_time_stats[cls]['sum_down']   += sample_down_time
            class_time_stats[cls]['sum_region'] += sample_region_time
            class_time_stats[cls]['sum_point']  += sample_point_time
            class_time_stats[cls]['sum_hybrid'] += sample_hybrid_time
            class_time_stats[cls]['sum_shap']   += sample_shap_time
            class_time_stats[cls]['count'] += 1
            overall_total += sample_total_time
            overall_down    += sample_down_time
            overall_region  += sample_region_time
            overall_point   += sample_point_time
            overall_hybrid  += sample_hybrid_time
            overall_shap    += sample_shap_time
            overall_count += 1

            sampled_data_list.append(ds_pc.astype('float32'))
            sampled_label_list.append(cls)
            print(f"Processed sample {i+1}/{num_samples}: final downsampled points {ds_pc.shape[0]}")
        print(f"Finished processing {len(sampled_data_list)} samples from {test_file}.")

        # ファイルマージ：既存ファイルがあれば読み込み、閉じた後に新しいサンプルと連結して保存
        if os.path.exists(output_filename):
            with h5py.File(output_filename, "r") as f:
                existing_data = f["data"][:]
                existing_label = f["label"][:]
                existing_feat = f["feat"][:] if "feat" in f else None
            # sampled_label_listを2次元に変換して連結
            new_data = np.concatenate([existing_data, np.array(sampled_data_list)], axis=0)
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
        print(f"Saved integrated downsampled SHAP point clouds for {test_file} to {output_filename}")

    # 最終的な処理時間の統計を出力
    time_output = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(time_output, "w") as time_f:
        time_f.write("Class\tDS_Points\tAvg_Region(sec)\tAvg_Point(sec)\tAvg_Hybrid(sec)\tAvg_Down(sec)\tAvg_Total(sec)\tR_region(%)\tR_point(%)\tR_hybrid(%)\tSample_Count\n")
        for cls in sorted(class_time_stats.keys()):
            stats = class_time_stats[cls]
            if stats['count'] > 0:
                avg_region = stats['sum_region'] / stats['count']
                avg_point  = stats['sum_point']  / stats['count']
                avg_hybrid = stats['sum_hybrid'] / stats['count']
                avg_down   = stats['sum_down']   / stats['count']
                avg_total  = stats['sum_total']  / stats['count']
                if avg_total == 0:
                    r_region = r_point = r_hybrid = 0.0      # すべてキャッシュ
                else:
                    r_region = 100 * avg_region / avg_total
                    r_point  = 100 * avg_point  / avg_total
                    r_hybrid = 100 * avg_hybrid / avg_total
                time_f.write(f"{class_names[cls]}\t{ds_points}\t"
                            f"{avg_region:.6f}\t{avg_point:.6f}\t{avg_hybrid:.6f}\t"
                            f"{avg_down:.6f}\t{avg_total:.6f}\t"
                            f"{r_region:.2f}\t{r_point:.2f}\t{r_hybrid:.2f}\t"
                            f"{stats['count']}\n")
        if overall_count > 0:
            overall_avg_shap = overall_shap / overall_count
            overall_avg_region = overall_region / overall_count
            overall_avg_point  = overall_point  / overall_count
            overall_avg_hybrid = overall_hybrid / overall_count
            overall_avg_down   = overall_down   / overall_count
            overall_avg_total  = overall_total  / overall_count
            if overall_avg_total == 0:
                Rr = Rp = Rh = 0.0      # すべてキャッシュ
            else:
                Rr = 100 * overall_avg_region / overall_avg_total
                Rp = 100 * overall_avg_point  / overall_avg_total
                Rh = 100 * overall_avg_hybrid / overall_avg_total
            time_f.write(f"ALL\t{ds_points}\t"
                        f"{overall_avg_region:.6f}\t{overall_avg_point:.6f}\t{overall_avg_hybrid:.6f}\t"
                        f"{overall_avg_down:.6f}\t{overall_avg_total:.6f}\t"
                        f"{Rr:.2f}\t{Rp:.2f}\t{Rh:.2f}\t"
                        f"{overall_count}\n")
    print(f"Processing times by class saved to {time_output}")

if __name__ == "__main__":
    main()
