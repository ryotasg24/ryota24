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

# 可視化用ライブラリ（今回は出力には使用しません）
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- PointNeXt 関連 ---
# PointNeXt本体のルート（例：/workspace/PointNeXt）をsys.pathに追加
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg  # モデル構築用関数
from openpoints.utils import EasyConfig  # 設定ファイル読み込み用

# --- provider.py （PointNet時代のもの） ---
# （同じディレクトリまたはPYTHONPATH上に配置しておく前提）
import provider

# ======================================================
# 保存用関数（後で読み込みと連結後、'w'モードで再作成する）
def save_h5_data(h5_filename, data, label):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)
# ======================================================

# ===============================
# 各種関数の定義
# ===============================

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
    # background_point_clouds: (B,1024,3) を仮定
    if background_point_clouds.ndim != 3:
        background_point_clouds = np.asarray(background_point_clouds)
    B, N, _ = background_point_clouds.shape
    X = background_point_clouds.reshape(B * N, 3)
    lbl = kmeans.predict(X)
    K = kmeans.n_clusters
    out = []
    for i in range(K):
        sel = (lbl == i)
        out.append(X[sel].mean(axis=0) if np.any(sel) else np.zeros(3, dtype=X.dtype))
    return out

def compute_point_baseline(bg_pcs, km_reg, subkm_dict, fanout):
    # fanout*k_region 個の baseline (各 Point クラスタ平均) をベクトル化
    k_region = km_reg.n_clusters
    B, N, _ = bg_pcs.shape
    X = bg_pcs.reshape(B * N, 3)
    reg_lbl_all = km_reg.predict(X)                    # (B*N,)
    baselines = []
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

def apply_block_mask_fixed_baseline(point_cloud, blocks, mask_vector, baseline_reg_Blocks):
    # （CPU版ヘルパは残すが、推論本体は下のGPU版でまとめて行う）
    masked_point_cloud = point_cloud.copy()
    for block_idx, mask in enumerate(mask_vector):
        if mask == 0:
            masked_point_cloud[blocks[block_idx]] = baseline_reg_Blocks[block_idx]
    return masked_point_cloud

# 前計算（各点→所属block, 各点→baseline座標）
def _precompute_pointwise_maps(blocks, baseline_reg_Blocks):
    N = sum(len(b) for b in blocks)
    block_ids = np.empty(N, dtype=np.int64)
    for b, idxs in enumerate(blocks):
        block_ids[np.array(idxs, dtype=np.int64)] = b
    base_pt = np.stack([baseline_reg_Blocks[b] for b in block_ids], axis=0).astype(np.float32)  # (N,3)
    return block_ids, base_pt

#  GPUベクトル化されたマスク推論（chunk化＋pinned memory）
def _predict_masks_in_chunks(mask_vectors, blocks, pc, baseline_reg_Blocks, model, device,
                            chunk_size: int = 512, pin_mem: bool = True, min_chunk_size: int = 8):
    model.eval()
    M = len(mask_vectors)
    if M == 0:
        return np.array([])

    block_ids, base_pt = _precompute_pointwise_maps(blocks, baseline_reg_Blocks)
    pc_t   = torch.from_numpy(pc.astype(np.float32)).to(device).unsqueeze(0)   # (1,N,3)
    base_t = torch.from_numpy(base_pt).to(device).unsqueeze(0)                 # (1,N,3)
    if pin_mem:
        pc_t = pc_t.contiguous()
        base_t = base_t.contiguous()

    out = []
    bid_t = torch.from_numpy(block_ids).to(device)                             # (N,)
    step = int(chunk_size)
    s = 0
    while s < M:
        e = min(M, s + step)
        mv = np.asarray(mask_vectors[s:e], dtype=np.float32)                   # (B,K)
        try:
            if pin_mem and device.type == "cuda":
                mv_t = torch.from_numpy(mv).pin_memory().to(device, non_blocking=True)
            else:
                mv_t = torch.from_numpy(mv).to(device)
            keep = mv_t[:, bid_t].unsqueeze(-1)                                # (B,N,1)
            pcB   = pc_t.expand(keep.shape[0], -1, -1)                         # (B,N,3)
            baseB = base_t.expand(keep.shape[0], -1, -1)                       # (B,N,3)
            masked = keep * pcB + (1.0 - keep) * baseB
            with torch.no_grad():
                logits = model(masked)                                         # (B,C)
            out.append(logits.detach().cpu().numpy())
            del mv_t, keep, pcB, baseB, masked, logits
            s = e
        except RuntimeError as err:
            oom = ("CUDA out of memory" in str(err)) or ("CUDA error: out of memory" in str(err))
            if not oom or step <= min_chunk_size:
                raise
            step = max(min_chunk_size, step // 2)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[WARN] OOM detected. Reduce mask_chunk: {s}-{e} step -> {step}")
    return np.concatenate(out, axis=0)

def pointnext_predict_with_block_mask_fixed(mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks,
                                            model, device, chunk_size=512, pin_mem=True, min_chunk_size=8):
    # GPU上でベクトル化（chunk化）して一括推論
    return _predict_masks_in_chunks(mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks,
                                    model, device, chunk_size=chunk_size, pin_mem=pin_mem,
                                    min_chunk_size=min_chunk_size)

def shap_predict_block_mask(mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks,
                            model, device, chunk_size=512, pin_mem=True, min_chunk_size=8):
    # SHAP の KernelExplainer 用ラッパー（GPU一括版）
    return pointnext_predict_with_block_mask_fixed(
        mask_vectors, blocks, original_point_cloud, baseline_reg_Blocks,
        model, device, chunk_size=chunk_size, pin_mem=pin_mem, min_chunk_size=min_chunk_size
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

# PointSHAP を Region の L1に合わせる（共通 L1 正規化）
def l1_match_point(point_shap: np.ndarray, region_per_pt: np.ndarray, eps: float):
    L1_reg = float(np.sum(np.abs(region_per_pt))) + eps
    L1_pnt = float(np.sum(np.abs(point_shap))) + eps
    scale  = L1_reg / L1_pnt
    return point_shap * scale

# ===============================
# main() 関数
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Downsample SHAP point clouds for ModelNet40 test files using PointNeXt."
    )
    parser.add_argument("--ds_points", type=int, default=500, help="Number of points to downsample each point cloud (default 500)")
    parser.add_argument("--pattern", type=int, default=100, help="Region-SHAP の 近似マスキングパターン数 (default 100)")
    parser.add_argument("--division_region", type=int, default=32, help="Region-SHAP 用 k (default 32)")
    parser.add_argument("--division_point",  type=int, default=32, help="Point-SHAP 用 k (Region の整数倍にして下さい)")
    parser.add_argument("--l1_eps", type=float, default=1e-6, help="PointSHAP の L1正規化用 ε")
    parser.add_argument("--cache_mode", type=str, default="auto", choices=["auto", "save", "load"], help="Hybrid-SHAP キャッシュ: auto=あればload/無ければsave, save=毎回再計算, load=常に読込")
    parser.add_argument("--num_groups", type=int, default=1, help="Number of groups to divide the 40 classes (default 1)")
    parser.add_argument("--group_index", type=int, default=0, help="Index of the group to process (0-indexed, default 0)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml", help="Path to the PointNeXt config file")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth", help="Path to the PointNeXt checkpoint file")
    # GPU推論用
    parser.add_argument("--mask_chunk", type=int, default=512, help="SHAP用マスクの推論チャンクサイズ")
    parser.add_argument("--min_mask_chunk", type=int, default=16, help="自動縮小時の最小チャンクサイズ")
    parser.add_argument("--no_pinmem", action="store_true", help="CPU→GPU転送でpinned memoryを使わない（デフォルトは使う）")

    args = parser.parse_args()
    ds_points = args.ds_points
    num_groups = args.num_groups
    group_index = args.group_index
    pattern = args.pattern
    k_region = args.division_region
    k_point  = args.division_point

    # fan-out 動的判定
    if k_point % k_region != 0:
        sys.exit("--division_point は --division_region の整数倍にして下さい")
    fanout = k_point // k_region
    print(f"[INFO] Region k={k_region}, Point k={k_point}, fan-out={fanout}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットディレクトリ
    DATA_DIR = "/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048"

    # クラス名の読み込み
    class_names_file = os.path.join(DATA_DIR, "shape_names.txt")
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
    output_folder = os.path.join("result", f"L1_kr{k_region}_kp{k_point}_p{pattern}_dsSHAP_PointNeXt_h5", str(ds_points))                    ####出力ファイルパスの修正
    #output_folder = os.path.join("result", f"p{pattern}_dsSHAP_PointNeXt_h5", str(ds_points))
    os.makedirs(output_folder, exist_ok=True)
    # ★ グローバル共通キャッシュ  (ds_points に依存させない)
    BASE_CACHE_DIR = os.path.join("result", "_shap_cache_global")
    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    #   （モデルや重み W が変わる可能性があればサブディレクトリで分離）
    CACHE_DIR = os.path.join(BASE_CACHE_DIR, f"L1_p{pattern}_kr{k_region}_kp{k_point}")    #Region マスキングパターン数によって        ###キャッシュファイルの指定(有:キャッシュ使用, 無：新規計算)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # テストファイルとトレーニングファイルのパス取得
    TRAIN_FILES = provider.getDataFiles(os.path.join(DATA_DIR, "train_files.txt"))
    TEST_FILES = provider.getDataFiles(os.path.join(DATA_DIR, "test_files.txt"))

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

    # モンテカルロシミュレーション用リスト宣言
    all_deltas = []      # 予測誤差 Δ
    all_s      = []      # 標本標準偏差 s
    all_se     = []      # 標準誤差 SE

    # 各テストファイルを処理（結果は同一出力ファイルに追記してマージ）
    for test_file in TEST_FILES:
        print(f"\n----- Processing test file: {test_file} -----")
        data, labels = provider.loadDataFile(test_file)
        num_samples = len(data)
        print(f"Total samples in {os.path.basename(test_file)}: {num_samples}")
        base_name = os.path.basename(test_file) # キャッシュ判定
        name_without_ext  = os.path.splitext(base_name)[0]
        subdir    = os.path.basename(DATA_DIR)
        output_filename = os.path.join(output_folder, subdir, base_name)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        sampled_data_list = []
        sampled_label_list = []

        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}")
            cls = int(labels[i])
            # --------- キャッシュ判定 ----------
            sample_id  = f"{name_without_ext}_{i:05d}_L1_r{k_region}_p{k_point}"
            cache_path  = os.path.join(CACHE_DIR, sample_id + ".npz")
            print(f"[DEBUG] Using cache file: {cache_path}")
            use_cache   = (args.cache_mode in ["auto","load"]) and os.path.exists(cache_path)
            save_cache  = (args.cache_mode in ["auto","save"]) and (not use_cache)
            hybrid_shap = None

            if cls not in selected_classes:
                continue
            gc.collect()
            bg_pc = backgrounds.get(cls, None)
            if bg_pc is None or bg_pc.size == 0:
                print(f"Skipping sample {i} (class {cls}): No background data available.")
                continue
            pc = data[i]
            # ==========================================================
            # 0)  キャッシュヒットなら即読み込み
            # ==========================================================
            if use_cache:
                print(f"  ↳ cache hit: {cache_path}")
                hybrid_shap          = np.load(cache_path)["shap"]
                # 速度計測は 0 扱い
                sample_region_time = sample_point_time = sample_hybrid_time = \
                sample_down_time   = sample_total_time = sample_shap_time   = 0.0

                # --- ダウンサンプリング用インデックス (cache hit 時) ---
                sorted_indices = np.argsort(hybrid_shap)[::-1]
                top_indices    = sorted_indices[:ds_points]
                ds_pc          = pc[top_indices, :]

            # ==========================================================
            # 1)  キャッシュが無い場合はヘビー計算して保存
            # ==========================================================
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
                    preds_A = []
                    def shap_predict_A(mv):
                        out = shap_predict_block_mask(
                            mv, reg_A, pc_A, baseline_reg_A, model, device,
                            chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem), min_chunk_size=args.min_mask_chunk
                        )
                        preds_A.append(out)
                        return out
                    explainer_A = shap.KernelExplainer(shap_predict_A, np.zeros((1,len(reg_A))))
                    rA0 = time.time()
                    with warnings.catch_warnings():                 # ★ 警告を抑制
                        warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                        block_A = (
                            explainer_A.shap_values(np.ones((1, len(reg_A))), nsamples=pattern, l1_reg="num_features(10)")
                            [target_explain_class_index].reshape(-1)
                        )
                        preds_A_np = np.concatenate(preds_A, axis=0)
                        delta_A    = preds_A_np[:, target_explain_class_index] - explainer_A.expected_value[target_explain_class_index]
                        s_A  = float(delta_A.std(ddof=1))
                        M_A  = delta_A.shape[0]
                        SE_A = s_A / np.sqrt(M_A)
                        all_deltas.append(delta_A)
                        all_s.append(s_A)
                        all_se.append(SE_A)
                    rA1 = time.time()
                    pA0 = time.time()
                    block_A_pt = np.repeat(block_A, fanout)
                    pt_shap_A  = compute_point_level_shap_values(
                        pc_A, pt_A, block_A_pt, baseline_pt_A,
                        model, target_explain_class_index,
                        device, scaling_factor=1.0)
                    pA1 = time.time()

                    # --- Region / Point for B ---
                    preds_B = []
                    def shap_predict_B(mv):
                        out = shap_predict_block_mask(
                            mv, reg_B, pc_B, baseline_reg_B, model, device,
                            chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem), min_chunk_size=args.min_mask_chunk
                        )
                        preds_B.append(out)
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
                        preds_B_np = np.concatenate(preds_B, axis=0)
                        delta_B    = preds_B_np[:, target_explain_class_index] - explainer_B.expected_value[target_explain_class_index]
                        s_B  = float(delta_B.std(ddof=1))
                        M_B  = delta_B.shape[0]
                        SE_B = s_B / np.sqrt(M_B)
                        all_deltas.append(delta_B)
                        all_s.append(s_B)
                        all_se.append(SE_B)
                    rB1 = time.time()
                    pB0 = time.time()
                    block_B_pt = np.repeat(block_B, fanout)
                    pt_shap_B  = compute_point_level_shap_values(
                        pc_B, pt_B, block_B_pt, baseline_pt_B,
                        model, target_explain_class_index,
                        device, scaling_factor=1.0)
                    pB1 = time.time()

                    # --- Hybrid ---
                    h0 = time.time()
                    # Region SHAP を子クラスタへブロードキャスト
                    region_A_exp = np.repeat(block_A, fanout)
                    region_A_per_pt = np.zeros_like(pt_shap_A)
                    for b_idx, idxs in enumerate(pt_A):
                        region_A_per_pt[idxs] = region_A_exp[b_idx]
                    region_B_exp = np.repeat(block_B, fanout)
                    region_B_per_pt = np.zeros_like(pt_shap_B)
                    for b_idx, idxs in enumerate(pt_B):
                        region_B_per_pt[idxs] = region_B_exp[b_idx]
                    # PointSHAP を Region と L1整合させてから合成
                    pt_adj_A = l1_match_point(pt_shap_A, region_A_per_pt, args.l1_eps)
                    pt_adj_B = l1_match_point(pt_shap_B, region_B_per_pt, args.l1_eps)
                    hybrid_A = region_A_per_pt + pt_adj_A
                    hybrid_B = region_B_per_pt + pt_adj_B
                    hybrid_shap = np.concatenate([hybrid_A, hybrid_B])
                    h1 = time.time()

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
                    preds_blk = []
                    def shap_predict_blk(mv):
                        out = shap_predict_block_mask(
                            mv, reg_blk, pc, baseline_reg_blk, model, device,
                            chunk_size=args.mask_chunk, pin_mem=(not args.no_pinmem), min_chunk_size=args.min_mask_chunk
                        )
                        preds_blk.append(out)
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
                        preds_blk_np = np.concatenate(preds_blk, axis=0)
                        delta_blk    = preds_blk_np[:, target_explain_class_index] - explainer.expected_value[target_explain_class_index]
                        s_blk  = float(delta_blk.std(ddof=1))
                        M_blk  = delta_blk.shape[0]
                        SE_blk = s_blk / np.sqrt(M_blk)
                        all_deltas.append(delta_blk)
                        all_s.append(s_blk)
                        all_se.append(SE_blk)
                    r1 = time.time()
                    p0 = time.time()
                    block_vals_pt = np.repeat(block_vals, fanout)
                    pt_shap_blk   = compute_point_level_shap_values(
                        pc, pt_blk, block_vals_pt, baseline_pt_blk,
                        model,target_explain_class_index,device,
                        scaling_factor=1.0)
                    p1 = time.time()
                    h0 = time.time()
                    region_blk_exp = np.repeat(block_vals, fanout)
                    region_blk_per_pt = np.zeros_like(pt_shap_blk)
                    for b_idx, idxs in enumerate(pt_blk):
                        region_blk_per_pt[idxs] = region_blk_exp[b_idx]
                    pt_adj_blk = l1_match_point(pt_shap_blk, region_blk_per_pt, args.l1_eps)
                    hybrid_shap = region_blk_per_pt + pt_adj_blk
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
                    np.savez_compressed(cache_path, shap=hybrid_shap)

                # --- ダウンサンプリング用インデックス ---
                sorted_indices = np.argsort(hybrid_shap)[::-1]
                top_indices    = sorted_indices[:ds_points]
                ds_pc          = pc[top_indices, :]

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
            # sampled_label_listを2次元に変換して連結
            new_data = np.concatenate([existing_data, np.array(sampled_data_list)], axis=0)
            new_label = np.concatenate([existing_label, np.array(sampled_label_list).reshape(-1, 1)], axis=0)
            save_h5_data(output_filename, new_data, new_label)
        else:
            save_h5_data(output_filename, np.array(sampled_data_list), np.array(sampled_label_list).reshape(-1, 1))
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

    # ===== Monte-Carlo 統計を Monte_Carlo.txt に保存 =====
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
