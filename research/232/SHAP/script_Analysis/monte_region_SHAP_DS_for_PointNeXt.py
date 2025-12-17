#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import gc
import numpy as np
import argparse
import shap
import csv
from sklearn.cluster import KMeans
import time
import torch
import warnings

N_BOOT = 10        # ブートストラップで Region-SHAP を再計算する回数

# --- PointNeXt 関連 ---
# PointNeXt本体のルート（例：/workspace/PointNeXt）をsys.pathに追加
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg  # モデル構築用関数
from openpoints.utils import EasyConfig  # 設定ファイル読み込み用

# --- provider.py （PointNet時代のもの） ---
# （同じディレクトリまたはPYTHONPATH上に配置しておく前提）
import provider


# ===============================
# 各種関数の定義
# ===============================

def get_class_names(file_path):
    """クラス名リストをファイルから読み込む関数"""
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def spatial_divide_into_blocks(points, num_blocks, kmeans=None):
    """点群を空間的に領域に分割（K-Means）する関数"""
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

def collect_background_point_clouds(target_label, train_files, num_point_clouds=50):
    """
    指定したクラスラベルに属する点群をトレーニングファイルから収集する関数
    ※ 各点群は1024点に統一される前提
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
    # 各クラスタ i (0..k-1) に対し、baseline点群すべてから「クラスタ i に属する点」を集めて平均し、(k, 3) のリストを返す。
    # それぞれ対応するクラスタに点を対応させ、baseline点群を形成させる。

    num_clusters = kmeans.n_clusters
    baseline_blocks = []
    for i in range(num_clusters):
        # 1つの巨大なリストにクラスタ i の点をためる
        cluster_points = []
        for pc in background_point_clouds:           # pc shape (N,3)
            labels = kmeans.predict(pc)              # (N,)
            pts_i  = pc[labels == i]                 # 該当クラスタの点
            if pts_i.size:
                cluster_points.append(pts_i)
        if cluster_points:                           # 点が1つ以上集まった
            cluster_points = np.concatenate(cluster_points, axis=0)
            centroid = cluster_points.mean(axis=0)   # (3,)
        else:                                        # 万一空ならゼロ
            centroid = np.zeros(3)

        baseline_blocks.append(centroid)

    return baseline_blocks

def apply_block_mask_fixed_baseline(point_cloud, blocks, mask_vector, baseline_blocks):
    """
    指定したマスク(mask_vector)に従い、マスクが0の領域の点を背景基準値に置換する
    """
    masked_point_cloud = point_cloud.copy()
    for block_idx, mask in enumerate(mask_vector):
        if mask == 0:
            masked_point_cloud[blocks[block_idx]] = baseline_blocks[block_idx]
    return masked_point_cloud

def _precompute_pointwise_maps(blocks, baseline_blocks):
    """各点の所属 block_id と per-point baseline を前計算（GPU一括適用用）。"""
    N = sum(len(b) for b in blocks)
    block_ids = np.empty(N, dtype=np.int64)
    for b, idxs in enumerate(blocks):
        if len(idxs) == 0: 
            continue
        block_ids[np.asarray(idxs, dtype=np.int64)] = b
    base_pt = np.stack([baseline_blocks[b] for b in block_ids], axis=0).astype(np.float32)  # (N,3)
    return block_ids, base_pt

def _predict_masks_in_chunks(mask_vectors, blocks, original_point_cloud, baseline_blocks,
                            model, device, chunk_size: int = 512, pin_mem: bool = True, min_chunk_size: int = 8):
    """(B,K)マスクをGPU上で(B,N,3)に展開→model()をバッチ推論。順序・I/Fは従来と同じ。"""
    model.eval()
    M = len(mask_vectors)
    if M == 0:
        return np.array([])
    pc = np.asarray(original_point_cloud, dtype=np.float32)  # (N,3)
    block_ids, base_pt = _precompute_pointwise_maps(blocks, baseline_blocks)     # (N,), (N,3)
    pc_t   = torch.from_numpy(pc).to(device).unsqueeze(0)      # (1,N,3)
    base_t = torch.from_numpy(base_pt).to(device).unsqueeze(0) # (1,N,3)
    bid_t  = torch.from_numpy(block_ids).to(device)            # (N,)
    if pin_mem:
        pc_t = pc_t.contiguous(); base_t = base_t.contiguous()
    out = []
    step = int(chunk_size); s = 0
    while s < M:
        e = min(M, s + step)
        mv = np.asarray(mask_vectors[s:e], dtype=np.float32)   # (B,K)
        try:
            if pin_mem and device.type == "cuda":
                mv_t = torch.from_numpy(mv).pin_memory().to(device, non_blocking=True)
            else:
                mv_t = torch.from_numpy(mv).to(device)
            keep  = mv_t[:, bid_t].unsqueeze(-1)               # (B,N,1)
            pcB   = pc_t.expand(keep.shape[0], -1, -1)         # (B,N,3)
            baseB = base_t.expand(keep.shape[0], -1, -1)       # (B,N,3)
            masked = keep * pcB + (1.0 - keep) * baseB         # (B,N,3)
            with torch.no_grad():
                logits = model(masked)                          # (B,C)
            out.append(logits.detach().cpu().numpy())
            del mv_t, keep, pcB, baseB, masked, logits
            s = e
        except RuntimeError as err:
            if ("out of memory" not in str(err).lower()) or (step <= min_chunk_size):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            new_step = max(min_chunk_size, step // 2)
            print(f"[WARN] CUDA OOM: chunk {step} -> {new_step}")
            step = new_step
    return np.concatenate(out, axis=0)

def pointnext_predict_with_block_mask_fixed(
        mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device,
         *, chunk_size=512, pin_mem=True, min_chunk_size=8):
    """
    各マスクパターンに対して、背景基準値で置換した点群の予測を取得する
    GPU一括推論版
    """
    return _predict_masks_in_chunks(
        mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device,
        chunk_size=chunk_size, pin_mem=pin_mem, min_chunk_size=min_chunk_size
    )

def shap_predict_block_mask(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device,
                            *, chunk_size=512, pin_mem=True, min_chunk_size=8):
    """
    SHAPのKernelExplainer用ラッパー関数
    KernelExplainer用：GPU一括推論を呼ぶだけ
    """
    return pointnext_predict_with_block_mask_fixed(
        mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device,
        chunk_size=chunk_size, pin_mem=pin_mem, min_chunk_size=min_chunk_size
    )

# モンテカルロシュミレーションに関係ないPoint-SHAP関連を削除
'''
def compute_point_level_contributions(point_cloud, blocks, baseline_blocks, model, target_class_index, device):
    """
    各点の寄与度を、入力点群の勾配と背景との差の内積から算出する
    """
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
        baseline = baseline_blocks[i]
        for idx in block:
            diff = point_cloud[idx] - baseline
            point_contrib[idx] = np.dot(grad_val[idx], diff)
    return point_contrib

def compute_point_level_shap_values(point_cloud, blocks, block_shap_values, baseline_blocks, model, target_class_index, device, scaling_factor=1.0):
    """
    統合SHAP値を、領域単位のSHAP値と各点の寄与度（Zスコア正規化した値）から算出する
    定義: Hybrid_SHAP = (Region_SHAP) + (scaling_factor × Point_SHAP)
    """
    point_contrib = compute_point_level_contributions(point_cloud, blocks, baseline_blocks, model, target_class_index, device)
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
'''
def _se_from_linear_reg(mask_mat, weights, y, phi):
    """
    最小二乗 φ 推定量の共分散から SE を計算。
    mask_mat : (M,k) 0/1 行列
    weights  : (M,)   KernelSHAP の重み
    y        : (M,)   目的クラスの予測値
    phi      : (k,)   shap_values で得た推定 φ
    """
    # (Xᵀ W X)^−1
    # DenseData → ndarray へ
    if hasattr(mask_mat, "data"):
        mask_mat = mask_mat.data
    # weights が無ければ一様重み
    if weights is None:
        weights = np.ones(mask_mat.shape[0])
    XtW         = mask_mat.T * weights                         # (k,M)
    XtWX        = XtW @ mask_mat                               # (k,k)
    try:
        XtWX_inv = np.linalg.inv(XtWX)                         # 正則なら通常の逆行列
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)                        # 特異なら擬逆で代用
    # 残差分散 σ²
    y_hat       = mask_mat @ phi
    resid       = y - y_hat
    rank_X      = np.linalg.matrix_rank(mask_mat)
    dof         = max(1, weights.sum() - rank_X - 1)           # 自由度を rank ベースに
    sigma2_hat  = (weights * resid**2).sum() / dof
    var_phi     = sigma2_hat * np.diag(XtWX_inv)          # (k,)
    return np.sqrt(var_phi)                               # SE(φᵢ)

# ---------- φ のブートストラップ半幅＆平均 ---------- #
def bootstrap_ci95_and_phi(explainer, mask, nsamples, target_idx, n_boot=N_BOOT, eps=1e-9, sample_idx=None, num_samples=None):
    """
    戻り値:  (half_widths, phi_mean)  ともに shape = (k,)
    """
    phi_stack = []
    for b in range(n_boot):
        # 進捗表示：「現在boot/全体boot  Processing files i/N」
        if (sample_idx is not None) and (num_samples is not None):
            print(f"boot: {b+1}/{n_boot}  Processing files: {sample_idx}/{num_samples}", flush=True)
        else:
            print(f"{b+1}/{n_boot}", flush=True)
        phi = explainer.shap_values(mask,
                                    nsamples=nsamples,
                                    l1_reg="num_features(10)")[target_idx].reshape(-1)
        phi_stack.append(phi)
    phi_mat   = np.vstack(phi_stack)               # (B,k)
    phi_mean  = phi_mat.mean(axis=0)               # (k,)
    se_phi    = phi_mat.std(axis=0, ddof=1)        # (k,)
    hw_phi    = 1.96 * se_phi                     # 半幅
    rel_err   = hw_phi / (np.abs(phi_mean)+eps) * 100.0  # %
    return hw_phi, rel_err, phi_mean

# ===============================
# main() 関数
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Downsample SHAP point clouds for ModelNet40 test files using PointNeXt."
    )
    parser.add_argument("--ds_points", type=int, default=50, help="Number of points to downsample each point cloud (default 500)")
    parser.add_argument("--pattern", type=int, default=100, help="Region-SHAP の 近似マスキングパターン数 (default 100)")
    parser.add_argument("--division", type=int, default=32, help="k-meansによる領域数 (default 32)")
    parser.add_argument("--num_groups", type=int, default=1, help="Number of groups to divide the 40 classes (default 1)")
    parser.add_argument("--group_index", type=int, default=0, help="Index of the group to process (0-indexed, default 0)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml", help="Path to the PointNeXt config file")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth", help="Path to the PointNeXt checkpoint file")
    # GPU推論の調整用オプション
    parser.add_argument("--mask_chunk", type=int, default=512, help="SHAP用マスクの推論チャンクサイズ")
    parser.add_argument("--min_mask_chunk", type=int, default=8, help="自動縮小時の最小チャンクサイズ")
    parser.add_argument("--no_pinmem", action="store_true", help="pinned memory を無効化")

    args = parser.parse_args()
    ds_points = args.ds_points
    num_groups = args.num_groups
    group_index = args.group_index
    pattern = args.pattern
    division = args.division


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

    output_folder = os.path.join("/workspace/PointNeXt/result", f"Monte_k{division}_p{pattern}_regionSHAP")
    os.makedirs(output_folder, exist_ok=True)

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
    all_ci95   = []      # φ の 95%CI 半幅 (1.96×SE)
    all_rel95  = []      # φ 95% 相対誤差 (%)
    all_phi    = []      # φ̂ᵢ のブートストラップ平均値

    # 各テストファイルを処理（結果は同一出力ファイルに追記してマージ）
    for test_file in TEST_FILES:
        print(f"\n----- Processing test file: {test_file} -----")
        data, labels = provider.loadDataFile(test_file)
        num_samples = len(data)
        print(f"Total samples in {os.path.basename(test_file)}: {num_samples}")

        for i in range(num_samples):
            print(f"Processing files {i+1}/{num_samples}")
            cls = int(labels[i])

            if cls not in selected_classes:
                continue
            gc.collect()
            bg_pc = backgrounds.get(cls, None)
            if bg_pc is None or bg_pc.size == 0:
                print(f"Skipping sample {i} (class {cls}): No background data available.")
                continue
            pc = data[i]

            # ======== Region-SHAP の計算（ブートストラップ含む）========
            if pc.shape[0] == 2048:
                # ---------- heavy 計算 (A/B) ----------
                pc_A = pc[:1024];  pc_B = pc[1024:]
                blocks_A, kmeans_A = spatial_divide_into_blocks(pc_A, division)
                blocks_B, kmeans_B = spatial_divide_into_blocks(pc_B, division)
                baseline_A = compute_background_baseline(bg_pc, kmeans_A)
                baseline_B = compute_background_baseline(bg_pc, kmeans_B)
                target_explain_class_index = cls

                t0 = time.time()
                # --- Region / Point for A ---
                preds_A = []
                def shap_predict_A(mv):
                    out = shap_predict_block_mask(
                        mv, blocks_A, pc_A, baseline_A, model, device,
                        chunk_size=args.mask_chunk,
                        pin_mem=(not args.no_pinmem),
                        min_chunk_size=args.min_mask_chunk
                    )
                    preds_A.append(out)
                    return out
                explainer_A = shap.KernelExplainer(shap_predict_A, np.zeros((1,len(blocks_A))))
                rA0 = time.time()
                with warnings.catch_warnings():                 # ★ 警告を抑制
                    warnings.filterwarnings("ignore", message=".*active set degenerate.*")
                    block_A = (
                        explainer_A.shap_values(np.ones((1, len(blocks_A))), nsamples=pattern, l1_reg="num_features(10)")
                        [target_explain_class_index].reshape(-1)
                    )
                    # --- φ のブートストラップ CI & 相対誤差 ---
                    hw_A, rel_A, phi_A = bootstrap_ci95_and_phi(
                        explainer_A,
                        np.ones((1, len(blocks_A))),
                        nsamples=pattern,
                        target_idx=target_explain_class_index,
                        sample_idx=i+1, num_samples=num_samples)
                    all_ci95.extend(hw_A)
                    all_rel95.extend(rel_A)
                    all_phi.extend(phi_A) # φ̂ᵢ を収集

                    preds_A_np = np.concatenate(preds_A, axis=0)
                    delta_A    = preds_A_np[:, target_explain_class_index] - explainer_A.expected_value[target_explain_class_index]
                    s_A  = float(delta_A.std(ddof=1))
                    M_A  = delta_A.shape[0]
                    SE_A = s_A / np.sqrt(M_A)
                    all_deltas.append(delta_A)
                    all_s.append(s_A)
                    all_se.append(SE_A)
                rA1 = time.time()
                pA0 = pA1 = rA1

                # --- Region / Point for B ---
                preds_B = []
                def shap_predict_B(mv):
                    out = shap_predict_block_mask(
                        mv, blocks_B, pc_B, baseline_B, model, device,
                        chunk_size=args.mask_chunk,
                        pin_mem=(not args.no_pinmem),
                        min_chunk_size=args.min_mask_chunk
                    )
                    preds_B.append(out)
                    return out
                explainer_B = shap.KernelExplainer(shap_predict_B, np.zeros((1,len(blocks_B))))
                rB0 = time.time()
                with warnings.catch_warnings():                 # ★ 同様に抑制
                    warnings.filterwarnings("ignore",
                                            message=".*active set degenerate.*")
                    block_B = (
                        explainer_B.shap_values(np.ones((1, len(blocks_B))), nsamples=pattern, l1_reg="num_features(10)")
                        [target_explain_class_index].reshape(-1)
                    )
                    preds_B_np = np.concatenate(preds_B, axis=0)
                    delta_B    = preds_B_np[:, target_explain_class_index] - explainer_B.expected_value[target_explain_class_index]
                    hw_B, rel_B, phi_B = bootstrap_ci95_and_phi(
                        explainer_B,
                        np.ones((1, len(blocks_B))),
                        nsamples=pattern,
                        target_idx=target_explain_class_index,
                        sample_idx=i+1, num_samples=num_samples)
                    all_ci95.extend(hw_B)
                    all_rel95.extend(rel_B)
                    all_phi.extend(phi_B)
                    s_B  = float(delta_B.std(ddof=1))
                    M_B  = delta_B.shape[0]
                    SE_B = s_B / np.sqrt(M_B)
                    all_deltas.append(delta_B)
                    all_s.append(s_B)
                    all_se.append(SE_B)
                rB1 = time.time()
                pB0 = pB1 = rB1

                # --- Hybrid ---(作成なし)
                h0 = h1 = pB1

                # 時間まとめ
                sample_region_time = (rA1-rA0) + (rB1-rB0)
                sample_point_time  = 0.0
                sample_hybrid_time = 0.0
                t_final            = time.time()
                sample_total_time  = t_final - t0
                sample_down_time   = sample_total_time - sample_region_time
                sample_shap_time   = sample_region_time

            else:
                # ---------- heavy 計算 (≤1024) ----------
                if pc.shape[0] > 1024:
                    pc = pc[np.random.choice(pc.shape[0],1024,replace=False)]
                elif pc.shape[0] < 1024:
                    pc = np.pad(pc,((0,1024-pc.shape[0]),(0,0)),'constant')

                blocks,kmeans = spatial_divide_into_blocks(pc, division)
                baseline_blk  = compute_background_baseline(bg_pc, kmeans)
                target_explain_class_index = cls

                t0 = time.time()
                preds_blk = []
                def shap_predict_blk(mv):
                    out = shap_predict_block_mask(
                        mv, blocks, pc, baseline_blk, model, device,
                        chunk_size=args.mask_chunk,
                        pin_mem=(not args.no_pinmem),
                        min_chunk_size=args.min_mask_chunk
                    )
                    preds_blk.append(out)
                    return out
                explainer = shap.KernelExplainer(shap_predict_blk,
                                                        np.zeros((1,len(blocks))))
                r0 = time.time()
                with warnings.catch_warnings():                 # ★
                    warnings.filterwarnings("ignore",
                                            message=".*active set degenerate.*")
                    block_vals = (
                        explainer.shap_values(np.ones((1, len(blocks))), nsamples=pattern, l1_reg="num_features(10)")
                        [target_explain_class_index].reshape(-1)
                    )
                    preds_blk_np = np.concatenate(preds_blk, axis=0)
                    delta_blk    = preds_blk_np[:, target_explain_class_index] - explainer.expected_value[target_explain_class_index]
                    hw_blk, rel_blk, phi_blk = bootstrap_ci95_and_phi(
                        explainer,
                        np.ones((1, len(blocks))),
                        nsamples=pattern,
                        target_idx=target_explain_class_index,
                        sample_idx=i+1, num_samples=num_samples)
                    all_ci95.extend(hw_blk)
                    all_rel95.extend(rel_blk)
                    all_phi.extend(phi_blk)
                    s_blk  = float(delta_blk.std(ddof=1))
                    M_blk  = delta_blk.shape[0]
                    SE_blk = s_blk / np.sqrt(M_blk)
                    all_deltas.append(delta_blk)
                    all_s.append(s_blk)
                    all_se.append(SE_blk)
                r1 = time.time()
                # Point/Hybrid は計算しない
                p0 = p1 = r1
                h0 = h1 = p1
                sample_region_time = r1-r0
                sample_point_time  = 0.0
                sample_hybrid_time = 0.0
                t_final            = time.time()
                sample_total_time  = t_final - t0
                sample_down_time   = sample_total_time - sample_region_time
                sample_shap_time   = sample_region_time

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

            print(f"Processed sample {i+1}/{num_samples}")

    # 最終的な処理時間の統計を出力
    time_output = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(time_output, "w") as time_f:
        time_f.write("Class\tDS_Points\tAvg_Region(sec)\tAvg_Point(sec)\tAvg_Hybrid(sec)\tAvg_Down(sec)\tAvg_Total(sec)\tR_region(%)\tR_point(%)\tR_hybrid(%)\tSample_Count\n")
        for cls in sorted(class_time_stats.keys()):
            stats = class_time_stats[cls]
            if stats['count'] > 0:
                avg_region = stats['sum_region'] / stats['count']
                avg_point  = 0.0
                avg_hybrid = 0.0
                avg_down   = stats['sum_down']   / stats['count']
                avg_total  = stats['sum_total']  / stats['count']
                if avg_total == 0:
                    r_region = r_point = r_hybrid = 0.0
                else:
                    r_region = 100 * avg_region / avg_total
                    r_point  = 0.0
                    r_hybrid = 0.0
                time_f.write(f"{class_names[cls]}\t{ds_points}\t"
                            f"{avg_region:.6f}\t{avg_point:.6f}\t{avg_hybrid:.6f}\t"
                            f"{avg_down:.6f}\t{avg_total:.6f}\t"
                            f"{r_region:.2f}\t{r_point:.2f}\t{r_hybrid:.2f}\t"
                            f"{stats['count']}\n")
        if overall_count > 0:
            overall_avg_shap = overall_shap / overall_count
            overall_avg_region = overall_region / overall_count
            overall_avg_point  = 0.0
            overall_avg_hybrid = 0.0
            overall_avg_down   = overall_down   / overall_count
            overall_avg_total  = overall_total  / overall_count
            if overall_avg_total == 0:
                Rr = Rp = Rh = 0.0
            else:
                Rr = 100 * overall_avg_region / overall_avg_total
                Rp = 0.0
                Rh = 0.0
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
            # …既存の書き込み処理…
            if all_ci95:
                f_mc.write("CI95_HW\t{:.6f}\t{:.6f}\t{:.6f}\n".format(np.mean(all_ci95), np.min(all_ci95), np.max(all_ci95)))
            if all_rel95:
                f_mc.write("CI95_REL%\t{:.2f}\t{:.2f}\t{:.2f}\n".format(np.mean(all_rel95), np.min(all_rel95), np.max(all_rel95)))
        else:
            f_mc.write("No Monte-Carlo data were collected.\n")
    print(f"Monte-Carlo statistics saved to {monte_output}")

    # 詳細 φ, CI, 相対誤差 を CSV に出力
    csv_output = os.path.join(output_folder, "shap_phi_ci_rel.csv")
    with open(csv_output, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["phi_mean","CI95_HW","CI95_REL%"])
        for phi_val, hw, rel in zip(all_phi, all_ci95, all_rel95):
            writer.writerow([phi_val, hw, rel])
    print(f"Detailed SHAP CSV saved to {csv_output}")

if __name__ == "__main__":
    main()