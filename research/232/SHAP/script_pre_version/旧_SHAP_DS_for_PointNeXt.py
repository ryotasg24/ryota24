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

"""
def compute_background_baseline(background_point_clouds, blocks, kmeans):

    # 各領域ごとに背景点群の平均値を計算し、基準値として返す

    baseline_blocks = []
    for block in blocks:
        block_points = background_point_clouds[:, block, :]  # (num_bg, block_size, 3)
        block_means = block_points.mean(axis=1)
        overall_block_mean = block_means.mean(axis=0)
        baseline_blocks.append(overall_block_mean)
    return baseline_blocks

# ★★★"compute_background_baseline"に関して、上記→下記に移植した。
"""

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

def pointnext_predict_with_block_mask_fixed(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device):
    """
    各マスクパターンに対して、背景基準値で置換した点群の予測を取得する
    """
    model.eval()
    predictions = []
    for mask_vector in mask_vectors:
        masked_point_cloud = apply_block_mask_fixed_baseline(original_point_cloud, blocks, mask_vector, baseline_blocks)
        pc_tensor = torch.tensor(masked_point_cloud, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred_val = model(pc_tensor)
        predictions.append(pred_val.cpu().numpy())
    if len(predictions) == 0:
        return np.array([])
    return np.concatenate(predictions, axis=0)

def shap_predict_block_mask(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device):
    """
    SHAPのKernelExplainer用ラッパー関数
    """
    predictions = pointnext_predict_with_block_mask_fixed(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device)
    return predictions

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

# ===============================
# main() 関数
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Downsample SHAP point clouds for ModelNet40 test files using PointNeXt."
    )
    parser.add_argument("--ds_points", type=int, default=500,
                        help="Number of points to downsample each point cloud (default 500)")
    parser.add_argument("--weight", type=float, default=1.0,
                        help="Hybrid-SHAP の Point-SHAP 部分への重み (default 1.0)")
    parser.add_argument("--pattern", type=int, default=100,
                        help="Region-SHAP の 近似マスキングパターン数 (default 100)")
    parser.add_argument("--division", type=int, default=32,
                        help="k-meansによる領域数 (default 32)")
    parser.add_argument("--num_groups", type=int, default=1,
                        help="Number of groups to divide the 40 classes (default 1)")
    parser.add_argument("--group_index", type=int, default=0,
                        help="Index of the group to process (0-indexed, default 0)")
    parser.add_argument("--cfg_file", type=str,
                        default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml",
                        help="Path to the PointNeXt config file")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth",
                        help="Path to the PointNeXt checkpoint file")
    args = parser.parse_args()
    ds_points = args.ds_points
    num_groups = args.num_groups
    group_index = args.group_index
    weight = args.weight
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

    # 出力先フォルダ（ds_points毎に出力ファイルは同一フォルダ内でマージ）
    #output_folder = os.path.join("result", f"w{weight}_dsSHAP_PointNeXt_h5", str(ds_points))
    output_folder = os.path.join("result", f"Monte Carlo_dsSHAP_PointNeXt_h5", str(ds_points))                                            ####出力ファイルパスの修正
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

    # モンテカルロシミュレーションの集計用リスト
    all_deltas = []      # 予測誤差 Δ の母集合
    all_s      = []      # 標本標準偏差 s   を見たいとき
    all_se     = []      # 標準誤差 SE      を見たいとき

    # 各テストファイルを処理（結果は同一出力ファイルに追記してマージ）
    for test_file in TEST_FILES:
        print(f"\n----- Processing test file: {test_file} -----")
        data, labels = provider.loadDataFile(test_file)
        num_samples = len(data)
        print(f"Total samples in {os.path.basename(test_file)}: {num_samples}")
        base_name = os.path.basename(test_file)
        subdir    = os.path.basename(DATA_DIR)
        output_filename = os.path.join(output_folder, subdir, base_name)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        sampled_data_list = []
        sampled_label_list = []

        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}")
            cls = int(labels[i])
            if cls not in selected_classes:
                continue
            gc.collect()
            bg_pc = backgrounds.get(cls, None)
            if bg_pc is None or bg_pc.size == 0:
                print(f"Skipping sample {i} (class {cls}): No background data available.")
                continue
            pc = data[i]
            # もし入力点群が2048点なら、2グループに分割して処理する
            if pc.shape[0] == 2048:
                pc_A = pc[:1024, :]
                pc_B = pc[1024:, :]
                blocks_A, kmeans_A = spatial_divide_into_blocks(pc_A, num_blocks=division)
                baseline_A = compute_background_baseline(bg_pc, kmeans_A)
                blocks_B, kmeans_B = spatial_divide_into_blocks(pc_B, num_blocks=division)
                baseline_B = compute_background_baseline(bg_pc, kmeans_B)
                target_explain_class_index = cls

                t0 = time.time()
                region_A0 = time.time()
                preds_A = [] #### モンテカルロシミュレーション用
                def shap_predict_block_mask_wrapper_A(mask_vectors):
                    return shap_predict_block_mask(mask_vectors, blocks_A, pc_A, baseline_A, model, device)
                def shap_predict_block_mask_wrapper_A(mask_vectors):    #### モンテカルロシミュレーション用
                    out = shap_predict_block_mask(mask_vectors, blocks_A, pc_A, baseline_A, model, device)
                    preds_A.append(out)
                    return out  #### モンテカルロシミュレーション用
                explainer_A = shap.KernelExplainer(shap_predict_block_mask_wrapper_A, np.zeros((1, len(blocks_A))))
                #block_shap_values_A = explainer_A.shap_values(np.ones((1, len(blocks_A))), nsamples=100)  #マスキングパターン
                block_shap_values_A = explainer_A.shap_values(np.ones((1, len(blocks_A))), nsamples=pattern)  #### モンテカルロシミュレーション用
                block_shap_values_A = block_shap_values_A[target_explain_class_index].reshape(-1)
                #### モンテカルロシミュレーションにて予測誤差を算出
                preds_A_np = np.concatenate(preds_A, axis=0)   # shape (M, num_class)
                delta_A = preds_A_np[:, target_explain_class_index]  - explainer_A.expected_value[target_explain_class_index]
                s_A  = float(delta_A.std(ddof=1))
                M_A  = delta_A.shape[0]
                SE_A = s_A / np.sqrt(M_A)
                print(f"[DEBUG] Region-A  s={s_A:.4f}, SE={SE_A:.4f}  (M={M_A})")
                all_deltas.append(delta_A)
                all_s.append(s_A);   all_se.append(SE_A)
                ####
                region_A1 = time.time()
                point_A0 = time.time()
                point_shap_vals_A = compute_point_level_shap_values(pc_A, blocks_A, block_shap_values_A, baseline_A,
                                                                    model, target_explain_class_index, device, scaling_factor=weight)
                point_A1 = time.time()

                region_B0 = time.time()
                preds_B = [] #### モンテカルロシミュレーション用
                def shap_predict_block_mask_wrapper_B(mask_vectors):
                    return shap_predict_block_mask(mask_vectors, blocks_B, pc_B, baseline_B, model, device)
                def shap_predict_block_mask_wrapper_B(mask_vectors):    #### モンテカルロシミュレーション用
                    out = shap_predict_block_mask(mask_vectors, blocks_B, pc_B, baseline_B, model, device)
                    preds_B.append(out)
                    return out  #### モンテカルロシミュレーション用
                explainer_B = shap.KernelExplainer(shap_predict_block_mask_wrapper_B, np.zeros((1, len(blocks_B))))
                #block_shap_values_B = explainer_B.shap_values(np.ones((1, len(blocks_B))), nsamples=100)  #マスキングパターン
                block_shap_values_B = explainer_B.shap_values(np.ones((1, len(blocks_B))), nsamples=pattern)  #### モンテカルロシミュレーション用
                block_shap_values_B = block_shap_values_B[target_explain_class_index].reshape(-1)
                #### モンテカルロシミュレーションにて予測誤差を算出
                preds_B_np = np.concatenate(preds_B, axis=0)   # shape (M, num_class)
                delta_B = preds_B_np[:, target_explain_class_index]  - explainer_B.expected_value[target_explain_class_index]
                s_B  = float(delta_B.std(ddof=1))
                M_B  = delta_B.shape[0]
                SE_B = s_B / np.sqrt(M_B)
                print(f"[DEBUG] Region-B  s={s_B:.4f}, SE={SE_B:.4f}  (M={M_B})")
                all_deltas.append(delta_B)
                all_s.append(s_B);   all_se.append(SE_B)
                ####
                region_B1 = time.time()
                point_B0 = time.time()
                point_shap_vals_B = compute_point_level_shap_values(pc_B, blocks_B, block_shap_values_B, baseline_B,
                                                                    model, target_explain_class_index, device, scaling_factor=weight)
                point_B1 = time.time()
                hybrid_t0 = time.time()
                mu_A, sigma_A = np.mean(point_shap_vals_A), np.std(point_shap_vals_A)
                mu_B, sigma_B = np.mean(point_shap_vals_B), np.std(point_shap_vals_B)
                norm_A = (point_shap_vals_A - mu_A) / sigma_A if sigma_A >= 1e-6 else np.zeros_like(point_shap_vals_A)
                norm_B = (point_shap_vals_B - mu_B) / sigma_B if sigma_B >= 1e-6 else np.zeros_like(point_shap_vals_B)
                combined_shap = np.concatenate([norm_A, norm_B])
                sorted_indices = np.argsort(combined_shap)[::-1]
                top_indices = sorted_indices[:ds_points]
                ds_pc = pc[top_indices, :]
                hybrid_t1 = time.time()
                t_final = time.time()

                sample_total_time  = t_final  - t0
                sample_region_time = (region_A1 - region_A0) + (region_B1 - region_B0)
                sample_point_time  = (point_A1 - point_A0) + (point_B1 - point_B0)
                sample_hybrid_time = hybrid_t1 - hybrid_t0
                sample_down_time   = sample_total_time - (sample_region_time + sample_point_time  + sample_hybrid_time)
                sample_shap_time   = sample_region_time + sample_point_time

            else:   # もし入力点群が1024点以下なら、グループ分けせずに処理する
                if pc.shape[0] > 1024:
                    sampled_indices = np.random.choice(pc.shape[0], 1024, replace=False)
                    pc = pc[sampled_indices, :]
                elif pc.shape[0] < 1024:
                    padding = 1024 - pc.shape[0]
                    pc = np.pad(pc, ((0, padding), (0, 0)), mode="constant")
                blocks, kmeans = spatial_divide_into_blocks(pc, num_blocks=division)
                baseline_blocks = compute_background_baseline(bg_pc, kmeans)
                target_explain_class_index = cls

                t0 = time.time()
                preds_blk = [] #### モンテカルロシミュレーション用
                def shap_predict_block_mask_wrapper(mask_vectors):
                    return shap_predict_block_mask(mask_vectors, blocks, pc, baseline_blocks, model, device)
                def shap_predict_block_mask_wrapper(mask_vectors):  #### モンテカルロシミュレーション用
                    out = shap_predict_block_mask(mask_vectors, blocks, pc, baseline_blocks, model, device)
                    preds_blk.append(out)           # ← 追加
                    return out  #### モンテカルロシミュレーション用
                explainer = shap.KernelExplainer(shap_predict_block_mask_wrapper, np.zeros((1, len(blocks))))
                region_t0 = time.time()
                #block_shap_values = explainer.shap_values(np.ones((1, len(blocks))), nsamples=100)  #マスキングパターン
                block_shap_values = explainer.shap_values(np.ones((1, len(blocks))), nsamples=pattern)    #### モンテカルロシミュレーション用
                block_shap_values = block_shap_values[target_explain_class_index].reshape(-1)
                #### モンテカルロシミュレーションにて予測誤差を算出
                preds_blk_np = np.concatenate(preds_blk, axis=0)        # shape (M, num_class)
                delta_blk = preds_blk_np[:, target_explain_class_index] - explainer.expected_value[target_explain_class_index]
                s_blk  = float(delta_blk.std(ddof=1))                    # 標本標準偏差
                M_blk  = delta_blk.shape[0]                              # マスク数
                SE_blk = s_blk / np.sqrt(M_blk)                          # 標準誤差
                print(f"[DEBUG] Region-Blocks s={s_blk:.4f}, SE={SE_blk:.4f}  (M={M_blk})")
                all_deltas.append(delta_blk)
                all_s.append(s_blk); all_se.append(SE_blk)
                ####
                region_t1 = time.time()
                point_t0 = time.time()
                point_shap_vals = compute_point_level_shap_values(pc, blocks, block_shap_values, baseline_blocks,
                                                                    model, target_explain_class_index, device, scaling_factor=weight)
                point_t1 = time.time()
                hybrid_t0 = time.time()
                sorted_indices = np.argsort(point_shap_vals)[::-1]
                top_indices = sorted_indices[:ds_points]
                ds_pc = pc[top_indices, :]
                hybrid_t1 = time.time()
                t_final = time.time()

                sample_total_time  = t_final  - t0
                sample_region_time = region_t1 - region_t0
                sample_point_time  = point_t1  - point_t0
                sample_hybrid_time = hybrid_t1 - hybrid_t0
                sample_down_time   = sample_total_time - (sample_region_time + sample_point_time  + sample_hybrid_time)
                sample_shap_time   = sample_region_time + sample_point_time

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
                r_region  = 100 * avg_region / avg_total
                r_point   = 100 * avg_point  / avg_total
                r_hybrid  = 100 * avg_hybrid / avg_total
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
            Rr = 100*overall_avg_region/overall_avg_total
            Rp = 100*overall_avg_point /overall_avg_total
            Rh = 100*overall_avg_hybrid/overall_avg_total
            time_f.write(f"ALL\t{ds_points}\t"
                        f"{overall_avg_region:.6f}\t{overall_avg_point:.6f}\t{overall_avg_hybrid:.6f}\t"
                        f"{overall_avg_down:.6f}\t{overall_avg_total:.6f}\t"
                        f"{Rr:.2f}\t{Rp:.2f}\t{Rh:.2f}\t"
                        f"{overall_count}\n")
    print(f"Processing times by class saved to {time_output}")

    # モンテカルロシミュレーションの集計結果表示
    if all_deltas:                          # 空でなければ
        concat = np.concatenate(all_deltas)
        mean_Δ = concat.mean()
        min_Δ  = concat.min()
        max_Δ  = concat.max()
        print("\n[Monte-Carlo Δ]  mean = {:.6f},  min = {:.6f},  max = {:.6f}"
            .format(mean_Δ, min_Δ, max_Δ))

        # もし s / SE も見たい場合
        if all_s:
            print("[Sample-std  s] mean = {:.6f},  min = {:.6f},  max = {:.6f}"
                .format(np.mean(all_s), np.min(all_s), np.max(all_s)))
        if all_se:
            print("[Std-error SE] mean = {:.6f},  min = {:.6f},  max = {:.6f}"
                .format(np.mean(all_se), np.min(all_se), np.max(all_se)))
    else:
        print("\n[Monte-Carlo]  Δ 集計対象がありません。")


if __name__ == "__main__":
    main()
