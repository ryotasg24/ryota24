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

def spatial_divide_into_blocks(points, num_blocks=32, kmeans=None):
    """点群を空間的にブロックに分割（K-Means）する関数"""
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

def compute_background_baseline(background_point_clouds, blocks, kmeans):
    """
    各ブロックごとに背景点群の平均値を計算し、基準値として返す
    """
    baseline_blocks = []
    for block in blocks:
        block_points = background_point_clouds[:, block, :]  # (num_bg, block_size, 3)
        block_means = block_points.mean(axis=1)
        overall_block_mean = block_means.mean(axis=0)
        baseline_blocks.append(overall_block_mean)
    return baseline_blocks

def apply_block_mask_fixed_baseline(point_cloud, blocks, mask_vector, baseline_blocks):
    """
    指定したマスク(mask_vector)に従い、マスクが0のブロックの点を背景基準値に置換する
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
    各点のSHAP値を、ブロック単位のSHAP値と各点の寄与度（Zスコア正規化した値）から算出する
    定義：SHAP値 = (ブロックのSHAP値) + (scaling_factor × 正規化された寄与度)
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
        description="Downsample SHAP point clouds for ModelNet40 test files using PointNeXt, " +
                    "preserving the original file order from the input h5 file."
    )
    parser.add_argument("--ds_points", type=int, default=500,
                        help="Number of points to downsample each point cloud (default 500)")
    parser.add_argument("--num_groups", type=int, default=40,
                        help="Number of groups to divide the 40 classes (default 40)")
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
    output_folder = os.path.join("result", "nonSort_SHAP", str(ds_points))
    os.makedirs(output_folder, exist_ok=True)

    # テストファイルとトレーニングファイルのパス取得
    TRAIN_FILES = provider.getDataFiles(os.path.join(DATA_DIR, "train_files.txt"))
    # TEST_FILES は入力h5ファイルの順番を保持するため、sortedでソートする（文字列順＝入力時順とする）
    TEST_FILES = sorted(provider.getDataFiles(os.path.join(DATA_DIR, "test_files.txt")))

    # 選択された各クラスの背景点群を収集
    backgrounds = {}
    for cls in selected_classes:
        print(f"Collecting background point clouds for class '{class_names[cls]}' (class index {cls})...")
        bg = collect_background_point_clouds(cls, TRAIN_FILES, num_point_clouds=50)
        backgrounds[cls] = bg

    # PointNeXtモデルのロード
    model, cfg = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)

    # 各クラスごとの処理時間を集計する辞書
    class_time_stats = {cls: {'sum_total': 0.0, 'sum_down': 0.0, 'sum_shap': 0.0, 'count': 0} for cls in selected_classes}
    overall_total = 0.0
    overall_down = 0.0
    overall_shap = 0.0
    overall_count = 0

    # 各テストファイルを処理（処理順はTEST_FILESの並び順＝入力h5ファイルの順番に準ずる）
    for test_file in TEST_FILES:
        print(f"\n----- Processing test file: {test_file} -----")
        data, labels = provider.loadDataFile(test_file)
        num_samples = len(data)
        print(f"Total samples in {os.path.basename(test_file)}: {num_samples}")
        base_name = os.path.basename(test_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_filename = os.path.join(output_folder, f"{name_without_ext}.h5")

        sampled_data_list = []
        sampled_label_list = []
        # サンプルは入力ファイルの順番 (0から順) でそのまま処理する
        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}")
            cls = int(labels[i])
            # 選択されていないクラスはスキップ（出力h5には含めない）
            if cls not in selected_classes:
                continue
            gc.collect()
            bg_pc = backgrounds.get(cls, None)
            if bg_pc is None or bg_pc.size == 0:
                print(f"Skipping sample {i} (class {cls}): No background data available.")
                continue
            pc = data[i]
            # 以下は既存の処理（点群が2048点の場合は2グループに分割、そうでなければそのまま処理）
            if pc.shape[0] == 2048:
                pc_A = pc[:1024, :]
                pc_B = pc[1024:, :]
                blocks_A, kmeans_A = spatial_divide_into_blocks(pc_A, num_blocks=32)
                baseline_A = compute_background_baseline(bg_pc, blocks_A, kmeans_A)
                blocks_B, kmeans_B = spatial_divide_into_blocks(pc_B, num_blocks=32)
                baseline_B = compute_background_baseline(bg_pc, blocks_B, kmeans_B)
                target_explain_class_index = cls

                t0 = time.time()
                def shap_predict_block_mask_wrapper_A(mask_vectors):
                    return shap_predict_block_mask(mask_vectors, blocks_A, pc_A, baseline_A, model, device)
                explainer_A = shap.KernelExplainer(shap_predict_block_mask_wrapper_A, np.zeros((1, len(blocks_A))))
                block_shap_values_A = explainer_A.shap_values(np.ones((1, len(blocks_A))), nsamples=100)
                block_shap_values_A = block_shap_values_A[target_explain_class_index].reshape(-1)
                point_shap_vals_A = compute_point_level_shap_values(pc_A, blocks_A, block_shap_values_A, baseline_A,
                                                                    model, target_explain_class_index, device, scaling_factor=1.0)

                def shap_predict_block_mask_wrapper_B(mask_vectors):
                    return shap_predict_block_mask(mask_vectors, blocks_B, pc_B, baseline_B, model, device)
                explainer_B = shap.KernelExplainer(shap_predict_block_mask_wrapper_B, np.zeros((1, len(blocks_B))))
                block_shap_values_B = explainer_B.shap_values(np.ones((1, len(blocks_B))), nsamples=100)
                block_shap_values_B = block_shap_values_B[target_explain_class_index].reshape(-1)
                point_shap_vals_B = compute_point_level_shap_values(pc_B, blocks_B, block_shap_values_B, baseline_B,
                                                                    model, target_explain_class_index, device, scaling_factor=1.0)
                t_intermediate = time.time()
                mu_A, sigma_A = np.mean(point_shap_vals_A), np.std(point_shap_vals_A)
                mu_B, sigma_B = np.mean(point_shap_vals_B), np.std(point_shap_vals_B)
                norm_A = (point_shap_vals_A - mu_A) / sigma_A if sigma_A >= 1e-6 else np.zeros_like(point_shap_vals_A)
                norm_B = (point_shap_vals_B - mu_B) / sigma_B if sigma_B >= 1e-6 else np.zeros_like(point_shap_vals_B)
                combined_shap = np.concatenate([norm_A, norm_B])
                sorted_indices = np.argsort(combined_shap)[::-1]
                top_indices = sorted_indices[:ds_points]
                ds_pc = pc[top_indices, :]
                t_final = time.time()

                sample_total_time = t_final - t0
                sample_down_time = t_final - t_intermediate
                sample_shap_time = t_intermediate - t0
            else:
                if pc.shape[0] > 1024:
                    sampled_indices = np.random.choice(pc.shape[0], 1024, replace=False)
                    pc = pc[sampled_indices, :]
                elif pc.shape[0] < 1024:
                    padding = 1024 - pc.shape[0]
                    pc = np.pad(pc, ((0, padding), (0, 0)), mode="constant")
                blocks, kmeans = spatial_divide_into_blocks(pc, num_blocks=32)
                baseline_blocks = compute_background_baseline(bg_pc, blocks, kmeans)
                target_explain_class_index = cls

                t0 = time.time()
                def shap_predict_block_mask_wrapper(mask_vectors):
                    return shap_predict_block_mask(mask_vectors, blocks, pc, baseline_blocks, model, device)
                explainer = shap.KernelExplainer(shap_predict_block_mask_wrapper, np.zeros((1, len(blocks))))
                block_shap_values = explainer.shap_values(np.ones((1, len(blocks))), nsamples=100)
                block_shap_values = block_shap_values[target_explain_class_index].reshape(-1)
                point_shap_vals = compute_point_level_shap_values(pc, blocks, block_shap_values, baseline_blocks,
                                                                  model, target_explain_class_index, device, scaling_factor=1.0)
                t_intermediate = time.time()
                sorted_indices = np.argsort(point_shap_vals)[::-1]
                top_indices = sorted_indices[:ds_points]
                ds_pc = pc[top_indices, :]
                t_final = time.time()

                sample_total_time = t_final - t0
                sample_down_time = t_final - t_intermediate
                sample_shap_time = t_intermediate - t0

            # 集計
            class_time_stats[cls]['sum_total'] += sample_total_time
            class_time_stats[cls]['sum_down'] += sample_down_time
            class_time_stats[cls]['sum_shap'] += sample_shap_time
            class_time_stats[cls]['count'] += 1
            overall_total += sample_total_time
            overall_down += sample_down_time
            overall_shap += sample_shap_time
            overall_count += 1

            # append時、forループのiの順番で追加するので、入力h5内の順番が保持される
            sampled_data_list.append(ds_pc.astype('float32'))
            sampled_label_list.append(cls)
            print(f"Processed sample {i+1}/{num_samples}: final downsampled points {ds_pc.shape[0]}")
        print(f"Finished processing {len(sampled_data_list)} samples from {test_file}.")

        # ファイルマージ：既存ファイルがあれば読み込み、連結して保存する
        if os.path.exists(output_filename):
            with h5py.File(output_filename, "r") as f:
                existing_data = f["data"][:]
                existing_label = f["label"][:]
            new_data = np.concatenate([existing_data, np.array(sampled_data_list)], axis=0)
            new_label = np.concatenate([existing_label, np.array(sampled_label_list).reshape(-1, 1)], axis=0)
            save_h5_data(output_filename, new_data, new_label)
        else:
            save_h5_data(output_filename, np.array(sampled_data_list), np.array(sampled_label_list).reshape(-1, 1))
        print(f"Saved integrated downsampled SHAP point clouds for {test_file} to {output_filename}")

    # 最終的な処理時間の統計を出力
    time_output = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(time_output, "w") as time_f:
        time_f.write("Class\tDS_Points\tAvg_SHAP_Time(sec)\tAvg_Downsampling_Time(sec)\tAvg_Total_Time(sec)\tSample_Count\n")
        for cls in sorted(class_time_stats.keys()):
            stats = class_time_stats[cls]
            if stats['count'] > 0:
                avg_shap = stats['sum_shap'] / stats['count']
                avg_down = stats['sum_down'] / stats['count']
                avg_total = stats['sum_total'] / stats['count']
                time_f.write(f"{class_names[cls]}\t{ds_points}\t{avg_shap:.6f}\t{avg_down:.6f}\t{avg_total:.6f}\t{stats['count']}\n")
        if overall_count > 0:
            overall_avg_shap = overall_shap / overall_count
            overall_avg_down = overall_down / overall_count
            overall_avg_total = overall_total / overall_count
            time_f.write(f"ALL\t{ds_points}\t{overall_avg_shap:.6f}\t{overall_avg_down:.6f}\t{overall_avg_total:.6f}\t{overall_count}\n")
    print(f"Processing times by class saved to {time_output}")

if __name__ == "__main__":
    main()

