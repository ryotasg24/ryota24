"""
for num in 50 75 100 250 500 750 1000; do python conventional_samplings.py --method RS --num_samples $num; done
"""



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import argparse
import h5py
import numpy as np
import time
import torch
import torch.nn.functional as F

# 可視化用ライブラリ（今回は出力には使用しません）
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# /workspace/PointNeXt を検索パスの先頭に追加
sys.path.insert(0, os.path.abspath("/workspace/PointNeXt"))

# --- FPS (Furthest Point Sampling) 用 ---
# ※FPSはGPU版の関数をそのまま使用する
from openpoints.models.layers.subsample import furthest_point_sample

# --- AVG 用：KMeans ---
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans   # ★追加：GPU 版 K‑Means 用

#############################################
# h5ファイルの読み込み／保存
#############################################
def load_h5_data(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
    return data, label

def save_h5_data(h5_filename, data, label):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)

#############################################
# FPS (GPU版)
#############################################
def fps_sampling(data, num_samples):
    """
    入力点群データに対して GPU版 Furthest Point Sampling を実施する関数である。
    :param data: np.array, 入力点群データ (B, N, 3) の形状
    :param num_samples: int, サンプリングする点数
    :return: np.array, サンプリング後の点群データ (B, num_samples, 3)
    """
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    sampled_indices = furthest_point_sample(data_tensor[:, :, :3].contiguous(), num_samples).long()
    sampled_data = torch.gather(
        data_tensor, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, data_tensor.shape[-1])
    )
    return sampled_data.cpu().numpy()

#############################################
# US (Uniform Sampling)
#############################################
def uniform_sample_o3d(pcd, num_samples):
    pts = np.asarray(pcd.points)
    N = pts.shape[0]
    if N < num_samples:
        raise ValueError("Uniform sampling: 入力点数が指定数より少ないため実施できません。")
    indices = np.linspace(0, N - 1, num_samples, dtype=int)
    new_pts = pts[indices]
    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(new_pts)
    return new_pc

def us_sampling(data, num_samples):
    B = data.shape[0]
    sampled_list = []
    for i in range(B):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[i])
        pcd_sampled = uniform_sample_o3d(pcd, num_samples)
        sampled_list.append(np.asarray(pcd_sampled.points))
    return np.stack(sampled_list, axis=0)

#############################################
# AVG (Average Voxel Grid Sampling) – KMeansによる実装
#############################################
'''
def avg_sampling(data, num_samples):
    """
    入力点群データ (B, N, 3) に対して、各点群に KMeans(n_clusters=num_samples) を適用し、
    各クラスタの重心をサンプリング点として返す。
    これにより、どのような形状の点群に対しても必ず指定点数分のボクセル重心が得られる。
    """
    B, N, D = data.shape
    sampled_data = np.empty((B, num_samples, D), dtype=data.dtype)
    for i in range(B):
        pts = data[i]
        kmeans = KMeans(n_clusters=num_samples, random_state=42).fit(pts)
        centroids = kmeans.cluster_centers_
        sampled_data[i] = centroids
    return sampled_data
'''
'''
#############################################
# AVG GPU版 (Average Voxel Grid Sampling) – KMeansによる実装
#############################################
def avg_sampling_gpu(data, num_samples):
    """
    GPU 版 KMeans (kmeans‑pytorch) を用いて各クラスタ重心を返す。
    入力: data (B, N, 3) numpy
    出力: (B, num_samples, 3) numpy
    """
    B, N, D = data.shape
    sampled = np.empty((B, num_samples, D), dtype=np.float32)
    for i in range(B):
        pts_t = torch.tensor(data[i], device='cuda')
        _, centroids = kmeans(
            X=pts_t,
            num_clusters=num_samples,
            distance='euclidean',
            device=torch.device('cuda')
        )
        sampled[i] = centroids.cpu().numpy()
    return sampled
'''
#############################################
# AVG (Average Voxel Grid Sampling) – Open3D 版
#############################################
def avg_sampling_o3d(data, num_samples, max_iter=10):
    """
    Open3D の voxel_down_sample で各ボクセルの平均点を取得し，
    目標点数 num_samples に近づくよう voxel_size を自動調整する。
    戻り値: (B, num_samples, 3) numpy
    """
    B, N, D = data.shape
    out = np.empty((B, num_samples, D), dtype=np.float32)

    for b in range(B):
        pts = data[b]
        # --- ① 初期 voxel_size 推定 ---
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        volume = np.prod(pmax - pmin)
        voxel_size = (volume / num_samples) ** (1/3) + 1e-6

        sampled_pts = None
        for _ in range(max_iter):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            down = pcd.voxel_down_sample(voxel_size)
            sampled = np.asarray(down.points)
            if sampled.shape[0] >= num_samples or voxel_size < 1e-6:
                sampled_pts = sampled
                break
            voxel_size *= 0.9   # 細かくして点数を増やす

        if sampled_pts is None or sampled_pts.shape[0] == 0:
            # フォールバック: ランダムサンプリング
            idx = np.random.choice(N, num_samples, replace=False)
            out[b] = pts[idx]
        elif sampled_pts.shape[0] > num_samples:
            idx = np.random.choice(sampled_pts.shape[0], num_samples, replace=False)
            out[b] = sampled_pts[idx]
        else:   # ちょうど or 少なめ → パディング
            pad = num_samples - sampled_pts.shape[0]
            if pad > 0:
                pad_idx = np.random.choice(sampled_pts.shape[0], pad, replace=True)
                sampled_pts = np.vstack([sampled_pts, sampled_pts[pad_idx]])
            out[b] = sampled_pts
    return out
#############################################
# RS (Random Sampling)
#############################################
def rs_sampling(data, num_samples):
    """
    入力点群データに対してランダムサンプリング(RS)を実施する関数である。
    :param data: np.array, 入力点群データ (B, N, 3) の形状
    :param num_samples: int, サンプリングする点数
    :return: np.array, サンプリング後の点群データ (B, num_samples, 3)
    """
    B, N, C = data.shape
    if N < num_samples:
        raise ValueError("Random sampling: 入力点数が指定点数より少ないため実施できません。")
    sampled_data = np.empty((B, num_samples, C), dtype=data.dtype)
    for b in range(B):
        indices = np.random.choice(N, num_samples, replace=False)
        sampled_data[b] = data[b][indices]
    return sampled_data

#############################################
# 入力パスの展開（ワイルドカード対応）
#############################################
def expand_input_paths(input_paths):
    expanded = []
    for path in input_paths:
        if '*' in path or not os.path.exists(path):
            matches = glob.glob(path)
            if matches:
                expanded.extend(matches)
            else:
                print(f"Warning: パターン {path} に一致するファイルはありません")
        else:
            expanded.append(path)
    return expanded

#############################################
# 単一ファイル処理：h5読み込み→各サンプルごとにサンプリング→h5保存→処理時間計測
#############################################
def process_file(input_h5_path, output_dir, num_samples, method):
    data, label = load_h5_data(input_h5_path)
    # 入力が (N, 3) の場合、バッチ次元を追加する
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    num_samples_in_file = data.shape[0]
    sampled_data_list = []
    sample_times = []
    method_upper = method.upper()
    for i in range(num_samples_in_file):
        sample = data[i:i+1, :, :]  # shape (1, N, 3)
        t0 = time.time()
        if method_upper == "FPS":
            sampled_sample = fps_sampling(sample, num_samples)
        elif method_upper == "US":
            sampled_sample = us_sampling(sample, num_samples)
        elif method_upper == "AVG":
#            sampled_sample = avg_sampling(sample, num_samples)     #　AVG(k-means) CPU版
#            sampled_sample = avg_sampling_gpu(sample, num_samples) # AVG(k-means) GPU版
            sampled_sample = avg_sampling_o3d(sample, num_samples)
        elif method_upper == "RS":
            sampled_sample = rs_sampling(sample, num_samples)
        else:
            print(f"Unknown method {method}. Using FPS as default.")
            sampled_sample = fps_sampling(sample, num_samples)
        t1 = time.time()
        dt = t1 - t0
        sample_times.append(dt)
        sampled_data_list.append(sampled_sample[0])
    sampled_data = np.array(sampled_data_list)
    file_name = os.path.basename(input_h5_path)
    output_h5_path = os.path.join(output_dir, file_name)
    save_h5_data(output_h5_path, sampled_data, label)
    print(f"Sampled data saved to {output_h5_path}")
    return sample_times

#############################################
# メイン処理
#############################################
def main(input_paths, output_base, num_samples, method):
    input_files = expand_input_paths(input_paths)
    # 出力ディレクトリ例: /workspace/PointNeXt/result/<METHOD>/<num_samples>/modelnet40_ply_hdf5_2048
    output_dir = os.path.join(output_base, method.upper(), str(num_samples), "modelnet40_ply_hdf5_2048")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"使用サンプリング手法: {method.upper()}, サンプリング点数: {num_samples}")
    print(f"出力ディレクトリ: {output_dir}")

    all_sample_times = []
    for input_path in input_files:
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith('.h5'):
                    file_path = os.path.join(input_path, file)
                    sample_times = process_file(file_path, output_dir, num_samples, method)
                    all_sample_times.extend(sample_times)
        else:
            sample_times = process_file(input_path, output_dir, num_samples, method)
            all_sample_times.extend(sample_times)

    if all_sample_times:
        overall_avg_time = np.mean(all_sample_times)
        total_samples = len(all_sample_times)
        summary_file = os.path.join(output_dir, f"average_downsampling_time_{num_samples}.txt")
        with open(summary_file, "w") as f:
            f.write(f"Average downsampling time: {overall_avg_time:.6f} sec\n")
            f.write(f"Total number of samples: {total_samples}\n")
        print(f"Overall average downsampling time saved to {summary_file}")

#############################################
# コマンドライン引数処理
#############################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='点群データのダウンサンプリング (FPS, US, AVG, RS)')
    parser.add_argument('--method', type=str, choices=['FPS', 'US', 'AVG', 'RS'],
                        default='AVG',
                        help='使用するサンプリング手法 (FPS, US, AVG, RS). デフォルトは AVG.')
    parser.add_argument('--num_samples', type=int, required=True,
                        help='ダウンサンプリング後の点数（例: 50）')
    parser.add_argument('--input', type=str, nargs='+',
                        default=['/workspace/PointNeXt/data/modelnet40_ply_hdf5_2048/ply_data_test*.h5'],
                        help='入力となる.h5ファイルまたはディレクトリのパス')
    parser.add_argument('--output', type=str,
                        default='/workspace/PointNeXt/result',
                        help='出力のベースディレクトリ')
    args = parser.parse_args()
    main(args.input, args.output, args.num_samples, args.method)
