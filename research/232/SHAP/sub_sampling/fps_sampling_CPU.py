#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import gc
import numpy as np
import argparse
import h5py
from sklearn.cluster import KMeans
import time
import torch
import torch.nn.functional as F
import glob

# 可視化用ライブラリ（今回は出力には使用しません）
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# まず、/workspace/PointNeXt を Python の検索パスの先頭に追加する
sys.path.insert(0, os.path.abspath("/workspace/PointNeXt"))

##############################################
# CPU版 Furthest Point Sampling の実装
##############################################
def fps_cpu(points, num_samples):
    """
    CPU版 Furthest Point Sampling を実装する関数である。
    :param points: numpy array, shape (N, 3)
    :param num_samples: int, サンプリングする点数
    :return: numpy array of indices, shape (num_samples,)
    """
    N = points.shape[0]
    selected_indices = np.zeros(num_samples, dtype=np.int64)
    # 最初の点はランダムに選ぶ
    selected_indices[0] = np.random.randint(0, N)
    distances = np.full(N, np.inf)
    for i in range(1, num_samples):
        current_point = points[selected_indices[i-1]]
        # 各点と現在の点とのユークリッド距離の2乗を計算
        dist = np.sum((points - current_point)**2, axis=1)
        distances = np.minimum(distances, dist)
        selected_indices[i] = np.argmax(distances)
    return selected_indices

def fps_sampling(data, num_samples):
    """
    入力点群データに対して CPU版 Furthest Point Sampling を実施する関数である。
    :param data: numpy array, 入力点群データ (B, N, 3) の形状
    :param num_samples: int, サンプリングする点数
    :return: numpy array, サンプリング後の点群データ (B, num_samples, 3)
    """
    B, N, C = data.shape
    sampled_data = np.zeros((B, num_samples, C), dtype=data.dtype)
    for b in range(B):
        indices = fps_cpu(data[b], num_samples)
        sampled_data[b] = data[b][indices]
    return sampled_data

##############################################
# H5ファイルの読み込み・保存関数
##############################################
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

##############################################
# ファイル処理関数
##############################################
def process_file(input_h5_path, output_dir, num_samples):
    """
    単一の.h5ファイルに対して、各点群サンプルごとにFPSを実施し、出力フォルダに同名で保存する関数である。
    また、各サンプルのFPS処理時間を計測し、そのリストを返す。
    """
    data, label = load_h5_data(input_h5_path)
    num_samples_in_file = data.shape[0]
    sampled_data_list = []
    sample_times = []
    
    for i in range(num_samples_in_file):
        sample = data[i:i+1, :, :]  # shape (1, N, 3)
        t0 = time.time()
        sampled_sample = fps_sampling(sample, num_samples)  # shape (1, num_samples, 3)
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

def expand_input_paths(input_paths):
    """
    入力パスにワイルドカードが含まれている場合は glob で展開する関数である。
    """
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

##############################################
# メイン処理
##############################################
def main(input_paths, output_base, num_samples):
    # 入力パスの展開
    input_files = expand_input_paths(input_paths)
    
    # 出力フォルダの作成（例: /workspace/PointNeXt/result/FPS/<num_samples>/modelnet40_ply_hdf5_2048）
    output_dir = os.path.join(output_base, str(num_samples), 'modelnet40_ply_hdf5_2048')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_sample_times = []
    for input_path in input_files:
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith('.h5'):
                    file_path = os.path.join(input_path, file)
                    sample_times = process_file(file_path, output_dir, num_samples)
                    all_sample_times.extend(sample_times)
        else:
            sample_times = process_file(input_path, output_dir, num_samples)
            all_sample_times.extend(sample_times)
    
    # 全サンプルのFPS処理時間の平均とサンプル数を1つのテキストファイルに出力する
    if all_sample_times:
        overall_avg_time = np.mean(all_sample_times)
        total_samples = len(all_sample_times)
        summary_file = os.path.join(output_dir, f"average_downsampling_time_{num_samples}.txt")
        with open(summary_file, "w") as f:
            f.write(f"Average downsampling time: {overall_avg_time:.6f} sec\n")
            f.write(f"Total number of samples: {total_samples}\n")
        print(f"Overall average downsampling time saved to {summary_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Furthest Point Samplingによる点群データのダウンサンプリング (CPU版)')
    parser.add_argument('--input', type=str, nargs='+',
                        default=['/workspace/PointNeXt/data/modelnet40_ply_hdf5_2048/ply_data_test*.h5'],
                        help='入力となる.h5ファイルまたはディレクトリのパス')
    parser.add_argument('--output', type=str,
                        default='/workspace/PointNeXt/result/FPS_CPU',
                        help='出力のベースディレクトリ')
    parser.add_argument('--num_samples', type=int, required=True,
                        help='ダウンサンプリング後の点数（例: 50）')
    
    args = parser.parse_args()
    main(args.input, args.output, args.num_samples)
