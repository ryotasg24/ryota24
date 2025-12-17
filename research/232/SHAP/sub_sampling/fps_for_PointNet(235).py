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
import glob

# 可視化用ライブラリ（今回は出力には使用しません）
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# まず、/workspace/PointNeXt を Python の検索パスの先頭に追加する
sys.path.insert(0, os.path.abspath("/workspace/PointNeXt"))

# --- openpoints モジュールのインポート ---
from openpoints.models.layers.subsample import furthest_point_sample

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

def fps_sampling(data, num_samples):
    """
    入力点群データに対して Furthest Point Sampling を実施する関数である。
    :param data: np.array, 入力点群データ (B, N, 3) の形状
    :param num_samples: int, サンプリングする点数
    :return: np.array, サンプリング後の点群データ (B, num_samples, 3)
    """
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    sampled_indices = furthest_point_sample(data_tensor[:, :, :3].contiguous(), num_samples).long()  # 型変換を追加
    sampled_data = torch.gather(
        data_tensor, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, data_tensor.shape[-1])
    )
    return sampled_data.cpu().numpy()

def process_file(input_h5_path, output_dir, num_samples):
    """
    単一の.h5ファイルに対して FPS を実施し、出力フォルダに同名で保存する関数である。
    """
    data, label = load_h5_data(input_h5_path)
    sampled_data = fps_sampling(data, num_samples)
    file_name = os.path.basename(input_h5_path)
    output_h5_path = os.path.join(output_dir, file_name)
    save_h5_data(output_h5_path, sampled_data, label)
    print(f"Sampled data saved to {output_h5_path}")

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

def main(input_paths, output_base, num_samples):
    # 入力パスの展開
    input_files = expand_input_paths(input_paths)
    
    # 出力フォルダの作成（例: /workspace/PointNeXt/result/FPS/<num_samples>/modelnet40_ply_hdf5_2048）
    output_dir = os.path.join(output_base, str(num_samples), 'modelnet40_ply_hdf5_2048')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for input_path in input_files:
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith('.h5'):
                    file_path = os.path.join(input_path, file)
                    process_file(file_path, output_dir, num_samples)
        else:
            process_file(input_path, output_dir, num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Furthest Point Samplingによる点群データのダウンサンプリング')
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        default=['/workspace/PointNeXt/data/modelnet40_ply_hdf5_2048/ply_data_test*.h5'],
        help='入力となる.h5ファイルまたはディレクトリのパス'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/workspace/PointNeXt/result/FPS',
        help='出力のベースディレクトリ'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        required=True,
        help='ダウンサンプリング後の点数（例: 50）'
    )
    
    args = parser.parse_args()
    main(args.input, args.output, args.num_samples)
