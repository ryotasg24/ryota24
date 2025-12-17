#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import glob
import argparse
import h5py
import numpy as np
import time

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# /workspace/PointNeXt をPythonの検索パスの先頭に追加する
sys.path.insert(0, os.path.abspath("/workspace/PointNeXt"))

# --- provider.py （PointNet時代のもの） ---
import provider

#############################################
# H5ファイルの読み込み／保存
#############################################
def load_h5_data(h5_filename):
    """
    h5ファイルを読み込み、データとラベルを返す。
    ラベルが多次元の場合は1次元に変換する。
    """
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]   # 例: (num_samples, N, 3)
        label = f['label'][:] # 例: (num_samples,) または (num_samples, 1)
    if label.ndim > 1:
        label = label.reshape(-1)
    return data, label

def save_h5_data(h5_filename, data, label):
    """
    h5ファイルにデータとラベルを書き出す。
    """
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)

#############################################
# PLY出力用関数
#############################################
def write_ply(filename, points, include_label=False, label=None):
    """
    点群(points)をPLY形式で出力する。
    points: (N,3) のfloat型numpy配列。
    include_labelがTrueかつlabelが指定されていれば、ファイル中にラベル情報も出力する。
    ※ 今回はファイル名にラベルを反映するため、出力内容は位置情報のみとする。
    """
    N = points.shape[0]
    if include_label and label is not None:
        header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property int label
end_header
"""
    else:
        header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        if include_label and label is not None:
            # 各行で座標とラベルを出力する場合
            for p, l in zip(points, label):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(l)}\n")
        else:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

#############################################
# 単一ファイル処理：.h5ファイル内の各サンプルをPLYに変換
#############################################
def process_h5_file(h5_filename, output_dir):
    """
    指定された.h5ファイルから点群データとラベルを読み込み、
    各サンプルを個別のPLYファイルとして出力する。
    出力ファイル名は、元のファイル名、サンプル番号、ラベル情報を含む。
    例: ply_data_test0_sample_000_label_0.ply
    """
    data, labels = load_h5_data(h5_filename)
    base_filename = os.path.splitext(os.path.basename(h5_filename))[0]
    num_samples = data.shape[0]
    for i in range(num_samples):
        sample_points = data[i]    # 形状: (N,3)
        # もともとのh5から読み込んだラベルをそのまま使用する（整数にキャスト）
        sample_label = int(labels[i])
        # ファイル名にラベルも反映する
        ply_filename = os.path.join(output_dir, f"{base_filename}_sample_{i:03d}_label_{sample_label}.ply")
        # include_label はFalseとして出力（ファイル内は位置のみ）
        write_ply(ply_filename, sample_points, include_label=False)
        print(f"Saved {ply_filename}")

#############################################
# メイン処理
#############################################
def main():
    parser = argparse.ArgumentParser(
        description="Convert all point cloud samples from h5 files (matching a pattern) to individual ply files, preserving original labels."
    )
    parser.add_argument("--input_dir", type=str,
                        default="/workspace/PointNeXt/result/AVG/1000/modelnet40_ply_hdf5_2048/ply_data_test*.h5",
                        help="入力となる.h5ファイルのパターン（例: ply_data_test*.h5）")
#    parser.add_argument("--input_dir", type=str,
#                        default="/workspace/PointNeXt/result/head1024/ply_data_test*.h5",
#                        help="入力となる.h5ファイルのパターン（例: ply_data_test*.h5）")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/PointNeXt/result/AVG/ply_files/1000",
                        help="出力PLYファイルを保存するディレクトリのパス")
    args = parser.parse_args()

    input_pattern = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 入力パターンにマッチするすべての.h5ファイルを取得
    h5_files = glob.glob(input_pattern)
    print(f"Found {len(h5_files)} h5 files matching pattern {input_pattern}.")

    for h5_file in h5_files:
        print(f"Processing file: {h5_file}")
        process_h5_file(h5_file, output_dir)

    print("All ply files have been saved.")

if __name__ == "__main__":
    main()

