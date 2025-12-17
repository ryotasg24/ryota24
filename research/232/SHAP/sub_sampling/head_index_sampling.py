#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import h5py
import numpy as np

def load_h5_data(h5_filename):
    """
    h5ファイルから点群データとラベルを読み込む。
    データは (num_samples, N, 3) の形状、ラベルは (num_samples,) または (num_samples,1) を想定。
    """
    with h5py.File(h5_filename, 'r') as f:
        data = f['data'][:]   # 点群データ
        label = f['label'][:] # ラベルデータ
    # ラベルが多次元の場合は1次元に変換
    if label.ndim > 1:
        label = label.reshape(-1)
    return data, label

def save_h5_data(h5_filename, data, label):
    """
    データとラベルをh5ファイルに保存する。
    ラベルは (num_samples,1) にリシェイプして保存する。
    """
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)

def downsample_points(points, target_count=1024):
    """
    1サンプル内の点群 (N,3) から、
    先頭 target_count 個の点をそのまま取り出す。
    点数が足りない場合は、足りない分を0でパディングする。
    """
    N = points.shape[0]
    if N >= target_count:
        return points[:target_count]
    else:
        pad_count = target_count - N
        pad = np.zeros((pad_count, 3), dtype=points.dtype)
        return np.concatenate([points, pad], axis=0)

def process_h5_file(input_h5, output_dir, target_count=1024):
    """
    指定されたh5ファイル内の各サンプルについて、先頭 target_count 点にダウンサンプリングし、
    同順・同形式に従い新たなh5ファイルとして保存する。
    """
    data, labels = load_h5_data(input_h5)
    num_samples = data.shape[0]
    # 各サンプルを先頭 target_count 点にダウンサンプリング
    downsampled_data = []
    for i in range(num_samples):
        sample = data[i]  # (N, 3)
        ds_sample = downsample_points(sample, target_count=target_count)
        downsampled_data.append(ds_sample)
    downsampled_data = np.array(downsampled_data)
    # ラベルはそのままの順序で保存
    output_filename = os.path.join(output_dir, os.path.basename(input_h5))
    save_h5_data(output_filename, downsampled_data, labels)
    print(f"Saved downsampled file: {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Downsample each point cloud sample in h5 files to the first 1024 points, preserving the original sample order."
    )
    parser.add_argument("--input_dir", type=str,
                        default="/workspace/PointNeXt/data/modelnet40_ply_hdf5_2048/ply_data_test*.h5",
                        help="入力となるh5ファイルのパターン（例: ply_data_test*.h5）")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/PointNeXt/result/head1024",
                        help="出力h5ファイルを保存するディレクトリ")
    parser.add_argument("--target_count", type=int, default=1024,
                        help="ダウンサンプリング後の点数（デフォルト1024）")
    args = parser.parse_args()

    # 出力先ディレクトリの作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 入力パスにマッチするh5ファイルリストを取得
    h5_files = sorted(glob.glob(args.input_dir))
    print(f"Found {len(h5_files)} h5 files matching pattern {args.input_dir}.")

    for h5_file in h5_files:
        print(f"Processing file: {h5_file}")
        process_h5_file(h5_file, args.output_dir, target_count=args.target_count)

if __name__ == "__main__":
    main()
