#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SampleNet の .txt 点群を ModelNet40 と同じ HDF5 形式へ変換
出力パス例:
  /workspace/PointNeXt/result/SampleNet/500/modelnet40_ply_hdf5_2048/ply_data_test0.h5

使い方（例）
  python txt_to_h5_samplenet.py \
    --input_root /mnt/data/SampleNet/log \
    --variants SampleNet50,SampleNet80,SampleNet125,SampleNet250,SampleNet500,SampleNet1000 \
    --shape_names /workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048/shape_names.txt \
    --output_root /workspace/PointNeXt/result/SampleNet \
    --shard_size 2048
"""

import os
import sys
import glob
import argparse
import re
from typing import List, Tuple
import numpy as np
import h5py

def load_shape_names(path: str) -> List[str]:
    with open(path, 'r') as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise RuntimeError(f"shape_names.txt が空です: {path}")
    return names

def read_points_txt(txt_path: str) -> np.ndarray:
    """1行 'x, y, z'（カンマ区切り）。点数はそのまま返す。"""
    try:
        arr = np.genfromtxt(txt_path, delimiter=',', dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"読み込み失敗: {txt_path}: {e}")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError(f"{txt_path}: 列数が3ではありません（shape={arr.shape}）")
    return arr  # (P,3)

def save_h5(h5_path: str, data: np.ndarray, labels: np.ndarray) -> None:
    """data:(N,P,3) float32, label:(N,) or (N,1) int64"""
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data', data=data.astype(np.float32))
        f.create_dataset('label', data=labels.astype(np.int64))

def write_meta_files(out_dir: str, shape_names: List[str], h5_paths: List[str]) -> None:
    """同ディレクトリに shape_names.txt と test_files.txt を出力"""
    shape_path = os.path.join(out_dir, 'shape_names.txt')
    with open(shape_path, 'w') as f:
        for n in shape_names:
            f.write(n + '\n')
    list_path = os.path.join(out_dir, 'test_files.txt')
    with open(list_path, 'w') as f:
        for p in h5_paths:
            f.write(os.path.abspath(p) + '\n')

def collect_txt_files(variant_root: str) -> List[Tuple[str, str]]:
    """
    variant_root: /mnt/data/SampleNet/log/SampleNet500
    戻り値: [(class_name, txt_path), ...]
    """
    class_root = os.path.join(variant_root, 'eval', 'class_data')
    if not os.path.isdir(class_root):
        raise FileNotFoundError(f"class_data ディレクトリが見つかりません: {class_root}")
    pairs = []
    for cls in sorted(os.listdir(class_root)):
        cls_dir = os.path.join(class_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for txt in sorted(glob.glob(os.path.join(cls_dir, '*.txt'))):
            pairs.append((cls, txt))
    return pairs

def variant_to_numeric_dirname(variant_name: str) -> str:
    """
    'SampleNet500' → '500' のように末尾の数字だけを取り出す。
    数字が見つからなければ、そのまま返す（保険）。
    """
    m = re.search(r'(\d+)', variant_name)
    return m.group(1) if m else variant_name

def build_dataset_for_variant(variant_root: str,
                              shape_names: List[str],
                              shard_size: int,
                              out_root: str,
                              split_name: str='test') -> None:
    """
    1 バリアントを処理して h5 を生成
    出力先: /workspace/PointNeXt/result/SampleNet/<数字>/modelnet40_ply_hdf5_2048
    """
    variant_name = os.path.basename(variant_root.rstrip('/'))      # 例: SampleNet500
    numeric_dir = variant_to_numeric_dirname(variant_name)         # 例: '500'
    out_dir = os.path.join(out_root, numeric_dir, 'modelnet40_ply_hdf5_2048')
    os.makedirs(out_dir, exist_ok=True)

    pairs = collect_txt_files(variant_root)
    if not pairs:
        print(f"[WARN] テキストが見つかりません: {variant_root}")
        return

    # 読み込み & ラベル化（点数は変更しない）
    data_list, label_list = [], []
    expected_P = None
    for cls, txt in pairs:
        if cls not in shape_names:
            print(f"[WARN] shape_names に無いクラスをスキップ: {cls} ({txt})")
            continue
        pts = read_points_txt(txt)  # (P,3)
        if expected_P is None:
            expected_P = pts.shape[0]
        elif pts.shape[0] != expected_P:
            raise ValueError(
                f"{variant_name}: 同一バリアント内で点数が混在しています。"
                f"\n  {txt} は {pts.shape[0]} 点 / 期待 {expected_P} 点。"
                "\n  （点数はそのまま保存する仕様のため、バリアント内で統一されている必要があります）"
            )
        data_list.append(pts)
        label_list.append(shape_names.index(cls))

    if not data_list:
        print(f"[WARN] データ 0 件: {variant_root}")
        return

    data_arr  = np.stack(data_list, axis=0).astype(np.float32)  # (N,P,3)
    label_arr = np.array(label_list, dtype=np.int64)            # (N,)

    # シャーディングして保存（ply_data_test{i}.h5）
    h5_paths = []
    N = data_arr.shape[0]
    shard_idx = 0
    for s in range(0, N, shard_size):
        e = min(s + shard_size, N)
        shard_data  = data_arr[s:e]
        shard_label = label_arr[s:e]
        h5_name = f"ply_data_{split_name}{shard_idx}.h5"
        h5_path = os.path.join(out_dir, h5_name)
        save_h5(h5_path, shard_data, shard_label)
        h5_paths.append(h5_path)
        print(f"[OK] {variant_name} → {h5_path}  samples {s}..{e-1}  shape={shard_data.shape}")
        shard_idx += 1

    # メタファイル（同ディレクトリ）
    write_meta_files(out_dir, shape_names, h5_paths)
    print(f"[DONE] 出力完了 → {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="SampleNet .txt → ModelNet40 互換 .h5 変換")
    parser.add_argument('--input_root', type=str, default='/mnt/data/SampleNet/log',
                        help='SampleNet ルート（直下に SampleNet500 などがある想定）')
    parser.add_argument('--variants', type=str,
                        default='SampleNet50,SampleNet80,SampleNet125,SampleNet250,SampleNet500,SampleNet1000',
                        help='カンマ区切りで処理するバリアント名')
    parser.add_argument('--shape_names', type=str,
                        default='/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048/shape_names.txt',
                        help='クラス名の並び（ラベル割当に使用）')
    parser.add_argument('--output_root', type=str,
                        default='/workspace/PointNeXt/result/SampleNet',
                        help='出力ルート（/workspace/PointNeXt/result/SampleNet）')
    parser.add_argument('--shard_size', type=int, default=2048,
                        help='1 つの h5 に入れるサンプル数（大きすぎる場合は分割）')
    parser.add_argument('--split_name', type=str, default='test',
                        help='出力ファイル接尾辞（ply_data_<split_name>{i}.h5）')
    args = parser.parse_args()

    shape_names = load_shape_names(args.shape_names)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    for v in variants:
        variant_root = os.path.join(args.input_root, v)
        if not os.path.isdir(variant_root):
            print(f"[SKIP] 見つからないためスキップ: {variant_root}")
            continue
        build_dataset_for_variant(
            variant_root=variant_root,
            shape_names=shape_names,
            shard_size=args.shard_size,
            out_root=args.output_root,
            split_name=args.split_name
        )

if __name__ == '__main__':
    main()