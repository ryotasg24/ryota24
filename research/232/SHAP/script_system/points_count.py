#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import open3d as o3d

def count_points_in_ply(ply_filename):
    """
    指定されたPLYファイルを読み込み、点数を返す関数。
    """
    pcd = o3d.io.read_point_cloud(ply_filename)
    return len(pcd.points)

def main():
    parser = argparse.ArgumentParser(
        description="Count the number of points in a PLY file or in multiple PLY files (using wildcards)."
    )
    parser.add_argument("--input", type=str,
                        default="/workspace/PointNeXt/result/nonSort_SHAP/ply_files/50/ply_data_test0_sample_000_label_4.ply",
                        help="対象となるPLYファイルのパスまたはパターン (例: data/*.ply)")
    args = parser.parse_args()

    # 入力パターンにマッチするファイルを取得
    ply_files = glob.glob(args.input)
    if not ply_files:
        print("指定のパターンに一致するPLYファイルが見つかりません。")
        return

    for ply_file in ply_files:
        count = count_points_in_ply(ply_file)
        print(f"{ply_file}: {count} points")

if __name__ == "__main__":
    main()
