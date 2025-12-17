#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コマンド:
python script_Analysis/Point-to-PlaneRMSE/P2Plane_RMSE.py \
-e /workspace/PointNeXt/result/switchSDCS-mlp-allOn_kr32_kp64_p10000_dsSDCS_PointNeXt_h5 \
--method HyCov-SD


H5 ↔ H5 でソート影響を完全排除して、参照（オリジナル）点群と評価（ダウンサンプル）点群の
対称 Point-to-Plane RMS を求めるスクリプト。

前提配置:
- 参照H5: <ref_dir>/ply_data_test{0,1,...}.h5  （data: (S,N,3)）
- 評価H5: <eval_base_dir>/<point_count>/(直下 または 下位の任意サブディレクトリ)/ply_data_test{0,1,...}.h5

各 test_id ごとに sample_idx をインデックスとして 1:1 対応させるため、ファイル名とサンプル順のみを使用し、
ファイル内の点の並び（インデックス）とは無関係に評価できます。

出力:
・・・
[OK] Point 400: pairs=2468, avg=0.024713, min=0.000000, max=0.202652, std=0.021152, psnr=40.84 dB
・・・
「avg=0.024713」: Point-to-Plane-RMSE。(求めていた指標)
「psnr=40.84 dB」: Point-to-Plane RSNR。
"""

import sys
import os
import re
import glob
import math
import argparse
from typing import Dict, List, Tuple

import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ===========================
# 引数
# ===========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute symmetric point-to-plane RMS between original (H5) and downsampled (H5) point clouds."
    )
    p.add_argument("-a", "--ref_dir", type=str,
        default="/workspace/PointNeXt/data/modelnet40_ply_hdf5_2048",
        help="参照（オリジナル）H5のディレクトリ。ply_data_test*.h5 を置く。"
    )
    p.add_argument("-e", "--eval_base_dir", type=str,
        default="/workspace/PointNeXt/result/AMA-mlp-gate_mlp_n100_200_kr32_kp64_p10000_dsSHAP_PointNeXt_h5",
        help="評価H5のベースディレクトリ。各点数ごとにサブディレクトリを持ち、その下に ply_data_test*.h5 がある想定。"
    )
    p.add_argument("-m", "--method", type=str,
        default="SD-GLM",
        help="手法名（出力txt/図タイトルに利用）"
    )
    p.add_argument("-p", "--point_counts", type=int, nargs='+',
        default=[100, 200 ,300, 400, 500, 600, 700, 800, 900, 1000],
        help="評価する点数のリスト（各サブディレクトリ名と一致）"
    )
    p.add_argument("-r", "--rtimes", type=float,
        default=0.03,
        help="法線推定半径の倍率（0ならKNN、>0で bbox 対角 × 倍率 を半径とする）"
    )
    p.add_argument("-k", "--knn", type=int,
        default=32,
        help="法線推定時のKNN数（Hybridでも max_nn として利用）"
    )
    p.add_argument("--ylim", type=float,
        default=0.3,
        help="棒グラフのY上限"
    )
    return p.parse_args()


# ===========================
# 参照読み込み（H5）
# ===========================
_H5_TEST_RE = re.compile(r"ply_data_test(\d+)\.h5$")

def index_reference_h5(ref_dir: str) -> Dict[int, np.ndarray]:
    """
    参照H5を test_id -> data(np.ndarray, shape=(S,N,3)) に読み込む。
    """
    pattern = os.path.join(ref_dir, "ply_data_test*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Error: 参照H5が見つかりません: {pattern}")
        sys.exit(1)
    ref_map: Dict[int, np.ndarray] = {}
    total = 0
    for path in files:
        m = _H5_TEST_RE.search(os.path.basename(path))
        if not m:
            continue
        test_id = int(m.group(1))
        with h5py.File(path, "r") as f:
            data = f["data"][:]  # (S, N, 3)
        ref_map[test_id] = data
        total += data.shape[0]
    if not ref_map:
        print("Error: 参照H5のインデックス化に失敗しました。ファイル名/中身を確認してください。")
        sys.exit(1)
    print(f"[REF] H5 indexed: test_ids={sorted(ref_map.keys())}, total_samples={total}")
    return ref_map


# ===========================
# 評価読み込み（H5）
# ===========================
def find_eval_h5_files(eval_dir_for_count: str) -> List[str]:
    """
    指定点数サブディレクトリ配下の ply_data_test*.h5 を列挙する。
    直下に無ければ、下位階層も探索（1+ 階層対応）。
    """
    direct = sorted(glob.glob(os.path.join(eval_dir_for_count, "ply_data_test*.h5")))
    if direct:
        return direct
    # 下位も探索（recursive）
    recursive = sorted(glob.glob(os.path.join(eval_dir_for_count, "**", "ply_data_test*.h5"), recursive=True))
    return recursive


def iter_eval_h5(eval_dir_for_count: str):
    """
    評価H5を列挙し、(test_id, sample_idx, points(np.ndarray(N,3))) を yield。
    """
    files = find_eval_h5_files(eval_dir_for_count)
    if not files:
        print(f"[WARN] 評価H5が見つかりません: {eval_dir_for_count}")
        return
    for path in files:
        m = _H5_TEST_RE.search(os.path.basename(path))
        if not m:
            continue
        test_id = int(m.group(1))
        with h5py.File(path, "r") as f:
            data = f["data"][:]  # (S, N, 3)
        S = data.shape[0]
        for i in range(S):
            yield test_id, i, data[i]


# ===========================
# Open3D 変換・法線推定
# ===========================
def _bbox_diag_from_pcd(pcd: o3d.geometry.PointCloud) -> float:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return 1.0
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def _estimate_normals_inplace(pcd: o3d.geometry.PointCloud, knn: int, rtimes: float):
    if rtimes > 0.0:
        radius = _bbox_diag_from_pcd(pcd) * rtimes
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn)
        )
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )
    pcd.normalize_normals()


def ndarray_to_pcd(points: np.ndarray, knn: int, rtimes: float) -> o3d.geometry.PointCloud:
    # points: (N,3) float
    pts = points.astype(np.float32, copy=False)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    _estimate_normals_inplace(pcd, knn, rtimes)
    return pcd


# ===========================
# 誤差計算
# ===========================
def compute_point_to_plane_error(src_pcd: o3d.geometry.PointCloud,
                                 tgt_pcd: o3d.geometry.PointCloud) -> float:
    src_pts = np.asarray(src_pcd.points)
    src_nrm = np.asarray(src_pcd.normals)
    tgt_pts = np.asarray(tgt_pcd.points)

    if src_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
        return 0.0

    tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    errs2 = np.empty(src_pts.shape[0], dtype=np.float64)

    for i in range(src_pts.shape[0]):
        p = src_pts[i]
        n = src_nrm[i]
        k, idx, _ = tgt_tree.search_knn_vector_3d(p, 1)
        if k > 0:
            q = tgt_pts[idx[0]]
            errs2[i] = (abs(np.dot(p - q, n))) ** 2
        else:
            errs2[i] = 0.0

    return math.sqrt(float(np.mean(errs2))) if errs2.size else 0.0


def compute_symmetric_p2plane(pcd_ref: o3d.geometry.PointCloud,
                              pcd_eval: o3d.geometry.PointCloud) -> float:
    e1 = compute_point_to_plane_error(pcd_ref, pcd_eval)
    e2 = compute_point_to_plane_error(pcd_eval, pcd_ref)
    return 0.5 * (e1 + e2)


def calc_psnr(rms: float, peak: float) -> float:
    if rms <= 0.0:
        return float('inf')
    return 20.0 * math.log10(peak / rms)


# ===========================
# 出力
# ===========================
def save_results_txt(method_name: str, results: dict, output_txt: str = None):
    if output_txt is None:
        output_txt = f"/workspace/PointNeXt/script_Analysis/Point-to-PlaneRMSE/evaluation_results_{method_name}.txt"
    header = "Method\tPoint_Count\tAverage\tMin\tMax\tStd\tPSNR\n"
    lines = [header]
    for pc in sorted(results.keys(), key=lambda x: int(x)):
        r = results[pc]
        lines.append(
            f"{method_name}\t{pc}\t{r['avg']:.6f}\t{r['min']:.6f}\t{r['max']:.6f}\t{r['std']:.6f}\t{r['psnr']:.4f}\n"
        )
    with open(output_txt, "w") as f:
        f.writelines(lines)
    print(f"[SAVE] {output_txt}")


def plot_bar(results: dict, method_name: str, ylim: float = 0.5):
    xs = sorted(results.keys(), key=lambda x: int(x))
    avg = [results[k]["avg"] for k in xs]
    lo  = [results[k]["avg"] - results[k]["min"] for k in xs]
    hi  = [results[k]["max"] - results[k]["avg"] for k in xs]

    plt.figure(figsize=(10, 6))
    plt.bar([str(k) for k in xs], avg, yerr=[lo, hi], capsize=8)
    plt.xlabel("Point Count")
    plt.ylabel("Point-to-Plane RMS Error")
    plt.title(f"Point-to-Plane RMS Error (original vs {method_name})")
    plt.ylim(0.0, ylim)
    plt.tight_layout()
    out = f"point_to_plane_{method_name}.png"
    plt.savefig(out)
    plt.show()
    print(f"[SAVE] {out}")


# ===========================
# メイン
# ===========================
def main():
    args = parse_args()

    # 参照（H5）を test_id→data に読み込み
    ref_map = index_reference_h5(args.ref_dir)

    results = {}

    for count in args.point_counts:
        eval_dir = os.path.join(args.eval_base_dir, str(count))
        if not os.path.isdir(eval_dir):
            print(f"[WARN] 点数 {count}: ディレクトリなし: {eval_dir}")
            continue

        errs: List[float] = []
        psnrs: List[float] = []

        # test_id ごとに参照を用意（Open3D への変換はサンプル毎）
        n_pairs = 0
        for test_id, sample_idx, eval_points in iter_eval_h5(eval_dir):
            ref_data = ref_map.get(test_id, None)
            if ref_data is None:
                # 参照に存在しない test_id はスキップ
                continue
            if sample_idx < 0 or sample_idx >= ref_data.shape[0]:
                # 参照側に同 index が無ければスキップ
                continue

            ref_points = ref_data[sample_idx]  # (N,3)

            # ndarray -> Open3D, 法線推定
            pcd_ref  = ndarray_to_pcd(ref_points, args.knn, args.rtimes)
            pcd_eval = ndarray_to_pcd(eval_points, args.knn, args.rtimes)

            # 対称Point-to-Plane
            e = compute_symmetric_p2plane(pcd_ref, pcd_eval)
            errs.append(e)

            peak = _bbox_diag_from_pcd(pcd_ref)  # 参照のスケールをピーク値に
            psnrs.append(calc_psnr(e, peak))

            n_pairs += 1

        if errs:
            errs_np = np.array(errs, dtype=np.float64)
            results[count] = {
                "avg": float(np.mean(errs_np)),
                "min": float(np.min(errs_np)),
                "max": float(np.max(errs_np)),
                "std": float(np.std(errs_np)),
                "psnr": float(np.mean(psnrs)) if psnrs else 0.0
            }
            print(
                f"[OK] Point {count}: pairs={n_pairs}, "
                f"avg={results[count]['avg']:.6f}, min={results[count]['min']:.6f}, "
                f"max={results[count]['max']:.6f}, std={results[count]['std']:.6f}, "
                f"psnr={results[count]['psnr']:.2f} dB"
            )
        else:
            print(f"[WARN] 点数 {count}: 有効なペアがありませんでした。")

    if not results:
        print("評価結果が空です。参照/評価の配置とフォーマットを確認してください。")
        sys.exit(1)

    save_results_txt(args.method, results)
    plot_bar(results, args.method, ylim=args.ylim)


if __name__ == "__main__":
    main()
