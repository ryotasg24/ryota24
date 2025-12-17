#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RR_animation_StackHistogram.py

目的：
- RoundRobin_SHAP_for_PointNeXt.py で使うのと同じ
  「Region-SHAP × Point-SHAP の Round-Robin 的な点選択ルール」に基づき、
  1つのサンプルに対して、選択点数 N を 1 から target_N まで増やしたときに、
  各 Region の棒がどのように積み重なっていくかをアニメーション化する。

前提：
- histogram_Region_ChoicePoint.py, animate_Region_StackHistogram.py と同様、
  MID キャッシュ（region_per_pt, point_adj, pc, label）を利用する。
- Region 分割は hierarchical_kmeans により k_region 個（例: 32）、
  各 Region に fanout 個（例: 2）の Point-SHAP サブ領域（合計 k_point=64）を作る。
- Round-Robin の点選択は RoundRobin_SHAP_for_PointNeXt.py の mid_cache 利用ルートと同じく：
    - Region スコア: region_per_pt を各 Region ブロックで平均
    - Point スコア: point_adj
    - Region スコアの降順でラウンドロビンに各 Region から 1点ずつ選択

出力：
- GIF:
    /workspace/PointNeXt/script_Analysis/Histogram/hist_result/RR/animation/
    配下に保存（既存ファイルがあれば上書き）。
    --out_gif で「ファイル名」を指定（省略時は `sample_id_N{target_N}.gif`）。
- PNG（最終 N=target_N の静止ヒストグラム）:
    /workspace/PointNeXt/script_Analysis/Histogram/hist_result/RR/final
    配下に保存（既存ファイルがあれば上書き）。
    --out_png で「ファイル名」を指定（省略時は `sample_id_N{target_N}_final.png`）。
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.cluster import KMeans

# PointNeXt / provider へのパス（環境に合わせて調整）
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
import provider

# 出力ルートディレクトリ（固定）
GIF_ROOT = "/workspace/PointNeXt/script_Analysis/Histogram/hist_result/RR/animation"
PNG_ROOT = "/workspace/PointNeXt/script_Analysis/Histogram/hist_result/RR/final"


# ===================== 共通ユーティリティ =====================

def _load_mid_npz_list(mid_cache_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(mid_cache_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No npz found in mid_cache_dir: {mid_cache_dir}")
    return files


def _try_load_pc_from_mid(z: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    if "pc" in z.files:
        pc = np.asarray(z["pc"], dtype=np.float32)
        if pc.ndim == 2 and pc.shape[1] == 3:
            return pc
    return None


def _find_dataset_files(dataset_root: str) -> List[str]:
    test_list = os.path.join(dataset_root, "test_files.txt")
    if not os.path.exists(test_list):
        raise RuntimeError(f"test_files.txt not found under dataset root: {dataset_root}")
    return provider.getDataFiles(test_list)


def _load_pc_from_dataset(dataset_root: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    ret = {}
    test_files = _find_dataset_files(dataset_root)
    for fp in test_files:
        data, labels = provider.loadDataFile(fp)
        base = os.path.splitext(os.path.basename(fp))[0]
        ret[base] = (data.astype(np.float32), labels.astype(np.int64))
    return ret


def _parse_sample_id_from_mid_path(mid_path: str) -> str:
    return Path(mid_path).stem


def _extract_file_and_index(sample_id: str) -> Tuple[str, int]:
    core = sample_id
    if "_r" in core:
        core = core[:core.rfind("_r")]
    idx_str = core[-5:]
    fname = core[:-6]
    try:
        idx = int(idx_str)
    except Exception:
        idx = -1
    return fname, idx


# ===================== clustering & RR 選択ロジック =====================

def hierarchical_kmeans(points: np.ndarray, k_region: int, fanout: int):
    """
    RoundRobin_SHAP_for_PointNeXt.py と同じ仕様の hierarchical_kmeans:

    - まず KMeans(k_region) で Region を決める
    - 各 Region 内でさらに fanout クラスタに分割し、point_blocks として返す

    return:
      region_blocks: 長さ k_region, 各要素は点インデックス(list)
      point_blocks : 長さ k_region*fanout, 各要素は点インデックス(list)
      km_reg       : 上位 Region 用 KMeans
      subkm_dict   : {region_id: KMeans} の辞書
    """
    km_reg = KMeans(n_clusters=k_region, random_state=42).fit(points)
    reg_lbl = km_reg.labels_
    region_blocks = [np.where(reg_lbl == r)[0].tolist() for r in range(k_region)]

    point_blocks = []
    subkm_dict = {}
    for r, idxs in enumerate(region_blocks):
        sub_pts = points[idxs]
        n_sub = fanout if len(sub_pts) >= fanout and fanout > 1 else max(1, len(sub_pts))
        km_sub = KMeans(n_clusters=n_sub, random_state=0).fit(sub_pts)
        subkm_dict[r] = km_sub
        for s in range(fanout):
            if s < n_sub:
                sub_idx = np.array(idxs)[km_sub.labels_ == s]
            else:
                sub_idx = np.array([], dtype=int)
            point_blocks.append(sub_idx.tolist())
    return region_blocks, point_blocks, km_reg, subkm_dict


def _merge_fanout_lists(sub_lists: List[np.ndarray], point_scores: np.ndarray) -> np.ndarray:
    """
    fanout 個のサブ領域を Point-SHAP 降順でマージする。
    sub_lists: list of np.ndarray(indices already sorted desc within each sub-block)
    point_scores: (num_points,) float
    """
    fanout = len(sub_lists)
    ptrs = [0] * fanout
    merged: List[int] = []

    while True:
        best_j = -1
        best_sc = -float("inf")
        for j in range(fanout):
            arr = sub_lists[j]
            p = ptrs[j]
            if p < len(arr):
                sc = float(point_scores[arr[p]])
                if sc > best_sc:
                    best_sc = sc
                    best_j = j
        if best_j < 0:
            break
        merged.append(int(sub_lists[best_j][ptrs[best_j]]))
        ptrs[best_j] += 1

    return np.asarray(merged, dtype=np.int32)


def build_region_point_lists(
    region_blocks: List[List[int]],
    point_blocks: List[List[int]],
    point_scores: np.ndarray,
    k_region: int,
    fanout: int,
) -> List[np.ndarray]:
    """
    各Region rについて、
      - そのRegionに対応する fanout 個の Pointサブ領域（point_blocks）を取り出し
      - 各サブ領域内で Point-SHAP 降順ソート
      - fanout 個を Point-SHAP 降順でマージ
    した「Regionごとの候補列」を作る。
    """
    R = len(region_blocks)
    assert R == k_region, f"region_blocks len mismatch: {R} vs k_region={k_region}"
    assert len(point_blocks) == k_region * fanout

    region_point_lists: List[np.ndarray] = []
    for r in range(k_region):
        sub_lists_sorted: List[np.ndarray] = []
        for s in range(fanout):
            pb_idx = r * fanout + s
            pts = np.asarray(point_blocks[pb_idx], dtype=np.int32)
            if len(pts) == 0:
                sub_lists_sorted.append(np.asarray([], dtype=np.int32))
                continue
            sc = point_scores[pts]
            order = np.argsort(-sc)  # 降順
            sub_lists_sorted.append(pts[order])
        merged = _merge_fanout_lists(sub_lists_sorted, point_scores)
        region_point_lists.append(merged)
    return region_point_lists


def roundrobin_select(
    region_scores: np.ndarray,
    region_point_lists: List[np.ndarray],
    target_N: int,
    debug: bool = False,
    point_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Region-SHAP 降順の region_order に従いラウンドロビンで選択。
    各Regionからは「region_point_lists[r] の k-th 点」を取る（k=周回番号）。
    """
    R = len(region_scores)
    region_order = np.argsort(-region_scores)  # 降順
    selected: List[int] = []
    k = 0

    if debug:
        print("[DEBUG_RR] region_order (desc by Region-SHAP):")
        print(region_order.tolist())

    while len(selected) < target_N:
        added_any = False
        for r in region_order:
            pts = region_point_lists[int(r)]
            if k < len(pts):
                pid = int(pts[k])
                selected.append(pid)
                added_any = True
                if debug and point_scores is not None and len(selected) <= 40:
                    print(
                        f"[DEBUG_RR] cycle={k:02d}, region={int(r):02d}, "
                        f"pid={pid}, point_score={float(point_scores[pid]):.6f}"
                    )
                if len(selected) >= target_N:
                    break
        if not added_any:
            break
        k += 1

    return np.asarray(selected[:target_N], dtype=np.int32)


# ===================== メインロジック =====================

def main():
    ap = argparse.ArgumentParser("Animate cumulative Region histogram (RoundRobin SHAP) for one sample.")
    ap.add_argument("--mid_cache_dir", type=str, required=True,
                    help="MID cache dir containing region_per_pt / point_adj / pc / label.")
    ap.add_argument("--dataset", type=str, default="",
                    help="Dataset root including test_files.txt; used if pc not stored in MID.")
    ap.add_argument("--division_region", type=int, default=32,
                    help="Region-SHAPの親領域数 k_region (例: 32)")
    ap.add_argument("--division_point", type=int, default=64,
                    help="Point-SHAPのサブ領域数 k_point (例: 64, fanout=2)")
    ap.add_argument("--target_N", type=int, default=300,
                    help="最終的に取りたい点数（このNまでのラウンドロビン順序で積み上げる）")
    ap.add_argument("--step", type=int, default=1,
                    help="Nを何点刻みで増やすか（1なら1点ずつ）")
    ap.add_argument("--sample_index", type=int, default=0,
                    help="mid_cache_dir 内で何番目のnpzを使うか（0-based）")
    ap.add_argument(
        "--out_gif",
        type=str,
        default="",
        help=(
            "出力するアニメーションGIFのファイル名（パスではなくファイル名）。"
            "空の場合は 'sample_id_N{target_N}.gif'。"
            f"保存先ディレクトリは固定で {GIF_ROOT} 。"
        ),
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default="",
        help=(
            "最終ヒストグラムPNGのファイル名（パスではなくファイル名）。"
            "空の場合は 'sample_id_N{target_N}_final.png'。"
            f"保存先ディレクトリは固定で {PNG_ROOT} 。"
        ),
    )
    ap.add_argument("--fps", type=int, default=10,
                    help="アニメーションのFPS")
    ap.add_argument("--debug_rr", action="store_true",
                    help="RoundRobinの最初の取り出し順を少し表示する")
    args = ap.parse_args()

    # 出力ルートディレクトリを作成
    os.makedirs(GIF_ROOT, exist_ok=True)
    os.makedirs(PNG_ROOT, exist_ok=True)

    k_region = args.division_region
    k_point = args.division_point
    if k_point % k_region != 0:
        raise ValueError("--division_point must be a multiple of --division_region")
    fanout = k_point // k_region

    # MID npz を列挙して sample_index 番目を使用
    mid_files = _load_mid_npz_list(args.mid_cache_dir)
    if args.sample_index < 0 or args.sample_index >= len(mid_files):
        raise RuntimeError(f"sample_index {args.sample_index} out of range (0..{len(mid_files)-1})")

    mid_path = mid_files[args.sample_index]
    sample_id = _parse_sample_id_from_mid_path(mid_path)
    print(f"[INFO] Use sample #{args.sample_index}: {sample_id}")

    # 必要に応じて dataset をロード
    dataset_dict: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    if args.dataset:
        dataset_dict = _load_pc_from_dataset(args.dataset)

    # MID から材料読み込み
    with np.load(mid_path) as z:
        region_per_pt = np.asarray(z["region_per_pt"], dtype=np.float32).reshape(-1)
        point_adj = np.asarray(z["point_adj"], dtype=np.float32).reshape(-1)
        lab = int(np.asarray(z["label"]).reshape(-1)[0]) if "label" in z.files else -1
        pc = _try_load_pc_from_mid(z)

    if pc is None:
        if dataset_dict is None:
            raise RuntimeError("pc not found in MID and no --dataset specified.")
        fname, idx = _extract_file_and_index(sample_id)
        if fname not in dataset_dict or idx < 0:
            raise RuntimeError(f"cannot map sample_id to dataset: {sample_id}")
        data, labels = dataset_dict[fname]
        if idx >= data.shape[0]:
            raise RuntimeError(f"index out of range for {fname}: {idx}")
        pc = data[idx]

    pc = pc.astype(np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise RuntimeError(f"invalid pc shape: {pc.shape} id={sample_id}")

    num_points = pc.shape[0]
    if args.target_N > num_points:
        print(f"[WARN] target_N={args.target_N} > num_points={num_points}, clamp to num_points.")
        target_N = num_points
    else:
        target_N = args.target_N

    print(f"[INFO] pc points: {num_points}, target_N: {target_N}, k_region={k_region}, k_point={k_point}, fanout={fanout}")

    # Region / Point ブロックの構築（ここでは A/B 分割せず、全体から直接 k_region × k_point を作る）
    region_blocks, point_blocks, _, _ = hierarchical_kmeans(pc, k_region, fanout)

    # Regionごとのスコア = region_per_pt の平均
    region_scores = np.zeros((k_region,), dtype=np.float32)
    for rid, idxs in enumerate(region_blocks):
        if len(idxs) > 0:
            region_scores[rid] = float(np.mean(region_per_pt[np.asarray(idxs, dtype=np.int32)]))
        else:
            region_scores[rid] = 0.0

    # Point スコア = MIDの point_adj（L1正規化後 or 生SHAP、MIDの中身に従う）
    point_scores = point_adj.astype(np.float32)

    # Regionごとの「Point候補リスト」（Point-SHAP降順）を生成
    region_point_lists = build_region_point_lists(
        region_blocks, point_blocks, point_scores, k_region, fanout
    )

    # 各点がどのRegionに属するか（ヒストグラム用）
    region_ids = np.full((num_points,), -1, dtype=np.int32)
    for rid, idxs in enumerate(region_blocks):
        region_ids[np.asarray(idxs, dtype=np.int32)] = rid
    num_regions = k_region

    # Round-Robin による選択順（長さ target_N のインデックス列）
    top_indices = roundrobin_select(
        region_scores, region_point_lists, target_N,
        debug=args.debug_rr, point_scores=point_scores
    )
    if top_indices.shape[0] < target_N:
        print(f"[WARN] RoundRobin selected only {top_indices.shape[0]} points (< target_N={target_N}).")
        target_N = top_indices.shape[0]

    sorted_regions = region_ids[top_indices]

    # フレームごとの N のリスト
    step = max(1, args.step)
    frame_N_list = list(range(1, target_N + 1, step))
    if frame_N_list[-1] != target_N:
        frame_N_list.append(target_N)

    # 描画のセットアップ
    x = np.arange(num_regions)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    bars = ax.bar(x, np.zeros_like(x, dtype=np.float32))
    ax.set_xlabel("Region ID")
    ax.set_ylabel("Selected points")
    ax.set_ylim(0, target_N)  # 上限は target_N で十分
    title_txt = ax.set_title(f"Sample {sample_id}, N=0 / target_N={target_N}")

    def init():
        for rect in bars:
            rect.set_height(0.0)
        title_txt.set_text(f"Sample {sample_id}, N=0 / target_N={target_N}")
        return list(bars) + [title_txt]

    def update(N_current: int):
        hist = np.bincount(sorted_regions[:N_current], minlength=num_regions)
        for rect, h in zip(bars, hist):
            rect.set_height(float(h))
        title_txt.set_text(
            f"Sample {sample_id}, N={N_current} / target_N={target_N} (class={lab})"
        )
        return list(bars) + [title_txt]

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_N_list,
        init_func=init,
        blit=False,
        repeat=False,
    )

    # GIF 出力パス（固定ルート＋ファイル名）
    if args.out_gif:
        out_gif_name = os.path.basename(args.out_gif)
    else:
        out_gif_name = f"{sample_id}_N{target_N}.gif"
    out_gif = os.path.join(GIF_ROOT, out_gif_name)

    if os.path.exists(out_gif):
        os.remove(out_gif)

    writer = PillowWriter(fps=args.fps)
    ani.save(out_gif, writer=writer)
    plt.close(fig)
    print(f"[SAVE] animation gif -> {out_gif}")

    # 最終ヒストグラムPNGの保存パス（固定ルート＋ファイル名）
    if args.out_png:
        out_png_name = os.path.basename(args.out_png)
    else:
        out_png_name = f"{sample_id}_N{target_N}_final.png"
    out_png = os.path.join(PNG_ROOT, out_png_name)

    if os.path.exists(out_png):
        os.remove(out_png)

    hist_final = np.bincount(sorted_regions[:target_N], minlength=num_regions)
    fig2, ax2 = plt.subplots(figsize=(8.0, 3.8))
    ax2.bar(x, hist_final)
    ax2.set_xlabel("Region ID")
    ax2.set_ylabel("Selected points (N=target_N)")
    ax2.set_title(f"Final Region Histogram @ N={target_N}\nSample {sample_id}, class={lab}")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig2)

    print(f"[SAVE] final histogram png -> {out_png}")


if __name__ == "__main__":
    main()
