#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
histogram_Region_ChoicePoint.py

目的：
- SD-GLM(AMA_SHAP_for_PointNeXt.py)で得られる downsampling の結果に対して、
  「各Nで、Hybridで選ばれた点がどのRegionから何点ずつ取られているか」を計測し、
  Region別の選択点数ヒストグラム（counts）を出力・可視化する。

前提：
- Region分割は AMA_SHAP_for_PointNeXt.py と同じ hierarchical_kmeans を用いて再構成する。
- **このスクリプト上のRegionは、Region-SHAPの親k_region領域（例:32）を必ず使う。**
  Point-SHAP側のleaf(k_point領域)は使わず、fanout個ずつ束ねた親領域で集計する。
- Hybrid-SHAP スコアは以下いずれかで取得する：
  (A) MIDキャッシュから region_per_pt / point_adj を読み、AMA（heuristic or mlp）で N ごとに再合成
  (B) globalキャッシュ（特定Nのhybrid_shap保存）を読む
  基本は(A)推奨。globalは N ごとに別ディレクトリなので扱いが煩雑なため。

使い方例：
1) MIDキャッシュを使って、複数Nのヒストグラムをまとめて作る（heuristic Gate）:
    python histogram_Region_ChoicePoint.py \
        --mid_cache_dir result/_shap_cache_mid/p10000_kr32_kp64_ckpt-XXXX \
        --division_region 32 --division_point 64 \
        --n_list 300 400 500 600 700 800 900 1000 \
        --ama --ama_mode heuristic \
        --out_dir result/_region_hist_debug

2) MIDキャッシュ + 学習済みGate-MLPでα,βを推定してヒストグラム:
    python histogram_Region_ChoicePoint.py \
        --mid_cache_dir result/_shap_cache_mid/p10000_kr32_kp64_ckpt-XXXX \
        --division_region 32 --division_point 64 \
        --n_list 300 400 500 600 700 800 900 1000 \
        --ama --ama_mode mlp \
        --gate_ckpt result/_gate_train_dump/gate_mlp_n300_400_...pth \
        --gate_scaler result/_gate_train_dump/gate_scaler_n300_400_...npz \
        --out_dir result/_region_hist_debug

3) datasetから直接読みたい場合（MIDにpcが無いときの保険）:
    python histogram_Region_ChoicePoint.py \
        --dataset /workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048 \
        --mid_cache_dir result/_shap_cache_mid/p10000_kr32_kp64_ckpt-XXXX \
        --division_region 32 --division_point 64 \
        --n_list 300 500 800 1000 \
        --ama --ama_mode heuristic

出力：
- out_dir/region_hist_per_sample.tsv
    1行 = 1サンプル×1N
    columns:
      sample_id, label, N, num_regions, nonempty_regions, max_count, min_nonzero, counts_csv
- out_dir/region_hist_summary.tsv
    Nごとの平均（counts_mean）、標準偏差（counts_std）、非空Region数統計
- out_dir/plots/mean_hist_N{N}.png       （--plot を付けた時のみ）

注意：
- このスクリプトは「どのRegionから取られたかの分布を見る」目的なので、
  SHAP/heavy再計算は行わない。MIDキャッシュの材料からHybrid再合成するだけ。
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import gc
import math
import glob
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt

# PointNeXt / AMA側の既存関数を再利用
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from AMA_SHAP_for_PointNeXt import (
    hierarchical_kmeans,
    gate_forward,
    AdaptiveGateMLP,
)

# provider を使ってdataset h5を読む（MIDにpcが無い場合のみ）
import provider


def _load_mid_npz_list(mid_cache_dir: str) -> List[str]:
    """mid_cache_dir 内の npz（sample材料）を列挙"""
    files = sorted(glob.glob(os.path.join(mid_cache_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No npz found in mid_cache_dir: {mid_cache_dir}")
    return files


def _try_load_pc_from_mid(z: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    """MIDにpcがあれば返す。なければNone"""
    if "pc" in z.files:
        pc = np.asarray(z["pc"], dtype=np.float32)
        if pc.ndim == 2 and pc.shape[1] == 3:
            return pc
    return None


def _find_dataset_files(dataset_root: str) -> List[str]:
    """dataset_root/test_files.txt から test h5 を取得"""
    test_list = os.path.join(dataset_root, "test_files.txt")
    if not os.path.exists(test_list):
        raise RuntimeError(f"test_files.txt not found under dataset root: {dataset_root}")
    return provider.getDataFiles(test_list)


def _load_pc_from_dataset(dataset_root: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    dataset_root 配下の test_files 全部を読んで辞書にする。
    key = base_name (e.g., ply_data_test0)
    value = (data, label)
    """
    ret = {}
    test_files = _find_dataset_files(dataset_root)
    for fp in test_files:
        data, labels = provider.loadDataFile(fp)
        base = os.path.splitext(os.path.basename(fp))[0]  # ply_data_test0
        ret[base] = (data.astype(np.float32), labels.astype(np.int64))
    return ret


def _parse_sample_id_from_mid_path(mid_path: str) -> str:
    """
    AMA_SHAP側 sample_id:
      f"{name_without_ext}_{i:05d}_r{k_region}_p{k_point}"
    midファイル名から sample_id を返す（拡張子除去）
    """
    return Path(mid_path).stem


def _extract_file_and_index(sample_id: str) -> Tuple[str, int]:
    """
    sample_id から testfile名と i を推定。
    例: ply_data_test0_00012_r32_p64 -> ("ply_data_test0", 12)
    """
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


def _extract_parent_region_blocks(hkm_out: Tuple, k_region: int, fanout: int) -> List[List[int]]:
    """
    hierarchical_kmeans の返り値から親Region(=k_region)のブロックを抽出する。

    優先度：
    1) 返り値タプルの中に「長さ k_region の list-of-lists」があればそれを親Regionとして採用。
    2) 無ければ、leaf(=k_region*fanout もしくは k_point)のブロックを fanout 個ずつ束ねて親Regionに復元する。

    return:
      parent_blocks: len=k_region のリスト。各要素は点インデックスのリスト。
    """
    # まず返り値の中から「親Regionらしいブロック」を探す
    for item in hkm_out:
        if isinstance(item, (list, tuple)) and len(item) == k_region:
            ok = True
            for x in item:
                if not isinstance(x, (list, tuple, np.ndarray)):
                    ok = False
                    break
            if ok:
                parent_blocks = []
                for x in item:
                    parent_blocks.append([int(i) for i in list(x)])
                return parent_blocks

    # 親が見つからない場合は leaf を束ねて親を作る
    leaf = hkm_out[0]
    if not isinstance(leaf, (list, tuple)):
        raise RuntimeError("hierarchical_kmeans output is not list/tuple; cannot infer blocks.")

    expected_leaf_len = k_region * fanout
    if len(leaf) != expected_leaf_len:
        raise RuntimeError(
            f"Cannot infer parent regions: leaf length {len(leaf)} != k_region*fanout {expected_leaf_len}."
        )

    parent_blocks = [[] for _ in range(k_region)]
    for leaf_id, idxs in enumerate(leaf):
        pid = leaf_id // fanout
        parent_blocks[pid].extend([int(i) for i in list(idxs)])

    # 重複があれば除去（安全策）
    for pid in range(k_region):
        if len(parent_blocks[pid]) > 0:
            parent_blocks[pid] = list(dict.fromkeys(parent_blocks[pid]))

    return parent_blocks


def _build_region_id_per_point(pc: np.ndarray, k_region: int, k_point: int) -> Tuple[np.ndarray, int]:
    """
    hierarchical_kmeans を使い、各点の region_id を作る。
    **必ず Region-SHAP の親k_region領域で集計する（leafは使わない）。**

    IMPORTANT:
    - 2048点でも A/B 分割はせず、全体に対して k_region の親領域を作る。
      これによりヒストグラムは常に k_region 本（例:32本）になる。

    return:
      region_id_per_point: (num_points,) int
      num_regions: int (=k_region)
    """
    num_points = pc.shape[0]

    if k_point % k_region != 0:
        raise ValueError("division_point must be a multiple of division_region")
    fanout = k_point // k_region

    # 2048でも分割せずフルで親領域のIDを作る
    pc_use = pc.copy()

    # hierarchical_kmeansの設計が1024前提ならここで合わせる（安全策）
    # ただし 2048 の場合は 2048 のまま処理する
    if num_points != 2048:
        if pc_use.shape[0] > 1024:
            pc_use = pc_use[:1024]
        elif pc_use.shape[0] < 1024:
            pad = 1024 - pc_use.shape[0]
            pc_use = np.pad(pc_use, ((0, pad), (0, 0)), mode="constant")

    hkm_out = hierarchical_kmeans(pc_use, k_region, fanout)
    parent_blocks = _extract_parent_region_blocks(hkm_out, k_region, fanout)

    # region_id を全点分に割り当て
    region_id = np.full((pc_use.shape[0],), -1, dtype=np.int32)
    for rid, idxs in enumerate(parent_blocks):
        region_id[np.asarray(idxs, dtype=np.int32)] = rid

    # pc_use のサイズが元pcと違う場合は、元pcの先頭に対応させる
    if region_id.shape[0] != num_points:
        # 元pcが1024より大きい場合は切り詰めに対応
        if num_points > region_id.shape[0]:
            # 先頭 region_id を元pcの先頭に適用し、それ以降は最後のIDで埋める
            last_id = int(region_id[-1]) if region_id.size > 0 else -1
            full_region_id = np.full((num_points,), last_id, dtype=np.int32)
            full_region_id[:region_id.shape[0]] = region_id
            region_id = full_region_id
        else:
            region_id = region_id[:num_points]

    num_regions = k_region
    return region_id, num_regions


def _load_gate_mlp_if_needed(args, device) -> Tuple[Optional[AdaptiveGateMLP], Optional[dict]]:
    """
    ama_mode=mlp かつ gate_ckpt がある場合のみ Gate-MLP と scaler を読む。
    """
    gate_model = None
    gate_scaler = None
    if args.ama and args.ama_mode == "mlp" and args.gate_ckpt:
        gate_model = AdaptiveGateMLP(hidden=args.gate_hidden)
        gate_model.load_state_dict(torch.load(args.gate_ckpt, map_location="cpu"))
        gate_model.to(device).eval()

        if args.gate_scaler and os.path.exists(args.gate_scaler):
            sc = np.load(args.gate_scaler)
            gate_scaler = {
                "mu": sc["mu"].astype(np.float32),
                "std": sc["std"].astype(np.float32),
            }
        else:
            print("[WARN] MLP mode without scaler: falling back to raw features")
    return gate_model, gate_scaler


def _compute_hybrid_from_mid(region_per_pt: np.ndarray, point_adj: np.ndarray,
                            N: int, args, gate_model, gate_scaler, device) -> Tuple[np.ndarray, float, float]:
    """
    MID材料から、Nに応じた hybrid_shap を再合成。
    return:
      hybrid_shap (len=num_points)
      alpha, beta
    """
    var_reg = float(np.var(region_per_pt))
    var_pnt = float(np.var(point_adj))

    if args.ama:
        alpha, beta = gate_forward(N, var_reg, var_pnt, args, gate_model, device, gate_scaler)
    else:
        alpha, beta = 0.5, 1.0

    hybrid = alpha * region_per_pt + (1.0 - alpha) * beta * point_adj
    return hybrid.astype(np.float32), float(alpha), float(beta)


def _bincount_hist(region_ids: np.ndarray, top_indices: np.ndarray, num_regions: int) -> np.ndarray:
    selected_regions = region_ids[top_indices]
    hist = np.bincount(selected_regions, minlength=num_regions).astype(np.int32)
    return hist


def _write_per_sample_tsv(out_path: str, rows: List[dict]):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join([
            "sample_id", "label", "N", "num_regions",
            "nonempty_regions", "max_count", "min_nonzero",
            "alpha", "beta", "counts_csv"
        ]) + "\n")
        for r in rows:
            f.write("\t".join([
                str(r["sample_id"]),
                str(r["label"]),
                str(r["N"]),
                str(r["num_regions"]),
                str(r["nonempty_regions"]),
                str(r["max_count"]),
                str(r["min_nonzero"]),
                f'{r["alpha"]:.6f}' if r["alpha"] is not None else "NA",
                f'{r["beta"]:.6f}'  if r["beta"]  is not None else "NA",
                r["counts_csv"],
            ]) + "\n")


def _write_summary_tsv(out_path: str, summary: Dict[int, dict]):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join([
            "N", "num_regions",
            "nonempty_mean", "nonempty_std",
            "maxcount_mean", "maxcount_std",
            "counts_mean_csv", "counts_std_csv"
        ]) + "\n")
        for N in sorted(summary.keys()):
            s = summary[N]
            f.write("\t".join([
                str(N),
                str(s["num_regions"]),
                f'{s["nonempty_mean"]:.6f}',
                f'{s["nonempty_std"]:.6f}',
                f'{s["maxcount_mean"]:.6f}',
                f'{s["maxcount_std"]:.6f}',
                s["counts_mean_csv"],
                s["counts_std_csv"],
            ]) + "\n")


def _plot_mean_hist(N: int, counts_mean: np.ndarray, out_png: str, title_extra: str = ""):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # 既存があれば必ず上書き（明示的に削除）
    if os.path.exists(out_png):
        try:
            os.remove(out_png)
        except Exception:
            pass

    x = np.arange(len(counts_mean))
    plt.figure(figsize=(8.0, 3.8))
    plt.bar(x, counts_mean)
    plt.xlabel("Region ID (parent / global)")
    plt.ylabel("Mean selected points")
    plt.title(f"Mean Region Histogram @ N={N} {title_extra}".strip())
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Analyze which Regions are chosen by Hybrid for each N.")
    ap.add_argument("--mid_cache_dir", type=str, required=True,
                    help="MID cache directory containing npz with region_per_pt/point_adj/pc/label.")
    ap.add_argument("--dataset", type=str, default="",
                    help="Dataset root including test_files.txt; used only if MID npz does not contain pc.")
    ap.add_argument("--division_region", type=int, default=32)
    ap.add_argument("--division_point", type=int, default=64)
    ap.add_argument("--n_list", type=int, nargs="+", default=[300, 400, 500, 600, 700, 800, 900, 1000])
    # AMA/Gate options (must match inference behavior)
    ap.add_argument("--ama", action="store_true")
    ap.add_argument("--ama_mode", choices=["heuristic", "mlp"], default="heuristic")
    ap.add_argument("--ama_eps", type=float, default=1e-6)
    ap.add_argument("--ama_Nmid", type=int, default=300)
    ap.add_argument("--ama_k", type=float, default=4.0)
    ap.add_argument("--gate_ckpt", type=str, default="")
    ap.add_argument("--gate_hidden", type=int, default=8)
    ap.add_argument("--gate_scaler", type=str, default="")
    # subset selection (optional)
    ap.add_argument("--selected_classes", type=int, nargs="*", default=[],
                    help="If specified, only analyze these class indices.")
    ap.add_argument("--max_samples", type=int, default=10**9,
                    help="Upper bound of npz samples to analyze.")
    # output
    ap.add_argument("--out_dir", type=str, default="result/_region_hist_debug")
    ap.add_argument("--plot", action="store_true",
                    help="Save mean histograms as png for each N.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_region = args.division_region
    k_point = args.division_point
    if k_point % k_region != 0:
        raise ValueError("--division_point must be a multiple of --division_region")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # MID files
    mid_files = _load_mid_npz_list(args.mid_cache_dir)
    if args.max_samples < len(mid_files):
        mid_files = mid_files[:args.max_samples]

    # dataset preload (if needed)
    dataset_dict = None
    if args.dataset:
        dataset_dict = _load_pc_from_dataset(args.dataset)

    # Gate-MLP if needed
    gate_model, gate_scaler = _load_gate_mlp_if_needed(args, device)

    per_sample_rows = []
    perN_hists: Dict[int, List[np.ndarray]] = {N: [] for N in args.n_list}
    perN_nonempty: Dict[int, List[int]] = {N: [] for N in args.n_list}
    perN_maxcount: Dict[int, List[int]] = {N: [] for N in args.n_list}

    print(f"[INFO] Analyze MID npz: {len(mid_files)} files")
    print(f"[INFO] N list: {args.n_list}")
    print(f"[INFO] AMA: {args.ama} mode={args.ama_mode}")
    print(f"[INFO] Using parent regions only: k_region={k_region}")

    for t, mid_path in enumerate(mid_files):
        sample_id = _parse_sample_id_from_mid_path(mid_path)
        with np.load(mid_path) as z:
            region_per_pt = np.asarray(z["region_per_pt"], dtype=np.float32).reshape(-1)
            point_adj     = np.asarray(z["point_adj"], dtype=np.float32).reshape(-1)
            lab           = int(np.asarray(z["label"]).reshape(-1)[0]) if "label" in z.files else -1

            if args.selected_classes and (lab not in args.selected_classes):
                continue

            pc = _try_load_pc_from_mid(z)

        if pc is None:
            if dataset_dict is None:
                print(f"[SKIP] pc not found in MID and no --dataset given: {sample_id}")
                continue
            fname, idx = _extract_file_and_index(sample_id)
            if fname not in dataset_dict or idx < 0:
                print(f"[SKIP] cannot map sample_id to dataset: {sample_id}")
                continue
            data, labels = dataset_dict[fname]
            if idx >= data.shape[0]:
                print(f"[SKIP] index out of range for {fname}: {idx}")
                continue
            pc = data[idx]

        pc = pc.astype(np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            print(f"[SKIP] invalid pc shape: {pc.shape} id={sample_id}")
            continue

        # (1) make region id per point (Region-SHAPの親k_region領域)
        region_ids, num_regions = _build_region_id_per_point(pc, k_region, k_point)

        # (2) for each N, compute hybrid and hist
        for N in args.n_list:
            hybrid, alpha, beta = _compute_hybrid_from_mid(
                region_per_pt, point_adj, N, args, gate_model, gate_scaler, device
            )

            top_indices = np.argsort(hybrid)[::-1][:N]

            hist = _bincount_hist(region_ids, top_indices, num_regions)
            nonempty = int(np.count_nonzero(hist))
            maxc = int(hist.max()) if hist.size > 0 else 0
            minnz = int(hist[hist > 0].min()) if nonempty > 0 else 0

            per_sample_rows.append({
                "sample_id": sample_id,
                "label": lab,
                "N": N,
                "num_regions": num_regions,
                "nonempty_regions": nonempty,
                "max_count": maxc,
                "min_nonzero": minnz,
                "alpha": alpha,
                "beta": beta,
                "counts_csv": ",".join(str(int(x)) for x in hist.tolist()),
            })

            perN_hists[N].append(hist.astype(np.float32))
            perN_nonempty[N].append(nonempty)
            perN_maxcount[N].append(maxc)

        if (t + 1) % 50 == 0:
            print(f"[INFO] processed {t+1}/{len(mid_files)} samples")
        gc.collect()

    per_sample_tsv = os.path.join(out_dir, "region_hist_per_sample.tsv")
    _write_per_sample_tsv(per_sample_tsv, per_sample_rows)
    print(f"[SAVE] per-sample hist -> {per_sample_tsv}")

    summary = {}
    for N in args.n_list:
        hlist = perN_hists[N]
        if len(hlist) == 0:
            continue
        Lmax = max(h.shape[0] for h in hlist)
        Hmat = np.stack([
            np.pad(h, (0, Lmax - h.shape[0]), mode="constant")
            for h in hlist
        ], axis=0)

        counts_mean = Hmat.mean(axis=0)
        counts_std  = Hmat.std(axis=0)

        nonempty_arr = np.asarray(perN_nonempty[N], dtype=np.float32)
        maxcount_arr = np.asarray(perN_maxcount[N], dtype=np.float32)

        summary[N] = {
            "num_regions": int(Lmax),
            "nonempty_mean": float(nonempty_arr.mean()),
            "nonempty_std": float(nonempty_arr.std()),
            "maxcount_mean": float(maxcount_arr.mean()),
            "maxcount_std": float(maxcount_arr.std()),
            "counts_mean_csv": ",".join(f"{x:.6f}" for x in counts_mean.tolist()),
            "counts_std_csv": ",".join(f"{x:.6f}" for x in counts_std.tolist()),
            "counts_mean_arr": counts_mean,
        }

        if args.plot:
            out_png = os.path.join(out_dir, "plots", f"mean_hist_N{N}.png")
            _plot_mean_hist(N, counts_mean, out_png)

    summary_tsv = os.path.join(out_dir, "region_hist_summary.tsv")
    _write_summary_tsv(summary_tsv, summary)
    print(f"[SAVE] summary hist -> {summary_tsv}")

    if args.plot:
        print(f"[SAVE] plots -> {os.path.join(out_dir, 'plots')}")


if __name__ == "__main__":
    main()
