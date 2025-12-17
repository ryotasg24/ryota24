#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
animate_Region_StackHistogram.py

目的：
- 1つのサンプルに対して、Hybrid-SHAP による点選択が
  「N=1 から target_N まで増えていくとき、
   各Regionの棒がどのように積み重なっていくか」をアニメーション化する。

前提：
- histogram_Region_ChoicePoint.py と同じ MID キャッシュ（region_per_pt, point_adj, pc, label）を利用。
- Region分割は Region-SHAP の親32領域（k_region=32, k_point=64なら親32）を使用。
- target_N のときの Gate (α,β) で hybrid_shap を計算し、
  その順位に沿って 1点ずつ（or stepごと）追加していく。

出力：
- --out_gif で指定したパスに gif を保存（既存ファイルは上書き）。
- --out_png で指定したパス（未指定時は out_gif のベース名 + "_final.png"）に、
  最終的な N=target_N のヒストグラムを PNG で保存（既存ファイルは上書き）。
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
import torch

sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from AMA_SHAP_for_PointNeXt import (
    hierarchical_kmeans,
    gate_forward,
    AdaptiveGateMLP,
)

import provider


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


def _extract_parent_region_blocks(hkm_out: Tuple, k_region: int, fanout: int) -> List[List[int]]:
    """
    hierarchical_kmeans の返り値から親Region(=k_region)のブロックを抽出する。

    優先度：
    1) 返り値タプルの中に「長さ k_region の list-of-lists」があればそれを親Regionとして採用。
    2) 無ければ、leaf(=k_region*fanout)のブロックを fanout 個ずつ束ねて親Regionに復元する。
    """
    # 親Regionらしいブロックを探す
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
    if len(leaf) != k_region * fanout:
        raise RuntimeError(
            f"Cannot infer parent regions: leaf length {len(leaf)} != k_region*fanout {k_region*fanout}."
        )

    parent_blocks = [[] for _ in range(k_region)]
    for leaf_id, idxs in enumerate(leaf):
        pid = leaf_id // fanout
        parent_blocks[pid].extend([int(i) for i in list(idxs)])

    # 重複があれば除去
    for pid in range(k_region):
        if len(parent_blocks[pid]) > 0:
            parent_blocks[pid] = list(dict.fromkeys(parent_blocks[pid]))

    return parent_blocks


def _build_region_id_per_point(pc: np.ndarray, k_region: int, k_point: int) -> Tuple[np.ndarray, int]:
    """
    Region-SHAP の親32領域で region_id を作成する（A/B分割は行わず、全体で k_region）。
    """
    if k_point % k_region != 0:
        raise ValueError("division_point must be a multiple of division_region")
    fanout = k_point // k_region

    pc_use = pc.astype(np.float32)
    hkm_out = hierarchical_kmeans(pc_use, k_region, fanout)
    reg_parent = _extract_parent_region_blocks(hkm_out, k_region, fanout)

    region_id = np.full((pc_use.shape[0],), -1, dtype=np.int32)
    for rid, idxs in enumerate(reg_parent):
        region_id[np.asarray(idxs, dtype=np.int32)] = rid

    num_regions = k_region
    return region_id, num_regions


def _load_gate_mlp_if_needed(args, device) -> Tuple[Optional[AdaptiveGateMLP], Optional[dict]]:
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
    var_reg = float(np.var(region_per_pt))
    var_pnt = float(np.var(point_adj))

    if args.ama:
        alpha, beta = gate_forward(N, var_reg, var_pnt, args, gate_model, device, gate_scaler)
    else:
        alpha, beta = 0.5, 1.0

    hybrid = alpha * region_per_pt + (1.0 - alpha) * beta * point_adj
    return hybrid.astype(np.float32), float(alpha), float(beta)


def main():
    ap = argparse.ArgumentParser("Animate cumulative Region histogram for one sample.")
    ap.add_argument("--mid_cache_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="",
                    help="Dataset root including test_files.txt; used if pc not stored in MID.")
    ap.add_argument("--division_region", type=int, default=32)
    ap.add_argument("--division_point", type=int, default=64)
    ap.add_argument("--target_N", type=int, default=300,
                    help="最終的に取りたい点数（このNでGateを評価し、1..Nまで積み上げる）")
    ap.add_argument("--step", type=int, default=1,
                    help="Nを何点刻みで増やすか（1なら1点ずつ）")
    ap.add_argument("--sample_index", type=int, default=0,
                    help="mid_cache_dir 内で何番目のnpzを使うか（0-based）")
    # AMA/Gate options
    ap.add_argument("--ama", action="store_true")
    ap.add_argument("--ama_mode", choices=["heuristic", "mlp"], default="heuristic")
    ap.add_argument("--ama_eps", type=float, default=1e-6)
    ap.add_argument("--ama_Nmid", type=int, default=300)
    ap.add_argument("--ama_k", type=float, default=4.0)
    ap.add_argument("--gate_ckpt", type=str, default="")
    ap.add_argument("--gate_hidden", type=int, default=8)
    ap.add_argument("--gate_scaler", type=str, default="")
    # 出力
    ap.add_argument("--out_gif", type=str, default="script_Analysis/Histogram/hist_result/SD-GLM/animation/region_stack_anim.gif")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out_png", type=str, default="",
                    help="最終フレーム(N=target_N)のヒストグラムPNGの保存先。"
                         "指定なしの場合は out_gif のベース名 + '_final.png'。")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_region = args.division_region
    k_point = args.division_point
    if k_point % k_region != 0:
        raise ValueError("--division_point must be a multiple of --division_region")

    mid_files = _load_mid_npz_list(args.mid_cache_dir)
    if args.sample_index < 0 or args.sample_index >= len(mid_files):
        raise RuntimeError(f"sample_index {args.sample_index} out of range (0..{len(mid_files)-1})")

    mid_path = mid_files[args.sample_index]
    sample_id = _parse_sample_id_from_mid_path(mid_path)
    print(f"[INFO] Use sample #{args.sample_index}: {sample_id}")

    dataset_dict = None
    if args.dataset:
        dataset_dict = _load_pc_from_dataset(args.dataset)

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

    # Region-SHAP 親32領域で region id を作成
    region_ids, num_regions = _build_region_id_per_point(pc, k_region, k_point)
    print(f"[INFO] num_regions (parent): {num_regions}")

    # Gate-MLP 読み込み
    gate_model, gate_scaler = _load_gate_mlp_if_needed(args, device)

    # target_N で hybrid_shap を計算（この順位で1..Nまで積み上げる）
    hybrid, alpha, beta = _compute_hybrid_from_mid(
        region_per_pt, point_adj, args.target_N, args, gate_model, gate_scaler, device
    )
    print(f"[INFO] alpha={alpha:.6f}, beta={beta:.6f}")

    sorted_indices = np.argsort(hybrid)[::-1][:args.target_N]
    sorted_regions = region_ids[sorted_indices]

    # フレームごとの N のリスト
    step = max(1, args.step)
    frame_N_list = list(range(1, args.target_N + 1, step))
    if frame_N_list[-1] != args.target_N:
        frame_N_list.append(args.target_N)

    x = np.arange(num_regions)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    bars = ax.bar(x, np.zeros_like(x, dtype=np.float32))
    ax.set_xlabel("Region ID (parent)")
    ax.set_ylabel("Selected points")
    ax.set_ylim(0, args.target_N)
    title_txt = ax.set_title(f"Sample {sample_id}, N=0 / target_N={args.target_N}")

    def init():
        for rect in bars:
            rect.set_height(0.0)
        title_txt.set_text(f"Sample {sample_id}, N=0 / target_N={args.target_N}")
        return list(bars) + [title_txt]

    def update(N_current: int):
        hist = np.bincount(sorted_regions[:N_current], minlength=num_regions)
        for rect, h in zip(bars, hist):
            rect.set_height(float(h))
        title_txt.set_text(
            f"Sample {sample_id}, N={N_current} / target_N={args.target_N} (class={lab})"
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

    out_gif = args.out_gif
    out_dir = os.path.dirname(out_gif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_gif):
        os.remove(out_gif)

    writer = PillowWriter(fps=args.fps)
    ani.save(out_gif, writer=writer)
    plt.close(fig)

    print(f"[SAVE] animation gif -> {out_gif}")

    # 最終フレーム (N=target_N) の静的ヒストグラム PNG を保存
    if args.out_png and len(args.out_png.strip()) > 0:
        out_png = args.out_png
    else:
        base, _ = os.path.splitext(out_gif)
        out_png = base + "_final.png"

    out_png_dir = os.path.dirname(out_png)
    if out_png_dir:
        os.makedirs(out_png_dir, exist_ok=True)
    if os.path.exists(out_png):
        os.remove(out_png)

    hist_final = np.bincount(sorted_regions[:args.target_N], minlength=num_regions)
    x2 = np.arange(num_regions)
    fig2, ax2 = plt.subplots(figsize=(8.0, 3.8))
    ax2.bar(x2, hist_final)
    ax2.set_xlabel("Region ID (parent)")
    ax2.set_ylabel("Selected points")
    ax2.set_ylim(0, args.target_N)
    ax2.set_title(
        f"Final Region Histogram, sample {sample_id}, N={args.target_N} (class={lab})"
    )
    fig2.tight_layout()
    fig2.savefig(out_png, dpi=200)
    plt.close(fig2)

    print(f"[SAVE] final histogram png -> {out_png}")


if __name__ == "__main__":
    main()
