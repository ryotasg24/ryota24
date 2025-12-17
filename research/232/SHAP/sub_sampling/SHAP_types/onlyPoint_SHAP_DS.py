#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import gc
import time
import argparse

import numpy as np
import h5py
import torch
from sklearn.cluster import KMeans

# --- PointNeXt 本体のルートを通す ---
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig

# --- provider.py (PointNet 時代のもの) ---
import provider

# ======================================================
# ヘルパー関数
# ======================================================

def save_h5_data(h5_filename, data, label):
    """HDF5ファイルにデータとラベルを保存する"""
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)

def get_class_names(file_path):
    """クラス名が記述されたファイルからクラス名のリストを取得する"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def read_file_list(list_path, data_root):
    """ファイルリストを読み込み、絶対パスのリストを返す"""
    files = []
    with open(list_path, 'r') as f:
        for line in f:
            rel = line.strip()
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            files.append(os.path.join(data_root, rel))
    return files

def spatial_divide_into_blocks(points, num_blocks):
    """K-Meansで点群を空間的にブロック分割する"""
    # 古いscikit-learnとの互換性のため n_init パラメータを指定しない
    kmeans = KMeans(n_clusters=num_blocks, random_state=42)
    labels = kmeans.fit_predict(points)

    blocks = []
    for i in range(num_blocks):
        block_indices = np.where(labels == i)[0].tolist()
        blocks.append(block_indices)
    return blocks, kmeans

def collect_background_point_clouds(target_label, train_files, num_point_clouds=50):
    """指定クラスの学習用点群を背景データとして収集する"""
    bgs = []
    for fp in train_files:
        data, labels = provider.loadDataFile(fp)
        labels = labels.flatten()
        for idx in np.where(labels == target_label)[0]:
            pc = data[idx]
            if pc.shape[0] > 1024:
                sel = np.random.choice(pc.shape[0], 1024, replace=False)
                pc = pc[sel, :]
            elif pc.shape[0] < 1024:
                continue
            bgs.append(pc)
            if len(bgs) >= num_point_clouds:
                return np.array(bgs)
    return np.array(bgs)

def load_pointnext_model(cfg_file, checkpoint_path, device):
    """PointNeXtモデルをロードする"""
    cfg = EasyConfig(); cfg.load(cfg_file)
    model = build_model_from_cfg(cfg["model"])
    model.to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    sd = ck.get('model_state_dict', ck.get('model', ck))
    model.load_state_dict(sd)
    model.eval()
    return model

def compute_vectorized_baseline(background_point_clouds, kmeans):
    """背景ベースラインをベクトル化して効率的に計算する"""
    if background_point_clouds.ndim < 3: return [] # 背景データがない場合
    B, N, _ = background_point_clouds.shape
    X = background_point_clouds.reshape(B * N, 3)
    lbl = kmeans.predict(X)
    K = kmeans.n_clusters

    out = []
    for i in range(K):
        sel = (lbl == i)
        out.append(X[sel].mean(axis=0) if np.any(sel) else np.zeros(3, dtype=X.dtype))
    return out

def compute_point_level_contributions(point_cloud, baseline_blocks, blocks, model, target_class_index, device):
    """Raw-SHAP（勾配ベースの生の寄与度）を計算する"""
    model.eval()
    t = torch.tensor(point_cloud, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
    t.retain_grad()

    out = model(t)
    score = out[0, target_class_index]
    score.backward()
    grad = t.grad.squeeze(0).cpu().numpy()

    contrib = np.zeros(point_cloud.shape[0], dtype=np.float32)
    for i, b_indices in enumerate(blocks):
        if not b_indices: continue
        base = baseline_blocks[i]
        diff = point_cloud[b_indices] - base
        contrib[b_indices] = np.sum(grad[b_indices] * diff, axis=1) # dot product
    return contrib

def z_score_normalize_in_blocks(raw_contribs, blocks):
    """生の寄与度をブロック内でZ-score正規化する"""
    z_scored_shap = np.zeros_like(raw_contribs)
    for b_indices in blocks:
        if not b_indices: continue
        vals = raw_contribs[b_indices]
        mu, sigma = vals.mean(), vals.std()

        if sigma < 1e-8:
            z_scored_shap[b_indices] = 0
        else:
            z_scored_shap[b_indices] = (vals - mu) / sigma
    return z_scored_shap

def l1_normalize_shap(shap_values, eps=1e-6):
    """SHAP値をL1正規化する"""
    l1_norm = np.sum(np.abs(shap_values))
    return shap_values / (l1_norm + eps)

# ======================================================
# main関数
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Downsample point clouds using Point-SHAP (optional L1 normalization).")
    parser.add_argument("--ds_points", type=int, default=500, help="ダウンサンプリング後の点群数 (default: 500)")
    parser.add_argument("--division_point", type=int, default=32, help="Point-SHAPのZ-score正規化で使う空間分割数 (k_point) (default: 32)")
    parser.add_argument("--l1_eps", type=float, default=1e-6, help="L1正規化のゼロ除算を防ぐためのε (default: 1e-6)")
    # L1 正規化の ON/OFF 切替（デフォルト: ON）
    parser.add_argument("--no_l1", dest="use_l1", action="store_false", help="L1 正規化を無効化（デフォルトは有効）")
    parser.set_defaults(use_l1=True)
    parser.add_argument("--cache_mode", type=str, default="auto", choices=["auto", "force", "disable"], help="キャッシュモード: 'auto'=あれば利用, 'force'=再計算して保存, 'disable'=利用しない")
    parser.add_argument("--num_groups", type=int, default=1, help="処理を並列化するためのクラス分割数 (default: 40)")
    parser.add_argument("--group_index", type=int, default=0, help="処理対象とするクラスグループのインデックス (0-indexed, default: 0)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 設定と準備 ---
    DATA_ROOT = "/workspace/PointNeXt/data"
    DATA_DIR = os.path.join(DATA_ROOT, "modelnet40_ply_hdf5_2048")

    k_point = args.division_point
    # 出力/キャッシュパス（L1 の有無でタグを変える）
    tag = "L1" if args.use_l1 else "noL1"
    output_base = f"/workspace/PointNeXt/result/SHAP_Types/PointSHAP_kp{k_point}_{tag}"
    cache_base_dir = f"/workspace/PointNeXt/result/_shap_cache_global/point_kp{k_point}_{tag}"

    output_folder = os.path.join(output_base, str(args.ds_points))
    cache_folder = cache_base_dir

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)
    print(f"Results will be saved in: {output_folder}")
    print(f"Cache will be stored in: {cache_folder}")

    class_names = get_class_names(os.path.join(DATA_DIR, "shape_names.txt"))
    train_files = read_file_list(os.path.join(DATA_DIR, "train_files.txt"), DATA_ROOT)
    test_files = read_file_list(os.path.join(DATA_DIR, "test_files.txt"), DATA_ROOT)

    groups = np.array_split(range(len(class_names)), args.num_groups)
    selected_classes = groups[args.group_index]

    model = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)
    backgrounds = {cls: collect_background_point_clouds(cls, train_files, 50) for cls in selected_classes}

    timing_stats = {cls: {'shap_time': 0.0, 'total_time': 0.0, 'count': 0} for cls in selected_classes}

    # --- 2. メインループ ---
    for tf in test_files:
        print(f"\nProcessing file: {os.path.basename(tf)}")
        data, labels = provider.loadDataFile(tf)
        labels = labels.flatten()

        downsampled_pcs, downsampled_labels = [], []

        for i, (pc, cls) in enumerate(zip(data, labels)):
            if cls not in selected_classes: continue

            bg_pc = backgrounds.get(cls)
            if bg_pc is None or bg_pc.size == 0: continue

            print(f"  -> Processing sample {i} (Class: {class_names[cls]})")

            t_start = time.time()

            cache_filename = f"{os.path.basename(tf).replace('.h5', '')}_{i}_cls{cls}.npz"
            cache_path = os.path.join(cache_folder, cache_filename)
            final_shap_values = None

            if args.cache_mode == "auto" and os.path.exists(cache_path):
                print("     - Loading SHAP values from cache.")
                try:
                    final_shap_values = np.load(cache_path)['shap']
                    t_shap_end = time.time()
                except Exception as e:
                    print(f"     - Cache loading failed: {e}. Recomputing...")

            if final_shap_values is None:
                if args.cache_mode != 'disable': print("     - Computing Point-SHAP values...")

                def compute_shap_for_part(part_pc):
                    blocks, kmeans = spatial_divide_into_blocks(part_pc, args.division_point)
                    baseline_blocks = compute_vectorized_baseline(bg_pc, kmeans)

                    raw_contribs = compute_point_level_contributions(part_pc, baseline_blocks, blocks, model, cls, device)
                    z_scored_shap = z_score_normalize_in_blocks(raw_contribs, blocks)
                    # L1 正規化はオプション
                    return (l1_normalize_shap(z_scored_shap, args.l1_eps)
                            if args.use_l1 else z_scored_shap)

                if pc.shape[0] == 2048:
                    shap_A = compute_shap_for_part(pc[:1024])
                    shap_B = compute_shap_for_part(pc[1024:])
                    final_shap_values = np.concatenate([shap_A, shap_B])
                else:
                    work_pc = pc.copy()
                    if work_pc.shape[0] != 1024:
                        if work_pc.shape[0] > 1024:
                            sel = np.random.choice(work_pc.shape[0], 1024, replace=False)
                            work_pc = work_pc[sel]
                        else:
                            pad = 1024 - work_pc.shape[0]
                            work_pc = np.pad(work_pc, ((0, pad), (0, 0)), mode="constant")
                    final_shap_values = compute_shap_for_part(work_pc)

                t_shap_end = time.time()

                if args.cache_mode != 'disable':
                    print("     - Saving SHAP values to cache.")
                    np.savez_compressed(cache_path, shap=final_shap_values)

            # --- 3. ダウンサンプリング ---
            sorted_indices = np.argsort(final_shap_values)[::-1]
            ds_pc = pc[sorted_indices[:args.ds_points]]
            t_end = time.time()

            timing_stats[cls]['shap_time'] += (t_shap_end - t_start)
            timing_stats[cls]['total_time'] += (t_end - t_start)
            timing_stats[cls]['count'] += 1

            downsampled_pcs.append(ds_pc.astype('float32'))
            downsampled_labels.append(cls)

        if downsampled_pcs:
            rel_path = os.path.relpath(tf, DATA_ROOT)
            output_h5_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

            new_data  = np.stack(downsampled_pcs).astype('float32')
            new_label = np.array(downsampled_labels).reshape(-1, 1)

            if os.path.exists(output_h5_path):
                with h5py.File(output_h5_path, "r") as f:
                    exist_data  = f["data"][:]
                    exist_label = f["label"][:]
                merged_data  = np.concatenate([exist_data,  new_data],  axis=0)
                merged_label = np.concatenate([exist_label, new_label], axis=0)
                save_h5_data(output_h5_path, merged_data, merged_label)
            else:
                save_h5_data(output_h5_path, new_data, new_label)
            print(f"Saved {len(downsampled_pcs)} samples to {output_h5_path}")
        gc.collect()

    # --- 4. 最終的な時間統計を出力 ---
    stats_file = os.path.join(output_folder, f"processing_times_group_{args.group_index}.txt")
    with open(stats_file, 'w') as f:
        f.write("Class\tDS_Points\tDivision_Point\tAvg_SHAP(sec)\tAvg_Total(sec)\tCount\n")
        total_shap_time, total_time, total_count = 0.0, 0.0, 0
        for cls_idx in sorted(timing_stats.keys()):
            stats, count = timing_stats[cls_idx], timing_stats[cls_idx]['count']
            if count > 0:
                avg_shap = stats['shap_time'] / count
                avg_total = stats['total_time'] / count
                f.write(f"{class_names[cls_idx]}\t{args.ds_points}\t{args.division_point}\t{avg_shap:.6f}\t{avg_total:.6f}\t{count}\n")
                total_shap_time += stats['shap_time']
                total_time += stats['total_time']
                total_count += count
        if total_count > 0:
            avg_shap_all = total_shap_time / total_count
            avg_total_all = total_time / total_count
            f.write(f"ALL\t{args.ds_points}\t{args.division_point}\t{avg_shap_all:.6f}\t{avg_total_all:.6f}\t{total_count}\n")
    print(f"\nProcessing finished. Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()