#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import gc
import time
import argparse

import numpy as np
import shap
import h5py
from sklearn.cluster import KMeans
import torch

# --- PointNeXt 本体のルートを通す ---
# ご自身の環境に合わせてパスを修正してください
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig

# --- provider.py (PointNet 時代のもの) ---
# このスクリプトと同じディレクトリに配置するか、PYTHONPATHを通してください
import provider

# ======================================================
# グローバル設定
# ======================================================
# データセットのルートディレクトリ
DATA_ROOT = "/workspace/PointNeXt/data"
DATA_DIR  = os.path.join(DATA_ROOT, "modelnet40_ply_hdf5_2048")
# ======================================================


# ======================================================
# ヘルパー関数
# ======================================================

def read_file_list(list_path):
    """ファイルリストを読み込み、絶対パスのリストを返す"""
    files = []
    with open(list_path, 'r') as f:
        for line in f:
            rel = line.strip()
            # "data/" プレフィックスがあれば削除
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            files.append(os.path.join(DATA_ROOT, rel))
    return files

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

def spatial_divide_into_blocks(points, num_blocks, kmeans=None):
    """
    K-Meansクラスタリングを用いて点群を空間的にブロック分割する
    ★★★ エラーの原因となった n_init='auto' を削除し、古いscikit-learnに対応 ★★★
    """
    if kmeans is None:
        # scikit-learnの古いバージョンとの互換性のために n_init パラメータを指定しない
        kmeans = KMeans(n_clusters=num_blocks, random_state=42)
        labels = kmeans.fit_predict(points)
    else:
        labels = kmeans.predict(points)

    blocks = []
    for i in range(num_blocks):
        block_indices = np.where(labels == i)[0].tolist()
        blocks.append(block_indices)
    return blocks, kmeans

def collect_background_point_clouds(target_label, train_files, num_point_clouds=50):
    """指定クラスの学習用点群を背景データとして収集する"""
    backgrounds = []
    for fp in train_files:
        data, labels = provider.loadDataFile(fp)
        # ラベル配列を平坦化して処理
        idxs = np.where(labels.flatten() == target_label)[0]
        for i in idxs:
            pc = data[i]
            # 点群を1024点に正規化
            if pc.shape[0] > 1024:
                sel = np.random.choice(pc.shape[0], 1024, replace=False)
                pc = pc[sel, :]
            elif pc.shape[0] < 1024:
                continue  # 1024点未満はスキップ

            backgrounds.append(pc)
            if len(backgrounds) >= num_point_clouds:
                return np.array(backgrounds)
    return np.array(backgrounds)

def load_pointnext_model(cfg_file, checkpoint_path, device):
    """PointNeXtモデルを設定ファイルとチェックポイントから読み込む"""
    cfg = EasyConfig()
    cfg.load(cfg_file)
    model = build_model_from_cfg(cfg["model"])
    model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # state_dictのキーが異なる場合に対応
    sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(sd)
    model.eval()
    return model

def compute_background_baseline(background_point_clouds, blocks):
    """背景データセットから各ブロックのベースライン（平均座標）を計算する"""
    baseline_blocks = []
    num_blocks = len(blocks)
    # 各ブロックに対して平均値を計算
    for i in range(num_blocks):
        block_indices = blocks[i]
        if not block_indices:
            # 空のブロックの場合はゼロベクトルを追加
            baseline_blocks.append(np.zeros(3, dtype=np.float32))
            continue
        # 全ての背景点群からこのブロックに属する点を収集
        pts_in_block = background_point_clouds[:, block_indices, :] # (num_bg_pcs, num_pts_in_block, 3)
        # 各背景点群内での平均を計算
        per_pc_mean = pts_in_block.mean(axis=1) # (num_bg_pcs, 3)
        # それらの平均を計算して最終的なベースラインとする
        overall_mean = per_pc_mean.mean(axis=0) # (3,)
        baseline_blocks.append(overall_mean)
    return baseline_blocks

def _precompute_pointwise_maps(blocks, baseline_blocks):
    """高速化のため、各点がどのブロックに属し、そのベースラインは何かを事前計算する"""
    num_points = sum(len(b) for b in blocks)
    block_ids = np.empty(num_points, dtype=np.int64)
    for i, indices in enumerate(blocks):
        if not indices: continue
        block_ids[np.array(indices, dtype=np.int64)] = i

    baseline_coords = np.stack([baseline_blocks[b] for b in block_ids], axis=0).astype(np.float32)
    return block_ids, baseline_coords

def predict_masks_gpu_batch(mask_vectors, pc, block_ids, baseline_coords, model, device,
                            chunk_size: int = 256, min_chunk_size: int = 16, use_pin_mem: bool = True):
    """
    マスクされた点群の予測をGPU上でバッチ処理する（OOMエラー対応版）。
    """
    model.eval()
    num_masks = len(mask_vectors)
    if num_masks == 0:
        return np.array([])

    # データをPyTorch Tensorに変換し、GPUへ送る
    pc_t = torch.from_numpy(pc.astype(np.float32)).to(device)
    base_t = torch.from_numpy(baseline_coords).to(device)
    block_ids_t = torch.from_numpy(block_ids).to(device)

    all_preds = []
    current_pos = 0
    current_chunk_size = chunk_size

    with torch.no_grad():
        while current_pos < num_masks:
            end_pos = min(current_pos + current_chunk_size, num_masks)
            chunk_masks = mask_vectors[current_pos:end_pos]

            try:
                mask_np = np.asarray(chunk_masks, dtype=np.float32)
                if use_pin_mem and device.type == "cuda":
                    mask_t = torch.from_numpy(mask_np).pin_memory().to(device, non_blocking=True)
                else:
                    mask_t = torch.from_numpy(mask_np).to(device)

                keep_mask = mask_t[:, block_ids_t].unsqueeze(-1) # (Batch, N, 1)
                masked_pc_batch = pc_t.unsqueeze(0) * keep_mask + base_t.unsqueeze(0) * (1.0 - keep_mask)

                logits = model(masked_pc_batch)
                all_preds.append(logits.cpu().numpy())

                current_pos = end_pos # 正常に処理できたら次に進む

            except RuntimeError as e:
                if "out of memory" in str(e) and current_chunk_size > min_chunk_size:
                    if device.type == "cuda": torch.cuda.empty_cache()
                    new_chunk_size = max(min_chunk_size, current_chunk_size // 2)
                    print(f"  [WARN] CUDA OOM. Reducing chunk size from {current_chunk_size} to {new_chunk_size}.")
                    current_chunk_size = new_chunk_size
                else:
                    raise e # OOM以外、または最小チャンクサイズでも失敗した場合はエラーを投げる

    return np.concatenate(all_preds, axis=0)


# ======================================================
# main関数
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Downsample point clouds using Region-SHAP with updated, efficient methods.")
    # --- 基本設定 ---
    parser.add_argument("--ds_points", type=int, default=500, help="ダウンサンプリング後の点群数 (default: 500)")
    parser.add_argument("--pattern", type=int, default=10000, help="Region-SHAP の近似に使用するマスキングパターンの数 (nsamples) (default: 100)")
    parser.add_argument("--division_region", type=int, default=32, help="Region-SHAP で使用する空間分割数 (K-Meansのk) (default: 32)")
    # --- 処理とファイルI/O関連 ---
    parser.add_argument("--cache_mode", type=str, default="auto", choices=["auto", "force", "disable"], help="キャッシュモード: 'auto'=あれば利用, 'force'=再計算して保存, 'disable'=利用しない (default: auto)")
    parser.add_argument("--num_groups", type=int, default=1, help="処理を並列化するためのクラス分割数 (default: 40)")
    parser.add_argument("--group_index", type=int, default=0, help="処理対象とするクラスグループのインデックス (0-indexed, default: 0)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml", help="PointNeXtのモデル設定ファイルへのパス")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth", help="学習済みモデルのチェックポイントファイルへのパス")
    # --- GPU推論のパフォーマンス設定 ---
    parser.add_argument("--mask_chunk", type=int, default=256, help="SHAP計算時のマスク推論バッチサイズ (default: 256)")
    parser.add_argument("--min_mask_chunk", type=int, default=16, help="OOM発生時に自動縮小する最小チャンクサイズ (default: 16)")
    parser.add_argument("--no_pinmem", action="store_true", help="CPU->GPU転送でpinned memoryを使用しない場合に指定")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 設定と準備 ---
    k_region = args.division_region
    pattern = args.pattern

    OUTPUT_BASE = f"/workspace/PointNeXt/result/SHAP_Types/Region_kr{k_region}_p{pattern}"
    CACHE_BASE_DIR = f"/workspace/PointNeXt/result/_shap_cache_global/region_p{pattern}_kr{k_region}"

    output_folder = os.path.join(OUTPUT_BASE, str(args.ds_points))
    cache_folder = CACHE_BASE_DIR

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)
    print(f"Results will be saved in: {output_folder}")
    print(f"Cache will be stored in: {cache_folder}")

    class_names = get_class_names(os.path.join(DATA_DIR, "shape_names.txt"))
    num_classes = len(class_names)
    TRAIN_FILES = read_file_list(os.path.join(DATA_DIR, "train_files.txt"))
    TEST_FILES  = read_file_list(os.path.join(DATA_DIR, "test_files.txt"))

    # クラスのグループ分け
    groups = np.array_split(range(num_classes), args.num_groups)
    selected_classes = groups[args.group_index]
    print(f"Processing group {args.group_index + 1}/{args.num_groups}: Classes {selected_classes}")

    # 時間計測用の辞書
    timing_stats = {cls: {'shap_time': 0.0, 'total_time': 0.0, 'count': 0} for cls in selected_classes}

    # --- 2. モデルと背景データのロード ---
    model = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)
    backgrounds = {cls: collect_background_point_clouds(cls, TRAIN_FILES, 50) for cls in selected_classes}

    # --- 3. メインループ ---
    for test_file in TEST_FILES:
        print(f"\nProcessing file: {os.path.basename(test_file)}")
        data, labels = provider.loadDataFile(test_file)
        labels = labels.flatten()

        downsampled_pcs, downsampled_labels = [], []

        for i, (pc, cls) in enumerate(zip(data, labels)):
            if cls not in selected_classes: continue

            bg_pc = backgrounds.get(cls)
            if bg_pc is None or bg_pc.size == 0:
                print(f"  [SKIP] Sample {i} (Class: {class_names[cls]}): No background data.")
                continue

            print(f"  -> Processing sample {i} (Class: {class_names[cls]})")

            t_start = time.time()

            cache_filename = f"{os.path.basename(test_file).replace('.h5', '')}_{i}_cls{cls}.npz"
            cache_path = os.path.join(cache_folder, cache_filename)
            point_shap_values = None

            use_cache = args.cache_mode == "auto" and os.path.exists(cache_path)
            if use_cache:
                print("     - Loading SHAP values from cache.")
                try:
                    point_shap_values = np.load(cache_path)['shap']
                    t_shap_end = time.time()
                except Exception as e:
                    print(f"     - Cache loading failed: {e}. Recomputing...")

            if point_shap_values is None:
                if args.cache_mode != 'disable': print("     - Computing SHAP values (cache not found or mode is 'force').")
                else: print("     - Computing SHAP values (cache disabled).")

                # --- SHAP値の計算 ---
                def compute_shap_for_part(part_pc, target_cls):
                    # K-Meansの分割数を引数から取得
                    blocks, _ = spatial_divide_into_blocks(part_pc, num_blocks=args.division_region)
                    baseline = compute_background_baseline(bg_pc, blocks)
                    block_ids, baseline_coords = _precompute_pointwise_maps(blocks, baseline)

                    def f_mask(mask_vectors):
                        return predict_masks_gpu_batch(
                            mask_vectors, part_pc, block_ids, baseline_coords, model, device,
                            chunk_size=args.mask_chunk, min_chunk_size=args.min_mask_chunk, use_pin_mem=not args.no_pinmem
                        )

                    explainer = shap.KernelExplainer(f_mask, np.zeros((1, args.division_region)))
                    shap_values_all_classes = explainer.shap_values(np.ones((1, args.division_region)), nsamples=args.pattern)

                    block_shaps = np.array(shap_values_all_classes[target_cls]).flatten()

                    part_point_shaps = np.zeros(part_pc.shape[0])
                    for block_idx, indices in enumerate(blocks):
                        if indices: part_point_shaps[indices] = block_shaps[block_idx]
                    return part_point_shaps

                if pc.shape[0] == 2048:
                    shap_A = compute_shap_for_part(pc[:1024], cls)
                    shap_B = compute_shap_for_part(pc[1024:], cls)
                    point_shap_values = np.concatenate([shap_A, shap_B])
                else:
                    if pc.shape[0] != 1024:
                        if pc.shape[0] > 1024:
                            sel = np.random.choice(pc.shape[0], 1024, replace=False)
                            pc = pc[sel, :]
                        else:
                            pad = 1024 - pc.shape[0]
                            pc = np.pad(pc, ((0, pad), (0, 0)), mode="constant")
                    point_shap_values = compute_shap_for_part(pc, cls)

                t_shap_end = time.time()

                if args.cache_mode != 'disable':
                    print("     - Saving SHAP values to cache.")
                    np.savez(cache_path, shap=point_shap_values)

            # --- 4. ダウンサンプリングと時間計測 ---
            sorted_indices = np.argsort(point_shap_values)[::-1]
            ds_pc = pc[sorted_indices[:args.ds_points]]
            t_end = time.time()

            timing_stats[cls]['shap_time'] += (t_shap_end - t_start)
            timing_stats[cls]['total_time'] += (t_end - t_start)
            timing_stats[cls]['count'] += 1

            downsampled_pcs.append(ds_pc.astype('float32'))
            downsampled_labels.append(cls)

        # --- 5. ファイルごとの結果を保存（既存があればマージして上書き） ---
        if downsampled_pcs:
            base_name = os.path.basename(test_file)                      # 例: test_*.h5
            subdir    = os.path.basename(DATA_DIR)                       # 例: modelnet40_ply_hdf5_2048
            output_h5_path = os.path.join(output_folder, subdir, base_name)
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
                print(f"Merged {len(new_data)} samples into {output_h5_path}")
            else:
                save_h5_data(output_h5_path, new_data, new_label)
                print(f"Saved {len(new_data)} samples to {output_h5_path}")

        gc.collect()

    # --- 6. 最終的な時間統計を出力 ---
    stats_file = os.path.join(output_folder, f"processing_times_group_{args.group_index}.txt")
    with open(stats_file, 'w') as f:
        f.write("Class\tDS_Points\tDivision_Region\tPattern\tAvg_SHAP(sec)\tAvg_Total(sec)\tCount\n")
        total_shap_time, total_time, total_count = 0.0, 0.0, 0
        for cls_idx in sorted(timing_stats.keys()):
            stats = timing_stats[cls_idx]
            count = stats['count']
            if count > 0:
                avg_shap = stats['shap_time'] / count
                avg_total = stats['total_time'] / count
                f.write(f"{class_names[cls_idx]}\t{args.ds_points}\t{args.division_region}\t{args.pattern}\t{avg_shap:.6f}\t{avg_total:.6f}\t{count}\n")
                total_shap_time += stats['shap_time']
                total_time += stats['total_time']
                total_count += count
        if total_count > 0:
            avg_shap_all = total_shap_time / total_count
            avg_total_all = total_time / total_count
            f.write(f"ALL\t{args.ds_points}\t{args.division_region}\t{args.pattern}\t{avg_shap_all:.6f}\t{avg_total_all:.6f}\t{total_count}\n")
    print(f"\nProcessing finished. Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()