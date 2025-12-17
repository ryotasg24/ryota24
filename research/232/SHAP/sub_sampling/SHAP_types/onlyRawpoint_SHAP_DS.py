#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import gc
import time
import argparse

import numpy as np
import h5py
from sklearn.cluster import KMeans
import torch

# --- PointNeXt 本体のルートを通す ---
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig

# --- provider.py (PointNet 時代のもの) ---
import provider

# ======================================================
# ディレクトリ設定
DATA_ROOT = "/workspace/PointNeXt/data"
DATA_DIR  = os.path.join(DATA_ROOT, "modelnet40_ply_hdf5_2048")
OUTPUT_BASE = "/workspace/PointNeXt/result/SHAP_Types/RawpointSHAP"
# ======================================================

def save_h5_data(h5_filename, data, label):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)

def get_class_names(file_path):
    with open(file_path, 'r') as f:
        return [l.strip() for l in f]

def read_file_list(list_path):
    files = []
    with open(list_path, 'r') as f:
        for line in f:
            rel = line.strip()
            # 先頭が "data/" なら削除しておく
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            files.append(os.path.join(DATA_ROOT, rel))
    return files

def spatial_divide_into_blocks(points, num_blocks=32, kmeans=None):
    if kmeans is None:
        kmeans = KMeans(n_clusters=num_blocks, random_state=42)
        labels = kmeans.fit_predict(points)
    else:
        labels = kmeans.predict(points)
    blocks = [np.where(labels == i)[0].tolist() for i in range(num_blocks)]
    return blocks, kmeans

def collect_background_point_clouds(target_label, train_files, num_point_clouds=50):
    bgs = []
    for fp in train_files:
        data, labels = provider.loadDataFile(fp)
        for idx in np.where(labels == target_label)[0]:
            pc = data[idx]
            if pc.shape != (1024, 3):
                if pc.shape[0] > 1024:
                    sel = np.random.choice(pc.shape[0], 1024, False)
                    pc = pc[sel, :]
                else:
                    continue
            bgs.append(pc)
            if len(bgs) >= num_point_clouds:
                return np.array(bgs)
    return np.array(bgs)

def load_pointnext_model(cfg_file, checkpoint_path, device):
    cfg = EasyConfig()
    cfg.load(cfg_file)
    model = build_model_from_cfg(cfg["model"])
    model.to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    sd = ck.get('model_state_dict', ck.get('model', ck))
    model.load_state_dict(sd)
    model.eval()
    return model

def compute_background_baseline(background_point_clouds, blocks, kmeans):
    bl = []
    for b in blocks:
        pts = background_point_clouds[:, b, :]
        avg1 = pts.mean(axis=1)
        avg2 = avg1.mean(axis=0)
        bl.append(avg2)
    return bl

def compute_point_level_contributions(point_cloud, blocks, baseline_blocks, model, target_class_index, device):
    model.eval()
    t = torch.tensor(point_cloud, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
    t.retain_grad()
    out = model(t)
    score = out[0, target_class_index]
    score.backward()
    grad = t.grad.squeeze(0).cpu().numpy()

    contrib = np.zeros(point_cloud.shape[0])
    for i, b in enumerate(blocks):
        base = baseline_blocks[i]
        for idx in b:
            diff = point_cloud[idx] - base
            contrib[idx] = np.dot(grad[idx], diff)
    return contrib

def main():
    parser = argparse.ArgumentParser(
        description="生の点レベル寄与度のみでダウンサンプリング"
    )
    parser.add_argument("--ds_points", type=int, default=500,
                        help="Number of points to downsample each point cloud (default 500)")
    parser.add_argument("--num_groups", type=int, default=40,
                        help="Number of groups to divide the 40 classes (default 40)")
    parser.add_argument("--group_index", type=int, default=0,
                        help="Index of the group to process (0-indexed, default 0)")
    parser.add_argument("--cfg_file", type=str,
                        default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml",
                        help="Path to the PointNeXt config file")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth",
                        help="Path to the PointNeXt checkpoint file")
    args = parser.parse_args()

    ds_points = args.ds_points
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # クラス名 & ファイルリスト
    class_names = get_class_names(os.path.join(DATA_DIR, "shape_names.txt"))
    num_classes = len(class_names)
    TRAIN_FILES = read_file_list(os.path.join(DATA_DIR, "train_files.txt"))
    TEST_FILES  = read_file_list(os.path.join(DATA_DIR, "test_files.txt"))

    # グループ分割
    base, rem, start = num_classes // args.num_groups, num_classes % args.num_groups, 0
    groups = []
    for g in range(args.num_groups):
        extra = 1 if g < rem else 0
        end   = start + base + extra
        groups.append(list(range(start, end)))
        start = end
    selected = groups[args.group_index]

    # 出力フォルダ
    output_folder = os.path.join(OUTPUT_BASE, str(ds_points))

    # 時間計測用
    class_time = {cls: {'shap':0.0,'down':0.0,'total':0.0,'count':0} for cls in selected}
    overall    = {'shap':0.0,'down':0.0,'total':0.0,'count':0}

    # モデルロード
    model = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)

    # 背景点群収集
    backgrounds = {cls: collect_background_point_clouds(cls, TRAIN_FILES, 50) for cls in selected}

    for tf in TEST_FILES:
        data, labels = provider.loadDataFile(tf)
        # test ファイルの DATA_ROOT 以下相対パスを取得
        rel_path = os.path.relpath(tf, DATA_ROOT)
        # <OUTPUT_BASE>/<ds_points>/<rel_path> に保存
        out_h5   = os.path.join(OUTPUT_BASE, str(ds_points), rel_path)
        os.makedirs(os.path.dirname(out_h5), exist_ok=True)

        samples, labs = [], []
        for i, pc in enumerate(data):
            cls = int(labels[i])
            if cls not in selected:
                continue
            bg = backgrounds[cls]
            if bg.size == 0:
                continue

            # 2048点なら2分割
            if pc.shape[0] == 2048:
                pc_A, pc_B = pc[:1024], pc[1024:]
                # 1) blocks, kmeans を取得
                blocks_A, km_A = spatial_divide_into_blocks(pc_A)
                blocks_B, km_B = spatial_divide_into_blocks(pc_B)
                # 2) baseline_blocks (各ブロックの平均点) を計算
                blA = compute_background_baseline(bg, blocks_A, km_A)
                blB = compute_background_baseline(bg, blocks_B, km_B)
                # 3) 生の寄与度を正しく計算
                t0 = time.time()
                contrib_A = compute_point_level_contributions(pc_A, blocks_A, blA, model, cls, device)
                contrib_B = compute_point_level_contributions(pc_B, blocks_B, blB, model, cls, device)
                t1 = time.time()

                combined = np.concatenate([contrib_A, contrib_B])
            else:
                # 1024点に合わせ
                if pc.shape[0] > 1024:
                    sel = np.random.choice(pc.shape[0], 1024, replace=False)
                    pc  = pc[sel]
                elif pc.shape[0] < 1024:
                    pad = 1024 - pc.shape[0]
                    pc  = np.pad(pc, ((0,pad),(0,0)), mode="constant")

                blocks, km = spatial_divide_into_blocks(pc)
                bls        = compute_background_baseline(bg, blocks, km)
                t0         = time.time()
                combined   = compute_point_level_contributions(pc, blocks, bls, model, cls, device)
                t1         = time.time()

            # ダウンサンプリング
            idxs = np.argsort(combined)[::-1][:ds_points]
            ds_pc = pc[idxs]
            t2    = time.time()

            # 時間集計
            class_time[cls]['shap']  += (t1 - t0)
            class_time[cls]['down']  += (t2 - t1)
            class_time[cls]['total'] += (t2 - t0)
            class_time[cls]['count'] += 1
            overall['shap']  += (t1 - t0)
            overall['down']  += (t2 - t1)
            overall['total'] += (t2 - t0)
            overall['count'] += 1

            samples.append(ds_pc.astype('float32'))
            labs.append(cls)

        # 保存 or マージ
        if os.path.exists(out_h5):
            with h5py.File(out_h5, 'r') as f:
                d0 = f['data'][:]; l0 = f['label'][:]
            new_d = np.concatenate([d0, np.array(samples)], axis=0)
            new_l = np.concatenate([l0, np.array(labs).reshape(-1,1)], axis=0)
            save_h5_data(out_h5, new_d, new_l)
        else:
            save_h5_data(out_h5, np.array(samples), np.array(labs).reshape(-1,1))

    # 処理時間統計を出力
    stats_file = os.path.join(output_folder, "processing_times_by_class.txt")
    with open(stats_file, 'w') as f:
        f.write("Class\tDS_Points\tAvg_SHAP(sec)\tAvg_DOWN(sec)\tAvg_Total(sec)\tCount\n")
        for cls in sorted(class_time.keys()):
            cnt = class_time[cls]['count']
            if cnt > 0:
                f.write(f"{class_names[cls]}\t{ds_points}\t"
                        f"{class_time[cls]['shap']/cnt:.6f}\t"
                        f"{class_time[cls]['down']/cnt:.6f}\t"
                        f"{class_time[cls]['total']/cnt:.6f}\t"
                        f"{cnt}\n")
        oc = overall['count']
        if oc > 0:
            f.write(f"ALL\t{ds_points}\t"
                    f"{overall['shap']/oc:.6f}\t"
                    f"{overall['down']/oc:.6f}\t"
                    f"{overall['total']/oc:.6f}\t"
                    f"{oc}\n")

if __name__ == "__main__":
    main()
