#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage
$ python oneFile_SHAP_DS_heatmap_Visualize.py --target_class airplane --sample_index 0
"""

import sys
import os
import gc
import time
import argparse
import numpy as np
import h5py
from sklearn.cluster import KMeans
import shap
import torch
import torch.nn.functional as F

# 可視化用ライブラリ
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- PointNeXt 関連 ---
sys.path.append(os.path.abspath("/workspace/PointNeXt"))
from openpoints.models.build import build_model_from_cfg
from openpoints.utils import EasyConfig

# --- provider.py ---
import provider


# ======================================================
# 補助関数（共通処理）
# ======================================================

def get_class_names(file_path):
    """クラス名リストをファイルから読み込む"""
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def spatial_divide_into_blocks(points, num_blocks, kmeans=None):
    """点群を空間的にブロックに分割（K-Means）する"""
    if kmeans is None:
        kmeans = KMeans(n_clusters=num_blocks, random_state=42)
        labels = kmeans.fit_predict(points)
    else:
        labels = kmeans.predict(points)
    blocks = []
    for i in range(num_blocks):
        block_indices = np.where(labels == i)[0].tolist()
        blocks.append(block_indices)
    return blocks, kmeans

def compute_background_baseline(background_point_clouds, kmeans):
    num_clusters = kmeans.n_clusters
    baseline_blocks = []
    for i in range(num_clusters):
        cluster_points = []
        for pc in background_point_clouds:      # pc shape (N,3)
            labels = kmeans.predict(pc)         # (N,)
            pts_i  = pc[labels == i]
            if pts_i.size:
                cluster_points.append(pts_i)
        centroid = np.mean(np.concatenate(cluster_points, axis=0), axis=0) if cluster_points else np.zeros(3)
        baseline_blocks.append(centroid)
    return baseline_blocks

def apply_block_mask_fixed_baseline(point_cloud, blocks, mask_vector, baseline_blocks):
    """
    指定のmask_vectorに従い、mask=0のブロックの点を背景基準値に置換する
    """
    masked_point_cloud = point_cloud.copy()
    for block_idx, mask in enumerate(mask_vector):
        if mask == 0:
            masked_point_cloud[blocks[block_idx]] = baseline_blocks[block_idx]
    return masked_point_cloud

def pointnext_predict_with_block_mask_fixed(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device):
    """
    各maskパターンに対し背景基準値で置換した点群の予測を取得する
    """
    model.eval()
    predictions = []
    for mask_vector in mask_vectors:
        masked_pc = apply_block_mask_fixed_baseline(original_point_cloud, blocks, mask_vector, baseline_blocks)
        pc_tensor = torch.tensor(masked_pc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred_val = model(pc_tensor)
        predictions.append(pred_val.cpu().numpy())
    if len(predictions) == 0:
        return np.array([])
    return np.concatenate(predictions, axis=0)

def shap_predict_block_mask(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device):
    """SHAPのKernelExplainer用ラッパー関数"""
    return pointnext_predict_with_block_mask_fixed(mask_vectors, blocks, original_point_cloud, baseline_blocks, model, device)

def compute_point_level_contributions(point_cloud, blocks, baseline_blocks, model, target_class_index, device):
    """
    各点の寄与度を、入力点群の勾配と背景との差の内積から算出する
    (point_cloud: (N,3))
    ※ unsqueeze後にretain_grad()を呼び出してgradが保持されるようにする
    """
    model.eval()
    pc_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=device, requires_grad=True)
    pc_tensor = pc_tensor.unsqueeze(0)
    pc_tensor.retain_grad()
    output = model(pc_tensor)
    target_output = output[0, target_class_index]
    target_output.backward()
    grad_val = pc_tensor.grad.squeeze(0).cpu().numpy()  # (N, 3)
    point_contrib = np.zeros(point_cloud.shape[0])
    for i, block in enumerate(blocks):
        baseline = baseline_blocks[i]
        for idx in block:
            diff = point_cloud[idx] - baseline
            point_contrib[idx] = np.dot(grad_val[idx], diff)
    return point_contrib

def compute_point_level_shap_values(point_cloud, blocks, block_shap_values, baseline_blocks, model, target_class_index, device, scaling_factor=1.0):
    """
    各点の統合SHAP値 = (ブロックのSHAP値) + (scaling_factor × ブロック内正規化された寄与度)  
    ※ 各ブロック内で寄与度をzスコア正規化して加算する
    """
    point_contrib = compute_point_level_contributions(point_cloud, blocks, baseline_blocks, model, target_class_index, device)
    integrated_shap = np.zeros(point_cloud.shape[0])
    for i, block in enumerate(blocks):
        block_contribs = point_contrib[block]
        mean_val = np.mean(block_contribs)
        std_val = np.std(block_contribs)
        if std_val < 1e-6:
            normalized_contrib = np.zeros_like(block_contribs)
        else:
            normalized_contrib = (block_contribs - mean_val) / std_val
        for j, idx in enumerate(block):
            integrated_shap[idx] = block_shap_values[i] + scaling_factor * normalized_contrib[j]
    return integrated_shap

def l1_match_point(point_shap: np.ndarray, region_per_pt: np.ndarray, eps: float = 1e-6):
    """
    Point-SHAP を L1 正規化で Region-SHAP（per-point 展開）にスケール合わせする
    返り値: (point_adj, L1_reg, L1_pnt)
    """
    L1_reg = float(np.sum(np.abs(region_per_pt))) + eps
    L1_pnt = float(np.sum(np.abs(point_shap))) + eps
    scale  = L1_reg / L1_pnt
    return point_shap * scale, L1_reg, L1_pnt


def load_pointnext_model(cfg_file, checkpoint_path, device):
    """
    PointNeXtモデルのロード（PyTorch版）
    cfg_file: PointNeXtのconfigファイルのパス
    checkpoint_path: 学習済みチェックポイントのパス
    """
    cfg = EasyConfig()
    cfg.load(cfg_file)
    model_cfg = cfg["model"]
    model = build_model_from_cfg(model_cfg)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


# ======================================================
# 可視化処理（Open3DおよびMatplotlib）
# ======================================================

def create_custom_colormap():
    """
    （参考用）青→赤の2色グラデーション（本実装では手動着色を使うため未使用でもOK）
    """
    return LinearSegmentedColormap.from_list("custom_rb", [(0, "blue"), (1, "red")])

def apply_color_map(values, colormap):
    """
    データ内の最小/最大に合わせて動的スケール：
      ・最大＋ = 真っ赤、最小－ = 真っ青
      ・0 だけグレー
      ・小さい正は薄赤、小さい負は薄青
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    n = v.shape[0]
    # 既定は全部グレー（0のときだけ最終的に残る）
    colors = np.tile(np.array([0.6, 0.6, 0.6], dtype=float), (n, 1))

    if n == 0:
        return colors

    vmax = np.max(v)
    vmin = np.min(v)

    # 正側：薄赤(=light_red)→赤
    mask_pos = v > 0
    if np.any(mask_pos) and vmax > 0:
        t = (v[mask_pos] / vmax).astype(float)  # 0..1
        light_red = np.array([1.0, 0.85, 0.85], dtype=float)
        red       = np.array([1.0, 0.0, 0.0],  dtype=float)
        colors[mask_pos] = light_red + (red - light_red) * t[:, None]

    # 負側：薄青(=light_blue)→青
    mask_neg = v < 0
    if np.any(mask_neg) and vmin < 0:
        t = (np.abs(v[mask_neg]) / np.abs(vmin)).astype(float)  # 0..1
        light_blue = np.array([0.85, 0.90, 1.0], dtype=float)
        blue       = np.array([0.0,  0.0,  1.0], dtype=float)
        colors[mask_neg] = light_blue + (blue - light_blue) * t[:, None]

    # 0 は明確にグレー
    # （浮動小数の丸め誤差を避けたい場合は == ではなく閾値に変更してください）
    mask_zero = (v == 0)
    if np.any(mask_zero):
        colors[mask_zero] = np.array([0.6, 0.6, 0.6], dtype=float)

    return colors

def visualize_blocks(point_cloud, blocks, window_name="Block Segmentation",
                    screenshot_file="result/heatmap_block_segmentation.png"):
    """
    K-means で分割されたブロック境界を確認する可視化。ブロックごとに異なる色を割り当てる（tab20 カラーマップ）。
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(blocks))   # 最大 20 色 → 足りないときは色循環
    point_colors = np.zeros((point_cloud.shape[0], 3))
    for i, idx_list in enumerate(blocks):
        color = np.array(cmap(i % cmap.N)[:3])   # RGBA → RGB
        point_colors[idx_list] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 画面表示＋保存
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Block segmentation heatmap saved to {screenshot_file}")

def visualize_blocks_with_baseline(point_cloud, blocks, baseline_blocks, window_name="Blocks + Baseline", screenshot_file="result/heatmap_blocks_baseline.png", radius=0.015):
    """
    • 点群をブロックごとに着色（tab20 パレット）
    • 各ブロック重心 (baseline) を濃い青の球メッシュで表示
    • radius はスケールに応じて調整
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(blocks))

    # ---------- 元点群（ブロック色） ----------
    colors_pc = np.zeros((point_cloud.shape[0], 3))
    for i, idx_list in enumerate(blocks):
        colors_pc[idx_list] = np.array(cmap(i % cmap.N)[:3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors_pc)

    # ---------- baseline 重心 (球メッシュ) ----------
    sphere_meshes = []
    for center in baseline_blocks:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        sphere.translate(center)                       # 座標へ移動
        sphere.paint_uniform_color([0.0, 0.25, 1.0])   # 濃い青
        sphere.compute_vertex_normals()
        sphere_meshes.append(sphere)

    # ---------- 表示 & スクリーンショット ----------
    geoms = [pcd] + sphere_meshes
    o3d.visualization.draw_geometries(geoms, window_name=window_name)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Block + baseline heatmap saved to {screenshot_file}")



# --- ヒートマップ可視化 ---
def visualize_shapley_blocks(point_cloud, blocks, blocks_shapley, window_name="Block SHAP Visualization", screenshot_file="result/block_shap_visualization.png"):
    """
    ブロックSHAPヒートマップ：各ブロックのSHAP値により色付け
    """
    custom_cmap = create_custom_colormap()
    block_colors = apply_color_map(np.array(blocks_shapley), custom_cmap)
    point_colors = np.zeros((point_cloud.shape[0], 3))
    for i, indices in enumerate(blocks):
        point_colors[indices] = block_colors[i]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Block SHAP heatmap saved to {screenshot_file}")

def visualize_point_contributions(point_cloud, point_contrib, window_name="Raw Point SHAP Visualization", screenshot_file="result/raw_point_shap_visualization.png"):
    """
    各点SHAP（生の寄与）ヒートマップ：生の寄与値により色付け
    """
    custom_cmap = create_custom_colormap()
    colors = apply_color_map(point_contrib, custom_cmap)
    zero_mask = (point_contrib == 0)
    colors[zero_mask] = np.array([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Raw point SHAP heatmap saved to {screenshot_file}")

def visualize_normalized_point_shap_values(point_cloud, normalized_point_shap, window_name="Normalized Point SHAP Visualization", screenshot_file="result/normalized_point_shap_visualization.png"):
    """
    ブロック内正規化後の各点SHAPヒートマップ：正規化後の値により色付け
    """
    custom_cmap = create_custom_colormap()
    colors = apply_color_map(normalized_point_shap, custom_cmap)
    zero_mask = (normalized_point_shap == 0)
    colors[zero_mask] = np.array([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Normalized point SHAP heatmap saved to {screenshot_file}")

def visualize_integrated_point_shap_values(point_cloud, integrated_shap, window_name="Integrated Point SHAP Visualization", screenshot_file="result/integrated_point_shap_visualization.png"):
    """
    統合後の各点SHAPヒートマップ：統合SHAP値により色付け
    """
    custom_cmap = create_custom_colormap()
    colors = apply_color_map(integrated_shap, custom_cmap)
    zero_mask = np.abs(integrated_shap) < 1e-6
    colors[zero_mask] = np.array([0.5, 0.5, 0.5])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Integrated point SHAP heatmap saved to {screenshot_file}")

# --- ヒストグラムプロット ---
def plot_shapley_histogram(block_shap_values, output_file="result/block_shap_histogram.png"):
    """
    ブロックSHAP値のヒストグラムをプロットしてPNGで保存
    """
    plt.figure(figsize=(8, 6))
    plt.hist(block_shap_values, bins=50, color='gray', edgecolor='black')
    plt.title("Block-level SHAP Values Distribution")
    plt.xlabel("SHAP Value")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    plt.close()
    print(f"Block SHAP histogram saved to {output_file}")

def plot_raw_point_shap_histogram(raw_point_contrib, output_file="result/raw_point_shap_histogram.png"):
    """
    生の各点SHAP（寄与）のヒストグラムをプロットしてPNGで保存
    """
    plt.figure(figsize=(8, 6))
    plt.hist(raw_point_contrib, bins=50, color='green', edgecolor='black')
    plt.title("Raw Point-level SHAP Values Distribution")
    plt.xlabel("Contribution Value")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    plt.close()
    print(f"Raw point SHAP histogram saved to {output_file}")

def plot_normalized_point_shap_histogram(normalized_point_shap, output_file="result/normalized_point_shap_histogram.png"):
    """
    ブロック内正規化後の各点SHAP値のヒストグラムをプロットしてPNGで保存
    """
    plt.figure(figsize=(8, 6))
    plt.hist(normalized_point_shap, bins=50, color='orange', edgecolor='black')
    plt.title("Normalized Point-level SHAP Values Distribution")
    plt.xlabel("Normalized SHAP Value")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    plt.close()
    print(f"Normalized point SHAP histogram saved to {output_file}")

def plot_integrated_point_shap_histogram(integrated_shap, output_file="result/integrated_point_shap_histogram.png"):
    """
    統合後の各点SHAP値のヒストグラムをプロットしてPNGで保存
    """
    plt.figure(figsize=(8, 6))
    plt.hist(integrated_shap, bins=50, color='blue', edgecolor='black')
    plt.title("Integrated Point-level SHAP Values Distribution")
    plt.xlabel("SHAP Value")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    plt.close()
    print(f"Integrated point SHAP histogram saved to {output_file}")

# ======================================================
# main() 関数
# ======================================================

def main():
    parser = argparse.ArgumentParser(
        description="指定クラス1サンプルに対して、PointNeXt(Pytorch)によるSHAP値計算と各種ヒートマップ・ヒストグラム、ダウンサンプリング後点群の出力を実行する"
    )
    parser.add_argument("--input_file", type=str, default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048/ply_data_test0.h5",
                        help="入力h5ファイルのパス")
    parser.add_argument("--class_names_file", type=str, default="/workspace/PointNeXt/data/ModelNet40Ply2048/modelnet40_ply_hdf5_2048/shape_names.txt",
                        help="shape_names.txt のパス")
    parser.add_argument("--target_class", type=str, default="airplane",
                        help="対象クラス名 (例: airplane, plant 等)")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="対象クラス内で処理するサンプルのインデックス")
    parser.add_argument("--output_ply", type=str, default="output.ply",
                        help="ダウンサンプリング結果のPLYファイル名")
    parser.add_argument("--ds_points", type=int, default=500,
                        help="ダウンサンプリングする点数")
    parser.add_argument("--weight", type=float, default=1.0,
                    help="Hybrid-SHAP の Point-SHAP 部分への重み")
    parser.add_argument("--pattern", type=int, default=100,
                        help="KernelExplainer の nsamples 設定")
    parser.add_argument("--division", type=int, default=32,
                        help="k-meansによる領域数 (default 32)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml",
                        help="PointNeXt configファイルのパス")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth",
                        help="PointNeXt チェックポイントのパス")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # クラス名読み込みと対象クラスの決定
    class_names = get_class_names(args.class_names_file)
    if args.target_class not in class_names:
        sys.exit(f"対象クラス '{args.target_class}' が見つかりません。")
    target_class_index = class_names.index(args.target_class)
    print(f"対象クラス '{args.target_class}' (index: {target_class_index}) を選択。")

    # 入力h5ファイルから点群データとラベルを読み込み
    data, labels = provider.loadDataFile(args.input_file)
    print(f"{args.input_file} から {len(data)} サンプルを読み込みました。")

    # 対象クラスのサンプル抽出
    target_indices = np.where(labels == target_class_index)[0]
    if len(target_indices) == 0:
        sys.exit(f"対象クラス '{args.target_class}' のサンプルが存在しません。")
    if args.sample_index >= len(target_indices):
        sys.exit("sample_index が範囲外です。")
    chosen_idx = target_indices[args.sample_index]
    pc = data[chosen_idx]
    if pc.shape[1] != 3:
        sys.exit("入力点群は3次元座標を持っていません。")
    print(f"サンプル index {args.sample_index} (グローバル: {chosen_idx}) の点群形状: {pc.shape}")

    # 背景点群の収集（同クラス内の他サンプル、最大50件）
    bg_indices = np.delete(target_indices, args.sample_index)
    if len(bg_indices) == 0:
        sys.exit("背景点群のサンプルが不足しています。")
    bg_pc_list = []
    num_bg = min(50, len(bg_indices))
    for idx in bg_indices[:num_bg]:
        candidate = data[idx]
        if candidate.shape[0] > 1024:
            sampled_idx = np.random.choice(candidate.shape[0], 1024, replace=False)
            candidate = candidate[sampled_idx, :]
        elif candidate.shape[0] < 1024:
            continue
        if candidate.shape[0] == 1024:
            bg_pc_list.append(candidate)
    if len(bg_pc_list) == 0:
        sys.exit("有効な背景点群が見つかりません。")
    bg_pc = np.array(bg_pc_list)
    print(f"背景点群として {bg_pc.shape[0]} 個のサンプルを収集しました。")

    # PointNeXtモデルのロード
    model, cfg = load_pointnext_model(args.cfg_file, args.checkpoint_path, device)
    print("PointNeXtモデルをロードしました。")

    # 2048点の場合は1024点ずつに分割して処理、その他は1024点にサンプリング
    if pc.shape[0] == 2048:
        pc_A = pc[:1024, :]
        pc_B = pc[1024:, :]

        # パートA
        blocks_A, kmeans_A = spatial_divide_into_blocks(pc_A, num_blocks=args.division)
        #visualize_blocks(pc_A, blocks_A, window_name="Blocks A", screenshot_file="result/heatmap_blocks_A.png")
        baseline_A = compute_background_baseline(bg_pc, kmeans_A)
#        visualize_blocks_with_baseline(pc_A, blocks_A, baseline_A, window_name="Regions A + Baseline", screenshot_file="result/heatmap_RegionsA_baseline.png", radius=0.015)
        def shap_predict_block_mask_wrapper_A(mask_vectors):
            return shap_predict_block_mask(mask_vectors, blocks_A, pc_A, baseline_A, model, device)
        explainer_A = shap.KernelExplainer(shap_predict_block_mask_wrapper_A, np.zeros((1, len(blocks_A))))
        block_shap_values_A = explainer_A.shap_values(np.ones((1, len(blocks_A))), nsamples=args.pattern, l1_reg="num_features(10)")
        block_shap_values_A = block_shap_values_A[target_class_index].reshape(-1)
        point_shap_vals_A = compute_point_level_shap_values(pc_A, blocks_A, block_shap_values_A, baseline_A,
                                                            model, target_class_index, device, scaling_factor=args.weight)
        # パートB
        blocks_B, kmeans_B = spatial_divide_into_blocks(pc_B, num_blocks=args.division)
        #visualize_blocks(pc_B, blocks_B, window_name="Blocks B", screenshot_file="result/heatmap_blocks_B.png")
        baseline_B = compute_background_baseline(bg_pc, kmeans_B)
#        visualize_blocks_with_baseline(pc_B, blocks_B, baseline_B, window_name="Regions B + Baseline", screenshot_file="result/heatmap_RegionsB_baseline.png", radius=0.015)
        def shap_predict_block_mask_wrapper_B(mask_vectors):
            return shap_predict_block_mask(mask_vectors, blocks_B, pc_B, baseline_B, model, device)
        explainer_B = shap.KernelExplainer(shap_predict_block_mask_wrapper_B, np.zeros((1, len(blocks_B))))
        block_shap_values_B = explainer_B.shap_values(np.ones((1, len(blocks_B))), nsamples=args.pattern, l1_reg="num_features(10)")
        block_shap_values_B = block_shap_values_B[target_class_index].reshape(-1)
        point_shap_vals_B = compute_point_level_shap_values(pc_B, blocks_B, block_shap_values_B, baseline_B,
                                                            model, target_class_index, device, scaling_factor=args.weight)
        # 連結（2048点全体）
        integrated_shap = np.concatenate([point_shap_vals_A, point_shap_vals_B])
        block_shap_values = np.concatenate([block_shap_values_A, block_shap_values_B])
        blocks = blocks_A + [[idx+1024 for idx in b] for b in blocks_B]
        combined_pc = pc
        baseline_combined = baseline_A + baseline_B
        raw_point_contrib = compute_point_level_contributions(combined_pc, blocks, baseline_combined, model, target_class_index, device)
    else:
        if pc.shape[0] > 1024:
            sampled_idx = np.random.choice(pc.shape[0], 1024, replace=False)
            pc = pc[sampled_idx, :]
        blocks, kmeans = spatial_divide_into_blocks(pc, num_blocks=args.division)
        #visualize_blocks(pc, blocks, window_name="Blocks (single)", screenshot_file="result/heatmap_blocks_single.png")
        baseline_blocks = compute_background_baseline(bg_pc, kmeans)
#        visualize_blocks_with_baseline(pc, blocks, baseline_blocks, window_name="Regions + Baseline", screenshot_file="result/heatmap_Regions_baseline.png", radius=0.015)
        def shap_predict_block_mask_wrapper(mask_vectors):
            return shap_predict_block_mask(mask_vectors, blocks, pc, baseline_blocks, model, device)
        explainer = shap.KernelExplainer(shap_predict_block_mask_wrapper, np.zeros((1, len(blocks))))
        block_shap_values = explainer.shap_values(np.ones((1, len(blocks))), nsamples=args.pattern, l1_reg="num_features(10)")
        block_shap_values = block_shap_values[target_class_index].reshape(-1)
        integrated_shap = compute_point_level_shap_values(pc, blocks, block_shap_values, baseline_blocks,
                                                        model, target_class_index, device, scaling_factor=args.weight)
        combined_pc = pc
        raw_point_contrib = compute_point_level_contributions(combined_pc, blocks, baseline_blocks, model, target_class_index, device)

    # --- L1正規化：Region-SHAP（per-point展開）と同一L1にスケール合わせした Point-SHAP_adj を作成 ---
    # per-point の Region-SHAP を作る（各ブロックSHAPを点にブロードキャスト）
    region_per_pt = np.zeros(combined_pc.shape[0], dtype=np.float32)
    for i, idxs in enumerate(blocks):
        region_per_pt[idxs] = block_shap_values[i]

    # integrated_shap（= block_SHAP + 正規化寄与）を L1 正規化で合わせる
    point_shap_adj, L1_reg, L1_pnt = l1_match_point(integrated_shap, region_per_pt, eps=1e-6)

    # （任意）Point-SHAP_adj のヒートマップを保存したい場合は以下を有効化
    # visualize_integrated_point_shap_values(
    #     combined_pc, point_shap_adj,
    #     window_name="Point-SHAP_adj Visualization",
    #     screenshot_file="result/heatmap_point_shap_adj.png"
    # )

    # ここで【ブロック内正規化後の各点SHAP】を算出（raw_point_contribを各ブロック内でzスコア正規化）
    normalized_point_shap = np.zeros_like(raw_point_contrib)
    for i, block in enumerate(blocks):
        block_vals = raw_point_contrib[block]
        mean_val = np.mean(block_vals)
        std_val = np.std(block_vals)
        if std_val < 1e-6:
            norm_vals = np.zeros_like(block_vals)
        else:
            norm_vals = (block_vals - mean_val) / std_val
        normalized_point_shap[block] = norm_vals

    # ダウンサンプリング：統合SHAP値が高い上位 ds_points を選択
    sorted_idx = np.argsort(integrated_shap)[::-1]
    top_indices = sorted_idx[:args.ds_points]
    ds_pc = combined_pc[top_indices, :]

    # 1. 各ヒストグラムの出力（MatplotlibによるPNG保存）
    # ヒストグラム1：ブロックSHAP
    plot_shapley_histogram(block_shap_values, output_file="result/histogram_block_shap.png")
    # ヒストグラム2：各点SHAP（生の寄与）
    plot_raw_point_shap_histogram(raw_point_contrib, output_file="result/histogram_raw_point_shap.png")
    # ヒストグラム3：ブロック内正規化後の各点SHAP
    plot_normalized_point_shap_histogram(normalized_point_shap, output_file="result/histogram_normalized_point_shap.png")
    # ヒストグラム4：統合後の各点SHAP
    plot_integrated_point_shap_histogram(integrated_shap, output_file="result/histogram_integrated_point_shap.png")
    # ヒストグラム5：L1正規化後の Point-SHAP_adj
    plot_integrated_point_shap_histogram(point_shap_adj, output_file="result/histogram_point_shap_adj.png")

    # 2. 各ヒートマップの出力（Open3Dによる可視化＆スクリーンショット保存）
    # ヒートマップ1：ブロックSHAP
    visualize_shapley_blocks(combined_pc, blocks, block_shap_values, window_name="Block SHAP Visualization", screenshot_file="result/heatmap_block_shap.png")
    # ヒートマップ2：各点SHAP（生の寄与）
    #visualize_point_contributions(combined_pc, raw_point_contrib, window_name="Raw Point SHAP Visualization", screenshot_file="result/heatmap_raw_point_shap.png")
    # ヒートマップ3：ブロック内正規化後の各点SHAP
    #visualize_normalized_point_shap_values(combined_pc, normalized_point_shap, window_name="Normalized Point SHAP Visualization", screenshot_file="result/heatmap_normalized_point_shap.png")
    # ヒートマップ4：統合後の各点SHAP
    #visualize_integrated_point_shap_values(combined_pc, integrated_shap, window_name="Integrated Point SHAP Visualization", screenshot_file="result/heatmap_integrated_point_shap.png")
    # ヒートマップ5：L1正規化後の Point-SHAP_adj
    #visualize_integrated_point_shap_values(combined_pc, point_shap_adj, window_name="Point-SHAP_adj Visualization", screenshot_file="result/heatmap_point_shap_adj.png")


    # 3. ダウンサンプリング後の点群の出力（PLY形式）と表示
    ds_output_path = os.path.join("result", "downsampled_point_cloud.ply")
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(ds_pc)
    # すべての点の色を赤に設定
    red_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (ds_pc.shape[0], 1))
    pcd_out.colors = o3d.utility.Vector3dVector(red_colors)
    o3d.io.write_point_cloud(ds_output_path, pcd_out)
    print(f"ダウンサンプリング後の点群を {ds_output_path} として保存しました。")
    # 表示（点の色は赤）
    o3d.visualization.draw_geometries([pcd_out], window_name="Downsampled Point Cloud (Red)")

if __name__ == "__main__":
    main()
