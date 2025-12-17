#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage
$  CUDA_VISIBLE_DEVICES=0 python espw_oneFile_SHAP_heatmap_Visualize.py --target_class airplane --sample_index 0 --alpha 1 --lambda_eps 0.05

$ CUDA_VISIBLE_DEVICES=0 python espw_oneFile_SHAP_heatmap_Visualize.py --target_class airplane --sample_index 0 --alpha_grid 0.01,0.1,0.3,0.5 --lambda_grid 0.05,0.1,0.15,0.2
"""

import sys
import os
import argparse
import time
import numpy as np
from sklearn.cluster import KMeans
import shap
import torch

# 可視化用ライブラリ
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator, NullLocator

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
    """点群を空間的にRegionに分割（K-Means）する"""
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
    指定のmask_vectorに従い、mask=0のRegionの点を背景基準値に置換する
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

def compute_point_level_shap_values(point_cloud, blocks, block_shap_values, baseline_blocks,
                                    model, target_class_index, device,
                                    alpha: float = 1.0, lambda_eps: float = 0.1):
    """
    espw (ε‑Smoothing + signed‑Power Weight) 版 Point‑SHAP

    φ_point(i)=φ_region(r)·ω_p
    ω_p = sign(φ_raw(p))*(|φ_raw(p)|+ε_r)^α / Σ_q sign(g_q)*(|g_q|+ε_r)^α

    * φ_raw(p) : Raw‑SHAP = ∇f·(x_i‑baseline_r)
    * ε_r = λ·μ_r ,  μ_r = mean_q |g_q|
    """
    raw = compute_point_level_contributions(point_cloud, blocks,
                                            baseline_blocks,
                                            model, target_class_index, device)
    point_shap = np.zeros_like(raw)
    weights_all = np.zeros_like(raw)
    for i, block in enumerate(blocks):
        if len(block) == 0:
            continue

        # --- ε‑smoothing --------------------------------------------------
        abs_r = np.abs(raw[block])
        mu_r  = abs_r.mean()
        q90   = np.quantile(abs_r, 0.90)
        base  = max(mu_r, q90)
        eps   = lambda_eps * (base if base > 0 else 1e-6)       # 常に ε>0

        # --- 符号付きパワー重み ------------------------------------------
        sign_r = np.sign(block_shap_values[i])
        if sign_r == 0:      # Region SHAP が 0 → 正符号を継承
            sign_r = 1.0           # Region=0 → +1
        abs_raw = np.abs(raw[block])
        signs   = np.where(abs_raw == 0, sign_r, np.sign(raw[block]))
        weights = signs * (abs_raw + eps) ** alpha
        weights = np.nan_to_num(weights, nan=0.0)
        weights_all[block] = weights

        denom = weights.sum()
        if np.abs(denom) < np.finfo(float).eps:  # 機械イプシロン ε 以下でのみ等分配
            point_shap[block] = sign_r * (eps**alpha) * block_shap_values[i] / max(len(block), 1)
            continue

        factor = block_shap_values[i] / denom
        point_shap[block] = weights * factor

    return point_shap, weights_all

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
    SHAP値[-1,0,1]に対し、「青 → ライトグレー → 赤」のグラデーションを作成するカラーマップ
    """
    return LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (0.0, "blue"),       # most negative → deep blue
            (0.5, "lightgray"),  # zero → neutral light gray
            (1.0, "red")         # most positive → deep red
        ]
    )

def apply_color_map(values, colormap):
    """
    入力のSHAP値（numpy array）を[-1,1]にクリップし[0,1]に正規化、カラーマップからRGB色を返す
    """
    norm_values = np.clip(values, -1, 1)
    norm_values = (norm_values + 1) / 2.0  # -1 -> 0, 0 -> 0.5, 1 -> 1
    colors = colormap(norm_values)
    return colors[:, :3]  # RGBAのAを除く

def visualize_blocks(point_cloud, blocks, window_name="Block Segmentation",
                    screenshot_file="result/heatmap_block_segmentation.png"):
    """
    K-means で分割されたRegion境界を確認する可視化。Regionごとに異なる色を割り当てる（tab20 カラーマップ）。
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


def visualize_blocks_with_baseline(point_cloud, blocks, baseline_blocks, window_name="Regions + Baseline", screenshot_file="result/heatmap_regions_baseline.png", radius=0.015):
    """
    • 点群をRegionごとに着色（tab20 パレット）
    • 各Region重心 (baseline) を濃い青の球メッシュで表示
    • radius はスケールに応じて調整
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(blocks))

    # ---------- 元点群（Region色） ----------
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
    print(f"Region + Baseline heatmap saved to {screenshot_file}")



# --- ヒートマップ可視化 ---
def visualize_region_shap(point_cloud, blocks, blocks_shapley, window_name="Region SHAP Visualization", screenshot_file="result/region_shap_visualization.png"):
    """
    各領域SHAPヒートマップ：Region-SHAPにより色付け
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
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Block SHAP heatmap saved to {screenshot_file}")

def visualize_raw_shap(point_cloud, point_contrib, window_name="Raw SHAP Visualization", screenshot_file="result/raw_shap_visualization.png"):
    """
    各点SHAP（生の寄与）ヒートマップ：Raw-SHAPにより色付け
    """
    custom_cmap = create_custom_colormap()
    colors = apply_color_map(point_contrib, custom_cmap)  # 0 でも緑になる
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Raw point SHAP heatmap saved to {screenshot_file}")

def visualize_weights_heatmap(point_cloud, weights, window_name="Weights Visualization", screenshot_file="result/weights_visualization.png"):
    """
    点ごとの重み ω をカラーマップで表示＆スクリーンショット保存
    """
    # 重みを [-1,1]→[0,1] に正規化
    max_abs = np.max(np.abs(weights)) or 1.0
    norm = (weights / max_abs + 1.0) / 2.0
    cmap = create_custom_colormap()
    colors = cmap(norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 画面表示
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    # 保存
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Weights ω_p heatmap saved to {screenshot_file}")

def visualize_point_shap(point_cloud, integrated_shap, window_name="Point SHAP Visualization", screenshot_file="result/point_shap_visualization.png"):
    """
    統合後の各点SHAPヒートマップ：統合したPoint-SHAP値により色付け
    """
    custom_cmap = create_custom_colormap()
    colors = apply_color_map(integrated_shap, custom_cmap)  # 0 でも緑になる
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, window_name=window_name)
    vis.add_geometry(pcd)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_file)
    vis.destroy_window()
    print(f"Point SHAP heatmap saved to {screenshot_file}")

# --- ヒストグラムプロット ---
def make_symlog_bins(data, nbins=50, linthresh=1e-6):
    """symlog軸上で等幅に見えるようにビンを作る"""
    lo, hi = data.min(), data.max()
    # 線形領域の閾値
    lt = linthresh
    # ビン数を線形・ログで分割
    n_lin = max(2, nbins//10)
    n_log = (nbins - n_lin)//2
    # 正領域と負領域でログ空間ビン
    pos_bins = np.logspace(np.log10(max(lt, data[data>lt].min())), np.log10(hi), n_log+1)
    neg_bins = -np.logspace(np.log10(max(lt, (-data[data< -lt]).min())), np.log10(-lo), n_log+1)[::-1]
    # 線形領域ビン（-lt〜+lt）
    lin_bins = np.linspace(-lt, lt, n_lin+1)
    # くっつけてユニーク化
    bins = np.unique(np.concatenate([neg_bins, lin_bins, pos_bins]))
    return bins

def plot_region_histogram(block_shap_values, output_file="result/region_shap_histogram.png"):
    """
    領域SHAP値のヒストグラムをプロットしてPNGで保存
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lo, hi = block_shap_values.min(), block_shap_values.max()
    lt = (hi - lo) * 1e-3
    bins = make_symlog_bins(block_shap_values, nbins=50, linthresh=lt)

    ax.hist(block_shap_values, bins=bins, color='gray', edgecolor='black')
    ax.set_xscale('symlog', linthresh=lt)

    # SymmetricalLogLocator で全ティック位置を取得し、
    # 先頭から間引いて10本だけ表示
    locator = SymmetricalLogLocator(base=10, linthresh=lt)
    all_ticks = locator.tick_values(lo, hi)
    # プロットするティックを間引き (最大10個)
    sel = np.linspace(0, len(all_ticks)-1, 10, dtype=int)
    ax.set_xticks(all_ticks[sel])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title("Region-level SHAP Values Distribution")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Region SHAP histogram saved to {output_file}")

def plot_raw_point_shap_histogram(raw_point_contrib, output_file="result/raw_shap_histogram.png"):
    """
    生の各点SHAP（寄与）のヒストグラムをプロットしてPNGで保存
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lo, hi = raw_point_contrib.min(), raw_point_contrib.max()
    lt = (hi - lo) * 1e-3
    bins = make_symlog_bins(raw_point_contrib, nbins=50, linthresh=lt)

    ax.hist(raw_point_contrib, bins=bins, color='green', edgecolor='black')
    ax.set_xscale('symlog', linthresh=lt)

    # SymmetricalLogLocator で全ティック位置を取得し、
    # 先頭から間引いて10本だけ表示
    locator = SymmetricalLogLocator(base=10, linthresh=lt)
    all_ticks = locator.tick_values(lo, hi)
    # プロットするティックを間引き (最大10個)
    sel = np.linspace(0, len(all_ticks)-1, 10, dtype=int)
    ax.set_xticks(all_ticks[sel])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title("Raw Point-level SHAP Values Distribution")
    ax.set_xlabel("Contribution Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Raw point SHAP histogram saved to {output_file}")

def plot_weights_histogram(weights, output_file="result/weights_raw_histogram.png"):
    """
    点ごとの重み ω のヒストグラムをプロットしてPNGで保存
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lo, hi = weights.min(), weights.max()
    lt = (hi - lo) * 1e-3
    bins = make_symlog_bins(weights, nbins=50, linthresh=lt)

    ax.hist(weights, bins=bins, color='red', edgecolor='black')
    ax.set_xscale('symlog', linthresh=lt)

    # SymmetricalLogLocator で全ティック位置を取得し、
    # 先頭から間引いて10本だけ表示
    locator = SymmetricalLogLocator(base=10, linthresh=lt)
    all_ticks = locator.tick_values(lo, hi)
    # プロットするティックを間引き (最大10個)
    sel = np.linspace(0, len(all_ticks)-1, 10, dtype=int)
    ax.set_xticks(all_ticks[sel])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title("Point-level Weight ω Distribution")
    ax.set_xlabel("ω_p (pre-normalization)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Weight histogram saved to {output_file}")

def plot_point_shap_histogram(integrated_shap, output_file="result/integrated_point_shap_histogram.png"):
    """
    統合後の各点SHAP値のヒストグラムをプロットしてPNGで保存
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    lo, hi = integrated_shap.min(), integrated_shap.max()
    lt = (hi - lo) * 1e-3
    bins = make_symlog_bins(integrated_shap, nbins=50, linthresh=lt)

    ax.hist(integrated_shap, bins=bins, color='blue', edgecolor='black')
    ax.set_xscale('symlog', linthresh=lt)

    # SymmetricalLogLocator で全ティック位置を取得し、
    # 先頭から間引いて10本だけ表示
    locator = SymmetricalLogLocator(base=10, linthresh=lt)
    all_ticks = locator.tick_values(lo, hi)
    # プロットするティックを間引き (最大10個)
    sel = np.linspace(0, len(all_ticks)-1, 10, dtype=int)
    ax.set_xticks(all_ticks[sel])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title("Integrated Point-level SHAP Values Distribution")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
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
    parser.add_argument("--ds_points", type=int, default=500,
                        help="ダウンサンプリングする点数")
    parser.add_argument("--pattern", type=int, default=100,
                        help="KernelExplainer の nsamples 設定")
    parser.add_argument("--division", type=int, default=32,
                        help="k-meansによる領域数 (default 32)")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="パワー重み |g|^a の a（0≦a）")
    parser.add_argument("--lambda_eps", type=float, default=0,
                        help="ε‑スムージング係数 λ (0.05〜0.2 推奨)")
    parser.add_argument("--cfg_file", type=str, default="/workspace/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml",
                        help="PointNeXt configファイルのパス")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/PointNeXt/log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT/checkpoint/modelnet40ply2048-train-pointnext-s-ngpus1-seed8793-20240703-161207-98MoJTGkRQAnXf5pBHvGbT_ckpt_best.pth",
                        help="PointNeXt チェックポイントのパス")
    # ★ デバッグ用グリッド探索（カンマ区切りで複数値を渡す）
    parser.add_argument("--alpha_grid", type=str, default="",
                        help="例 \"0.2,0.4,0.6\"  ※空なら探索しない")
    parser.add_argument("--lambda_grid", type=str, default="",
                        help="例 \"0.05,0.10,0.15\"  ※空なら探索しない")
    parser.add_argument("--tau_kappa", type=float, default=0.05,
                        help="near‑zero 判定係数 κ (φ のメディアンに対する割合)")
    args = parser.parse_args()

    # ★ グリッド用リストを準備（空→デフォルト単値）
    alpha_list   = [float(s) for s in args.alpha_grid.split(",") if s.strip()] or [args.alpha]
    lambda_list  = [float(s) for s in args.lambda_grid.split(",") if s.strip()] or [args.lambda_eps]
    do_grid      = (len(alpha_list) > 1 or len(lambda_list) > 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 画像・PLY を保存するディレクトリを確保
    os.makedirs("result", exist_ok=True)

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
        #visualize_blocks(pc_A, blocks_A, window_name="Regions A", screenshot_file="result/heatmap_regions_A.png")
        baseline_A = compute_background_baseline(bg_pc, kmeans_A)
        #visualize_blocks_with_baseline(pc_A, blocks_A, baseline_A, window_name="Regions A + Baseline", screenshot_file="result/heatmap_RegionsA_baseline.png", radius=0.015)
        def shap_predict_block_mask_wrapper_A(mask_vectors):
            return shap_predict_block_mask(mask_vectors, blocks_A, pc_A, baseline_A, model, device)
        explainer_A = shap.KernelExplainer(shap_predict_block_mask_wrapper_A, np.zeros((1, len(blocks_A))))
        block_shap_values_A = explainer_A.shap_values(np.ones((1, len(blocks_A))), nsamples=args.pattern, l1_reg=0.0)
        block_shap_values_A = block_shap_values_A[target_class_index].reshape(-1)
        point_shap_vals_A, _ = compute_point_level_shap_values(pc_A, blocks_A, block_shap_values_A,
                                                            baseline_A, model, target_class_index, device,
                                                            alpha=args.alpha, lambda_eps=args.lambda_eps)
        # パートB
        blocks_B, kmeans_B = spatial_divide_into_blocks(pc_B, num_blocks=args.division)
        #visualize_blocks(pc_B, blocks_B, window_name="Regions B", screenshot_file="result/heatmap_regions_B.png")
        baseline_B = compute_background_baseline(bg_pc, kmeans_B)
        #visualize_blocks_with_baseline(pc_B, blocks_B, baseline_B, window_name="Regions B + Baseline", screenshot_file="result/heatmap_RegionsB_baseline.png", radius=0.015)
        def shap_predict_block_mask_wrapper_B(mask_vectors):
            return shap_predict_block_mask(mask_vectors, blocks_B, pc_B, baseline_B, model, device)
        explainer_B = shap.KernelExplainer(shap_predict_block_mask_wrapper_B, np.zeros((1, len(blocks_B))))
        block_shap_values_B = explainer_B.shap_values(np.ones((1, len(blocks_B))), nsamples=args.pattern, l1_reg=0.0)
        block_shap_values_B = block_shap_values_B[target_class_index].reshape(-1)
        point_shap_vals_B, _ = compute_point_level_shap_values(pc_B, blocks_B, block_shap_values_B,
                                                            baseline_B, model, target_class_index, device,
                                                            alpha=args.alpha, lambda_eps=args.lambda_eps)
        # 連結（2048点全体）
        integrated_shap = np.concatenate([point_shap_vals_A, point_shap_vals_B])
        block_shap_values = np.concatenate([block_shap_values_A, block_shap_values_B])
        blocks = blocks_A + [[idx+1024 for idx in b] for b in blocks_B]
        baseline_combined = baseline_A + baseline_B
        baseline_all      = baseline_combined        # ←後段で共通名を参照
        combined_pc       = pc
        raw_point_contrib = None     # 後で near‑zero 最適 α,λ が決まってから計算
    else:
        if pc.shape[0] > 1024:
            sampled_idx = np.random.choice(pc.shape[0], 1024, replace=False)
            pc = pc[sampled_idx, :]
        blocks, kmeans = spatial_divide_into_blocks(pc, num_blocks=args.division)
        #visualize_blocks(pc, blocks, window_name="Regions (single)", screenshot_file="result/heatmap_regions_single.png")
        baseline_blocks = compute_background_baseline(bg_pc, kmeans)
        visualize_blocks_with_baseline(pc, blocks, baseline_blocks, window_name="Regions + Baseline", screenshot_file="result/heatmap_Regions_baseline.png", radius=0.015)
        def shap_predict_block_mask_wrapper(mask_vectors):
            return shap_predict_block_mask(mask_vectors, blocks, pc, baseline_blocks, model, device)
        explainer = shap.KernelExplainer(shap_predict_block_mask_wrapper, np.zeros((1, len(blocks))))
        block_shap_values = explainer.shap_values(np.ones((1, len(blocks))), nsamples=args.pattern, l1_reg=0.0)
        block_shap_values = block_shap_values[target_class_index].reshape(-1)
        # ---------- ★ グリッド探索 ----------
        best_diff     = 1e9
        best_pair     = (args.alpha, args.lambda_eps)
        best_shap     = None
        max_var = -np.inf
        max_pair_var = (None, None)
        target_rate   = 0.4               # 30‑50 % →中心 0.4 を最適化
        for a in alpha_list:
            for l in lambda_list:
                shap_tmp, _ = compute_point_level_shap_values(
                            pc, blocks, block_shap_values,
                            baseline_blocks, model, target_class_index, device,
                            alpha=a, lambda_eps=l)
                # --- 分散を計算して最大を記録 ---
                var_tmp = np.var(shap_tmp)
                if var_tmp > max_var:
                    max_var = var_tmp
                    max_pair_var = (a, l)
                #print(f"[VAR] α={a}, λ={l} → var={var_tmp:.6f}")

                tau_abs  = args.tau_kappa * np.median(np.abs(shap_tmp))
                nz_rate  = np.mean(np.abs(shap_tmp) < tau_abs)
                #print(f"[GRID] α={a}, λ={l} → near‑zero={nz_rate*100:.1f}%")
                if 0.30 <= nz_rate <= 0.50:             # ★ 条件を満たせば即採用
                    best_pair = (a, l); best_shap = shap_tmp; best_diff = 0
                    break
                diff = abs(nz_rate - target_rate)
                if diff < best_diff:
                    best_diff = diff; best_pair = (a, l); best_shap = shap_tmp
            if best_diff == 0:
                break

        alpha_sel, lambda_sel = best_pair
        # --- 最大分散ペアをログ出力 ---
        print(f"[VARIANCE] max variance at α={max_pair_var[0]}, λ={max_pair_var[1]} → var={max_var:.6f}")
        integrated_shap = best_shap
        tau_sel = args.tau_kappa * np.median(np.abs(integrated_shap))
        print(f"[SELECT] α={alpha_sel}, λ={lambda_sel} を採用 (near-zero≈{np.mean(np.abs(integrated_shap)<tau_sel)*100:.1f}%)")

        # ---------- Region別 near‑zero 率をログ ----------
        for bi, blk in enumerate(blocks):
            if not blk:
                continue
            tau_blk = args.tau_kappa * np.median(np.abs(integrated_shap[blk]))
            rate_blk = np.mean(np.abs(integrated_shap[blk]) < tau_blk)
            #print(f"    block {bi:02d}: {rate_blk*100:.1f}% near-zero (<{tau_blk:.3e})")

        # 可視化やヒストグラムで使う寄与値をここで計算
        baseline_all       = baseline_blocks          # 共通名に束ねる
        combined_pc        = pc
        raw_point_contrib  = None     # 後で計算

    # ==========================================================
    # ★ ここから先は点数に依存しない共通処理  (α–λ グリッド探索を含む)
    # ==========================================================
    if do_grid:                          # 2048 でも 1024 でも同じ
        best_diff   = 1e9
        best_pair   = (args.alpha, args.lambda_eps)
        best_shap   = integrated_shap      # 既定値
        # --- 分散最大ペア探索用変数 ---
        max_var = -np.inf
        max_pair_var = (None, None)
        target_rate = 0.1           # 最適なexact‑zeroの設定値
        for a in alpha_list:
            for l in lambda_list:
                shap_tmp, _ = compute_point_level_shap_values(
                                combined_pc, blocks, block_shap_values,
                                baseline_all, model, target_class_index, device,
                                alpha=a, lambda_eps=l)
                # --- 分散計算・記録 ---
                var_tmp = np.var(shap_tmp)
                if var_tmp > max_var:
                    max_var = var_tmp
                    max_pair_var = (a, l)
                #print(f"[VAR] α={a}, λ={l} → var={var_tmp:.6f}")

                tau_abs = args.tau_kappa * np.median(np.abs(shap_tmp))
                nz      = np.mean(np.abs(shap_tmp) < tau_abs)
                #print(f"[GRID] α={a}, λ={l} → near‑zero={nz*100:.1f}%")
                if 0.30 <= nz <= 0.50:
                    best_pair, best_shap, best_diff = (a, l), shap_tmp, 0
                    break
                diff = abs(nz - target_rate)
                if diff < best_diff:
                    best_pair, best_shap, best_diff = (a, l), shap_tmp, diff
            if best_diff == 0:
                break
        alpha_sel, lambda_sel = best_pair
        integrated_shap = best_shap
        tau_sel2 = args.tau_kappa * np.median(np.abs(integrated_shap))
        zero_frac = np.mean(integrated_shap == 0.0)
        zero_cnt  = np.sum(integrated_shap == 0.0)

        print(f"α: {alpha_list}")
        print(f"λ: {lambda_list}")
        print(f"[ZERO] α={alpha_sel}, λ={lambda_sel} を採用 (near‑zero≈{np.mean(np.abs(integrated_shap)<tau_sel2)*100:.1f}%)")
        #print(f"[METRIC] exact‑zero points: {zero_frac*100:.2f}% ({zero_cnt}/{integrated_shap.size})")
        print(f"[VARIANCE] max variance at α={max_pair_var[0]}, λ={max_pair_var[1]} → var={max_var:.6f}")

    # ---------- Region別 near‑zero 率をログ ----------
    for bi, blk in enumerate(blocks):
        if not blk:         # 空Region
            continue
        tau_blk = args.tau_kappa * np.median(np.abs(integrated_shap[blk]))
        r       = np.mean(np.abs(integrated_shap[blk]) < tau_blk)
        #print(f"    block {bi:02d}: {r*100:.1f}% near-zero (<{tau_blk:.3e})")

    # Raw‑SHAP は α,λ が確定してから一度だけ計算
    if raw_point_contrib is None:
        raw_point_contrib = compute_point_level_contributions(combined_pc, blocks, baseline_all, model, target_class_index, device)
        # 完全0の点数をデバッグ出力
        zero_cnt = np.sum(raw_point_contrib == 0.0)
        print(f"[DEBUG] Raw-SHAP 完全ゼロ点数: {zero_cnt}/{raw_point_contrib.size}")

    # point_shap と同時に ω_p（weights）も取得
    integrated_shap, weights = compute_point_level_shap_values(
        combined_pc, blocks, block_shap_values, baseline_all,
        model, target_class_index, device,
        alpha=args.alpha, lambda_eps=args.lambda_eps
    )

    # --- 完全ゼロ点数をログ出力 (例: 123/2048) ---
    zero_cnt = int(np.sum(integrated_shap == 0.0))
    total_pts = integrated_shap.size
    print(f"[DEBUG] Point-SHAP 完全ゼロ点: {zero_cnt}/{total_pts}")

    # 「Raw SHAP（raw_point_contrib）* 極小化重み」を加算 → Point-SHAP同率（例：Point-SHAP=0 同士）でも、一貫した順位付け
    eps = np.finfo(float).eps
    sorted_idx = np.argsort(integrated_shap + raw_point_contrib * eps)[::-1]

    # ダウンサンプリング：統合SHAP値が高い上位 ds_points を選択
    top_indices = sorted_idx[:args.ds_points]
    ds_pc = combined_pc[top_indices, :]

    # 1. 各ヒストグラムの出力（MatplotlibによるPNG保存）
    # ヒストグラム1：各領域SHAP
    plot_region_histogram(block_shap_values, output_file="result/histogram_region_shap.png")
    # ヒストグラム2：各点SHAP（生の寄与）
    plot_raw_point_shap_histogram(raw_point_contrib, output_file="result/histogram_raw_shap.png")
    # ヒストグラム3：符号付きパワー重み ω_p
    plot_weights_histogram(weights, output_file="result/histogram_weights.png")
    # ヒストグラム4：Point‑SHAP (比例分配)
    plot_point_shap_histogram(integrated_shap, output_file="result/histogram_point_shap.png")


    # 2. 各ヒートマップの出力（Open3Dによる可視化＆スクリーンショット保存）
    # ヒートマップ1：RegionSHAP
#    visualize_region_shap(combined_pc, blocks, block_shap_values, window_name="Region SHAP Visualization", screenshot_file="result/heatmap_region_shap.png")
    # ヒートマップ2：各点SHAP（生の寄与）
#    visualize_raw_shap(combined_pc, raw_point_contrib, window_name="Raw SHAP Visualization", screenshot_file="result/heatmap_raw_shap.png")
    # ヒートマップ3：符号付きパワー重み ω_p
#    visualize_weights_heatmap(combined_pc, weights, window_name="Weight SHAP Visualization", screenshot_file="result/heatmap_weights.png")
    # ヒートマップ4：Point‑SHAP (比例分配 = Hybrid-SHAP)
#    visualize_point_shap(combined_pc, integrated_shap, window_name="Point SHAP Visualization", screenshot_file="result/heatmap_ point_shap.png")


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
#    o3d.visualization.draw_geometries([pcd_out], window_name="Downsampled Point Cloud (Red)")

if __name__ == "__main__":
    main()
