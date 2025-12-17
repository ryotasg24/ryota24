#!/usr/bin/env python
# coding: utf-8
"""
Visualize part-segmentation result of a single ShapeNet Part point cloud
using a pretrained PointNeXt checkpoint.

例:
  python examples/shapenetpart/visualize_partseg.py \
      --cfg cfgs/shapenetpart/pointnext-s.yaml \
      --pretrained_path  /workspace/PointNeXt/log/shapenetpart/shapenetpart-train-pointnext-s-ngpus1-seed2621-20250612-074205-9QFombJZY6hbJFtx2VVHML/checkpoint/shapenetpart-train-pointnext-s-ngpus1-seed2621-20250612-074205-9QFombJZY6hbJFtx2VVHML_ckpt_latest.pth \
      --vis_class airplane \
      --vis_idx 0

class: Airplane, Bag, Cap, Car, Chair, Earphone, Guitar, Knife, Lamp, Laptop, Motorbike, Mug, Pistol, Rocket, Skateboard, Table
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, yaml, torch, numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from openpoints.utils import EasyConfig, set_random_seed
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.models import build_model_from_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--vis_class', default='chair')
    parser.add_argument('--vis_idx', type=int, default=0)
    parser.add_argument('--save_ply', default='', help='output ply path (auto if empty)')
    parser.add_argument('--seed', type=int, default=0)
    args, opts = parser.parse_known_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.batch_size = 1
    cfg.is_training = False
    set_random_seed(args.seed)

    val_loader = build_dataloader_from_cfg(
        batch_size=1,
        dataset_cfg=cfg.dataset,
        dataloader_cfg=cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split='val',
        distributed=False)

    cls2idx = {name.capitalize(): idx for idx, name in enumerate(val_loader.dataset.classes)}
    cls_idx = cls2idx[args.vis_class.capitalize()]

    cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda().eval()
    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)

    samples = [d for d in val_loader.dataset if d['cls'].item() == cls_idx]
    assert len(samples) > args.vis_idx, 'vis_idx exceeds samples in class'
    sample = samples[args.vis_idx]

    batch = {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v) for k, v in sample.items()}
    batch['x'] = get_features_by_keys(batch, cfg.feature_keys)

    with torch.no_grad():
        logits = model(batch)
    pred = logits.argmax(1)[0].cpu().numpy()

    num_parts = pred.max() + 1
    palette = plt.get_cmap('tab20')(np.linspace(0, 1, num_parts))[:, :3]
    colors = palette[pred]

    pos_vis = batch['pos'][0].cpu().numpy()            # (N,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos_vis)
    pcd.colors = o3d.utility.Vector3dVector(colors)


    # 1) 出力ファイル名が指定されていなければ自動生成
    if not args.save_ply:
        args.save_ply = f'/workspace/PointNeXt/partseg_result/visualize/{args.vis_class}_{args.vis_idx}.ply'

    # 2) 保存してメッセージ
    o3d.io.write_point_cloud(args.save_ply, pcd)
    print(f'Saved to {args.save_ply}')

    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
