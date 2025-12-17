# クラス 'airplane' の 3 番目 (0,1,2,3… の 3)
# $ python visualize_h5.py <h5_file_path> airplane 3

import sys
import h5py
import numpy as np
import open3d as o3d

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel",
    "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood",
    "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand",
    "vase", "wardrobe", "xbox"
]

def show_point_cloud_of_class(h5_file_path, class_name, instance_idx=0):
    """
    HDF5 ファイルから <class_name> クラスの instance_idx 番目の点群を表示する。
    instance_idx を省略すると 0 (最初)。
    """
    if class_name not in MODELNET40_CLASSES:
        print(f"クラス名 '{class_name}' は ModelNet40 にありません。")
        return

    target_label = MODELNET40_CLASSES.index(class_name)

    # データ読み込み
    with h5py.File(h5_file_path, "r") as h5_file:
        data   = h5_file["data"][:]   # (N, P, C)
        labels = h5_file["label"][:]  # (N,) or (N,1)

    # 指定クラスのインデックス一覧を取得
    class_indices = [
        i for i, lab in enumerate(labels)
        if int(lab[0] if hasattr(lab, "__len__") and len(lab) > 1 else lab) == target_label
    ]

    if not class_indices:
        print(f"クラス '{class_name}' はファイル内にありません。")
        return
    if instance_idx < 0 or instance_idx >= len(class_indices):
        print(f"instance_idx={instance_idx} は範囲外です。最大 {len(class_indices)-1} まで。")
        return

    target_index = class_indices[instance_idx]
    print(f"クラス '{class_name}' の {instance_idx} 番目はデータインデックス {target_index} です。")

    points = data[target_index, :, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    red = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(red)

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python h5_Visualize.py <h5_file_path> airplane 0")
        sys.exit(1)

    h5_path    = sys.argv[1]
    cls_name   = sys.argv[2]
    inst_idx   = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    show_point_cloud_of_class(h5_path, cls_name, inst_idx)
