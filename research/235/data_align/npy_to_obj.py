import numpy as np

# Numpyファイルのパス
npy_file = 'reconstructions_test_set_multi_0016.npy'

# Numpyファイルを読み込む
point_cloud = np.load(npy_file)

def save_point_cloud_as_obj(point_cloud, obj_file):
    with open(obj_file, 'w') as f:
        for point in point_cloud:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")

# OBJファイルのパス
obj_file = 'SampleNet00/ds.obj'

# OBJファイルに変換
save_point_cloud_as_obj(point_cloud, obj_file)
