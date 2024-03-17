#入力点群をまとめたH5ファイルに対して、各点群ファイルにFPSを適用し、新たなH5ファイルとして保存するコード

import numpy as np
import os
import h5py


def load_h5(h5_filename):
    f = h5py.File(h5_filename, "r")
    data = np.array(f["data"][:])
    label = np.array(f["label"][:])
    f.close()
    return data, label

"""
# US -> FPS'
def sample_fps_new(point, n_samples):
    #point(numpy) --sampling--> selected_pts(numpy, n_samples)

    selected_pts = np.zeros((n_samples, 3))
    remaining_pts = np.copy(point)

    # Randomly pick a start
    #start_idx = np.random.randint(low=0, high=point.shape[0] - 1)
    #selected_pts[0] = remaining_pts[start_idx]
    #n_selected_pts = 1

    #最初に、先頭700点をサンプリング点とする
    if remaining_pts.shape[0] > 700:
        selected_pts[:700] = remaining_pts[:700]
        remaining_pts = remaining_pts[700:]
        n_selected_pts = 700
    else:
        selected_pts[:remaining_pts.shape[0]] = remaining_pts
        return selected_pts
    
    while n_selected_pts < n_samples:
        # Calculate distances from remaining points to selected points
        dist_pts_to_selected = np.linalg.norm(remaining_pts - selected_pts[n_selected_pts-1 : n_selected_pts], axis=1)
        #dist_pts_to_selected = np.linalg.norm(remaining_pts - selected_pts[np.random.randint(0, n_selected_pts)], axis=1)
        # Find the point with the maximum distance
        res_selected_idx = np.argmax(dist_pts_to_selected)
        selected_pts[n_selected_pts] = remaining_pts[res_selected_idx]

        n_selected_pts += 1

        # Remove the selected point from the remaining points
        remaining_pts = np.delete(remaining_pts, res_selected_idx, axis=0)

    return selected_pts
"""
# AVG
def avg_voxel_grid_sampling(point, voxel_size, avg_samples):
    # 入力点群データをボクセルサイズで分割
    voxel_grid = {}
    for i, pt in enumerate(point):
        voxel_coord = tuple((pt / voxel_size).astype(int))
        if voxel_coord in voxel_grid:
            voxel_grid[voxel_coord].append(i)
        else:
            voxel_grid[voxel_coord] = [i]

    # 各ボクセル内で座標平均を取り、代表点を得る
    sampled_pts = []
    for voxel_pts in voxel_grid.values():
        voxel_pts_np = np.array(voxel_pts)
        avg_coord = np.mean(point[voxel_pts_np], axis=0)
        sampled_pts.append(avg_coord)

    sampled_pts = np.array(sampled_pts)

    # 先頭の700点をサンプリング点とする
    if sampled_pts.shape[0] > avg_samples:
        sampled_pts = sampled_pts[:avg_samples]

    return sampled_pts

# AVG -> FPS
def sample_fps_new(point, n_samples, voxel_size, avg_samples):
    # AVGによるサンプリング
    avg_sampled_pts = avg_voxel_grid_sampling(point, voxel_size, avg_samples)
    
    # FPSによるダウンサンプリング
    selected_pts = np.zeros((n_samples, 3))     # initialize output_pc
    remaining_pts = np.copy(point)              # Never selected

    n_selected_pts = min(avg_samples, avg_sampled_pts.shape[0]) #AVGでのサンプリング点数
    selected_pts[:n_selected_pts] = avg_sampled_pts[:n_selected_pts]

    n_fps_selected_samples = n_samples - n_selected_pts #FPSでのサンプリング点数

    if n_fps_selected_samples > 0:
        N, D = remaining_pts.shape
        xyz = remaining_pts[:,:3]
        centroids = np.zeros((n_fps_selected_samples,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)

        for i in range(n_fps_selected_samples):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        selected_pts[n_selected_pts:] = point[centroids.astype(int)]

    return selected_pts


"""
# AVG -> FPS'
def sample_fps_new(point, n_samples, voxel_size):
    # AVGによるサンプリング
    avg_sampled_pts = avg_voxel_grid_sampling(point, voxel_size)
    
    # FPSによるダウンサンプリング
    selected_pts = np.zeros((n_samples, 3))
    remaining_pts = np.copy(point)


    n_selected_pts = min(n_samples, avg_sampled_pts.shape[0]) # =AVGで指定した点数
    selected_pts[:n_selected_pts] = avg_sampled_pts[:n_selected_pts]

    while n_selected_pts < n_samples:
        # Calculate distances from remaining points to selected points
        dist_pts_to_selected = np.linalg.norm(remaining_pts - selected_pts[n_selected_pts-1 : n_selected_pts], axis=1)
        # Find the point with the maximum distance
        res_selected_idx = np.argmax(dist_pts_to_selected)
        selected_pts[n_selected_pts] = remaining_pts[res_selected_idx]

        n_selected_pts += 1

        # Remove the selected point from the remaining points
        remaining_pts = np.delete(remaining_pts, res_selected_idx, axis=0)

    return selected_pts
"""
###############################################################################
def AF_sampling(pts, k):
    
    batch_size = pts.shape[0]
    a = [] 

    for i in range(batch_size):
        #print(pts[i])
        b = sample_fps_new(pts[i], k, 0.00001, 700)  # numpy -> numpy
        c = b.tolist()  # numpy -> list
        a.append(c)  # list -> list[list]

    return np.array(a)  # list[list] -> numpy


def save_h5(filename, data, label):
    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("label", data=label)


def load_and_save_h5(input_filename, output_filename, n_samples):
    if not os.path.exists(output_filename):
        print(f"Creating a new file: {output_filename}")
        data, label = load_h5(input_filename)
        #print(type(data))
        #print(data.shape)
        downsampled_data = AF_sampling(data, n_samples)
        #print(type(downsampled_data))
        #print(downsampled_data.shape)
        #print(downsampled_data[4])
        save_h5(output_filename, downsampled_data, label)
    else:
        print(f"File already exists: {output_filename}")


input_filename = "data/modelnet40_ply_hdf5_2048/ply_data_test0.h5"
output_filename = "data/modelnet40_ply_hdf5_2048/AF_test0.h5"
n_samples = 1024
load_and_save_h5(input_filename, output_filename, n_samples)