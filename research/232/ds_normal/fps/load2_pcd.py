import open3d as o3d
import numpy as np
import os
import math


def __points_on_circle__(radius, num_pts):
    pi = math.pi

    # a circle in 2D
    result_xy = np.asarray([(math.cos(2 * pi / num_pts * x) * radius,
                             math.sin(2 * pi / num_pts * x) * radius) for x in range(0, num_pts)])

    # put it in 3D by adding z = 0
    result_xyz = np.append(result_xy, np.zeros((num_pts, 1)), axis=1)
    return result_xyz


def __points_on_eclipse__(radius, num_pts):
    pi = math.pi
    eclipse_factor = 2

    # a circle in 2D
    result_xy = np.asarray([(math.cos(2 * pi / num_pts * x) * radius,
                             eclipse_factor * math.sin(2 * pi / num_pts * x) * radius) for x in range(0, num_pts)])

    # put it in 3D by adding z = 0
    result_xyz = np.append(result_xy, np.zeros((num_pts, 1)), axis=1)
    return result_xyz

def load_pcd(data_folder="../data/modelnet40/FPS"):
    pcd_xyz = None

    # Assuming there is only one pcd file in the specified folder
    for file in os.listdir(data_folder):
        if file.endswith(".pcd"):
            file_path = os.path.join(data_folder, file)
            pcd = o3d.io.read_point_cloud(file_path, format='pcd')
            pcd_xyz = np.asarray(pcd.points)
            break

    return pcd_xyz

if __name__ == '__main__':
    data_xyz = load_pcd("../data/modelnet40/FPS")
    print("Loaded data with shape:", data_xyz.shape)
