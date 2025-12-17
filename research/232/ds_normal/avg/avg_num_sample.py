import os
import open3d as o3d
import numpy as np

def uniform_downsample_and_save(input_path, output_path, target_points):
    for root, _, files in os.walk(input_path):
        for filename in files:
            if filename.endswith(".txt"):
                input_file_path = os.path.join(root, filename)
                output_subdirectory = os.path.relpath(root, input_path)
                output_subdirectory_path = os.path.join(output_path, output_subdirectory)
                os.makedirs(output_subdirectory_path, exist_ok=True)
                output_file_path = os.path.join(output_subdirectory_path, filename)

                point_cloud = np.loadtxt(input_file_path, delimiter=',').astype(np.float32)
                num_points = point_cloud.shape[0]

                if num_points < target_points:
                    print(f"Skipping {input_file_path}: Insufficient points for downsampling")
                    continue

                # target_points
                target = int(num_points / target_points)
                target = max(target, 1)  # without =0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

                # uniform_down_sample
                downsampled_pcd = pcd.uniform_down_sample(target)

                downsampled_points = np.asarray(downsampled_pcd.points)

                #random for selected_points
                if downsampled_points.shape[0] > target_points:
                    indices = np.random.choice(downsampled_points.shape[0], target_points, replace=False)
                    downsampled_points = downsampled_points[indices, :]

                np.savetxt(output_file_path, downsampled_points, delimiter=',')
                print(f"Downsampled {input_file_path} and saved to {output_file_path}")

input_directory = '80/txt'
output_directory = '5000/txt25'
target_points = 50
uniform_downsample_and_save(input_directory, output_directory, target_points)

