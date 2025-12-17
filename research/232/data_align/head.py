import os
import open3d as o3d
import numpy as np

def reduce_and_save_point_cloud(input_folder, output_folder, target_points_to_keep):
    # Iterate over files in the input folder
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                # Load the point cloud from the txt file
                file_path = os.path.join(subdir, file)
                point_cloud = np.loadtxt(file_path)

                # Take the specified number of points
                reduced_point_cloud = point_cloud[:target_points_to_keep]

                # Convert the numpy array to an Open3D point cloud
                ptCloud = o3d.geometry.PointCloud()
                ptCloud.points = o3d.utility.Vector3dVector(reduced_point_cloud)

                # Define the save path for the PLY file
                save_path = os.path.join(output_folder, "Head_ds/1000/ply", f"{file[:-4]}.ply")

                # Save the point cloud as a PLY file
                o3d.io.write_point_cloud(save_path, ptCloud)

# Example usage
input_folder_path = "../data/xyz_ModelNet40"
output_folder_path = "data/modelnet40"
target_points_to_keep = 1000
reduce_and_save_point_cloud(input_folder_path, output_folder_path, target_points_to_keep)

