import os
import open3d as o3d
import numpy as np

def convert_txt_to_pcd(txt_file_path, output_dir):
    # Load data from txt file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        points = [list(map(float, line.strip().split(', '))) for line in lines]

    # Convert to NumPy array
    points_array = np.array(points)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)

    # Get output file path
    file_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    output_file_path = os.path.join(output_dir, file_name + ".pcd")

    # Save point cloud as pcd file
    o3d.io.write_point_cloud(output_file_path, pcd)

def convert_all_txt_to_pcd(input_dir, output_dir):
    # Iterate over subdirectories
    for subdir_name in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir_name)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Create output subdirectory if not exists
            output_subdir = os.path.join(output_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)

            # Iterate over txt files in the subdirectory
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".txt"):
                    txt_file_path = os.path.join(subdir_path, file_name)
                    convert_txt_to_pcd(txt_file_path, output_subdir)

if __name__ == '__main__':
    # Specify input and output directories
    input_directory = "/code/loss9_SampleNet1000"
    output_directory = "/code/ds_normal/data/modelnet40/SampleNet_ds/Propose1000_pcd/"

    # Convert all txt files to pcd in the specified directories
    convert_all_txt_to_pcd(input_directory, output_directory)

