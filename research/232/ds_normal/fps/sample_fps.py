import open3d as o3d
import numpy as np
import os
import time
import argparse
from fps_v1 import FPS
from load2_pcd import load_pcd

def farthest_point_sample(point, n_samples, output_folder, file_name):
    
    output_file_path = os.path.join(output_folder, file_name.replace(".pcd", f".ply"))

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        #print(f"File '{output_file_path}' already exists. Skipping sampling.")
        return 0.0  # Return 0 processing time for skipped file
    
    pcd_selected = o3d.geometry.PointCloud()

    start_time = time.time()

    fps = FPS(point, n_samples)
    fps.fit()  # Get all samples.
    print("FPS sampling finished.")

    pcd_selected.points = o3d.utility.Vector3dVector(fps.get_selected_pts())
    # Save sampled point cloud with the same name as the input file
    o3d.io.write_point_cloud(output_file_path, pcd_selected)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.6f} seconds")

    return elapsed_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--manually_step", type=bool, default=False,
                        help="Hit \"N/n\" key to step sampling forward once.")
    ############################################################################
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples to obtain using FPS.")
    ############################################################################
    args = parser.parse_args()

    #data_folder = "../"
    #SampleNet
    data_folder = "/code/ds_normal/data/modelnet40/SampleNet_ds/Propose1000_pcd/"
    output_folder = "/code/ds_normal/data/modelnet40/SampleNet_ds/Add_1000/ply"
    ####################################################################batch_size###########
    manually_step = args.manually_step
    n_samples = args.n_samples

    # Assuming each subdirectory contains a pcd file
    subdirs = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

    total_time = 0.0
    total_samples = 0

    for subdir in subdirs:
        subdir_path = os.path.join(data_folder, subdir)
        input_point_cloud = load_pcd(subdir_path)

        if input_point_cloud is not None:
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".pcd"):
                    elapsed_time = farthest_point_sample(input_point_cloud, n_samples, output_folder, file_name)

                    total_time += elapsed_time
                    total_samples += n_samples

    average_time = total_time / total_samples if total_samples > 0 else 0.0
    print(f"\nAverage processing time per sample: {average_time:.6f} seconds")

