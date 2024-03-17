# sampling.py

import os
import ds_function as func
import open3d as o3d
import numpy as np
import time

def process_folder(input_folder, output_folder):
    total_rs_time = 0.0
    total_us_time = 0.0
    total_avg_time = 0.0
    num_files = 0

    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                pcd = o3d.io.read_point_cloud(file_path, format='xyz')
                
                # RS
                start = time.time()
                #func.random_sample(pcd, output_folder, file)
                end = time.time()
                rs_time_diff = end - start
                total_rs_time += rs_time_diff

                # US
                start = time.time()
                #func.uniform_sample(pcd, output_folder, file)
                end = time.time()
                us_time_diff = end - start
                total_us_time += us_time_diff

                # AVG
                start = time.time()
                func.average_voxel_grid_sample(pcd, output_folder, file)
                end = time.time()
                avg_time_diff = end - start
                total_avg_time += avg_time_diff

                num_files += 1
                
                #print(f"{file} RS_sample time: {rs_time_diff:.6f} seconds")
                #print(f"{file} US_sample time: {uni_time_diff:.6f} seconds")
                #print(f"{file} AVG_sample time: {avg_time_diff:.6f} seconds")
    
    #avg_rs_time = total_rs_time / num_files if num_files > 0 else 0.0
    #avg_us_time = total_us_time / num_files if num_files > 0 else 0.0
    avg_avg_time = total_avg_time / num_files if num_files > 0 else 0.0

    #print(f"Total RS_sample time: {total_rs_time:.6f} seconds")
    #print(f"Total US_sample time: {total_us_time:.6f} seconds")
    print(f"Total AVG_sample time: {total_avg_time:.6f} seconds")
    #print(f"Number of files processed: {num_files}")
    #print(f"Average RS_sample time per file: {avg_rs_time:.6f} seconds")
    #print(f"Average US_sample time per file: {avg_us_time:.6f} seconds")
    print(f"Average AVG_sample time per file: {avg_avg_time:.6f} seconds")

# Set input and output folders
input_folder = "../data/xyz_ModelNet40"
output_folder = "data/modelnet40"

# Process the folders
process_folder(input_folder, output_folder)

