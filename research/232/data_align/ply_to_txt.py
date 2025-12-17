import os
import numpy as np
from plyfile import PlyData

def convert_ply_to_txt(input_folder, output_folder):
    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".ply"):
                ply_path = os.path.join(subdir, file)
                ply_data = PlyData.read(ply_path)
                vertex_data = ply_data['vertex']

                points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
                txt_path = os.path.join(output_folder, file.replace(".ply", ".txt"))
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)

                with open(txt_path, "w") as txt_file:
                    for point in points:
                        txt_file.write(f"{point[0]:.6f},{point[1]:.6f},{point[2]:.6f}\n")

# Set input and output folders
#input_folder = "/code/shape_net_core_uniform_samples_2048/ply"
input_folder = "/code/ds_normal/data/gpcc/S25/ply"
#output_folder = "/code/shape_net_core_uniform_samples_2048/txt"
output_folder = "/code/ds_normal/data/gpcc/S25/txt"

# Convert .ply to .txt
convert_ply_to_txt(input_folder, output_folder)

