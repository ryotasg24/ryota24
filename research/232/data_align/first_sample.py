import os
import numpy as np

def sample_and_save(input_directory, output_directory, num_points=1024):
    for root, _, files in os.walk(input_directory):
        for filename in files:
            if filename.endswith(".txt"):
                input_file_path = os.path.join(root, filename)
                output_subdirectory = os.path.relpath(root, input_directory)
                output_subdirectory_path = os.path.join(output_directory, output_subdirectory)
                os.makedirs(output_subdirectory_path, exist_ok=True)
                output_file_path = os.path.join(output_subdirectory_path, filename)

                # Read point cloud data from input file
                point_cloud = np.loadtxt(input_file_path, delimiter=',').astype(np.float32)

                # Sample the first 'num_points' points
                sampled_points = point_cloud[:num_points, :]

                # Save the sampled points to the output file
                np.savetxt(output_file_path, sampled_points, delimiter=',')
                print(f"Sampled {num_points} points from {input_file_path} and saved to {output_file_path}")

# Example usage
input_directory = 'raw_modelnet40'  # Replace with the path to your input directory
output_directory = '1024modelnet40'  # Replace with the path to your output directory
sample_and_save(input_directory, output_directory)

