import os
import open3d as o3d

def convert_txt_to_pcd(input_folder, output_folder):
    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        # Process each txt file in the current directory
        for file_name in files:
            if file_name.endswith(".txt"):
                # Construct full file paths
                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(output_folder, os.path.relpath(input_file_path, input_folder).replace(".txt", ".pcd"))

                # Read txt file and create a PointCloud
                points = []
                with open(input_file_path, 'r') as txt_file:
                    for line in txt_file:
                        x, y, z = map(float, line.split())
                        points.append([x, y, z])

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # Write PointCloud to pcd file
                o3d.io.write_point_cloud(output_file_path, pcd)

                #print(f"Converted: {input_file_path} -> {output_file_path}")

if __name__ == '__main__':
    input_directory = "/code/ds_normal/data/modelnet40/SampleNet/SampleNet1000"      
    output_directory = "/code/ds_normal/data/modelnet40/SampleNet_ds/SampleNet1000_pcd"  

    convert_txt_to_pcd(input_directory, output_directory)

