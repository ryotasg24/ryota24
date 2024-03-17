import os
import open3d as o3d

def convert_txt_to_pcd(input_dir, output_dir):
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)

        for file in os.listdir(subdir_path):
            if file.endswith(".txt"):
                file_path = os.path.join(subdir_path, file)
                pcd = o3d.io.read_point_cloud(file_path, format='xyz')

                output_subdir = os.path.join(output_dir, subdir)
                os.makedirs(output_subdir, exist_ok=True)

                output_file_path = os.path.join(output_subdir, file.replace(".txt", ".pcd"))
                o3d.io.write_point_cloud(output_file_path, pcd)

                print(f"Converted and saved: {output_file_path}")

if __name__ == '__main__':
    input_directory = "../../data/xyz_ModelNet40"
    output_directory = "../data/modelnet40/FPS"

    convert_txt_to_pcd(input_directory, output_directory)

