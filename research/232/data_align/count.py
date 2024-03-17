import numpy as np

def count_points_in_txt_file(file_path):
    try:
        point_cloud = np.loadtxt(file_path, delimiter=',')
        num_points = point_cloud.shape[0]
        return num_points
    except Exception as e:
        print(f"Error counting points in {file_path}: {str(e)}")
        return None


txt_file_path = 'modelnet40/Head_ds/50/txt/curtain_0102.txt'


num_points = count_points_in_txt_file(txt_file_path)

if num_points is not None:
    print(f"The number of points in {txt_file_path} is: {num_points}")
else:
    print(f"Failed to count points in {txt_file_path}")

