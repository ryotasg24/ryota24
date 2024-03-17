"""
#import open3d as o3d
#import numpy as np


def uniform_sample(point):
    import open3d as o3d

    # Down-sampling by uniform downsample
    ptCloud = point.uniform_down_sample(every_k_points=200) ###2500=4, 1000=10, 500=20, 250=40, 125=80,80=125, 50=200, 25=400

    # Saving point cloud
    print(ptCloud)
    
    #o3d.io.write_point_cloud("data/bunny/ds_bunny/UNI_ds.ply", ptCloud)
    o3d.io.write_point_cloud("data/modelnet40/test/US_001.ply", ptCloud)

def average_voxel_grid_sample(point):
    import open3d as o3d

    # Down-sampling by average voxel grid
    ptCloud = point.voxel_down_sample(voxel_size=0.58) ###2500=0.02671 , 1000=0.044 500=0.07138, 250=0.0995, 125=0.16234,80 =0.19,  50=0.29, 25=0.58

    # Saving point cloud
    print(ptCloud)
    
    #o3d.io.write_point_cloud("data/bunny/ds_bunny/AVG_ds.ply", ptCloud)
    o3d.io.write_point_cloud("data/modelnet40/test/AVG_001.ply", ptCloud)


##SampleNet : 235docker(sugi_test) -> reconstruction
"""
# ds_function.py

import os
import open3d as o3d
import numpy as np

def random_sample(point, output_folder, file_name):
    # Down-sampling by random downsample
    ptCloud = point.random_down_sample(1000/10000)

    # Saving point cloud
    save_path = os.path.join(output_folder, "RS_ds/1000/ply", f"{file_name[:-4]}.ply")
    o3d.io.write_point_cloud(save_path, ptCloud)

def uniform_sample(point, output_folder, file_name):
    # Down-sampling by uniform downsample
    ptCloud = point.uniform_down_sample(every_k_points=400) 

    # Saving point cloud
    save_path = os.path.join(output_folder, "US_ds/25/ply", f"{file_name[:-4]}.ply")
    o3d.io.write_point_cloud(save_path, ptCloud)

def average_voxel_grid_sample(point, output_folder, file_name):
    # Down-sampling by average voxel grid
    ptCloud = point.voxel_down_sample(voxel_size=0.19)

    # Saving point cloud
    save_path = os.path.join(output_folder, "AVG_ds/80/ply", f"{file_name[:-4]}.ply")
    o3d.io.write_point_cloud(save_path, ptCloud)

