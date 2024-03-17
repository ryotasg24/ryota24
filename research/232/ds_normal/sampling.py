#import Module 
import ds_function as func
from FPS_ds import main_sample as fps
import open3d as o3d
import numpy as np
import time

#import Dataset
print("Loading point cloud (format:ply)")
#pcd = o3d.io.read_point_cloud("data/bunny/raw_bunny/bun000.ply", format='ply')
pcd = o3d.io.read_point_cloud("../data/xyz_ModelNet40/chair/chair_0001.txt", format='xyz')

#Raw data
print(pcd)

#UNI
start = time.time()
func.uniform_sample(pcd)
end = time.time()
time_dff = end-start
print("UNI_sample time : ", end="")
print(time_dff)

#AVG
start = time.time()
func.average_voxel_grid_sample(pcd)
end = time.time()
time_dff = end-start
print("AVG_sample time : ",end="")
print(time_dff)

#FPS
start = time.time()
#fps.farthest_point_sample(pcd)
end = time.time()
time_dff = end-start
print("FPS_sample time : ",end="")
print(time_dff)
