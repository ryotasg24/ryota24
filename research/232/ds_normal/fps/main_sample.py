import open3d as o3d
import numpy as np
import argparse

from FPS_ds.load_pcd import load_pcd
#from FPS_ds.fps_v0 import FPS      # Simple loop
from FPS_ds.fps_v1 import FPS      # Utilise broadcasting


def farthest_point_sample(point):
    
    #convert .ply->.pcd
    o3d.io.write_point_cloud("./FPS_ds/pcd.pcd", point)
    
    #parser : Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./FPS_ds/pcd.pcd",
                        help="Load some points data, choices are \"bunny\", \"circle\", \"eclipse\", "
                             "or \"a_path_to_your_ply_file\".")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples we would like to draw.")
    parser.add_argument("--manually_step", type=bool, default=False,
                        help="Hit \"N/n\" key to step sampling forward once.")

    args = parser.parse_args()

    example_data = args.data
    n_samples = args.n_samples
    manually_step = args.manually_step

    pcd_xyz = load_pcd(example_data)
    print("Loaded ", example_data, "with shape: ", pcd_xyz.shape)

    if n_samples > pcd_xyz.shape[0]:
        print("WARNING: required {0:d} samples but the loaded point cloud only has {1:d} points.\n "
              "Change the n_sample to {2:d}.".format(n_samples, pcd_xyz.shape[0], pcd_xyz.shape[0]))
        print("WARNING: sampling")
        n_samples = pcd_xyz.shape[0]

    fps = FPS(pcd_xyz, n_samples)
    print("Initialised FPS sampler successfully.")
    print("Running FPS over {0:d} points and geting {1:d} samples.".format(pcd_xyz.shape[0], n_samples))

    # Init visualisation
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(fps.pcd_xyz)
    #pcd_all.paint_uniform_color([0, 1, 0])  # original: green

    pcd_selected = o3d.geometry.PointCloud()

    if manually_step is False:
        fps.fit()  # Get all samples.
        print("FPS sampling finished.")

        pcd_selected.points = o3d.utility.Vector3dVector(fps.get_selected_pts())
        pcd_selected.paint_uniform_color([1, 0, 0])  # selected: red

        #o3d.visualization.draw_geometries([pcd_all, pcd_selected])
        #print("You can step the sampling process by setting \"--manually_step\" to True and press \"N/n\".")

        #Save sampled point cloud
        o3d.io.write_point_cloud("data/bunny/ds_bunny/FPS_ds.ply", pcd_selected)

    else:

        def fit_step_callback(vis):
            fps.step() # Get ONE new sample

            pcd_selected.points = o3d.utility.Vector3dVector(fps.get_selected_pts())
            pcd_selected.paint_uniform_color([1, 0, 0])  # selected:  red
            vis.update_geometry()


        key_to_callback = {ord("N"): fit_step_callback}

        # Draw the first sampled points.
        #pcd_selected.points = o3d.utility.Vector3dVector(fps.get_selected_pts())
        #pcd_selected.paint_uniform_color([1, 0, 0])  # selected: red

        # Draw a new sample points every time press "N/n" key.
       # o3d.visualization.draw_geometries_with_key_callbacks([pcd_all, pcd_selected], key_to_callback)
