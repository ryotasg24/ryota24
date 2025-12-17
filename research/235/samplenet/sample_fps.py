#Tensorflowで書かれた点群データに対して、tfを用いてFPSを適用し、Tensorflow配列で出力するコードテスト

import numpy as np
import os
import time
import argparse
#from fps_v1 import FPS
#from load2_pcd import load_pcd
import tensorflow as tf

def sample_fps():
    #point(numpy) --sampling--> selected_pts(numpy, n_samples)
    input_point_cloud = tf.constant([[0.1, 0.1, 0.1],[1.0, 1.0, 1.0],[2.1, 2.1, 2.1],[3.0, 3.0, 3.0],[4.0, 4.0, 4.0],[5.0, 5.0, 5.0]])
    n_samples=3
    selected_pts = tf.zeros((n_samples, 3))
    remaining_pts = tf.identity(input_point_cloud)

    # Randomly pick a start
    start_idx = 1
    selected_pts[0] = remaining_pts[start_idx]
    n_selected_pts = 1
    
    while n_selected_pts < n_samples:
        # Calculate distances from remaining points to selected points
        dist_pts_to_selected = tf.norm(remaining_pts - selected_pts[n_selected_pts-1 : n_selected_pts], axis=1)
        
        # Find the point with the maximum distance
        res_selected_idx = tf.argmax(dist_pts_to_selected)
        selected_pts[n_selected_pts] = remaining_pts[res_selected_idx]

        n_selected_pts += 1

        # Remove the selected point from the remaining points
        remaining_pts = tf.gather(remaining_pts, indices=tf.range(tf.shape(remaining_pts)[0]), axis=0)
        remaining_pts = tf.gather(remaining_pts, indices=tf.where(tf.not_equal(tf.range(tf.shape(remaining_pts)[0]), res_selected_idx)), axis=0)


    return selected_pts


if __name__ == '__main__':
    sess = tf.Session()
    sampled_points = sess.run(sample_fps())
    print(sampled_points)