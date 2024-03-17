from builtins import range
import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
sys.path.append("/SampleNet/classification")
import tf_util
from structural_losses.tf_nndistance import nn_distance
from structural_losses.tf_approxmatch import approx_match

######################loss4###########################################################

def sample_fps(point, n_samples):
    #point(numpy) --sampling--> selected_pts(numpy, n_samples)

    selected_pts = np.zeros((n_samples, 3))
    remaining_pts = np.copy(point)

    # Randomly pick a start
    start_idx = np.random.randint(low=0, high=point.shape[0] - 1)
    selected_pts[0] = remaining_pts[start_idx]
    n_selected_pts = 1
    
    while n_selected_pts < n_samples:
        # Calculate distances from remaining points to selected points
        dist_pts_to_selected = np.linalg.norm(remaining_pts - selected_pts[n_selected_pts-1 : n_selected_pts], axis=1)
        
        # Find the point with the maximum distance
        res_selected_idx = np.argmax(dist_pts_to_selected)
        selected_pts[n_selected_pts] = remaining_pts[res_selected_idx]

        n_selected_pts += 1

        # Remove the selected point from the remaining points
        remaining_pts = np.delete(remaining_pts, res_selected_idx, axis=0)

    return selected_pts

def farthest_point_sample(pts, k):
    #Tensorflow配列をnumpy配列に変換
    #FPS関数を実行して点群削減, numpy配列で出力
    #b(numpy)->c(リスト型座標)->a(リスト型点群)->return(Tensorflow) 
    with tf.device("/GPU:0"):
        batch_size = pts.shape[0]
        num_points = pts.shape[1]
        a = [] 
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True  # Allow GPU memory growth
    #config.gpu_options.visible_device_list = '0'  # Specify GPU device
    #with tf.Session(config=config) as sess:
        for i in range(batch_size):
            print(pts[i])
            pts_np = pts[i].eval(session=tf.compat.v1.Session())#sess)
            b = sample_fps(pts_np, k)  # numpy -> numpy
            c = b.tolist()  # numpy -> list
            a.append(c)  # list -> list[list]

    return tf.convert_to_tensor(a, dtype=tf.float32)  # list[list] -> TensorFlow


###上記処理、データロードに移行のため不使用
#######################################################################################


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(
    point_cloud, is_training, num_output_points, bottleneck_size, bn_decay=None
):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    #NUM_IN_POINT=1024→2048に変更、入力点群数を2048点とする。=(バッチ、点数、3)
    #FPSによって点数を1024点に削減
    #print(tf.shape(point_cloud))
    #point_cloud = random_point_dropout(point_cloud, 1024)
    #point_cloud = average_voxel_grid_sampling(point_cloud, 1024)
    print("############################get_model_InputData.shape#################################")
    print(point_cloud.shape)
    #point_cloud = farthest_point_sample(point_cloud, 1024)
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(
        input_image,
        64,
        [1, 3],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv1",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        64,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv2",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        64,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv3",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        128,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv4",
        bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net,
        bottleneck_size,
        [1, 1],
        padding="VALID",
        stride=[1, 1],
        bn=True,
        is_training=is_training,
        scope="conv5",
        bn_decay=bn_decay,
    )

    net = tf_util.max_pool2d(net, [num_point, 1], padding="VALID", scope="maxpool")

    ###############################loss4
    # FPS Downsampling
    #sampled_points = furthest_point_sample(net, num_output_points)

    # Feature propagation (MLP implemented as fully connected layers)
    net = tf.reshape(net, [batch_size, -1])
    #net = tf.reshape(sampled_points, [batch_size, -1])

    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope="fc11b", bn_decay=bn_decay
    )
    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope="fc12b", bn_decay=bn_decay
    )
    net = tf_util.fully_connected(
        net, 256, bn=True, is_training=is_training, scope="fc13b", bn_decay=bn_decay
    )
    net = tf_util.fully_connected(
        net,
        3 * num_output_points,
        bn=True,
        is_training=is_training,
        scope="fc14b",
        bn_decay=bn_decay,
        activation_fn=None,
    )

    out_point_cloud = tf.reshape(net, [batch_size, -1, 3])
    
    #return point_cloud 
    return out_point_cloud

#######################################################################################

def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def fps_from_given_pc(pts, k, given_pc):
    farthest_pts = np.zeros((k, 3))
    t = np.size(given_pc) // 3
    farthest_pts[0:t] = given_pc

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    for i in range(t, k):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def unique(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def nn_matching(full_pc, idx, k, complete_fps=True):
    batch_size = np.size(full_pc, 0)
    out_pc = np.zeros((full_pc.shape[0], k, 3))
    for ii in range(0, batch_size):
        best_idx = idx[ii]
        if complete_fps:
            best_idx = unique(best_idx)
            out_pc[ii] = fps_from_given_pc(full_pc[ii], k, full_pc[ii][best_idx])
        else:
            out_pc[ii] = full_pc[ii][best_idx]
    return out_pc[:, 0:k, :]


def emd_matching(full_pc, gen_pc, sess):
    batch_size = np.size(full_pc, 0)
    k = np.size(gen_pc, 1)
    out_pc = np.zeros_like(gen_pc)

    match_mat_tensor = approx_match(
        tf.convert_to_tensor(full_pc), tf.convert_to_tensor(gen_pc)
    )
    pc1_match_idx_tensor = tf.cast(tf.argmax(match_mat_tensor, axis=2), dtype=tf.int32)

    pc1_match_idx = pc1_match_idx_tensor.eval(session=sess)

    for ii in range(0, batch_size):
        best_idx = unique(pc1_match_idx[ii])
        out_pc[ii] = fps_from_given_pc(full_pc[ii], k, full_pc[ii][best_idx])

    return out_pc


def get_nn_indices(ref_pc, samp_pc):
    _, idx, _, _ = nn_distance(samp_pc, ref_pc)
    return idx


def get_simplification_loss(ref_pc, samp_pc, pc_size, gamma=1, delta=0):
    cost_p1_p2, _, cost_p2_p1, _ = nn_distance(samp_pc, ref_pc)
    max_cost = tf.reduce_max(cost_p1_p2, axis=1)
    max_cost = tf.reduce_mean(max_cost)
    cost_p1_p2 = tf.reduce_mean(cost_p1_p2)
    cost_p2_p1 = tf.reduce_mean(cost_p2_p1)
    loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1

    tf.summary.scalar("cost_p1_p2", cost_p1_p2)
    tf.summary.scalar("cost_p2_p1", cost_p2_p1)
    tf.summary.scalar("max_cost", max_cost)

    return loss



####################loss4#######################################################
#ランダムサンプリングによる点群をpointnet層に入力
def random_point_dropout(point_cloud, num_points):
    
    batch_size, num_input_points, _ = point_cloud.get_shape().as_list()

    # Generate random indices for selecting points
    indices = tf.range(0, num_input_points)
    indices = tf.random.shuffle(indices)[:num_points]

    # Gather the selected points
    sampled_point_cloud = tf.gather(point_cloud, indices, axis=1)

    return sampled_point_cloud

#AVGによる点群をpointnet層に入力
def average_voxel_grid_sampling(point_cloud, num_output_points):
    
    batch_size, num_points, _ = point_cloud.get_shape().as_list()

    # Calculate voxel size to achieve the desired output point count
    voxel_size = num_points / num_output_points

    # Normalize point cloud to voxel coordinates
    normalized_coords = point_cloud / voxel_size

    # Floor each coordinate to get voxel index
    voxel_indices = tf.floor(normalized_coords)

    # Calculate voxel-wise average
    unique_voxels, _ = tf.unique(tf.cast(voxel_indices, dtype=tf.int32), axis=1)
    sampled_point_cloud = tf.TensorArray(dtype=tf.float32, size=num_output_points)

    def gather_voxel_average(i, sampled_point_cloud):
        voxel_mask = tf.reduce_all(tf.equal(voxel_indices, unique_voxels[:, i:i+1, :]), axis=-1)
        voxel_points = tf.boolean_mask(point_cloud, voxel_mask)
        voxel_average = tf.reduce_mean(voxel_points, axis=0)
        sampled_point_cloud = sampled_point_cloud.write(i, voxel_average)
        return i + 1, sampled_point_cloud

    _, sampled_point_cloud = tf.while_loop(
        cond=lambda i, sampled_point_cloud: i < tf.minimum(tf.shape(unique_voxels)[1], num_output_points),
        body=gather_voxel_average,
        loop_vars=[0, sampled_point_cloud]
    )

    sampled_point_cloud = sampled_point_cloud.stack()

    return sampled_point_cloud

#######################################################################################

"""
def fps(pts, num_output_points):
    
    batch_size, num_points, _ = tf.unstack(tf.shape(pts))

    # Initialize sampled points array
    sampled_pts = tf.TensorArray(dtype=tf.float32, size=num_output_points)

    # Choose the first point randomly
    farthest = tf.random.uniform(shape=[], maxval=num_points, dtype=tf.int32)
    farthest = tf.expand_dims(farthest, axis=0)

    def cond(i, sampled_pts, farthest):
        return i < num_output_points

    def body(i, sampled_pts, farthest):
        # Gather the farthest point
        farthest_point = tf.gather_nd(pts, farthest)

        # Update the sampled points tensor
        sampled_pts = sampled_pts.write(i, farthest_point)

        # Update the distance
        distances = tf.reduce_sum((pts - farthest_point) ** 2, axis=-1)
        distances = tf.expand_dims(distances, axis=-1)
        
        # Choose the next farthest point
        farthest = tf.argmax(distances, axis=1)
        farthest = tf.cast(farthest, dtype=tf.int32)
        farthest = tf.expand_dims(farthest, axis=0)

        return i + 1, sampled_pts, farthest

    # Loop to perform FPS
    _, sampled_pts, _ = tf.while_loop(
        cond, body, [tf.constant(1), sampled_pts, farthest],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape(None), tf.TensorShape([1, None])]
    )

    # Stack the TensorArray into a single tensor
    sampled_pts = sampled_pts.stack()

    return sampled_pts
def compute_fps_simplification_loss(raw_data, sampled_data, num_cycles, learning_rate=0.001):
    # Placeholder for input and output point clouds
    raw_pc, sampled_pc = tf.placeholder(tf.float32, shape=(None, None, 3)), tf.placeholder(tf.float32, shape=(None, None, 3))

    # Placeholder for cycle index
    cycle_index = tf.placeholder(tf.int32)

    # Downsample the raw point cloud using FPS
    downsampled_pc = downsample_point_cloud(raw_pc, is_training=True, num_output_points=tf.shape(sampled_pc)[1])

    # Compute nearest neighbor distances
    nn_distances = nn_distance(downsampled_pc, sampled_pc)[0]

    # Average NN distances for each point
    avg_nn_distances = tf.reduce_mean(nn_distances, axis=1)

    # Compute mean of average NN distances across all points
    mean_avg_nn_distance = tf.reduce_mean(avg_nn_distances)

    # Loss term penalizing larger mean average NN distances
    distance_penalty = mean_avg_nn_distance

    # Loss term penalizing larger cycles
    cycle_penalty = cycle_index * 0.01  # You may adjust the penalty coefficient

    # Total loss
    fps_simplification_loss = distance_penalty + cycle_penalty

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(fps_simplification_loss)

    # Training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training iterations
        for cycle in range(num_cycles):
            _, loss_value, avg_nn_dist = sess.run([train_op, fps_simplification_loss, mean_avg_nn_distance],
                                                  feed_dict={raw_pc: raw_data, sampled_pc: sampled_data, cycle_index: cycle})
            print(f"Cycle: {cycle + 1}, Loss: {loss_value}, Mean Avg NN Distance: {avg_nn_dist}")

        # Final evaluation
        final_loss, final_avg_nn_dist = sess.run([fps_simplification_loss, mean_avg_nn_distance],
                                                 feed_dict={raw_pc: raw_data, sampled_pc: sampled_data, cycle_index: num_cycles})

        print(f"Final Loss: {final_loss}, Final Mean Avg NN Distance: {final_avg_nn_dist}")

#################################################################################################

def tensor_scatter_nd_update(ref, indices, updates):
    # Calculate the shape of the reference tensor
    ref_shape = tf.shape(ref)

    # Create a mask with ones at the update positions
    mask = tf.one_hot(indices[:, -1], depth=ref_shape[-1], dtype=tf.float32)

    # Calculate the updated tensor
    mask = tf.expand_dims(mask, axis=1)  # Add a new dimension with size 1
    updated_ref = ref * (1.0 - mask) + updates * mask

    return updated_ref

def farthest_point_sample(xyz, npoint, random=True):

    B, N, C = tf.unstack(tf.shape(xyz))
    samples = tf.zeros((B, npoint, C), dtype=tf.float32)

    prev_dist = tf.ones((B, N)) * 1e10

    if random:
        farthest = tf.random_uniform((B,), minval=0, maxval=N, dtype=tf.int32)
    else:
        farthest = tf.zeros((B,), dtype=tf.int32)

    batch_indices = tf.range(B, dtype=tf.int32)

    for i in range(npoint):
        # Gather the farthest points
        farthest_points = tf.gather_nd(xyz, tf.stack([batch_indices, farthest], axis=-1)[:, tf.newaxis])
        # Update the samples tensor using basic TensorFlow operations
        samples = tensor_scatter_nd_update(samples, tf.concat([tf.expand_dims(batch_indices, 1), tf.expand_dims(farthest, 1)], axis=-1), tf.expand_dims(farthest_points, 1))

        centroid = farthest_points[:, tf.newaxis, :]
        current_dist = tf.reduce_sum((xyz - centroid) ** 2, axis=-1)
        current_dist = tf.expand_dims(current_dist, axis=-1)  # Add a new dimension with size 1

        mask = current_dist < prev_dist
        prev_dist = tf.where(mask, current_dist, prev_dist)

        farthest = tf.argmax(prev_dist, axis=-1)

    return samples
"""
"""
def farthest_point_sample(xyz, num_samples, scope='furthest_point_sample'):
    with tf.variable_scope(scope):
        batch_size = xyz.get_shape()[0]
        num_points = xyz.get_shape()[1]
        indices = tf.zeros((batch_size, num_samples), dtype=tf.int32)
        distance = tf.ones((batch_size, num_points)) * 1e10

        # Choose the first point randomly
        indices = tf.range(num_points)
        indices = tf.random.shuffle(indices)
        
        # Use the first 'num_samples' shuffled indices as initial farthest points
        farthest = indices[:num_samples]
        farthest = tf.squeeze(farthest)
        mask = tf.one_hot(farthest, num_points, on_value=True, off_value=False, dtype=tf.bool)
        
        farthest = farthest[:, tf.newaxis]
        indices = indices[tf.newaxis, :, tf.newaxis]

        farthest = tf.expand_dims(farthest, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)
        indices = tf.where(mask, farthest, indices)

        for i in range(num_samples - 1):
            # Update the distance
            cur_point = tf.gather(xyz, indices[:, i], axis=1)
            cur_point_broadcast = tf.expand_dims(cur_point, 2)
            dist = tf.reduce_sum((xyz - cur_point_broadcast) ** 2, axis=-1)

            # Choose the farthest point
            farthest = tf.argmax(dist, axis=1)
            farthest = tf.cast(farthest, dtype=tf.int32)
            mask = tf.one_hot(farthest, num_points, on_value=True, off_value=False, dtype=tf.bool)
            
            # Corrected lines below
            farthest = farthest[:, tf.newaxis, tf.newaxis]
            indices = tf.where(mask, farthest, indices)

        sampled_points = tf.gather(xyz, indices, axis=1)
        return sampled_points


def farthest_point_sample(xyz, num_samples, scope='furthest_point_sample'):
    with tf.variable_scope(scope):
        batch_size = xyz.get_shape()[0]
        num_points = xyz.get_shape()[1]

        indices = tf.zeros((batch_size, num_samples), dtype=tf.int32)
        distance = tf.ones((batch_size, num_points)) * 1e10

        # Choose the first point randomly
        indices = tf.range(num_points)
        indices = tf.random.shuffle(indices)
        farthest = indices[:batch_size]
        farthest = tf.squeeze(farthest)
        mask = tf.one_hot(farthest, num_points, on_value=True, off_value=False, dtype=tf.bool)
        farthest = farthest[:, tf.newaxis]
        indices = indices[:, tf.newaxis, tf.newaxis]

        farthest = tf.expand_dims(farthest, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)
        indices = tf.where(mask, farthest, indices)

        for i in range(num_samples - 1):
            # Update the distance
            cur_point = tf.gather(xyz, indices[:, i], axis=1)
            cur_point_broadcast = tf.expand_dims(cur_point, 2)
            dist = tf.reduce_sum((xyz - cur_point_broadcast) ** 2, axis=-1)

            # Choose the farthest point
            farthest = tf.argmax(dist, axis=1)
            farthest = tf.cast(farthest, dtype=tf.int32)
            mask = tf.one_hot(farthest, num_points, on_value=True, off_value=False, dtype=tf.bool)

            # Update the dimensions
            farthest = tf.expand_dims(farthest, axis=-1)
            mask = tf.expand_dims(mask, axis=-1)
            indices = tf.where(mask, farthest, indices)

        sampled_points = tf.gather(xyz, indices[:, :, 0], axis=1)
        return sampled_points


def downsample_point_cloud(current_data, is_training, num_output_points, bottleneck_size, bn_decay=None):
    batch_size = current_data.shape[0]  # Use shape of current_data to get batch size
    sampled_points_list = []

    for i in range(batch_size):
        # Get points for the current batch
        points = current_data[i]

        # Initialize sampled points array
        sampled_points = tf.zeros((num_output_points, 3), dtype=tf.float32)

        # Perform Farthest Point Sampling
        sampled_points = fps_from_given_pc_tf(points, num_output_points, sampled_points)

        # Append sampled points to the list
        sampled_points_list.append(sampled_points)

    # Convert the list to a TensorFlow tensor
    sampled_points_tensor = tf.convert_to_tensor(sampled_points_list, dtype=tf.float32)

    return sampled_points_tensor


def fps_from_given_pc_tf(pts, num_output_points, given_pc):
    #farthest_pts = tf.TensorArray(dtype=tf.float32, size=k, dynamic_size=False, clear_after_read=False)
    farthest_pts = tf.TensorArray(dtype=tf.float32, size=num_output_points, dynamic_size=False, clear_after_read=False)

    # Initialize with the provided points
    farthest_pts = farthest_pts.write(0, given_pc)
    
    # Choise first point randomly
    num_points = tf.shape(pts)[0]
    initial_idx = tf.random.uniform(shape=[], maxval=num_points, dtype=tf.int32)

    def cond(i, farthest_pts):
        return i < num_output_points

    def body(i, farthest_pts):
        distances = calc_distances_tf(farthest_pts.read(i - 1), pts)
        max_idx = tf.argmax(tf.reduce_min(distances, axis=0), output_type=tf.int32)
        farthest_pts = farthest_pts.write(i, tf.gather(pts, max_idx))
        return i + 1, farthest_pts

    # Initialize the farthest_pts TensorArray
    farthest_pts = tf.TensorArray(dtype=tf.float32, size=num_output_points, dynamic_size=False, clear_after_read=False)
    #farthest_pts = farthest_pts.write(0, tf.expand_dims(tf.gather(pts, initial_idx), 0))
    #farthest_pts = farthest_pts.write(i, tf.gather(pts, max_idx))


    # Loop to find the farthest
    _, farthest_pts_final = tf.while_loop(
            cond, body, [tf.constant(1), farthest_pts], shape_invariants=[tf.TensorShape([]), tf.TensorShape(None)]
            )

    # Stack the TensorArray into a single tensor
    farthest_pts_final = farthest_pts_final.stack()

    return farthest_pts_final

def calc_distances_tf(p0, points):
    return tf.map_fn(lambda x: tf.reduce_sum((p0[:, tf.newaxis, :] - x) ** 2, axis=2), points, dtype=tf.float32)


def get_simplification_loss_fps(ref_pc, samp_pc, pc_size, gamma=1, delta=0):
    #distances = tf.map_fn(lambda x: tf.reduce_sum((ref_pc[:, tf.newaxis, :] - x) ** 2, axis=2), samp_pc, dtype=tf.float32)
    #cost_p1_p2, _, cost_p2_p1, _ = tf.unstack(distances, axis=-1, num=pc_size)
    cost_p1_p2, _, cost_p2_p1, _  = nn_distance(samp_pc, rec_pc)

    # Create a TensorFlow session
    #with tf.Session() as sess:
        # Evaluate tensors within the session
        #cost_p1_p2_eval = sess.run(cost_p1_p2)
        #cost_p2_p1_eval = sess.run(cost_p2_p1)

    #max_cost = np.max(cost_p1_p2_eval, axis=1)
    #max_cost = np.mean(max_cost)
    #cost_p1_p2 = np.mean(cost_p1_p2_eval)
    #cost_p2_p1 = np.mean(cost_p2_p1_eval)
    
    max_cost = tf.reduce_max(cost_p1_p2, axis=1)
    max_cost = tf.reduce_mean(max_cost)
    cost_p1_p2 = tf.reduce_mean(cost_p1_p2)
    cost_p2_p1 = tf.reduce_mean(cost_p2_p1)
    loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1

    tf.summary.scalar("cost_p1_p2", cost_p1_p2)
    tf.summary.scalar("cost_p2_p1", cost_p2_p1)
    tf.summary.scalar("max_cost", max_cost)

    return loss
"""
###########################################################




