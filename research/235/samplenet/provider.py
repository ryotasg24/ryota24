from builtins import range
import os
import sys
import numpy as np
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")):
    www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    zipfile = os.path.basename(www)
    os.system("wget %s; unzip %s" % (www, zipfile))
    os.system("mv %s %s" % (zipfile[:-4], DATA_DIR))
    os.system("rm %s" % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def noisey_point_cloud(batch_data, ratio=0.1):
    """ Randomly replace points with noise.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    assert ratio < 1 and ratio >= 0
    B, N, C = batch_data.shape
    noise = np.random.rand(B, N, C) * 2 - 1

    rand_idx = np.random.permutation(list(range(0, N)))
    rand_idx = rand_idx[: int(N * ratio)]

    noisey_data = batch_data
    noisey_data[:, rand_idx, :] = noise[:, rand_idx, :]

    # noiseThresh = np.random.rand(B, N)
    # noiseThresh=(noiseThresh > ratio).choose(1,0)
    # for i in range(C):
    #     noise[:, :, i] = noise[:, :, i] * noiseThresh
    #
    # noisey_data =(noise > 0).choose(batch_data,noise)
    return noisey_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

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
    
    batch_size = pts.shape[0]
    num_points = pts.shape[1]
    a = [] 
    fps_flag=0
    for i in range(batch_size):
        #print(pts[i])
        b = sample_fps(pts[i], k)  # numpy -> numpy
        c = b.tolist()  # numpy -> list
        a.append(c)  # list -> list[list]
        
        #if (fps_flag==13):
            #ファイルの上書き保存ができていないよ、そのまま追加しちゃうから気を付けて
        #    with open("test.txt", "w") as file:
        #        for points in a:
                # リスト内の各座標を文字列に変換して改行で結合
        #            line = "\n".join([f"{x}, {y}, {z}" for x, y, z in points])
        #            file.write(line + "\n")  # 各要素の後に改行を追加
        
        #fps_flag = fps_flag + 1
        #print(fps_flag)

    return np.array(a)  # list[list] -> numpy

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f["data"][:]
    #print(type(data))
    #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #data = farthest_point_sample(data, 250)
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    label = f["label"][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f["data"][:]
    label = f["label"][:]
    seg = f["pid"][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)