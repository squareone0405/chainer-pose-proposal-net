import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as colors
import math

import ctypes
from ctypes import cdll

lib = cdll.LoadLibrary('../skeleton_optimizer/cmake-build-debug/libskeleton.so')
ceres_refine = lib.refine_skeleton


def refine_skeleton(init_points, expect_points, bone_length, confidence, initial_cost, final_cost):
    ceres_refine(ctypes.c_void_p(init_points.ctypes.data),
                 ctypes.c_void_p(expect_points.ctypes.data),
                 ctypes.c_void_p(bone_length.ctypes.data),
                 ctypes.c_void_p(confidence.ctypes.data),
                 ctypes.c_void_p(initial_cost.ctypes.data),
                 ctypes.c_void_p(final_cost.ctypes.data))


KEYPOINT_NAMES = [
    'head_top',
    'upper_neck',
    'l_shoulder',
    'r_shoulder',
    'l_elbow',
    'r_elbow',
    'l_wrist',
    'r_wrist',
    'l_hip',
    'r_hip',
    'l_knee',
    'r_knee',
    'l_ankle',
    'r_ankle',
]

EDGES_BY_NAME = [
    ['upper_neck', 'head_top'],
    ['upper_neck', 'l_shoulder'],
    ['upper_neck', 'r_shoulder'],
    ['upper_neck', 'l_hip'],
    ['upper_neck', 'r_hip'],
    ['l_shoulder', 'l_elbow'],
    ['l_elbow', 'l_wrist'],
    ['r_shoulder', 'r_elbow'],
    ['r_elbow', 'r_wrist'],
    ['l_hip', 'l_knee'],
    ['l_knee', 'l_ankle'],
    ['r_hip', 'r_knee'],
    ['r_knee', 'r_ankle'],
]

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

COLOR_LIST = np.array([
    [255, 0, 0, 255],
    [255, 85, 0, 255],
    [255, 170, 0, 255],
    [255, 255, 0, 255],
    [170, 255, 0, 255],
    [85, 255, 0, 255],
    [0, 127, 0, 255],
    [0, 255, 85, 255],
    [0, 170, 170, 255],
    [0, 255, 255, 255],
    [0, 170, 255, 255],
    [0, 85, 255, 255],
    [0, 0, 255, 255],
    [85, 0, 255, 255]
]) / 255.0


def load_data(path):
    skeleton_data = np.loadtxt(path, delimiter=',')
    zero_or_types = skeleton_data[:, 0].astype(np.int8)
    human_num = np.where(zero_or_types == 0)[0].shape[0]
    skeleton_points = np.zeros((human_num, 14, 3))
    time_stamps = np.zeros(human_num)
    human_idx = 0
    data_idx = 0
    while data_idx < len(zero_or_types):
        skeleton_num = skeleton_data[data_idx, 2].astype(np.int8)
        time_stamps[human_idx] = skeleton_data[data_idx, 3]
        data_idx += 1
        types = skeleton_data[data_idx: data_idx + skeleton_num, 0].astype(np.int8)
        points = skeleton_data[data_idx: data_idx + skeleton_num, 1:]
        for i in range(types.shape[0]):
            skeleton_points[human_idx, types[i] - 1, :] = points[i]
        human_idx += 1
        data_idx += skeleton_num
    return skeleton_points, time_stamps


def get_distance(point1, point2):
    diff = np.array(point1) - np.array(point2)
    return math.sqrt(np.sum(np.multiply(diff, diff)))


def exponential_smoothing(data, alpha): # TODO only smooth z when predict 3d point online
    data_len = data.shape[0]
    data_exp = np.zeros_like(data)
    data_exp[0, :] = data[0, :]
    for i in np.arange(1, data_len):
        data_exp[i, :] = data_exp[i - 1, :] * (1 - alpha) + data[i] * alpha
    return data_exp


def get_bone_length(points):
    bone_length = np.zeros(13)
    for edge_idx in range(len(EDGES)):
        s, t = EDGES[edge_idx]
        types = np.where(points[:, 2] != 0)[0]
        if s in types and t in types:
            bone_length[edge_idx] = \
                get_distance(points[s, :], points[t, :])
    return bone_length


def plot_skeleton(points, save=False, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1, 3)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    types = np.where(points[:, 2] != 0)[0]
    valid_points = points[types]
    ax.scatter(valid_points[:, 2], -valid_points[:, 1], valid_points[:, 0], 'x', c=COLOR_LIST[types])
    for edge_idx in range(len(EDGES)):
        s, t = EDGES[edge_idx]
        if s in types and t in types:
            ax.plot([points[s, 0], points[t, 0]],
                    [points[s, 2], points[t, 2]],
                    [-points[s, 1], -points[t, 1]],
                    color=COLOR_LIST[s])
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_skeletons(points1, points2 ,save=False, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    '''ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1, 3)
    ax.set_zlim(-1.5, 1.5)'''
    ax.set_xlim(1, 3)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 2.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    types = np.where(points1[:, 2] != 0)[0]
    valid_points = points1[types]
    # ax.scatter(valid_points[:, 2], -valid_points[:, 1], valid_points[:, 0], 'x', c=COLOR_LIST[types])
    ax.scatter(valid_points[:, 1], valid_points[:, 2], valid_points[:, 0], 'x', c=COLOR_LIST[types])
    for edge_idx in range(len(EDGES)):
        s, t = EDGES[edge_idx]
        if s in types and t in types:
            '''ax.plot([points1[s, 0], points1[t, 0]],
                    [points1[s, 2], points1[t, 2]],
                    [-points1[s, 1], -points1[t, 1]],
                    color=COLOR_LIST[s]'''
            ax.plot([points1[s, 0], points1[t, 0]],
                    [points1[s, 1], points1[t, 1]],
                    [points1[s, 2], points1[t, 2]],
                    color=COLOR_LIST[s])
    types = np.where(points2[:, 2] != 0)[0]
    valid_points = points2[types]
    # ax.scatter(valid_points[:, 2], -valid_points[:, 1], valid_points[:, 0], 'x', c=COLOR_LIST[types])
    ax.scatter(valid_points[:, 1], valid_points[:, 2], valid_points[:, 0], 'x', c=COLOR_LIST[types])
    for edge_idx in range(len(EDGES)):
        s, t = EDGES[edge_idx]
        if s in types and t in types:
            '''ax.plot([points2[s, 0], points2[t, 0]],
                    [points2[s, 2], points2[t, 2]],
                    [-points2[s, 1], -points2[t, 1]],
                    color='black')'''
            ax.plot([points2[s, 0], points2[t, 0]],
                    [points2[s, 1], points2[t, 1]],
                    [points2[s, 2], points2[t, 2]],
                    color='black')
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


class KalmanFilter:
    def __init__(self, bone_length_init, variance_init=np.zeros(13), Q=0.00005, R=0.2):
        self.bone_length = bone_length_init
        self.Q = Q
        self.R = R
        self.variance_kalman = variance_init

    def update(self, bone_length_measure):
        non_zero_idx = np.where(bone_length_measure != 0)[0]
        variance_pred = self.variance_kalman[non_zero_idx] + self.Q
        K = variance_pred / (variance_pred + self.R)
        self.bone_length[non_zero_idx] = self.bone_length[non_zero_idx] + K * \
                                         (bone_length_measure[non_zero_idx] - self.bone_length[non_zero_idx])
        self.variance_kalman[non_zero_idx] = (1 - K) * variance_pred
        return self.bone_length


if __name__ == '__main__':
    skeleton_points, time_stamps = load_data('skeleton10.csv') # TODO add confidence term to handle skeleton incomplete
    bone_length = np.zeros((skeleton_points.shape[0], len(EDGES)))
    for human_idx in range(skeleton_points.shape[0]):
        for edge_idx in range(len(EDGES)):
            s, t = EDGES[edge_idx]
            types = np.where(skeleton_points[human_idx, :, 2] != 0)[0]
            if s in types and t in types:
                bone_length[human_idx][edge_idx] = \
                    get_distance(skeleton_points[human_idx, s, :], skeleton_points[human_idx, t, :])

    bone_length_mean = np.sum(bone_length, axis=0) / np.count_nonzero(bone_length, axis=0)
    print(bone_length_mean)
    print(skeleton_points.shape)

    print(bone_length.shape)
    outlier_idx = np.where(np.max(bone_length, axis=1) > 2)[0]
    outlier_time = time_stamps[np.where(np.max(bone_length, axis=1) > 2)[0]]
    print(outlier_idx)
    for t in outlier_time:
        print(t)

    last_visible_time = np.zeros(14)
    skeleton_velocity = np.zeros((14, 3)) # TODO add threshold to velocity

    curr_points = skeleton_points[0] # TODO init skeleton must be complete
    last_points = skeleton_points[0]

    kalman_filter = KalmanFilter(bone_length[0])
    bone_length_kalman = np.zeros_like(bone_length)
    for i in range(bone_length.shape[0]):
        # print(kalman_filter.update(bone_length[0]))
        bone_length_kalman[i, :] = kalman_filter.update(bone_length[i])
    plt.plot(range(bone_length_kalman.shape[0]), bone_length_kalman)
    plt.plot(range(bone_length.shape[0]), bone_length)
    plt.show()

    kalman_filter = KalmanFilter(bone_length[0])

    confidence = np.ones(14)

    # for i in range(skeleton_points.shape[0]):
    for i in range(520):
        if i in outlier_idx:
            continue
        if i > 0:
            curr_points = curr_points + skeleton_velocity * (time_stamps[i] - time_stamps[i - 1])
            confidence = np.exp(last_visible_time - time_stamps[i])
            print('*' * 20 + str(i))
            print(confidence)
        last_points = curr_points
        initial_cost = np.zeros(1, dtype=np.float64)
        final_cost = np.zeros(1, dtype=np.float64)
        refine_skeleton(curr_points, skeleton_points[i], kalman_filter.update(bone_length[i]),
                        confidence, initial_cost, final_cost)
        skeleton_velocity = np.divide(curr_points - last_points,
                                      (time_stamps[i] - last_visible_time).reshape(-1, 1) + 0.0001)
        skeleton_velocity = np.clip(skeleton_velocity, a_min=-0.2, a_max=0.2)
        non_zero_idx = np.where(skeleton_points[i, :, 2] != 0)[0]
        last_visible_time[non_zero_idx] = time_stamps[i]
        plot_skeletons(curr_points, skeleton_points[i], True, './result/%04d.png' % i)
        '''if i in np.arange(300, 310):
            print(non_zero_idx)
            plot_skeletons(curr_points, skeleton_points[i])
            print(initial_cost)
            print(final_cost)'''


    '''type = 1
    head_point = skeleton_points[:, type, :]
    head_point = head_point[head_point[:, 0] != 0]
    head_point_smoothed = exponential_smoothing(head_point, 0.1)
    type = 2
    neck_point = skeleton_points[:, type, :]
    neck_point = neck_point[neck_point[:, 0] != 0]
    neck_point_smoothed = exponential_smoothing(neck_point, 0.1)
    plt.figure()
    plt.plot(range(head_point.shape[0]), head_point[:, 2])
    plt.plot(range(head_point.shape[0]), head_point_smoothed[:, 2])
    plt.plot(range(neck_point.shape[0]), neck_point[:, 2])
    plt.plot(range(neck_point.shape[0]), neck_point_smoothed[:, 2])
    plt.show()'''

    '''plt.figure()
    for i in range(len(EDGES)):
        plt.plot(range(bone_length.shape[0]), bone_length[:, i], label='%d' % i)
    plt.legend(loc='best')
    plt.show()'''

    '''for i in range(skeleton_points.shape[0]):
        plot_skeleton(skeleton_points[i], True, './result/%04d.png' % i)'''

