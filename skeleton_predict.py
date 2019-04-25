import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import math

'''COLOR_LIST = np.array([
    (255, 0, 0, 255),
    (255, 85, 0, 255),
    (255, 170, 0, 255),
    (255, 255, 0, 255),
    (170, 255, 0, 255),
    (85, 255, 0, 255),
    (0, 127, 0, 255),
    (0, 255, 85, 255),
    (0, 170, 170, 255),
    (0, 255, 255, 255),
    (0, 170, 255, 255),
    (0, 85, 255, 255),
    (0, 0, 255, 255),
    (85, 0, 255, 255)
]) / 255.0'''

KEYPOINT_NAMES = [
    'instance',
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

class skeleton_animation:
    def __init__(self, humans):
        self.humans = humans
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.graph, = self.ax.plot([], [], [], 'x')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(1, 3)
        self.ax.set_zlim(-1.5, 1.5)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        '''self.graph = self.ax.plot(self.humans[0]['points'][:, 0],
                                      self.humans[0]['points'][:, 1],
                                      self.humans[0]['points'][:, 2],
                                      c=COLOR_LIST[humans[0]['types'] - 1])'''
        self.ani = animation.FuncAnimation(fig=self.fig, func=self.animate, frames=len(self.humans),
                                           init_func=self.init, interval=50, blit=False)

    def init(self):
        idx = 0
        self.graph.set_data(self.humans[idx]['points'][:, 0],
                            self.humans[idx]['points'][:, 2])
        self.graph.set_3d_properties(-self.humans[idx]['points'][:, 1])

    def animate(self, idx):
        self.graph.set_data(self.humans[idx]['points'][:, 0],
                            self.humans[idx]['points'][:, 2])
        self.graph.set_3d_properties(-self.humans[idx]['points'][:, 1])

    def save(self):
        self.ani.save('skeleton.mp4', writer='ffmpeg', fps=5)


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

def plot_skeleton(points, types, save, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1, 3)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(points[:, 2], -points[:, 1], points[:, 0], 'x', color=COLOR_LIST[types - 1])
    for edge_idx in range(len(EDGES)):
        s, t = EDGES[edge_idx]
        if s in types and t in types:
            s_idx = np.where(types == s)
            t_idx = np.where(types == t)
            ax.plot([points[s_idx, 0][0, 0], points[t_idx, 0][0, 0]],
                    [points[s_idx, 2][0, 0], points[t_idx, 2][0, 0]],
                    [-points[s_idx, 1][0, 0], -points[t_idx, 1][0, 0]],
                    color=COLOR_LIST[types[t_idx] - 1][0])
    if save is True:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    skeleton_data = np.loadtxt('skeleton.csv', delimiter=',')
    zero_or_types = skeleton_data[:, 0].astype(np.int8)
    humans = []
    idx = 0
    while idx < len(zero_or_types):
        skeleton_num = skeleton_data[idx, 2].astype(np.int8)
        time_stamp = skeleton_data[idx, 3]
        idx += 1
        types = skeleton_data[idx: idx + skeleton_num, 0].astype(np.int8)
        points = skeleton_data[idx: idx + skeleton_num, 1:]
        idx += skeleton_num
        humans.append({'num': skeleton_num, 'time_stamp': time_stamp, 'types': types, 'points': points})
    # ani = skeleton_animation(humans)
    # plt.show()

    points_list = np.array([human['points'] for human in humans])
    types_list = np.array([human['types'] for human in humans])

    points_flatten = np.concatenate(points_list).ravel().reshape((-1, 3))
    types_flatten = np.concatenate(types_list).ravel()

    bone_length = np.zeros((len(humans), len(EDGES)))
    for human_idx in range(len(humans)):
        for edge_idx in range(len(EDGES)):
            s, t = EDGES[edge_idx]
            if s in types_list[human_idx] and t in types_list[human_idx]:
                s_idx = np.where(types_list[human_idx] == s)
                t_idx = np.where(types_list[human_idx] == t)
                bone_length[human_idx][edge_idx] = \
                    get_distance(points_list[human_idx][s_idx], points_list[human_idx][t_idx])

    print(bone_length[:, 0])

    '''head_depth = np.array([])
        for human_idx in range(len(humans)):
            if type in types_list[human_idx]:
                head_depth = np.hstack((head_depth, points_list[human_idx][np.where(types_list[human_idx] == type)[0], 2]))'''

    type = 1
    head_point = points_flatten[types_flatten == type]
    head_point_smoothed = exponential_smoothing(head_point, 0.1)
    type = 2
    neck_point = points_flatten[types_flatten == type]
    neck_point_smoothed = exponential_smoothing(neck_point, 0.1)
    plt.figure()
    plt.plot(range(head_point.shape[0]), head_point[:, 2])
    plt.plot(range(head_point.shape[0]), head_point_smoothed[:, 2])
    plt.plot(range(neck_point.shape[0]), neck_point[:, 2])
    plt.plot(range(neck_point.shape[0]), neck_point_smoothed[:, 2])
    plt.show()

    '''plt.figure()
    for i in range(len(EDGES)):
        plt.plot(range(bone_length.shape[0]), bone_length[:, i], label='%d' % i)
    plt.legend(loc='best')
    plt.show()'''

    '''for i in range(points_list.shape[0]):
        plot_skeleton(points_list[i], types_list[i], True, './result/%04d.png' % i)'''


