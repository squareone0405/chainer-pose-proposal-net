#!/usr/bin/python2

import numpy as np
import math
import Queue
import rospy
import std_msgs
from geometry_msgs.msg import PoseStamped
from ppn.msg import Skeleton
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tf import transformations

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

COLOR_LIST = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 127, 0),
    (0, 255, 85),
    (0, 170, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255)
]

cx_ = 346.46478271484375
cy_ = 200.3520050048828
fx_ = 334.70111083984375
fy_ = 334.70111083984375
k1 = 0.0
k2 = 0.0
K = np.zeros((3, 3))
K[0, 0] = fx_
K[1, 1] = fy_
K[0, 2] = cx_
K[1, 2] = cy_
K[2, 2] = 1.0

pose_queue = Queue.Queue(50)
pose_time_queue = Queue.Queue(50)
pose_lastest = None


def pix2cam(point):
    x = point[2] * (point[0] - cx_) / fx_
    y = point[2] * (point[1] - cy_) / fy_
    z = point[2]
    return [x, y, z]


def pose_to_numpy(pose):
    q_ori = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    position = [pose.position.x, pose.position.y, pose.position.z]
    eular = transformations.euler_from_quaternion(q_ori)
    eular_rect = np.array([-eular[2], eular[1], eular[0] + math.pi])
    print(eular_rect)
    R_rect = transformations.quaternion_matrix(
        transformations.quaternion_from_euler(-eular[2], eular[1], eular[0] + math.pi))
    R_rect = np.linalg.inv(R_rect)
    T_rect = np.dot(transformations.translation_matrix(position), R_rect)
    print(T_rect)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return T_rect


def point_transform(point, pose):
    if point[2] == 0:
        return np.zeros(3)
    T_pose = pose_to_numpy(pose)
    point_body = np.array([point[2], point[0], -point[1]])
    print(point_body)
    pose_homo = np.hstack((point_body, [1]))
    # pose_homo = np.dot(np.linalg.inv(T_pose), pose_homo)
    pose_homo = np.dot(T_pose, pose_homo)
    print(pose_homo)
    return pose_homo[:-1]


def parse_skeleton_msg(skeleton_msg):
    if skeleton_msg.human_num == 0:
        return np.array([]), 0
    skeleton_points = np.zeros((skeleton_msg.human_num, 14, 3))
    msg_point_idx = 0
    for human_idx in range(skeleton_msg.human_num):
        for skeleton_idx in range(skeleton_msg.skeleton_num[human_idx]):
            skeleton_points[human_idx, skeleton_msg.types[msg_point_idx] - 1, 0] = skeleton_msg.points[msg_point_idx].x
            skeleton_points[human_idx, skeleton_msg.types[msg_point_idx] - 1, 1] = skeleton_msg.points[msg_point_idx].y
            skeleton_points[human_idx, skeleton_msg.types[msg_point_idx] - 1, 2] = skeleton_msg.points[msg_point_idx].z
            msg_point_idx = msg_point_idx + 1
    return skeleton_points, skeleton_msg.header.stamp.to_sec()


def draw_skeleton(skeleton_points):
    msg = MarkerArray()
    msg.markers = [None] * (skeleton_points.shape[0] * 14)
    if skeleton_points.shape[0] > 0:
        for human_idx in range(skeleton_points.shape[0]):
            for skeleton_idx in range(14):
                invisible_marker = Marker()
                invisible_marker.header = std_msgs.msg.Header()
                invisible_marker.header.frame_id = 'map'
                invisible_marker.header.stamp = rospy.Time.now()
                invisible_marker.id = skeleton_idx + 1
                msg.markers[human_idx * 14 + skeleton_idx] = invisible_marker
            for skeleton_idx in range(14):
                if skeleton_points[human_idx, skeleton_idx, 2] != 0:
                    index = human_idx * 14 + skeleton_idx
                    marker = Marker()
                    marker.header = std_msgs.msg.Header()
                    marker.header.frame_id = 'map'
                    marker.header.stamp = rospy.Time.now()
                    marker.id = index

                    point_temp = skeleton_points[human_idx, skeleton_idx, :]

                    marker.pose.position.x = point_temp[0]
                    marker.pose.position.y = point_temp[1]
                    marker.pose.position.z = point_temp[2]
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0
                    marker.color.r = COLOR_LIST[skeleton_idx][0] / 255.0
                    marker.color.g = COLOR_LIST[skeleton_idx][1] / 255.0
                    marker.color.b = COLOR_LIST[skeleton_idx][2] / 255.0
                    msg.markers[index] = marker
    marker_pub.publish(msg)


def pose_interpolate(t):
    '''global pose_queue
    global pose_time_queue
    timestamps = np.array(list(pose_time_queue.queue))
    poses = np.array(list(pose_queue.queue))
    time_diff = np.fabs(timestamps - t + 1433786.95)
    idx = np.argmin(time_diff)
    # return poses[idx]'''
    global pose_lastest
    return pose_lastest


def skeleton_callback(skeleton_msg):
    skeleton_points, timestamp = parse_skeleton_msg(skeleton_msg)
    pose_interp = pose_interpolate(timestamp)
    print(skeleton_points)
    print('--------------------------------------------')
    for human_idx in range(skeleton_points.shape[0]):
        for skeleton_idx in range(14):
            point_temp = skeleton_points[human_idx, skeleton_idx, :]
            point_temp = pix2cam(point_temp)
            point_temp = point_transform(point_temp, pose_interp)
            skeleton_points[human_idx, skeleton_idx, :] = point_temp
    # draw_skeleton(skeleton_points)
    human_tracker.refine(skeleton_points, timestamp)


def pose_callback(pose_msg):
    '''global pose_queue
    global pose_time_queue
    pose_queue.put(pose_msg.pose)
    pose_time_queue.put(pose_msg.header.stamp.to_sec())'''

    global pose_lastest
    pose_lastest = pose_msg.pose


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


class HumanTracker():
    def __init__(self):
        self.bone_length = np.zeros(13)
        self.skeleton_points = np.zeros((14, 3))
        self.kalman_filter = None
        self.last_visible_time = np.zeros(14)
        self.skeleton_velocity = np.zeros((14, 3))
        self.confidence = np.ones(14)
        self.human_distance_thres = 0.5
        self.timestamp = 0

    def refine(self, human_points, timestamp):
        print(timestamp)
        print(self.bone_length)
        if human_points.shape[0] == 0:
            return
        for human_idx in range(human_points.shape[0]):
            if np.any(self.bone_length == 0):
                if human_points.shape[0] == 1:
                    bone_length_new = self.get_bone_length(np.squeeze(human_points[human_idx, :, :]))
                    bone_zero_idx = np.where(self.bone_length < 1e-10)[0]
                    print(bone_length_new)
                    print(bone_zero_idx)
                    # if bone_zero_idx.shape[0] > 6:
                    if bone_zero_idx.shape[0] > 0:
                        self.bone_length[bone_zero_idx] = bone_length_new[bone_zero_idx]
                    self.bone_length = self.bone_length * 0.8 + bone_length_new * 0.2 # exp smooth
                    self.skeleton_points = np.squeeze(human_points[0, :, :])
                    self.timestamp = timestamp
                    point_nz_idx = np.where(human_points[human_idx, :, 2] == 0)[0]
                    self.last_visible_time[point_nz_idx] = timestamp
                    if not np.any(self.bone_length < 1e-10):
                        self.kalman_filter = KalmanFilter(self.bone_length)
            else:
                if self.get_human_distance(np.squeeze(human_points[human_idx, :, :])) > self.human_distance_thres:
                    continue
                ''' update confidence, last_visible_time, bone_length '''
                curr_points = self.skeleton_points + self.skeleton_velocity * (timestamp - self.timestamp)
                self.confidence = np.exp(self.last_visible_time - timestamp)
                point_nz_idx = np.where(human_points[human_idx, :, 2] == 0)[0]
                self.last_visible_time[point_nz_idx] = timestamp
                bone_length_new = self.get_bone_length(np.squeeze(human_points[human_idx, :, :]))
                self.kalman_filter.update(bone_length_new)

                ''' refine '''
                last_points = curr_points
                initial_cost = np.zeros(1, dtype=np.float64)
                final_cost = np.zeros(1, dtype=np.float64)
                refine_skeleton(curr_points, human_points[human_idx, :, :], self.bone_length,
                                self.confidence, initial_cost, final_cost)
                self.skeleton_points = curr_points

                ''' update vel and clip '''
                self.skeleton_velocity = np.divide(curr_points - last_points,
                                              (timestamp - self.last_visible_time).reshape(-1, 1) + 0.0001)
                self.skeleton_velocity = np.clip(self.skeleton_velocity, a_min=-0.2, a_max=0.2)

                point4draw = np.zeros((1, 14, 3))
                point4draw[0, :, :] = self.skeleton_points
                # print(self.skeleton_points)
                # print(point4draw)
                draw_skeleton(point4draw)

    def get_human_distance(self, skeleton_new):
        non_zero_idx = np.where(skeleton_new[:, 2] != 0)[0]
        if non_zero_idx.shape[0] == 0:
            return np.inf
        skeleton_new_nz = skeleton_new[non_zero_idx]
        target_nz = self.skeleton_points[non_zero_idx]
        print(np.mean(np.fabs(skeleton_new_nz - target_nz), axis=0))
        print('distance: %f' % np.mean(np.mean(np.fabs(skeleton_new_nz - target_nz), axis=0)))
        return np.mean(np.mean(np.fabs(skeleton_new_nz - target_nz), axis=0))

    def get_distance(self, point1, point2):
        diff = np.array(point1) - np.array(point2)
        return math.sqrt(np.sum(np.multiply(diff, diff)))

    def get_bone_length(self, points):
        bone_length = np.zeros(13)
        for edge_idx in range(len(EDGES)):
            s, t = EDGES[edge_idx]
            types = np.where(points[:, 2] != 0)[0]
            if s in types and t in types:
                bone_length[edge_idx] = \
                    self.get_distance(points[s, :], points[t, :])
        return bone_length


if __name__ == '__main__':
    human_tracker = HumanTracker()
    rospy.init_node('skeleton_draw', anonymous=True)
    marker_pub = rospy.Publisher('skeleton_marker', MarkerArray, queue_size=1000)
    skeleton_sub = rospy.Subscriber('skeleton', Skeleton, skeleton_callback)
    pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, pose_callback)
    rospy.spin()

