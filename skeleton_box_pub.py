#!/usr/bin/python2

import rospy
import std_msgs
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from ppn.msg import Skeleton
from cv_bridge import CvBridge
from tf import transformations

import matplotlib.pyplot as plt
from scipy import signal
import math

import ctypes
from ctypes import cdll

lib = cdll.LoadLibrary('./build/ssd.so')
ssd_cuda = lib.ssd


def compute_ssd(window, target, out, window_width, window_height,
                target_width, target_height, out_width, out_height, num):
    ssd_cuda(ctypes.c_void_p(window.ctypes.data),
             ctypes.c_void_p(target.ctypes.data),
             ctypes.c_void_p(out.ctypes.data),
             ctypes.c_void_p(window_width.ctypes.data),
             ctypes.c_void_p(window_height.ctypes.data),
             ctypes.c_void_p(target_width.ctypes.data),
             ctypes.c_void_p(target_height.ctypes.data),
             ctypes.c_int32(out_width), ctypes.c_int32(out_height), ctypes.c_int32(num))

import configparser
import os
import Queue
import threading
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import chainer
import cv2
import numpy as np
from PIL import Image

from predict import get_feature, get_humans_by_feature, draw_humans, create_model
from utils import parse_size

import pyzed.sl as sl

QUEUE_SIZE = 5

bridge = CvBridge()

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

BaseLine = 119.971
CV_VGA = 0.0129813
RX_VGA = -0.000799271
RZ_VGA = -5.96179e-5
T = np.array([-BaseLine, 0, 0])
Rz, _ = cv2.Rodrigues(np.array([0, 0, RZ_VGA]))
Ry, _ = cv2.Rodrigues(np.array([0, CV_VGA, 0]))
Rx, _ = cv2.Rodrigues(np.array([RX_VGA, 0, 0]))
R = np.dot(Rz, np.dot(Ry, Rx))


class Capture(threading.Thread):
    def __init__(self, insize):
        super(Capture, self).__init__()
        self.insize = insize
        self.kx = 1.0 * 672 / insize[0]
        self.ky = 1.0 * 376 / insize[1]
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Capture'

    def pix2cam(self, point):
        x = point[2] * (point[0] - cx_) / fx_
        y = point[2] * (point[1] - cy_) / fy_
        z = point[2]
        return [x, y, z]

    def run(self):
        while not self.stop_event.is_set():
            global left_queue
            global right_queue
            global pose_list
            global timestamp_list
            if left_queue.qsize > 0 and right_queue.qsize() > 0 and len(timestamp_list) > 0:
                left_t, left_image = left_queue.get()
                right_t, right_image = right_queue.get()
                print(left_t - right_t)
                diff = np.fabs(timestamp_list - left_t)
                min_idx = np.argmin(diff)
                print(timestamp_list[min_idx] - left_t)
                pose = pose_list[min_idx]
                # timestamp_list = timestamp_list[min_idx:]
                # pose_list = pose_list[min_idx:]
                side_by_side = np.zeros((376, 672 * 2, 3), dtype=np.uint8)
                side_by_side[:, :672, :] = left_image
                side_by_side[:, 672:, :] = right_image
                try:
                    image = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)
                    self.queue.put((image, pose), timeout=1)
                except Queue.Full:
                    pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


class Predictor(threading.Thread):

    def __init__(self, model, cap):
        super(Predictor, self).__init__()
        self.cap = cap
        self.model = model
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Predictor'

        self.text_file = open("skeleton10.csv", "w")

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw_image, pose = self.cap.get()
                image_left = raw_image[:, 0:raw_image.shape[1] / 2, :]
                image_right = raw_image[:, raw_image.shape[1] / 2:, :]
                image = cv2.resize(image_left, self.cap.insize)
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
                self.queue.put((image, feature_map, image_left, image_right, pose), timeout=1)
            except Queue.Full:
                pass
            except Queue.Empty:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()

    def save_skeleton(self, humans_left, image_left, image_right, pose):
        if len(humans_left) > 0:
            skeleton_num = [0] * len(humans_left)
            for i in range(len(humans_left)):
                skeleton_num[i] = len(humans_left[i]) - 1
                # print('skeleton num')
                # print(skeleton_num[i])
                points = []
                types = []
                '''if skeleton_num[i] > 0:
                    points, types = self.get_points(humans_left[i], image_left, image_right, pose)
                self.text_file.write(str(0) + ',' + str(0) + ',' + str(len(types)) + ',' + str(time.time()) + '\n')
                for i in range(len(types)):
                    self.text_file.write(str(types[i]) + ',' + str(points[i][0]) + ','
                                         + str(points[i][1]) + ',' + str(points[i][2]) + '\n')'''

    def get_head_depth(self, humans_left, image_left, image_right):
        if len(humans_left) > 0:
            skeleton_num = [0] * len(humans_left)
            for i in range(len(humans_left)):
                skeleton_num[i] = len(humans_left[i]) - 1
                # print('skeleton num')
                # print(skeleton_num[i])
                points = []
                types = []
                '''if skeleton_num[i] > 0:
                    points, types = self.get_points(humans_left[i], image_left, image_right)
                if 1 in types:
                    return points[types == 1][2]'''
        return 0

    def pub_skeleton(self, humans_left, conf, image_left, image_right, pose):
        global img_time_queue
        msg = Skeleton()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = img_time_queue.get()
        msg.human_num = len(humans_left)
        if len(humans_left) > 0:
            skeleton_num = [0] * len(humans_left)
            types = []
            bboxes = []
            depths = []
            confidences = []
            for i in range(len(humans_left)):
                skeleton_num[i] = len(humans_left[i]) - 1
                print('skeleton num: %d' % skeleton_num[i])
                if skeleton_num[i] > 0:
                    bboxes_temp, depths_temp, types_temp = self.get_points(humans_left[i], image_left, image_right, pose)
                    confidences.extend(list(np.array(list(conf[i].values())[1:]).astype(np.float64)))
                    types.extend(types_temp)
                    bboxes.extend(bboxes_temp)
                    depths.extend(depths_temp)
            msg.skeleton_num = skeleton_num
            msg.types = types
            msg.bboxes = bboxes
            msg.depths = depths
            msg.confidences = confidences
            pub.publish(msg)
            print('done pub')

    def get_points(self, human, image_left, image_right, pose):
        plot = False

        BaseLine = 0.12
        FocalLength = fx_

        types = []
        depths = np.zeros((len(human) - 1), dtype='float64')
        bboxes = np.zeros((len(human) - 1, 4), dtype='float64')

        mask_size = 3
        interp_range = (0.1, 1)

        start_time = time.time()
        image_left_ori = image_left
        image_right_ori = image_right
        kx_ori = self.cap.kx
        ky_ori = self.cap.ky

        scale_ratio = 8 if image_left.shape[1] > 1000 else 4
        image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                                 int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
        image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                                   int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

        if plot is True:
            plt.imshow(image_left, cmap='gray')
            plt.show()
            plt.imshow(image_right, cmap='gray')
            plt.show()

        kx = kx_ori / scale_ratio
        ky = ky_ori / scale_ratio

        search_xmin = -20
        search_xmax = 0
        search_ymin = -2
        search_ymax = 2

        window_all = np.array([], dtype='int32')
        target_all = np.array([], dtype='int32')
        window_width = np.zeros((len(human) - 1), dtype='int32')
        window_height = np.zeros((len(human) - 1), dtype='int32')
        target_width = np.zeros((len(human) - 1), dtype='int32')
        target_height = np.zeros((len(human) - 1), dtype='int32')

        ssd = np.zeros((len(human) - 1, search_ymax - search_ymin + 1, search_xmax - search_xmin + 1), dtype='int32')
        idx = 0
        for key in sorted(human):
            if key != 0:
                ymin_f, xmin_f, ymax_f, xmax_f = human[key]  # TODO change the range of head and hip

                if key == 1:  # head top
                    ymin_f = ymin_f * 0.7 + ymax_f * 0.3
                if key in [9, 10]:  # left hip or right hip
                    xmin_f = xmin_f - (xmax_f - xmin_f) * 0.3
                    xmax_f = xmax_f + (xmax_f - xmin_f) * 0.3

                ymin = max(int(ymin_f * ky), 0)
                xmin = max(int(xmin_f * kx), 0)
                ymax = min(int(ymax_f * ky), image_left.shape[0])
                xmax = min(int(xmax_f * kx), image_left.shape[1])

                window = image_left[ymin:ymax, xmin:xmax].astype(np.int32)
                # print(ymin, ymax, xmin, xmax)

                target = np.full((window.shape[0] + search_ymax - search_ymin,
                                  window.shape[1] + search_xmax - search_xmin), 512, dtype='int32')
                target_ymin = max(ymin + search_ymin, 0)
                target_ymax = min(ymax + search_ymax, image_right.shape[0])
                target_xmin = max(xmin + search_xmin, 0)
                target_xmax = min(xmax + search_xmax, image_right.shape[1])
                target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
                clip_ymin = max(-(ymin + search_ymin), 0)
                clip_ymax = max(ymax + search_ymax - image_right.shape[0], 0)

                target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                    target_from_right

                window_all = np.append(window_all, window.flatten())
                target_all = np.append(target_all, target.flatten())
                window_width[idx] = window.shape[1]
                window_height[idx] = window.shape[0]
                target_width[idx] = target.shape[1]
                target_height[idx] = target.shape[0]
                idx = idx + 1

        # print('cpu:' + str(time.time() - start_time))
        out = np.zeros_like(ssd, dtype='int32')
        compute_ssd(window_all, target_all, out, window_width, window_height,
                    target_width, target_height, ssd.shape[2], ssd.shape[1], len(human) - 1)
        ssd = out
        # print('cuda ssd:' + str(time.time() - start_time))

        heat_map = ssd.sum(axis=0)

        offset_window = 0
        offset_target = 0
        if plot is True:
            for i in range(len(human) - 1):
                plt.subplot(311)
                plt.imshow(window_all[offset_window: offset_window + window_width[i] * window_height[i]]
                           .reshape((window_height[i], window_width[i])), cmap='gray')
                plt.subplot(312)
                plt.imshow(target_all[offset_target: offset_target + target_width[i] * target_height[i]]
                           .reshape((target_height[i], target_width[i])), cmap='gray')
                plt.subplot(313)
                plt.imshow(ssd[i], cmap='gray')
                plt.show()
                offset_window += window_width[i] * window_height[i]
                offset_target += target_width[i] * target_height[i]
            plt.imshow(heat_map, cmap='gray')
            plt.show()

        mask = np.ones((mask_size, mask_size), dtype='int32')
        convolved = signal.convolve2d(heat_map, mask, 'valid')
        min_idx = np.argmin(convolved.reshape(-1))
        center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
        # print(center_x, center_y)

        offset_x = (search_xmax - search_xmin - center_x) * scale_ratio
        offset_y = (search_ymin + center_y) * scale_ratio

        # print('search range:%d, %d' % (search_ymax - search_ymin + 1, search_xmax - search_xmin + 1))
        # print('round 1:***************************' + str(time.time() - start_time))

        '''round 2'''
        scale_ratio = 1
        image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                                 int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
        image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                                   int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

        if plot is True:
            plt.imshow(image_left, cmap='gray')
            plt.show()
            plt.imshow(image_right, cmap='gray')
            plt.show()

        kx = kx_ori / scale_ratio
        ky = ky_ori / scale_ratio

        search_xmin = -offset_x - max(4, int(offset_x / 8))
        search_xmax = -offset_x + max(4, int(offset_x / 8))
        search_ymin = offset_y - 3
        search_ymax = offset_y + 3

        # print(search_xmin, search_xmax, search_ymin, search_ymax)

        window_all = np.array([], dtype='int32')
        target_all = np.array([], dtype='int32')
        window_width = np.zeros((len(human) - 1), dtype='int32')
        window_height = np.zeros((len(human) - 1), dtype='int32')
        target_width = np.zeros((len(human) - 1), dtype='int32')
        target_height = np.zeros((len(human) - 1), dtype='int32')

        ssd = np.zeros((len(human) - 1, search_ymax - search_ymin + 1, search_xmax - search_xmin + 1), dtype='int32')
        idx = 0
        for key in sorted(human):
            if key != 0:
                types.append(key)
                ymin_f, xmin_f, ymax_f, xmax_f = human[key]  # TODO change the range of head and hip
                bboxes[idx] = [ymin_f, xmin_f, ymax_f, xmax_f]
                if key == 1:  # head top
                    ymin_f = ymin_f * 0.7 + ymax_f * 0.3
                if key in [9, 10]:  # left hip or right hip
                    xmin_f = xmin_f - (xmax_f - xmin_f) * 0.3
                    xmax_f = xmax_f + (xmax_f - xmin_f) * 0.3

                ymin = max(int(ymin_f * ky), 0)
                xmin = max(int(xmin_f * kx), 0)
                ymax = min(int(ymax_f * ky), image_left.shape[0])
                xmax = min(int(xmax_f * kx), image_left.shape[1])

                window = image_left[ymin:ymax, xmin:xmax].astype(np.int32)
                # print(ymin, ymax, xmin, xmax)

                target = np.full((window.shape[0] + search_ymax - search_ymin,
                                  window.shape[1] + search_xmax - search_xmin), 512, dtype='int32')
                target_ymin = max(ymin + search_ymin, 0)
                target_ymax = min(ymax + search_ymax, image_right.shape[0])
                target_xmin = max(xmin + search_xmin, 0)
                target_xmax = min(xmax + search_xmax, image_right.shape[1])
                target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
                clip_ymin = max(-(ymin + search_ymin), 0)
                clip_ymax = max(ymax + search_ymax - image_right.shape[0], 0)

                target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                    target_from_right

                window_all = np.append(window_all, window.flatten())
                target_all = np.append(target_all, target.flatten())
                window_width[idx] = window.shape[1]
                window_height[idx] = window.shape[0]
                target_width[idx] = target.shape[1]
                target_height[idx] = target.shape[0]

                idx = idx + 1

        # print('cpu:' + str(time.time() - start_time))
        out = np.zeros_like(ssd, dtype='int32')
        compute_ssd(window_all, target_all, out, window_width, window_height,
                    target_width, target_height, ssd.shape[2], ssd.shape[1], len(human) - 1)
        ssd = out
        # print('cuda ssd:' + str(time.time() - start_time))

        heat_map = ssd.sum(axis=0)

        offset_window = 0
        offset_target = 0
        if plot is True:
            for i in range(len(human) - 1):
                plt.subplot(311)
                plt.imshow(window_all[offset_window: offset_window + window_width[i] * window_height[i]]
                           .reshape((window_height[i], window_width[i])), cmap='gray')
                plt.subplot(312)
                plt.imshow(target_all[offset_target: offset_target + target_width[i] * target_height[i]]
                           .reshape((target_height[i], target_width[i])), cmap='gray')
                plt.subplot(313)
                plt.imshow(ssd[i], cmap='gray')
                plt.show()
                offset_window += window_width[i] * window_height[i]
                offset_target += target_width[i] * target_height[i]
            plt.imshow(heat_map, cmap='gray')
            plt.show()

        # print('search range:%d, %d' % (search_ymax - search_ymin + 1, search_xmax - search_xmin + 1))
        # print('round 2:+++++++++++++++++++++++++++' + str(time.time() - start_time))

        mask = np.ones((mask_size, mask_size), dtype='int32')
        convolved = signal.convolve2d(heat_map, mask, 'valid')
        min_idx = np.argmin(convolved.reshape(-1))
        center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
        # print(center_x, center_y)

        offset_x = -(search_xmax - search_xmin - center_x) * scale_ratio + search_xmax
        offset_y = (search_ymin + center_y) * scale_ratio + (search_ymin + search_ymax) / 2

        candidate_range_x = max(3, int(math.fabs(offset_x / 16)))  # pm
        candidate_range_y = max(1, int(math.fabs(offset_y / 16)))  # pm

        '''print(offset_x)
        print(offset_y)
        print(candidate_range_x)
        print(candidate_range_y)'''
        # print('offset: %d, %d' % (offset_y, offset_x))
        # print('candidate_range:%d, %d' % (2 * candidate_range_y + 1, 2 * candidate_range_x + 1))

        for i in range(ssd.shape[0]):  # TODO weighted interpolation
            candidate_ymin = max(center_y - candidate_range_y, 0)
            candidate_ymax = min(center_y + candidate_range_y + 1, ssd.shape[1])
            candidate_xmin = max(center_x - candidate_range_x, 0)
            candidate_xmax = min(center_x + candidate_range_x + 1, ssd.shape[2])
            candidate_rigion = ssd[i, candidate_ymin: candidate_ymax, candidate_xmin: candidate_xmax]
            if plot is True:
                plt.imshow(candidate_rigion, cmap='gray')
                plt.show()
            best_candidate_x = np.argmin(candidate_rigion.reshape(-1)) % (candidate_xmax - candidate_xmin)
            best_candidate_y = np.argmin(candidate_rigion.reshape(-1)) / (candidate_xmax - candidate_xmin)
            neighbor_xmin = max(best_candidate_x - 1, 0)
            neighbor_xmax = min(best_candidate_x + 2, candidate_rigion.shape[1])
            neighbor_ymin = max(best_candidate_y - 1, 0)
            neighbor_ymax = min(best_candidate_y + 2, candidate_rigion.shape[0])
            min_neighbor = candidate_rigion[neighbor_ymin: neighbor_ymax, neighbor_xmin: neighbor_xmax]
            min_neighbor = np.interp(min_neighbor, (np.min(min_neighbor.reshape(-1)), np.max(min_neighbor.reshape(-1))),
                                     interp_range)
            weight = 1.0 / min_neighbor
            norm_factor = np.sum(weight.reshape(-1))
            weight = weight / norm_factor
            disparity_center = candidate_range_x - best_candidate_x - offset_x
            disparity_left = disparity_center + 1 if neighbor_xmin == best_candidate_x - 1 else disparity_center
            disparity_right = disparity_left - weight.shape[1] + 1
            disparity_mat = np.repeat(np.linspace(disparity_left, disparity_right, num=weight.shape[1])[:, np.newaxis],
                                      weight.shape[0], axis=1).transpose()
            # print(disparity_mat)
            disparity = np.sum(np.multiply(weight, disparity_mat).reshape(-1))
            # print(disparity)
            depth = (FocalLength * BaseLine) / (disparity + 0.0000001)
            # print(depth)
            depths[i] = depth
        # print(time.time() - start_time)
        '''for i in range(len(points)):
            points[i] = self.cap.pix2cam(points[i])
            points[i] = point_transform(points[i], pose)'''
        '''if 1 in types:
            points[0] = point_transform(points[0], pose)'''
        return bboxes.flatten().tolist(), depths.tolist(), types


left_queue = Queue.Queue(1000)
right_queue = Queue.Queue(1000)
img_time_queue = Queue.Queue(1000)
pose_list = np.array([])
timestamp_list = np.array([])


def left_image_callback(image_msg):
    nparr = np.fromstring(image_msg.data, np.uint8)
    left_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    t = image_msg.header.stamp.to_sec()
    left_queue.put((t, left_image))
    img_time_queue.put(rospy.Time.now())


def right_image_callback(image_msg):
    nparr = np.fromstring(image_msg.data, np.uint8)
    right_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    t = image_msg.header.stamp.to_sec()
    right_queue.put((t, right_image))


def pose_callback(pose_msg):
    global pose_list
    global timestamp_list
    pose_list = np.hstack((pose_list, pose_msg.pose))
    timestamp_list = np.hstack((timestamp_list, pose_msg.header.stamp.to_sec()))
    pose_msg_cp = pose_msg
    pose_msg_cp.header.stamp = rospy.Time.now()
    pose_pub.publish(pose_msg)


def pose_to_numpy(pose):
    '''q_bw_ned = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    position = [pose.position.x, pose.position.y, pose.position.z]
    print(transformations.euler_from_quaternion(q_bw_ned))
    print(transformations.quaternion_matrix(q_bw_ned))
    T_bw_ned = transformations.quaternion_matrix(q_bw_ned)
    T_bw_enu = np.eye(4)
    T_bw_enu[0:3, 0:3] = np.matmul(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), T_bw_ned[0:3, 0:3])
    T_bw = np.dot(transformations.translation_matrix(position), T_bw_enu)
    T_cb = np.eye(4)
    T_cb[0:3, 0:3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
    return np.dot(T_cb, T_bw)'''

    q_ori = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    position = [pose.position.x, pose.position.y, pose.position.z]
    eular = transformations.euler_from_quaternion(q_ori)
    eular_rect = np.array([-eular[2], eular[1], eular[0] + math.pi])
    R_rect = transformations.quaternion_matrix(
        transformations.quaternion_from_euler(-eular[2], eular[1], eular[0] + math.pi))
    R_rect = np.linalg.inv(R_rect)
    T_rect = np.dot(transformations.translation_matrix(position), R_rect)
    return T_rect


def point_transform(point, pose):
    T_pose = pose_to_numpy(pose)
    point_body = np.array([point[2], point[0], -point[1]])
    pose_homo = np.hstack((point_body, [1]))
    # pose_homo = np.dot(np.linalg.inv(T_pose), pose_homo)
    pose_homo = np.dot(T_pose, pose_homo)
    return pose_homo[:-1]


def main():
    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)

    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    capture = Capture(model.insize)
    predictor = Predictor(model=model, cap=capture)

    capture.start()
    predictor.start()

    fps_time = 0
    degree = 0

    main_event = threading.Event()

    try:
        while not main_event.is_set():
            degree += 5
            degree = degree % 360
            try:
                image, feature_map, image_left, image_right, pose = predictor.get()
                humans, confidences = get_humans_by_feature(model, feature_map)
            except Queue.Empty:
                continue
            except Exception:
                break
            predictor.pub_skeleton(humans, confidences, image_left, image_right, pose)
            # predictor.save_skeleton(humans, image_left, image_right, pose)
            # head_depth = predictor.get_head_depth(humans, image_left, image_right)
            pilImg = Image.fromarray(image)
            pilImg = draw_humans(
                model.keypoint_names,
                model.edges,
                pilImg,
                humans,
                mask=mask.rotate(degree) if mask else None
            )
            img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
            msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
            msg += ' ' + config.get('model_param', 'model_name')
            cv2.putText(img_with_humans, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            '''cv2.putText(img_with_humans, 'depth: %lf' % (head_depth),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)'''
            img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
            cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
            fps_time = time.time()
            # press Esc to exit
            if cv2.waitKey(1) == 27:
                main_event.set()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        predictor.text_file.close()
        main_event.set()

    capture.stop()
    predictor.stop()

    capture.join()
    predictor.join()


if __name__ == '__main__':
    rospy.init_node('skeleton_pub', anonymous=True)
    pub = rospy.Publisher('skeleton', Skeleton, queue_size=1000)
    left_sub = rospy.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage, left_image_callback)
    right_sub = rospy.Subscriber('/zed/right/image_rect_color/compressed', CompressedImage, right_image_callback)
    pose_sub = rospy.Subscriber('/mavros/local_position/pose2', PoseStamped, pose_callback)
    pose_pub = rospy.Publisher('/mavros/local_position/pose', PoseStamped, queue_size=1000)
    main()
