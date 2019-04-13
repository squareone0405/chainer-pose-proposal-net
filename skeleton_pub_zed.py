#!/usr/bin/python2

import rospy
import std_msgs
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from ppn.msg import Skeleton
from cv_bridge import CvBridge

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

"""
Bonus script
If you have good USB camera which gets image as well as 60 FPS,
this script will be helpful for realtime inference
"""


class Capture(threading.Thread):
    def _init_zed(self):
        self.zed = sl.Camera()
        self.init = sl.InitParameters()
        self.init.camera_resolution = sl.RESOLUTION.RESOLUTION_VGA
        self.init.camera_fps = 30
        self.init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_NONE
        self.init.coordinate_units = sl.UNIT.UNIT_METER

        err = self.zed.open(self.init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.calibration_params = self.zed.get_camera_information().calibration_parameters

        self.runtime = sl.RuntimeParameters()
        self.runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

    def __init__(self, insize):
        super(Capture, self).__init__()
        self._init_zed()
        self.insize = insize
        self.kx = 1.0 * self.zed.get_resolution().width / insize[0]
        self.ky = 1.0 * self.zed.get_resolution().height / insize[1]
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Capture'

    def pix2cam(self, point):
        x = point[2] * (point[0] - self.calibration_params.left_cam.cx) / self.calibration_params.left_cam.fx
        y = point[2] * (point[1] - self.calibration_params.left_cam.cy) / self.calibration_params.left_cam.fy
        z = point[2]
        return [x, y, z]

    def run(self):
        image_size = self.zed.get_resolution()
        width_zed = image_size.width
        height_zed = image_size.height
        print("ZED will capture at %d x %d" % (width_zed, height_zed))
        image_zed = sl.Mat(width_zed, height_zed, sl.MAT_TYPE.MAT_TYPE_8U_C4)
        while not self.stop_event.is_set():
            try:
                err = self.zed.grab(self.runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image_zed, sl.VIEW.VIEW_SIDE_BY_SIDE, sl.MEM.MEM_CPU,
                                            int(width_zed), int(height_zed))
                    image = image_zed.get_data()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.queue.put(image, timeout=1)
            except Queue.Full:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.zed.close()
        self.stop_event.set()


class Predictor(threading.Thread):

    def __init__(self, model, cap):
        super(Predictor, self).__init__()
        self.cap = cap
        self.model = model
        self.stop_event = threading.Event()
        self.queue = Queue.Queue(QUEUE_SIZE)
        self.name = 'Predictor'

    def run(self):
        while not self.stop_event.is_set():
            try:
                raw_image = self.cap.get()
                image_left = raw_image[:, 0:raw_image.shape[1] / 2, :]
                image_right = raw_image[:, raw_image.shape[1] / 2:, :]
                image = cv2.resize(image_left, self.cap.insize)
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
                self.queue.put((image, feature_map, image_left, image_right), timeout=1)
            except Queue.Full:
                pass
            except Queue.Empty:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()

    def pub_skeleton(self, humans_left, image_left, image_right):
        msg = Skeleton()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = rospy.Time.now()
        msg.human_num = len(humans_left)
        if len(humans_left) > 0:
            skeleton_num = [0] * len(humans_left)
            types = []
            points = []
            for i in range(len(humans_left)):
                skeleton_num[i] = len(humans_left[i]) - 1
                print('skeleton num')
                print(skeleton_num[i])
                if skeleton_num[i] > 0:
                    point_list, type_list = self.get_points(humans_left[i], image_left, image_right)
                    types.extend(type_list)
                    for point in point_list:
                        p = Point()
                        p.x = point[0]
                        p.y = point[1]
                        p.z = point[2]
                        points.append(p)
            msg.skeleton_num = skeleton_num
            msg.types = types
            msg.points = points
            pub.publish(msg)
            print('done pub')

    def get_points(self, human, image_left, image_right):
        BaseLine = 0.12
        FocalLength = self.cap.calibration_params.left_cam.fx

        types = []
        points = np.zeros((len(human) - 1, 3), dtype='float64')
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

        '''plt.imshow(image_left, cmap='gray')
        plt.show()
        plt.imshow(image_right, cmap='gray')
        plt.show()'''

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
                types.append(key)
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

                '''plt.subplot(311)
                plt.imshow(window, cmap='gray')
                plt.subplot(312)
                plt.imshow(target, cmap='gray')
                plt.subplot(313)
                plt.imshow(ssd[idx], cmap='gray')
                plt.show()'''

        print('cpu:' + str(time.time() - start_time))
        out = np.zeros_like(ssd, dtype='int32')
        compute_ssd(window_all, target_all, out, window_width, window_height,
                    target_width, target_height, ssd.shape[2], ssd.shape[1], len(human) - 1)
        ssd = out
        print('cuda ssd:' + str(time.time() - start_time))
        heat_map = ssd.sum(axis=0)
        '''plt.imshow(heat_map, cmap='gray')
        plt.show()'''
        mask = np.ones((mask_size, mask_size), dtype='int32')
        convolved = signal.convolve2d(heat_map, mask, 'valid')
        min_idx = np.argmin(convolved.reshape(-1))
        center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1
        # print(center_x, center_y)

        offset_x = (search_xmax - search_xmin - center_x) * scale_ratio
        offset_y = (search_ymin + center_y) * scale_ratio

        '''round 2'''
        scale_ratio = 1
        image_left = cv2.resize(image_left_ori, (int(image_left_ori.shape[1] / scale_ratio),
                                                 int(image_left_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)
        image_right = cv2.resize(image_right_ori, (int(image_right_ori.shape[1] / scale_ratio),
                                                   int(image_right_ori.shape[0] / scale_ratio)), cv2.INTER_CUBIC)

        kx = kx_ori / scale_ratio
        ky = ky_ori / scale_ratio

        search_xmin = -offset_x - max(4, int(offset_x / 8))
        search_xmax = -offset_x + max(4, int(offset_x / 8))
        search_ymin = offset_y - 3
        search_ymax = offset_y + 3

        # print(search_xmin, search_xmax, search_ymin, search_ymax)

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
                points[idx, 0] = (xmin_f + xmax_f) * kx / 2
                points[idx, 1] = (ymin_f + ymax_f) * ky / 2
                idx = idx + 1

                '''plt.subplot(311)
                plt.imshow(window, cmap='gray')
                plt.subplot(312)
                plt.imshow(target, cmap='gray')
                plt.subplot(313)
                plt.imshow(ssd[idx], cmap='gray')
                plt.show()'''

        print('cpu:' + str(time.time() - start_time))
        out = np.zeros_like(ssd, dtype='int32')
        compute_ssd(window_all, target_all, out, window_width, window_height,
                    target_width, target_height, ssd.shape[2], ssd.shape[1], len(human) - 1)
        ssd = out
        print('cuda ssd:' + str(time.time() - start_time))
        heat_map = ssd.sum(axis=0)
        '''plt.imshow(heat_map, cmap='gray')
        plt.show()'''
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

        for i in range(ssd.shape[0]):  # TODO weighted interpolation
            candidate_ymin = max(center_y - candidate_range_y, 0)
            candidate_ymax = min(center_y + candidate_range_y + 1, ssd.shape[1])
            candidate_xmin = max(center_x - candidate_range_x, 0)
            candidate_xmax = min(center_x + candidate_range_x + 1, ssd.shape[2])
            candidate_rigion = ssd[i, candidate_ymin: candidate_ymax, candidate_xmin: candidate_xmax]
            '''plt.imshow(candidate_rigion, cmap='gray')
            plt.show()'''
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
            points[i, 2] = depth
        print(time.time() - start_time)
        for i in range(len(points)):
            points[i] = self.cap.pix2cam(points[i])
        print(points)
        return points.tolist(), types


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
                image, feature_map, image_left, image_right = predictor.get()
                humans = get_humans_by_feature(model, feature_map)
            except Queue.Empty:
                continue
            except Exception:
                break
            predictor.pub_skeleton(humans, image_left, image_right)
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
            img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
            cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
            fps_time = time.time()
            # press Esc to exit
            if cv2.waitKey(1) == 27:
                main_event.set()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        main_event.set()

    capture.stop()
    predictor.stop()

    capture.join()
    predictor.join()


if __name__ == '__main__':
    rospy.init_node('skeleton_pub', anonymous=True)
    pub = rospy.Publisher('skeleton', Skeleton, queue_size=1000)
    main()
