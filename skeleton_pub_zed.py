#!/usr/bin/python2

import rospy
import std_msgs
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from ppn.msg import Skeleton
from cv_bridge import CvBridge

import matplotlib.pyplot as plt
from scipy import signal

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
        self.init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE
        self.init.coordinate_units = sl.UNIT.UNIT_METER

        err = self.zed.open(self.init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

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
        types = []
        points = np.zeros((len(human) - 1, 3), dtype='float64')
        BaseLine = 0.12
        FocalLength = 350
        start_time = time.time()
        search_range_x = 30
        search_range_y = 3  # pm
        offset_y = 0
        mask_size = 3
        candidate_range_x = 3  # pm
        candidate_range_y = 1  # pm
        interp_range = (0.1, 1)
        ssd = np.zeros((len(human) - 1, search_range_y * 2 + 1, search_range_x + 1), dtype='int32')
        idx = 0
        for key in sorted(human):
            if key != 0:
                types.append(key)
                ymin_f, xmin_f, ymax_f, xmax_f = human[key]  # TODO range check
                ymin = int(ymin_f * self.cap.ky) if int(ymin_f * self.cap.ky) > 0 else 0
                xmin = int(xmin_f * self.cap.kx) if int(xmin_f * self.cap.kx) > 0 else 0
                ymax = int(ymax_f * self.cap.ky) if int(ymax_f * self.cap.ky) < image_left.shape[0] \
                    else image_left.shape[0]
                xmax = int(xmax_f * self.cap.kx) if int(xmax_f * self.cap.kx) < image_left.shape[1] \
                    else image_left.shape[1]
                window = image_left[ymin:ymax, xmin:xmax]

                target = np.full((window.shape[0] + search_range_y * 2, window.shape[1] + search_range_x), 512)
                target_ymin = offset_y + ymin - search_range_y if offset_y + ymin - search_range_y > 0 else 0
                target_ymax = offset_y + ymax + search_range_y if offset_y + ymax + search_range_y < \
                                                                  image_right.shape[0] else image_right.shape[0]
                target_xmin = xmin - search_range_x if xmin - search_range_x > 0 else 0
                target_xmax = xmax
                target_from_right = image_right[target_ymin:target_ymax, target_xmin:target_xmax]
                clip_ymin = 0 if offset_y + ymin - search_range_y > 0 else -(offset_y + ymin - search_range_y)
                clip_ymax = 0 if offset_y + ymax + search_range_y < image_right.shape[0] \
                    else (offset_y + ymax + search_range_y) - image_right.shape[0]

                target[clip_ymin: target.shape[0] - clip_ymax, target.shape[1] - (target_xmax - target_xmin):] = \
                    target_from_right

                for i in range(search_range_y * 2 + 1):
                    for j in range(search_range_x + 1):
                        diff = target[i: i + window.shape[0], j: j + window.shape[1]] - window
                        ssd[idx, i, j] = np.sum(np.multiply(diff, diff).reshape(-1))

                '''plt.subplot(311)
                plt.imshow(window, cmap='gray')
                plt.subplot(312)
                plt.imshow(target, cmap='gray')
                plt.subplot(313)
                plt.imshow(ssd[idx], cmap='gray')
                plt.show()'''
                points[idx, 0] = (xmin_f + xmax_f) * self.cap.kx / 2
                points[idx, 1] = (ymin_f + ymax_f) * self.cap.ky / 2
                idx = idx + 1
        heat_map = ssd.sum(axis=0)
        '''plt.imshow(heat_map, cmap='gray')
        plt.show()'''
        mask = np.ones((mask_size, mask_size), dtype='int32')
        convolved = signal.convolve2d(heat_map, mask, 'valid')
        min_idx = np.argmin(convolved.reshape(-1))
        center_x, center_y = min_idx % convolved.shape[1] + 1, min_idx / convolved.shape[1] + 1

        for i in range(ssd.shape[0]):  # TODO weighted interpolation
            candidate_ymin = center_y - candidate_range_y if center_y - candidate_range_y > 0 else 0
            candidata_ymax = center_y + candidate_range_y + 1 if center_y + candidate_range_y + 1 < ssd.shape[1] \
                else ssd.shape[1]
            candidata_xmin = center_x - candidate_range_x if center_x - candidate_range_x > 0 else 0
            candidata_xmax = center_x + candidate_range_x + 1 if center_x + candidate_range_x + 1 < ssd.shape[2] \
                else ssd.shape[2]
            candidate_rigion = ssd[i, candidate_ymin: candidata_ymax, candidata_xmin: candidata_xmax]

            '''plt.imshow(candidate_rigion, cmap='gray')
            plt.show()'''
            best_candidate_x = (np.argmin(candidate_rigion.reshape(-1)) % (candidate_rigion.shape[1]))
            best_candidate_y = (np.argmin(candidate_rigion.reshape(-1)) / (candidate_rigion.shape[1]))
            neignbor_xmin = best_candidate_x - 1 if best_candidate_x - 1 > 0 else 0
            neignbor_xmax = best_candidate_x + 2 if best_candidate_x + 2 < candidate_rigion.shape[1] \
                else candidate_rigion.shape[1]
            neignbor_ymin = best_candidate_y - 1 if best_candidate_y - 1 > 0 else 0
            neignbor_ymax = best_candidate_y + 2 if best_candidate_y + 2 < candidate_rigion.shape[0] \
                else candidate_rigion.shape[0]
            min_neignbor = candidate_rigion[neignbor_ymin: neignbor_ymax, neignbor_xmin: neignbor_xmax]
            min_neignbor = np.interp(min_neignbor, (np.min(min_neignbor.reshape(-1)), np.max(min_neignbor.reshape(-1))),
                                     interp_range)
            weight = 1.0 / min_neignbor
            norm_factor = np.sum(weight.reshape(-1))
            weight = weight / norm_factor
            disparity_center = search_range_x - (center_x - candidate_range_x + best_candidate_x)
            disparity_left = disparity_center + 1 if disparity_center + 1 < search_range_x else disparity_center
            disparity_right = disparity_left - weight.shape[1] + 1 if disparity_left - weight.shape[1] + 1 >= 0 else 0
            disparity_mat = np.repeat(np.linspace(disparity_left, disparity_right, num=weight.shape[1])[:, np.newaxis],
                                      weight.shape[0], axis=1).transpose()
            disparity = np.sum(np.multiply(weight, disparity_mat).reshape(-1))
            depth = (FocalLength * BaseLine) / (disparity + 0.0000001)
            points[i, 2] = depth
        print(time.time() - start_time)  # TODO speed up with cuda
        print('returning points***********************')
        print(points)
        print(types)
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

    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))

    capture = Capture(model.insize)
    predictor = Predictor(model=model, cap=capture)

    capture.start()
    predictor.start()

    fps_time = 0
    degree = 0

    main_event = threading.Event()

    try:
        while not main_event.is_set() and cap.isOpened():
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
